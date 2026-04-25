"""
Alert Enrichment — NetBox Lookup + Enrichment Logic
===================================================
Handles device resolution, tag checking, and context enrichment.
Designed for batch lookups to minimize NetBox API calls.
"""

import os
import sys
import json
import subprocess
from dataclasses import dataclass, field
from typing import Optional
from functools import lru_cache

# ────────────────────────────────────────────────────────────────
# MCP PORTER — NetBox access
# ────────────────────────────────────────────────────────────────

MCPORTER = "/home/jourdan/.npm-global/bin/mcporter"
NETBOX_SERVER = "netbox-mcp"

# Tags that escalate severity
CRITICAL_TAGS = {"core_device", "customer_critical_site"}

# Severity priority (lower = more severe)
PRIORITY_RANK = {"critical": 0, "high": 1, "average": 2, "warning": 3, "info": 4}
OPSGENIE_RANK = {"P1": 0, "P2": 1, "P3": 2, "P4": 3, "P5": 4}


def mcporter_call(tool: str, args: dict, timeout: int = 30) -> dict:
    """
    Call a NetBox MCP tool via mcporter.
    Uses 'key=value' flag syntax to avoid shell JSON-escaping issues.
    """
    import json as _json

    # Build flag-style arguments
    cmd_parts = [MCPORTER, "call", f"{NETBOX_SERVER}.{tool}"]
    for k, v in args.items():
        if isinstance(v, dict):
            # Dicts must be JSON-encoded as strings for key=value syntax
            cmd_parts.append(f"{k}={_json.dumps(v)}")
        elif isinstance(v, list):
            cmd_parts.append(f"{k}={_json.dumps(v)}")
        elif isinstance(v, str):
            cmd_parts.append(f"{k}={v}")
        elif isinstance(v, bool):
            cmd_parts.append(f"{k}={str(v).lower()}")
        elif v is None:
            pass  # skip None values
        else:
            cmd_parts.append(f"{k}={v}")

    result = subprocess.run(
        cmd_parts,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        print(f"[NetBox] ERROR calling {tool}: {result.stderr[:200]}", flush=True)
        return {}
    try:
        response = _json.loads(result.stdout)
        # NetBox list endpoints return {count, results, next, previous}
        # Extract results for convenience
        if isinstance(response, dict) and "results" in response:
            return response["results"]
        return response
    except Exception:
        return {}


# ────────────────────────────────────────────────────────────────
# DEVICE LOOKUP
# ────────────────────────────────────────────────────────────────

@dataclass
class NbDevice:
    """Normalized NetBox device record."""
    id: int
    name: str
    status: str
    role: str
    role_slug: str
    site: str
    site_slug: str
    tags: list[str]
    device_type: str = ""
    platform: str = ""
    serial: str = ""
    ip_address: str = ""

    @property
    def is_critical(self) -> bool:
        return bool(CRITICAL_TAGS & set(self.tags))

    @property
    def severity_escalation(self) -> str:
        """Return escalation multiplier: critical > high > medium."""
        if "core_device" in self.tags:
            return "critical"
        if "customer_critical_site" in self.tags:
            return "high"
        return "none"


def netbox_lookup(hostname: str) -> Optional[NbDevice]:
    """
    Resolve a hostname (Zabbix alias) → NetBox device record.
    Uses netbox_search_objects with field projection for minimal payload.
    Returns None if device not found in NetBox.
    """
    results = mcporter_call("netbox_search_objects", {
        "query": hostname,
        "object_types": ["dcim.device"],
        "fields": ["id", "name", "status", "role", "site", "tags"],
        "limit": 5
    })

    devices = results.get("dcim.device", [])
    if not devices:
        print(f"[NetBox] No device found for: {hostname}", flush=True)
        return None

    # Prefer exact name match, else take first result
    exact = next((d for d in devices if d.get("name") == hostname), None)
    d = exact or devices[0]

    # Resolve full device record with tags
    full = netbox_get_device(d["id"])
    return full


def netbox_get_device(device_id: int) -> Optional[NbDevice]:
    """Get full device record by ID with all needed fields."""
    # Use field projection to minimize payload
    fields = [
        "id", "name", "status", "serial",
        "role", "site", "tags",
        "device_type", "platform",
        "primary_ip"
    ]
    result = mcporter_call("netbox_get_object_by_id", {
        "object_type": "dcim.device",
        "object_id": device_id,
        "fields": fields
    })

    if not result or result.get("id") is None:
        return None

    # Extract role and site (they're nested objects)
    role_obj = result.get("role", {}) or {}
    site_obj = result.get("site", {}) or {}
    tags_raw = result.get("tags", []) or []
    device_type_obj = result.get("device_type", {}) or {}
    platform_obj = result.get("platform", {}) or {}
    primary_ip_obj = result.get("primary_ip", {}) or {}

    tags = [t.get("slug") or t.get("name", "").lower().replace(" ", "-")
            for t in tags_raw]

    ip_address = ""
    if primary_ip_obj:
        ip_address = primary_ip_obj.get("address", "")

    return NbDevice(
        id=result["id"],
        name=result.get("name", ""),
        status=result.get("status", {}).get("value", "unknown"),
        role=role_obj.get("display", role_obj.get("name", "")),
        role_slug=role_obj.get("slug", ""),
        site=site_obj.get("display", site_obj.get("name", "")),
        site_slug=site_obj.get("slug", ""),
        tags=tags,
        device_type=device_type_obj.get("display", ""),
        platform=platform_obj.get("slug", ""),
        serial=result.get("serial", ""),
        ip_address=ip_address,
    )


def _extract_id(ref) -> Optional[int]:
    """Extract integer ID from a NetBox reference dict or bare int/str."""
    if ref is None:
        return None
    if isinstance(ref, int):
        return ref
    if isinstance(ref, str):
        try:
            return int(ref)
        except ValueError:
            return None
    if isinstance(ref, dict):
        val = ref.get("id")
        if isinstance(val, int):
            return val
        if isinstance(val, str):
            try:
                return int(val)
            except ValueError:
                return None
    return None


def get_connected_devices(device_id: int) -> list[dict]:
    """Get devices connected via cables (from NetBox interfaces).

    Uses link_peers (NetBox 3.5+ native connectivity) when available,
    with cable-tracing as fallback. link_peers avoids extra API calls.
    """
    interfaces = mcporter_call("netbox_get_objects", {
        "object_type": "dcim.interface",
        "filters": {"device_id": device_id},
        "limit": 50
    })
    results = interfaces if isinstance(interfaces, list) else []
    if not results:
        return []

    connected = []
    seen = set()

    for iface in results:
        # Try link_peers first (NetBox 3.5+ — no extra API call needed)
        link_peers = iface.get("link_peers", []) or []
        for peer in link_peers:
            peer_device_ref = peer.get("device", {})
            peer_id = _extract_id(peer_device_ref)
            if peer_id and peer_id != device_id and peer_id not in seen:
                seen.add(peer_id)
                connected.append({
                    "device_id": peer_id,
                    "device_name": peer.get("device", {}).get("display") or peer.get("device", {}).get("name", ""),
                    "interface": peer.get("name", ""),
                })

        # Fallback: trace via cable
        cable_ref = iface.get("cable")
        cable_id = _extract_id(cable_ref)
        if not cable_id:
            continue

        cable = mcporter_call("netbox_get_object_by_id", {
            "object_type": "dcim.cable",
            "object_id": cable_id
        })
        if not cable:
            continue

        peer = _resolve_cable_peer(cable, device_id)
        if peer and peer["device_id"] not in seen:
            seen.add(peer["device_id"])
            connected.append(peer)

    return connected


def _resolve_cable_peer(cable: dict, my_device_id: int) -> Optional[dict]:
    """
    Resolve the device on the other end of a cable.
    NetBox Cable has a_terminations and b_terminations (lists of termination objects).
    Each termination has: {object_type, object_id, object: {device, name, ...}}
    """
    for side in ["a", "b"]:
        my_terminations = cable.get(f"{side}_terminations", [])
        peer_terminations = cable.get(f"{'b' if side == 'a' else 'a'}_terminations", [])

        # Find which side is our device
        my_iface = None
        for term in my_terminations:
            obj = term.get("object", {})
            if obj.get("device", {}).get("id") == my_device_id:
                my_iface = obj
                break

        if not my_iface:
            continue

        # Get the peer termination
        for term in peer_terminations:
            obj = term.get("object", {})
            peer_device_ref = obj.get("device")
            if not peer_device_ref:
                continue
            peer_device_id = peer_device_ref.get("id")
            if peer_device_id is None:
                continue

            # Skip self
            if peer_device_id == my_device_id:
                continue

            # Resolve peer device
            peer_device = netbox_get_device(peer_device_id)
            if peer_device:
                return {
                    "device_id": peer_device.id,
                    "device_name": peer_device.name,
                    "device_role": peer_device.role,
                    "device_site": peer_device.site,
                    "interface": obj.get("display", obj.get("name", "unknown")),
                }
    return None


def get_cables(device_id: int) -> list[dict]:
    """Get all cables connected to a device."""
    interfaces = mcporter_call("netbox_get_objects", {
        "object_type": "dcim.interface",
        "filters": {"device_id": device_id},
        "fields": ["id", "name", "cable"],
        "limit": 50
    })
    # mcporter_call already unwraps {results: [...]} → [...]
    results = interfaces if isinstance(interfaces, list) else []
    cables = []
    seen = set()
    for iface in results:
        cable_ref = iface.get("cable")
        if not cable_ref:
            continue
        cable_id = cable_ref.get("id")
        if not cable_id or cable_id in seen:
            continue
        seen.add(cable_id)

        # Fetch full cable record (no field projection — NetBox drops nested data)
        cable = mcporter_call("netbox_get_object_by_id", {
            "object_type": "dcim.cable",
            "object_id": cable_id,
        })
        if not cable:
            continue

        # Get my interface name
        my_iface_name = iface.get("name", "")
        peer_iface_name = ""
        peer_device_name = ""

        # Find peer termination
        for side in ["a", "b"]:
            my_terms = cable.get(f"{side}_terminations", [])
            peer_terms = cable.get(f"{'b' if side == 'a' else 'a'}_terminations", [])
            for term in my_terms:
                obj = term.get("object", {})
                if obj.get("device", {}).get("id") == device_id:
                    my_iface_name = obj.get("display", obj.get("name", my_iface_name))
                    for pterm in peer_terms:
                        pobj = pterm.get("object", {})
                        pd = pobj.get("device", {})
                        if pd.get("id") and pd.get("id") != device_id:
                            peer_iface_name = pobj.get("display", "unknown")
                            peer_device_name = pd.get("name", "unknown")
                            break
                    break

        def _get(obj, key, default=None):
            v = obj.get(key, default) if isinstance(obj, dict) else default
            return v

        cables.append({
            "id": cable_id,
            "label": cable.get("label", ""),
            "type": _get(cable.get("type"), "value", str(cable.get("type", "")) if not isinstance(cable.get("type"), dict) else ""),
            "status": _get(cable.get("status"), "value", "connected"),
            "interface": my_iface_name,
            "peer_device": peer_device_name,
            "peer_interface": peer_iface_name,
        })
    return cables


# ────────────────────────────────────────────────────────────────
# BATCH LOOKUP (for enrichment pipeline)
# ────────────────────────────────────────────────────────────────

def batch_lookup_devices(hostnames: list[str]) -> dict[str, Optional[NbDevice]]:
    """
    Batch lookup multiple hostnames → NetBox devices.
    One NetBox search per unique hostname.
    Returns {hostname: NbDevice or None}
    """
    results = {}
    for hostname in set(hostnames):
        results[hostname] = netbox_lookup(hostname)
    return results


# ────────────────────────────────────────────────────────────────
# SEVERITY COMPUTATION
# ────────────────────────────────────────────────────────────────

def compute_severity(
    alert_severity: str,
    device: Optional[NbDevice],
    is_opsgenie: bool = False
) -> str:
    """
    Compute final severity with device tag escalation.
    - Critical-tagged devices escalate P2→P1, P3→P2
    - Returns mock lab severity (critical/high/average/warning/info)
    """
    if is_opsgenie:
        # Convert Opsgenie P1-P5 to mock severity
        alert_severity = _opsgenie_to_mock(alert_severity)

    if device and device.is_critical:
        if alert_severity == "high":
            return "critical"
        if alert_severity == "average":
            return "high"

    return alert_severity


def _opsgenie_to_mock(priority: str) -> str:
    mapping = {"P1": "critical", "P2": "high", "P3": "average", "P4": "warning", "P5": "info"}
    return mapping.get(priority, "average")


def compute_delivery(severity: str) -> str:
    """Map severity to delivery mode."""
    delivery = {
        "critical": "immediate",
        "high": "2min",
        "average": "digest_15m",
        "warning": "digest_1h",
        "info": "jira_only",
    }
    return delivery.get(severity, "digest_15m")


# ────────────────────────────────────────────────────────────────
# ALERT TYPE DETECTION (from message text)
# ────────────────────────────────────────────────────────────────

def detect_alert_types(alerts: list) -> set[str]:
    """Infer alert types from a list of alert records or raw dicts."""
    types = set()
    for alert in alerts:
        msg = ""
        if isinstance(alert, dict):
            msg = f"{alert.get('message', '')} {alert.get('description', '')}".lower()
            alert_type = alert.get("type", alert.get("alert_type", ""))
        else:
            msg = " ".join(
        getattr(a, "message", "") or getattr(a, "description", "") or ""
        for a in alerts
    ).lower()
            alert_type = alert.alert_type if hasattr(alert, "alert_type") else ""

        if alert_type:
            types.add(alert_type)
            continue

        # Fallback: infer from message
        if "power" in msg or "ps0" in msg or "ps1" in msg:
            types.add("power-alert")
        elif "bgp" in msg or "neighbor" in msg:
            types.add("bgp-alert")
        elif "interface" in msg or "eth" in msg or "port" in msg:
            types.add("interface-alert")
        elif "cpu" in msg:
            types.add("cpu-alert")
        elif "memory" in msg:
            types.add("memory-alert")
        elif "temperature" in msg or "temp" in msg:
            types.add("environment-alert")
        elif "link" in msg or "cable" in msg:
            types.add("link-alert")
        else:
            types.add("unknown-alert")
    return types


# ────────────────────────────────────────────────────────────────
# TEST
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[Test] Looking up DC1-CORE-RTR-01...")
    device = netbox_lookup("DC1-CORE-RTR-01")
    if device:
        print(f"  Name: {device.name}")
        print(f"  Role: {device.role}")
        print(f"  Site: {device.site}")
        print(f"  Tags: {device.tags}")
        print(f"  Is Critical: {device.is_critical}")
        print(f"  Severity Escalation: {device.severity_escalation}")
    else:
        print("  NOT FOUND")

    print("\n[Test] Looking up DC1-CORE-01...")
    device2 = netbox_lookup("DC1-CORE-01")
    if device2:
        print(f"  Name: {device2.name}")
        print(f"  Tags: {device2.tags}")
        print(f"  Is Critical: {device2.is_critical}")
        print(f"  Connected devices: {get_connected_devices(device2.id)}")
