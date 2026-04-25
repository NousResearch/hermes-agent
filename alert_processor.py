"""
Alert Enrichment — Integrated Alert Processor
=============================================
Single process that:
  1. Receives alerts (from Flask webhook, or direct function call)
  2. Looks up device context via NetBox MCP (mcporter)
  3. Passes raw NetBox output + alert to MiniMax LLM for interpretation
  4. Sends enriched briefing to Telegram

Architecture:
  [Alert] → [Hermes/Gateway] → [Here: alert_processor.py] → [NetBox MCP] → [MiniMax LLM] → [Telegram]
                (webhook)              (this file)            (context)       (synthesis)
"""

import os
import json
import subprocess
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Optional

# Load .env file manually — NOT auto-loaded by Python.
# This is needed because:
#   1. Flask/webhook_receiver.py imports alert_processor.py as a module.
#      The calling process (Flask/ssh/cron) does NOT automatically source ~/.hermes/.env.
#   2. The mcporter subprocess and urllib calls to MiniMax inherit os.environ,
#      NOT shell variables. If the key isn't in os.environ, it reads as empty.
# Fix: parse ~/.hermes/.env ourselves at import time (lines 22-30).
_env_path = os.path.expanduser("~/.hermes/.env")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)


# ─────────────────────────────────────────────────────────────────────────────
# NOC PIPELINE — imports and LAB_MGMT_IPS guard
# ─────────────────────────────────────────────────────────────────────────────
# noc_md_processor: writes ~/noc/inbox/{alert_id}.json → ~/noc/context/{alert_id}.md
try:
    from noc_md_processor import process_alert_from_inbox
    _NOC_MD_AVAILABLE = True
except ImportError:
    _NOC_MD_AVAILABLE = False
    print("[AlertProcessor] noc_md_processor not available — NOC context pipeline disabled", flush=True)

# LAB_MGMT_IPS: explicit allow-list of management IPs for lab routers safe to SSH.
LAB_MGMT_IPS: set = set()

# router_diagnostics: AI agent that SSHes to lab routers and runs read-only commands.
try:
    from router_diagnostics import run_router_diagnostics, format_diagnostic_report, send_diagnostic_telegram
except ImportError:
    run_router_diagnostics = None
    format_diagnostic_report = None
    send_diagnostic_telegram = None

# ────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8764046749:AAHOX8PsdHiAFiUrzSD8LgDUFDd44zRBbCA")
HERMES_ALERTS_CHANNEL_ID = os.environ.get("HERMES_ALERTS_CHANNEL_ID", "-1003506715170")
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
NETBOX_SERVER = "netbox-mcp"
# ────────────────────────────────────────────────────────────────
# DATA MODELS
# ────────────────────────────────────────────────────────────────

@dataclass
class AlertRecord:
    """A normalized alert from any source (mock lab, Opsgenie, Zabbix, etc.)."""
    alert_id: str
    device: str                       # hostname — key for NetBox lookup
    alert_type: str
    severity: str
    message: str
    site: str = ""
    timestamp: str = ""
    raw_payload: dict = field(default_factory=dict)
    is_mock_lab: bool = False
    # Additional context from source (adjacent/nested data from the alert itself)
    netbox_context: dict = field(default_factory=dict)   # already-enriched context from source
    netbox_impact: dict = field(default_factory=dict)    # impact info from source
    netbox_cables: list = field(default_factory=list)   # cable info from source
    metrics: dict = field(default_factory=dict)         # live metrics from source

    @classmethod
    def from_mock_lab(cls, alert: dict) -> "AlertRecord":
        """Parse mock lab /simulate/* format."""
        nb_ctx = alert.get("netbox_context", {})
        nb_impact = alert.get("netbox_impact", {})
        nb_cable = alert.get("netbox_cable")
        return cls(
            alert_id=alert.get("alert_id", ""),
            device=alert.get("device", alert.get("netbox_device", "")),
            alert_type=alert.get("type", alert.get("alert_type", "unknown")),
            severity=alert.get("severity", "unknown"),
            message=alert.get("description", alert.get("message", "")),
            site=alert.get("site", nb_ctx.get("site", "")),
            timestamp=alert.get("timestamp", ""),
            raw_payload=alert,
            is_mock_lab=True,
            netbox_context=nb_ctx,
            netbox_impact=nb_impact,
            netbox_cables=[nb_cable] if nb_cable else [],
            metrics=alert.get("metrics", {}),
        )

    @classmethod
    def from_opsgenie(cls, alert: dict, action: str) -> "AlertRecord":
        """Parse Opsgenie webhook payload."""
        msg = f"{alert.get('message', '')} {alert.get('description', '')}"
        return cls(
            alert_id=alert.get("alertID", ""),
            device=alert.get("alias", msg.split()[0] if msg else "unknown"),
            alert_type=_detect_alert_type(msg),
            severity=_opsgenie_priority_to_mock(alert.get("priority", "P3")),
            message=alert.get("description", alert.get("message", "")),
            site="",  # Will resolve via NetBox
            timestamp=alert.get("updatedAt", ""),
            raw_payload=alert,
            is_mock_lab=False,
        )

    @classmethod
    def from_checkmk(cls, payload: dict) -> "AlertRecord":
        """
        Parse Checkmk alert handler webhook payload.

        Checkmk POSTs application/x-www-form-urlencoded with fields:
          context=<JSON>   — JSON with host/service alert details
          host_name        — hostname
          service_description — service name (empty for host alerts)
          event_id         — unique event ID
          (plus other flat fields)

        The Flask receiver normalises form-encoded data into a flat dict
        (request.form access), and also extracts the parsed JSON from the
        'context' field so it appears as a nested dict in payload.
        """
        import json as _json

        # 'context' is a JSON string in form-encoded payloads; normalize to dict
        ctx = payload.get("context", {})
        if isinstance(ctx, str):
            try:
                ctx = _json.loads(ctx)
            except Exception:
                ctx = {}

        host = ctx.get("host_name", payload.get("host_name", ""))
        service = ctx.get("service_description", payload.get("service_description", ""))

        # Checkmk state: 0=OK, 1=WARN, 2=CRIT, 3=UNKNOWN
        raw_state = ctx.get("service_state", ctx.get("host_state", "UNKNOWN"))
        try:
            state_num = int(raw_state)
        except (ValueError, TypeError):
            state_num = 3
        severity_map = {0: "info", 1: "warning", 2: "critical", 3: "warning"}
        severity = severity_map.get(state_num, "warning")

        # Build a human-readable message
        svc_output = ctx.get("service_output", ctx.get("host_output", ""))
        if service:
            message = f"[{service}] {svc_output}"
        else:
            message = svc_output

        # Attempt number (e.g. "1/3")
        attempt = ctx.get("attempt", payload.get("attempt", ""))
        if attempt and "/" not in str(attempt):
            attempt = f"{attempt}/1"

        # Unix timestamp from Checkmk
        ts = ctx.get("time", payload.get("time", ""))
        if ts:
            import datetime
            try:
                ts = datetime.datetime.fromtimestamp(int(ts)).isoformat()
            except Exception:
                pass

        return cls(
            alert_id=payload.get("event_id", f"checkmk-{host}-{service or 'host'}"),
            device=host,
            alert_type=_detect_alert_type(service or host),
            severity=severity,
            message=message,
            site="",  # Resolve via NetBox
            timestamp=str(ts),
            raw_payload=payload,
            is_mock_lab=False,
        )


def _detect_alert_type(msg: str) -> str:
    msg_lower = msg.lower()
    if "power" in msg_lower or "ps0" in msg_lower or "ps1" in msg_lower:
        return "power-alert"
    if "bgp" in msg_lower or "neighbor" in msg_lower:
        return "bgp-alert"
    if "interface" in msg_lower or "eth" in msg_lower or "port" in msg_lower or " GigabitEthernet" in msg_lower:
        return "interface-alert"
    if "cpu" in msg_lower:
        return "cpu-alert"
    if "memory" in msg_lower or "mem" in msg_lower:
        return "memory-alert"
    if "temperature" in msg_lower or "temp" in msg_lower or "thermal" in msg_lower:
        return "environment-alert"
    if "link" in msg_lower or "cable" in msg_lower:
        return "link-alert"
    if "device.*down" in msg_lower or "device-down" in msg_lower or "unreachable" in msg_lower:
        return "device-down-alert"
    if "hardware" in msg_lower or "fan" in msg_lower or "psu" in msg_lower:
        return "hardware-alert"
    if "ospf" in msg_lower or "isis" in msg_lower or "eigrp" in msg_lower:
        return "routing-alert"
    if "stp" in msg_lower or "spanning" in msg_lower:
        return "stp-alert"
    if "ntp" in msg_lower or "clock" in msg_lower or "sync" in msg_lower:
        return "ntp-alert"
    if "disk" in msg_lower or "storage" in msg_lower or "raid" in msg_lower:
        return "storage-alert"
    if "acl" in msg_lower or "firewall" in msg_lower or "policy" in msg_lower:
        return "security-alert"
    return "unknown-alert"


def _opsgenie_priority_to_mock(priority: str) -> str:
    return {"P1": "critical", "P2": "high", "P3": "average", "P4": "warning", "P5": "info"}.get(priority, "average")


# ────────────────────────────────────────────────────────────────
# NETBOX MCP (mcporter)
# ────────────────────────────────────────────────────────────────

# mcporter is a CLI bridge for MCP (Model Context Protocol) servers.
# It connects to forge:8082 where the MCP server runs, and exposes tools as CLI subcommands.
#
# WHY CLI vs NATIVE MCP:
#   - Native MCP via config.yaml had breakage (google-workspace broken, path issues).
#   - mcporter is more reliable and has 121 tools available.
#   - mcporter is already connected and authenticated for NetBox.
#
# TOOL NAME FORMAT:
#   mcporter call <server>.<tool>
#   server name = "netbox-mcp"  (logical name, not hostname — maps to the MCP server config)
#   tool       = e.g. "netbox_search_objects", "netbox_get_object_by_id"
#
# MCPorter binary location: /home/jourdan/.npm-global/bin/mcporter
MCPORTER = "/home/jourdan/.npm-global/bin/mcporter"


def mcporter_call(tool: str, args: dict, timeout: int = 30) -> dict:
    """
    Call a NetBox MCP tool via mcporter CLI subprocess.

    WHY subprocess:
    # mcporter uses stdio transport — we spawn it as a CLI command with JSON arguments.
    # The subprocess inherits os.environ from this Python process (which we seeded
    # with vars from ~/.hermes/.env at import time). If env vars were missing,
    # mcporter would also fail to authenticate.

    NETBOX TOOL CONVENTIONS:
    - Search:     netbox_search_objects  → returns {"results": [...], "dcim.device": [...]}
    - Get one:    netbox_get_object_by_id → returns single object dict
    - List many:  netbox_get_objects      → returns [obj, obj, ...] or {"results": [...]}

    ERROR BEHAVIOUR:
    - If return code != 0: print error to stderr, return empty dict {}
    - If JSON parse fails: return empty dict {}
    - If "results" key present in response: unwrap and return the list (normalizes list endpoints)
    - Empty {} is treated as "not found" by all callers — they fall back gracefully.
    """
    cmd_parts = [MCPORTER, "call", f"{NETBOX_SERVER}.{tool}"]
    for k, v in args.items():
        if isinstance(v, (dict, list)):
            cmd_parts.append(f"{k}={json.dumps(v)}")
        elif isinstance(v, bool):
            cmd_parts.append(f"{k}={str(v).lower()}")
        elif v is not None:
            cmd_parts.append(f"{k}={v}")

    result = subprocess.run(cmd_parts, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        print(f"[NetBox MCP] ERROR {tool}: {result.stderr[:200]}", flush=True)
        return {}
    try:
        resp = json.loads(result.stdout)
        # Unwrap {results: [...]} from list endpoints
        if isinstance(resp, dict) and "results" in resp:
            return resp["results"]
        return resp
    except Exception:
        return {}


def netbox_lookup_device(hostname: str) -> dict:
    """
    Look up a device in NetBox.
    Returns raw NetBox dict — pass directly to LLM for interpretation.
    """
    results = mcporter_call("netbox_search_objects", {
        "query": hostname,
        "object_types": ["dcim.device"],
        "fields": ["id", "name", "status", "role", "site", "tags", "device_type", "platform", "serial", "primary_ip"],
        "limit": 5
    })

    devices = results.get("dcim.device", []) if isinstance(results, dict) else results
    if not devices:
        return {}

    # Prefer exact name match (case-insensitive)
    hostname_lower = hostname.lower()
    exact = next((d for d in devices if (d.get("name") or "").lower() == hostname_lower), None)
    device = exact or (devices[0] if devices else {})

    device_id = device.get("id")
    if not device_id:
        return {}

    # Get full device record with all fields
    full = mcporter_call("netbox_get_object_by_id", {
        "object_type": "dcim.device",
        "object_id": device_id,
        "fields": [
            "id", "name", "status", "serial", "role", "site", "tags",
            "device_type", "platform", "primary_ip", "vcpus", "memory",
            "config_context", "comments"
        ]
    })
    return full if full else device


# ────────────────────────────────────────────────────────────────
# VM LOOKUP  (virtualization.virtualmachine via NetBox MCP)
# ────────────────────────────────────────────────────────────────

def netbox_lookup_vm(hostname: str) -> dict:
    """
    Look up a VM in NetBox's virtualization.virtualmachine table.
    Returns full VM record dict, or {} if not found.

    VM notes from NetBox mock lab (2026-04-20):
      - VMs live in clusters (not sites) — cluster provides DC context
      - role/platform/tenant are often null on VMs
      - primary_ip is usually null unless manually assigned
      - vcpus/memory are populated; disk is often null
    """
    hostname_lower = hostname.lower()

    # Search VMs
    results = mcporter_call("netbox_search_objects", {
        "query": hostname,
        "object_types": ["virtualization.virtualmachine"],
        "fields": ["id", "name", "status", "cluster", "site", "tags", "role", "vcpus", "memory"],
        "limit": 5
    })

    vms = results.get("virtualization.virtualmachine", []) if isinstance(results, dict) else results
    if not vms:
        return {}

    # Prefer exact name match (case-insensitive)
    exact = next((v for v in vms if (v.get("name") or "").lower() == hostname_lower), None)
    vm = exact or (vms[0] if vms else {})

    vm_id = vm.get("id")
    if not vm_id:
        return {}

    # Get full VM record
    full = mcporter_call("netbox_get_object_by_id", {
        "object_type": "virtualization.virtualmachine",
        "object_id": vm_id,
        "fields": [
            "id", "name", "status", "serial", "site", "cluster", "role",
            "platform", "tenant", "tags", "vcpus", "memory", "disk",
            "primary_ip", "primary_ip4", "primary_ip6",
            "config_context", "comments", "description"
        ]
    })

    return full if full else vm


def netbox_lookup_host(hostname: str) -> dict:
    """
    Unified host lookup — tries NetBox dcim.device first, falls back to
    virtualization.virtualmachine.  Returns a raw NetBox record dict (device
    or VM) with an added field _host_type = "device" | "vm" so callers can
    distinguish which table the result came from.
    """
    # Try device first
    dev = netbox_lookup_device(hostname)
    if dev and dev.get("id"):
        dev["_host_type"] = "device"
        return dev

    # Fall back to VM
    vm = netbox_lookup_vm(hostname)
    if vm and vm.get("id"):
        vm["_host_type"] = "vm"
        return vm

    return {}


def netbox_get_connected_devices(device_id: int) -> list[dict]:
    """Get devices connected via cables/interfaces. Returns raw list — pass to LLM."""
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
        # Try link_peers first (NetBox 3.5+)
        link_peers = iface.get("link_peers", []) or []
        for peer in link_peers:
            peer_dev = peer.get("device", {})
            peer_id = _extract_id(peer_dev)
            if peer_id and peer_id != device_id and peer_id not in seen:
                seen.add(peer_id)
                connected.append({
                    "device_id": peer_id,
                    "device_name": peer_dev.get("display") or peer_dev.get("name", ""),
                    "interface": peer.get("name", ""),
                    "interface_type": peer.get("type", {}).get("value", ""),
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


def netbox_get_cables(device_id: int) -> list[dict]:
    """Get all cables connected to a device. Returns raw list — pass to LLM."""
    interfaces = mcporter_call("netbox_get_objects", {
        "object_type": "dcim.interface",
        "filters": {"device_id": device_id},
        "fields": ["id", "name", "cable"],
        "limit": 50
    })
    results = interfaces if isinstance(interfaces, list) else []
    cables = []
    seen = set()

    for iface in results:
        cable_ref = iface.get("cable")
        if not cable_ref:
            continue
        cable_id = _extract_id(cable_ref)
        if not cable_id or cable_id in seen:
            continue
        seen.add(cable_id)

        cable = mcporter_call("netbox_get_object_by_id", {
            "object_type": "dcim.cable",
            "object_id": cable_id
        })
        if not cable:
            continue

        # Resolve both ends
        my_iface = iface.get("name", "")
        peer_iface = ""
        peer_device = ""

        for side in ["a", "b"]:
            my_terms = cable.get(f"{side}_terminations", [])
            peer_side = "b" if side == "a" else "a"
            peer_terms = cable.get(f"{peer_side}_terminations", [])

            for term in my_terms:
                obj = term.get("object", {})
                if obj.get("device", {}).get("id") == device_id:
                    my_iface = obj.get("display") or obj.get("name", my_iface)
                    for pt in peer_terms:
                        pobj = pt.get("object", {})
                        pd = pobj.get("device", {})
                        if pd.get("id") and pd.get("id") != device_id:
                            peer_iface = pobj.get("display") or "unknown"
                            peer_device = pd.get("name", "unknown")

        cable_type = cable.get("type", {})
        if isinstance(cable_type, dict):
            cable_type = cable_type.get("value", str(cable_type))

        cables.append({
            "cable_id": cable_id,
            "label": cable.get("label", ""),
            "type": cable_type,
            "status": (cable.get("status", {}) or {}).get("value", "connected"),
            "my_interface": my_iface,
            "peer_device": peer_device,
            "peer_interface": peer_iface,
        })

    return cables


def netbox_get_vm_hosts_at_risk(device_id: int) -> list[dict]:
    """Find VM hosts connected to this device (for blast radius). Returns raw — pass to LLM."""
    connected = netbox_get_connected_devices(device_id)
    vm_hosts = []
    for peer in connected:
        peer_id = peer.get("device_id")
        if not peer_id:
            continue
        peer_device = mcporter_call("netbox_get_object_by_id", {
            "object_type": "dcim.device",
            "object_id": peer_id,
            "fields": ["id", "name", "role", "site", "status"]
        })
        if not peer_device:
            continue
        role = peer_device.get("role", {}) or {}
        role_slug = role.get("slug", "")
        if role_slug == "vm-host" or "vm" in role_slug:
            # Get VMs on this host
            vms = mcporter_call("netbox_get_objects", {
                "object_type": "virtualization.virtual-machine",
                "filters": {"cluster_id": peer_device.get("cluster", {}).get("id")},
                "limit": 20
            })
            vm_list = vms if isinstance(vms, list) else (vms.get("results", []) if isinstance(vms, dict) else [])
            vm_hosts.append({
                "host": peer_device.get("name", ""),
                "site": (peer_device.get("site", {}) or {}).get("name", ""),
                "vms": [vm.get("name", "") for vm in vm_list]
            })
    return vm_hosts


def _extract_id(ref) -> Optional[int]:
    """Extract integer ID from NetBox reference dict or bare int/str."""
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


def _resolve_cable_peer(cable: dict, my_device_id: int) -> Optional[dict]:
    """Resolve the device on the other end of a cable."""
    for side, peer_side in [("a", "b"), ("b", "a")]:
        my_terms = cable.get(f"{side}_terminations", [])
        peer_terms = cable.get(f"{peer_side}_terminations", [])

        for term in my_terms:
            obj = term.get("object", {})
            if obj.get("device", {}).get("id") != my_device_id:
                continue
            for pt in peer_terms:
                pobj = pt.get("object", {})
                pd = pobj.get("device", {})
                peer_id = pd.get("id")
                if not peer_id or peer_id == my_device_id:
                    continue
                return {
                    "device_id": peer_id,
                    "device_name": pd.get("name", ""),
                    "device_role": "",
                    "interface": pobj.get("display") or pobj.get("name", "unknown"),
                }
    return None


# ────────────────────────────────────────────────────────────────
# MINIMAX LLM CALL
# ────────────────────────────────────────────────────────────────

MINIMAX_API_URL = "https://api.minimax.io/anthropic/v1/messages"


def call_minimax(prompt: str, max_tokens: int = 600) -> str:
    """Call MiniMax LLM via the Anthropic-compatible API endpoint."""
    if not MINIMAX_API_KEY:
        return _fallback_briefing(prompt)

    payload = {
        "model": "MiniMax-M2.7",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        req = urllib.request.Request(
            MINIMAX_API_URL,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {MINIMAX_API_KEY}",
                "Content-Type": "application/json",
                "x-api-key": MINIMAX_API_KEY,
                "anthropic-version": "2023-06-01"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            # MiniMax returns content blocks: [{"type":"thinking",...}, {"type":"text","text":"..."}]
            # Only extract from "text" type blocks
            text_parts = [block.get("text", "") for block in result.get("content", []) if block.get("type") == "text"]
            return "".join(text_parts)
    except Exception as e:
        return f"[LLM Error: {e}]\n\n{_fallback_briefing(prompt)}"


def _fallback_briefing(prompt: str) -> str:
    """Fallback when LLM is unavailable — extract key info from prompt."""
    lines = prompt.strip().split("\n")
    device = next((l for l in lines if "DEVICE:" in l or "📍" in l), "Unknown device")
    alert = next((l for l in lines if "ALERT:" in l or "MESSAGE:" in l), "")
    return f"""⚠️ *Manual Review Required*

{device}
{alert}

LLM enrichment unavailable — please review alert manually.
"""


# ────────────────────────────────────────────────────────────────
# TELEGRAM
# ────────────────────────────────────────────────────────────────

SEVERITY_EMOJI = {
    "critical": "🔴", "high": "🟠", "average": "🟡", "warning": "🔵", "info": "⚪"
}

SEVERITY_HEADER = {
    "critical": "🔴 *CRITICAL* — Immediate Action Required",
    "high":     "🟠 *HIGH* — Urgent Attention",
    "average":  "🟡 *AVERAGE* — Review Required",
    "warning":  "🔵 *WARNING* — Monitor",
    "info":     "⚪ *INFO*",
}


def send_telegram(text: str, chat_id: str = "") -> bool:
    """Send message to Telegram."""
    if not TELEGRAM_BOT_TOKEN:
        print("[Telegram] ERROR: TELEGRAM_BOT_TOKEN not set", flush=True)
        return False

    chat_id = chat_id or HERMES_ALERTS_CHANNEL_ID
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    params = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }

    try:
        data = json.dumps(params).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if result.get("ok"):
                msg_id = result.get("result", {}).get("message_id", "?")
                print(f"[Telegram] Sent msg {msg_id} to {chat_id}", flush=True)
                return True
            print(f"[Telegram] API error: {result.get('description')}", flush=True)
            return False
    except Exception as e:
        print(f"[Telegram] Error: {e}", flush=True)
        return False


# ────────────────────────────────────────────────────────────────
# BRIEFING PROMPT BUILDER — pass raw NetBox output to LLM
# ────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────
# REDUNDANCY & SEVERITY HELPERS
# ────────────────────────────────────────────────────────────────

# Tags that mark a device as critical (escalates severity)
CRITICAL_ROLE_SLUGS = {"core-router", "border-router", "core-switch", "aggregation-switch"}
CRITICAL_TAG_SLUGS = {"critical", "customer-critical", "tier-1", "redundancy-group-1"}

# Role → expected redundancy level (for computing single-points-of-failure)
ROLE_REDUNDANCY = {
    "border-router": "HAS_REDUNDANCY",     # dual ISP BGP peers
    "core-router": "HAS_REDUNDANCY",        # dual PS, dual uplinks
    "edge-router": "PARTIAL_REDUNDANCY",    # some redundancy but not full
    "distribution-switch": "HAS_REDUNDANCY",
    "access-switch": "PARTIAL_REDUNDANCY",
    "vm-host": "HAS_REDUNDANCY",
}


def _compute_redundancy_status(nb_device: dict, connected: list, cables: list, vm_hosts: list) -> str:
    """
    Compute a human-readable redundancy status for the device.
    Returns one of: HAS_REDUNDANCY | PARTIAL_REDUNDANCY | SINGLE_POINTS_OF_FAILURE | UNKNOWN
    """
    role = nb_device.get("role", {}) or {}
    role_slug = role.get("slug", "").lower()
    tags = nb_device.get("tags", []) or []
    tag_slugs = {t.get("slug", "").lower() for t in tags}

    # Check explicit critical tag
    is_critical = bool(tag_slugs & CRITICAL_TAG_SLUGS)

    # Check role
    role_level = ROLE_REDUNDANCY.get(role_slug, "UNKNOWN")

    # Check physical redundancy signals
    ps_count = sum(1 for c in cables if "ps" in c.get("label", "").lower() or "power" in c.get("label", "").lower())
    uplink_count = len([d for d in connected if d.get("interface", "").lower().startswith(("te", "ge", "xe", "et"))])
    vm_count = len(vm_hosts)

    if is_critical and uplink_count <= 1:
        return "⚠️ SINGLE_POINTS_OF_FAILURE — Critical device with only one active uplink"
    if is_critical and ps_count <= 1:
        return "⚠️ SINGLE_POINTS_OF_FAILURE — Critical device with no power redundancy"
    if role_level == "HAS_REDUNDANCY" and uplink_count >= 2:
        return "✅ HAS_REDUNDANCY — Multiple uplinks active"
    if role_level == "PARTIAL_REDUNDANCY" and uplink_count >= 1:
        return "🟡 PARTIAL_REDUNDANCY — Limited redundancy"
    if uplink_count == 0 and vm_count > 0:
        return "⚠️ UNKNOWN — No uplinks detected in NetBox"
    if role_level == "UNKNOWN":
        return "❓ UNKNOWN — Device role not classified for redundancy assessment"
    return f"🟡 {role_level}"


def _build_connected_section(connected: list, nb_device: dict, max_items: int = 8) -> str:
    """Build the connected devices section with role context."""
    if not connected:
        return "  No connected devices found in NetBox"

    lines = []
    for p in connected[:max_items]:
        iface = p.get("interface", "?")
        dev_name = p.get("device_name", "?")
        # Tag border/core with warning
        role = p.get("device_role", "").lower()
        warning = " ⚠️ CORE" if role in {"core-router", "border-router", "core-switch"} else ""
        lines.append(f"  • *{dev_name}* — via {iface}{warning}")
    return "\n".join(lines)


def _build_cables_section(cables: list, max_items: int = 6) -> str:
    """Build the cables section with ISP circuit IDs pulled out."""
    if not cables:
        return "  No cables found in NetBox"

    lines = []
    for c in cables[:max_items]:
        label_raw = c.get("label") or c.get("cable_id", "?")
        label = str(label_raw) if not isinstance(label_raw, str) else label_raw
        cable_type = c.get("type", "?")
        status = c.get("status", "connected")
        my_iface = c.get("my_interface", "?")
        peer_dev = c.get("peer_device", "?")
        peer_iface = c.get("peer_interface", "?")
        # Flag ISP circuits
        isp_flag = " 📡ISP" if any(x in label.upper() for x in ["ISP", "CIR-", "TRANSIT", "UPSTREAM"]) else ""
        status_flag = f" ⚠️ {status.upper()}" if status not in ("connected", "active") else ""
        if peer_dev == "?":
            peer_str = f"↔ {my_iface} (peer unknown)"
        else:
            peer_str = f"{my_iface} ↔ {peer_dev}/{peer_iface}"
        lines.append(f"  • `{label}`{isp_flag} — {cable_type} {status_flag}\n    {peer_str}")
    return "\n".join(lines)


def _build_vm_section(vm_hosts: list) -> str:
    """Build the VM hosts section."""
    if not vm_hosts:
        return "  No VM hosts connected to this device in NetBox"
    lines = []
    for h in vm_hosts:
        host = h.get("host", "?")
        site = h.get("site", "")
        vms = h.get("vms", [])
        vm_str = ", ".join(vms[:5]) if vms else "no VMs tracked"
        if len(vms) > 5:
            vm_str += f" (+{len(vms)-5} more)"
        lines.append(f"  • *{host}* ({site}) — VMs: {vm_str}")
    return "\n".join(lines)


def _build_source_context(alert: AlertRecord) -> str:
    """Build the ## SOURCE PROVIDED sections from mock lab / source data."""
    sections = []
    if alert.netbox_context:
        sections.append("## SOURCE PROVIDED CONTEXT")
        for k, v in alert.netbox_context.items():
            sections.append(f"  {k}: {v}")
    if alert.netbox_impact:
        sections.append("\n## SOURCE PROVIDED IMPACT")
        for k, v in alert.netbox_impact.items():
            sections.append(f"  {k}: {v}")
    if alert.netbox_cables:
        sections.append("\n## SOURCE PROVIDED CABLES")
        for c in alert.netbox_cables:
            sections.append(f"  {c}")
    if alert.metrics:
        sections.append("\n## SOURCE PROVIDED METRICS")
        for k, v in alert.metrics.items():
            sections.append(f"  {k}: {v}")
    return "\n".join(sections) if sections else ""


def _build_tags_section(tags: list) -> str:
    """Format tags for display."""
    if not tags:
        return "none"
    return ", ".join(sorted(t.get("slug") or t.get("name", "") for t in tags))


# ────────────────────────────────────────────────────────────────
# MAIN PROMPT BUILDER
# ────────────────────────────────────────────────────────────────

def build_enrichment_prompt(alert: AlertRecord, nb_device: dict, connected: list, cables: list, vm_hosts: list) -> str:
    """
    Build a high-quality NOC briefing prompt.

    Changes from previous version:
    - Adds redundancy_status computed from device role + uplinks + power
    - Adds ISP/circuit labels pulled from cable labels
    - Adds tags section
    - Adds source-provided context block
    - Gives LLM stronger tone/persona guidance
    - Adds "Failure Mode Hypothesis" task
    - Strictly caps output for Telegram single-message delivery
    - Handles unknown/not-found devices gracefully
    """
    device_name = alert.device
    role = nb_device.get("role", {}) or {}
    site = nb_device.get("site", {}) or {}
    device_type = nb_device.get("device_type", {}) or {}
    status = nb_device.get("status", {}) or {}
    primary_ip = nb_device.get("primary_ip") or {}

    role_display = role.get("display") or role.get("name", "Unknown")
    site_display = site.get("display") or site.get("name", "Unknown")
    device_type_display = device_type.get("display") or device_type.get("model", "")
    status_display = status.get("value", "unknown") if isinstance(status, dict) else str(status)
    ip_address = primary_ip.get("address", "") if isinstance(primary_ip, dict) else ""
    tags_str = _build_tags_section(nb_device.get("tags", []))

    redundancy_status = _compute_redundancy_status(nb_device, connected, cables, vm_hosts)

    # If device came back empty (not found in NetBox)
    device_found = bool(nb_device.get("id"))
    not_found_note = ""
    if not device_found:
        not_found_note = (
            "\n⚠️ *DEVICE NOT FOUND IN NETBOX* — proceeding with alert data only. "
            "On-call engineer should verify device exists and NetBox is up to date."
        )

    connected_section = _build_connected_section(connected, nb_device)
    cables_section = _build_cables_section(cables)
    vm_section = _build_vm_section(vm_hosts)
    source_context = _build_source_context(alert)

    # Severity escalation hint
    is_critical_tagged = any(
        t.get("slug", "").lower() in CRITICAL_TAG_SLUGS
        for t in nb_device.get("tags", [])
    )
    escalation_hint = ""
    if is_critical_tagged and alert.severity in ("high", "average"):
        escalation_hint = "\n🏷️ *Note: This device is tagged critical — severity may be escalated.*"

    source_label = "Mock Lab" if alert.is_mock_lab else "Opsgenie/Production"

    prompt = f"""You are a senior NOC engineer writing a concise, actionable Telegram alert briefing.
Write like a calm, experienced on-call engineer — precise, no fluff, situation-first.

---
## RAW ALERT
  Alert ID:   {alert.alert_id}
  Device:     {device_name}
  Type:       {alert.alert_type}
  Severity:   {alert.severity}
  Message:    {alert.message}
  Time:       {alert.timestamp}
  Source:     {source_label}
{not_found_note}
{escalation_hint}

---
## NETBOX DEVICE CONTEXT
  Name:         {nb_device.get('name', device_name) if device_found else device_name + " (NOT IN NETBOX)"}
  Role:         {role_display}
  Site:         {site_display}
  Device Type:  {device_type_display}
  Status:       {status_display}
  IP Address:   {ip_address}
  Tags:         {tags_str}
  Serial:       {nb_device.get('serial', 'N/A')}
  Redundancy:   {redundancy_status}

{source_context}
---
## CONNECTED DEVICES ({len(connected)} found in NetBox)
{connected_section}

---
## CABLING ({len(cables)} cables in NetBox)
{cables_section}

---
## VM HOSTS AT RISK ({len(vm_hosts)} hosts)
{vm_section}

---
## YOUR TASK
Generate a Telegram briefing with these exact sections:

1. **Header** — Severity emoji + type + device name + site
2. **Impact Summary** — 2-3 sentences. What is affected? Blast radius? Redundancy status?
3. **Connected Devices at Risk** — Name specific devices/customers that could be affected. If this is a border router, name the ISP circuit IDs and what transit is at risk.
4. **VM Workloads at Risk** — List affected VMs if any. If none, write "No VM workloads tracked in NetBox — confirm with VM team if relevant."
5. **Failure Mode Hypothesis** — What is the most likely root cause? (1 sentence)
6. **Recommended Actions** — 2-3 concrete next steps. Be specific: which command to run, which interface to check, whether to escalate.

TONE: Calm, authoritative, precise. Imagine you are texting a colleague at 3 AM.
OUTPUT: Your entire response MUST fit in a single Telegram message (max 4096 chars).
FORMAT: Clean Telegram Markdown. Use emojis for severity and warnings. Bold critical info.
DO NOT exceed 400 words total.
"""
    return prompt


# ────────────────────────────────────────────────────────────────
# MAIN ENRICHMENT FUNCTION
# ────────────────────────────────────────────────────────────────

def enrich_alert(alert: AlertRecord) -> dict:
    """
    Full enrichment pipeline for a single alert:
      1. NetBox lookup (device + connected + cables + VM hosts)
      2. Build prompt with raw NetBox data
      3. Call MiniMax LLM
      4. Send to Telegram
      5. Return result dict

    Returns: {"status": "ok"|"error", "briefing": str, "sent": bool, ...}
    """
    print(f"[Enrich] Processing alert {alert.alert_id} for device {alert.device}", flush=True)

    # Step 1: NetBox lookup (device first, then VM)
    nb_host = netbox_lookup_host(alert.device)
    if not nb_host:
        print(f"[Enrich] Host '{alert.device}' not found in NetBox — proceeding with alert only", flush=True)
        nb_host = {"name": alert.device, "role": {}, "site": {}, "device_type": {}, "status": {}, "tags": [], "primary_ip": {}, "_host_type": "unknown"}
    else:
        print(f"[Enrich] Found {nb_host.get('_host_type', '?')} '{alert.device}' in NetBox (id={nb_host.get('id')})", flush=True)

    nb_device = nb_host  # alias for clarity in the rest of the function
    device_id = nb_device.get("id")
    connected = []
    cables = []
    vm_hosts = []

    if device_id:
        connected = netbox_get_connected_devices(device_id)
        cables = netbox_get_cables(device_id)
        vm_hosts = netbox_get_vm_hosts_at_risk(device_id)
        print(f"[Enrich] NetBox: {len(connected)} connected, {len(cables)} cables, {len(vm_hosts)} VM hosts", flush=True)

    # Step 2: Build prompt with RAW NetBox data (LLM interprets)
    prompt = build_enrichment_prompt(alert, nb_device, connected, cables, vm_hosts)
    print(f"[Enrich] Prompt built ({len(prompt)} chars) — calling MiniMax...", flush=True)

    # Step 3: Call MiniMax LLM
    briefing = call_minimax(prompt)
    print(f"[Enrich] LLM response ({len(briefing)} chars)", flush=True)

    # Step 4: Send to Telegram
    severity_header = SEVERITY_HEADER.get(alert.severity, f"📋 *ALERT*")
    header = f"{severity_header}\n{'─' * 40}\n"
    full_text = header + briefing

    sent = send_telegram(full_text)
    print(f"[Enrich] Telegram: {'✅' if sent else '❌'}", flush=True)

    # Step 5: Router diagnostics — AI agent SSHes to lab routers for critical/warning alerts
    # Only runs if device has a management IP in LAB_MGMT_IPS (safety: only touches lab hardware)
    diag_sent = False
    if run_router_diagnostics and alert.severity in ("critical", "warning") and nb_device:
        primary_ip = nb_device.get("primary_ip") or {}
        mgmt_ip_raw = primary_ip.get("address", "") if isinstance(primary_ip, dict) else ""
        mgmt_ip = mgmt_ip_raw.split("/")[0] if mgmt_ip_raw else ""
        if mgmt_ip and mgmt_ip in LAB_MGMT_IPS:
            hostname = nb_device.get("name", alert.device)
            print(f"[Enrich] Running router diagnostics on {hostname} ({mgmt_ip})...", flush=True)
            diag_results = run_router_diagnostics(
                hostname=hostname,
                mgmt_ip=mgmt_ip,
                severity=alert.severity,
            )
            diag_report = format_diagnostic_report(
                diag_results,
                alert_message=alert.message,
                severity=alert.severity,
            )
            if diag_report:
                diag_sent = send_diagnostic_telegram(diag_report)
                print(f"[Enrich] Router diagnostics Telegram: {'✅' if diag_sent else '❌'}", flush=True)

    return {
        "status": "ok" if sent else "error",
        "alert_id": alert.alert_id,
        "device": alert.device,
        "severity": alert.severity,
        "briefing": briefing,
        "sent": sent,
        "diag_sent": diag_sent,
        "netbox": {
            "device": nb_device.get("name"),
            "connected_count": len(connected),
            "cables_count": len(cables),
            "vm_hosts_count": len(vm_hosts),
        }
    }


def enrich_alert_from_dict(alert_dict: dict, source: str = "mock_lab", noc_notify: bool = False) -> dict:
    """
    Parse a raw alert dict into an AlertRecord and run the enrichment pipeline.

    SOURCE FORMAT ROUTING:
        "mock_lab"  → AlertRecord.from_mock_lab(alert_dict)
                       Handles: alert_id, device, type, severity, description,
                       netbox_context, netbox_impact, netbox_cable, metrics

        "opsgenie"  → AlertRecord.from_opsgenie(alert_dict, action)
                       Handles: alertID, alias (→ device), priority (→ severity),
                       updatedAt (→ timestamp), message+description (→ message)

        fallback    → Generic AlertRecord with no source-specific parsing.
                       Used when source="opsgenie" but format doesn't match above.
                       Still runs full enrichment — graceful degradation.

    AFTER PARSING:
        Always calls enrich_alert(record) which does:
          1. NetBox lookup (device + connected + cables + VM hosts)
          2. Build prompt
          3. Call MiniMax LLM
          4. Send to Telegram
          5. Return result dict

    RETRY SAFETY:
        If this is called twice with the same alert_dict (e.g. Opsgenie retries
        after a 500), the LLM will generate the same briefing. Safe to retry.
    """
    # Check source FIRST to avoid false matches with alert_id field
    if source == "checkmk":
        record = AlertRecord.from_checkmk(alert_dict)
    elif source == "mock_lab" or ("alert_id" in alert_dict and "alerts" not in alert_dict):
        record = AlertRecord.from_mock_lab(alert_dict)
    elif "action" in alert_dict and "alert" in alert_dict:
        record = AlertRecord.from_opsgenie(alert_dict.get("alert", {}), alert_dict.get("action", ""))
    elif "context" in alert_dict or "host_name" in alert_dict:
        record = AlertRecord.from_checkmk(alert_dict)
        source = "checkmk"
    else:
        # Generic fallback
        record = AlertRecord(
            alert_id=alert_dict.get("alert_id", alert_dict.get("id", "unknown")),
            device=alert_dict.get("device", alert_dict.get("hostname", alert_dict.get("alias", "unknown"))),
            alert_type=alert_dict.get("type", alert_dict.get("alert_type", "unknown")),
            severity=alert_dict.get("severity", "average"),
            message=alert_dict.get("message", alert_dict.get("description", "")),
            site=alert_dict.get("site", ""),
            timestamp=alert_dict.get("timestamp", ""),
            raw_payload=alert_dict,
        )

    result = enrich_alert(record)

    # ── 6. NOC MD pipeline: save enriched context for on-demand NOC skill access ──
    # Writes ~/noc/inbox/{alert_id}.json → ~/noc/context/{alert_id}.md
    # and saves raw SSH diagnostics to ~/noc/diagnostics_raw/{alert_id}.json
    if noc_notify and result.get("alert_id"):
        _trigger_noc_md_pipeline(result.get("alert_id", ""), alert_dict, source)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# NOC MD pipeline — save enriched context for on-demand NOC skill access
# ─────────────────────────────────────────────────────────────────────────────

def _trigger_noc_md_pipeline(alert_id: str, alert_dict: dict, source: str) -> None:
    """
    Write alert to ~/noc/inbox/{alert_id}.json then trigger noc_md_processor
    to produce ~/noc/context/{alert_id}.md and ~/noc/diagnostics_raw/{alert_id}.json.

    This function is fire-and-forget — errors are logged but never raise,
    to avoid disrupting the main enrichment pipeline.
    """
    if not _NOC_MD_AVAILABLE:
        return

    import logging
    _noc_logger = logging.getLogger("noc_md_trigger")
    _noc_logger.setLevel(logging.INFO)

    inbox_data = {
        "alert": {
            "alert_id": alert_id,
            "device": alert_dict.get("device", ""),
            "severity": alert_dict.get("severity", "warning"),
            "message": alert_dict.get("message", alert_dict.get("description", "")),
            "site": alert_dict.get("site", ""),
            "timestamp": alert_dict.get("timestamp", ""),
        },
        "source": source,
    }

    inbox_path = Path.home() / "noc" / "inbox" / f"{alert_id}.json"
    try:
        inbox_path.parent.mkdir(parents=True, exist_ok=True)
        inbox_path.write_text(json.dumps(inbox_data, indent=2), encoding="utf-8")
        _noc_logger.info(f"[NOC Trigger] Wrote inbox: %s", inbox_path)

        # Run noc_md_processor (blocking — but webhook already returned 200 to source)
        result = process_alert_from_inbox(inbox_path)
        _noc_logger.info(
            "[NOC Trigger] alert_id=%s → md=%s had_diagnostics=%s errors=%s",
            alert_id,
            result.get("md_path", ""),
            result.get("had_diagnostics", False),
            result.get("errors", []),
        )
    except Exception as exc:
        _noc_logger.warning(f"[NOC Trigger] Failed for {alert_id}: {exc}")


from pathlib import Path


# ────────────────────────────────────────────────────────────────
# TEST
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test with mock lab payload (simulating what the webhook would receive)
    test_alert = {
        "alert_id": "LAB-POWER-001",
        "device": "DC1-CORE-RTR-01",
        "type": "power-alert",
        "severity": "high",
        "timestamp": "2026-04-18T19:25:00+08:00",
        "description": "Power supply PS1 failure on core router - primary PSU offline",
        "site": "DC1 - Singapore",
        "netbox_device": "DC1-CORE-RTR-01",
        "netbox_context": {
            "role": "Core Router",
            "device_type": "MX204",
            "connected_to": ["DC1-SPINE-01", "DC1-SPINE-02"],
            "customer": None,
            "isp_circuit": None,
            "isp_noc": None
        },
        "netbox_impact": {
            "redundancy": "DEGRADED — single PS active",
            "recommended_action": "Replace PS1 within 24h - no redundancy until resolved"
        },
        "netbox_cable": None,
        "metrics": {
            "ps1_status": "offline",
            "ps2_status": "active",
            "power_draw_watts": 380
        }
    }

    print("[Test] Running enrichment pipeline...", flush=True)
    result = enrich_alert_from_dict(test_alert, source="mock_lab")
    print(f"\nResult: {json.dumps(result, indent=2, default=str)}")
