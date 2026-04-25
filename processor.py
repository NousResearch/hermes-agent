"""
Alert Enrichment — Main Processor
=================================
Processes buffered alerts: cluster → NetBox enrichment → LLM briefing → Telegram.
Designed to run as a cron job every 5 minutes.
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional

# Local modules
from webhook_receiver import alert_buffer, buffer_lock, last_processed_at, is_processing
from netbox_lookup import (
    netbox_lookup, get_connected_devices, get_cables,
    compute_severity, compute_delivery, NbDevice,
    batch_lookup_devices, detect_alert_types,
)
from enrichment_prompts import build_cluster_briefing_prompt, complete, EnrichmentContext
from telegram_dispatcher import send_alert, HERMES_ALERTS_CHANNEL_ID, DEFAULT_DM_CHAT_ID

# ────────────────────────────────────────────────────────────────
# CLUSTERING
# ────────────────────────────────────────────────────────────────

def cluster_by_device(alerts: list) -> dict[str, list]:
    """
    Group alerts by device hostname (v1 clustering).
    Returns {device_name: [alert_records]}
    """
    clusters = {}
    for alert in alerts:
        # Handle both AlertRecord objects and plain dicts
        if hasattr(alert, "device"):
            device = alert.device
        else:
            device = alert.get("device", alert.get("alias", "unknown"))

        if device not in clusters:
            clusters[device] = []
        clusters[device].append(alert)
    return clusters


# ────────────────────────────────────────────────────────────────
# ENRICHMENT
# ────────────────────────────────────────────────────────────────

def enrich_cluster(device_name: str, alerts: list) -> dict:
    """
    Enrich a cluster of alerts for one device.
    Passes raw NetBox data directly to the LLM for interpretation.
    No pre-digestion — the LLM handles the nested/adjacent data.
    """
    from alert_processor import (
        netbox_lookup_device as nb_lookup,
        netbox_get_connected_devices as nb_connected,
        netbox_get_cables as nb_cables,
        netbox_get_vm_hosts_at_risk as nb_vm_hosts,
    )

    # Step 1: NetBox lookup — raw output goes to LLM
    nb_device = nb_lookup(device_name)

    if not nb_device:
        print(f"[Enrich] Device '{device_name}' not found in NetBox", flush=True)
        nb_device = {"name": device_name, "role": {}, "site": {}, "device_type": {}, "status": {}, "tags": [], "primary_ip": {}}

    device_id = nb_device.get("id")
    connected = nb_connected(device_id) if device_id else []
    cables = nb_cables(device_id) if device_id else []
    vm_hosts = nb_vm_hosts(device_id) if device_id else []

    # Step 2: Get live metrics from mock lab
    metrics = _fetch_mock_metrics(device_name)

    # Step 3: Detect alert types (only for routing/delivery logic, not for LLM)
    alert_types = detect_alert_types(alerts)

    # Step 4: Determine highest severity in cluster
    severities = []
    for a in alerts:
        sev = (a.severity if hasattr(a, "severity") else a.get("severity", "average"))
        severities.append(sev)
    top_severity = min(severities, key=lambda s: _severity_rank(s))

    # Step 5: Compute final escalated severity (for delivery routing, not for LLM)
    final_severity = compute_severity(top_severity, None)  # NbDevice compat removed; severity escalation handled by LLM

    # Step 6: Build a lightweight context for the legacy build_cluster_briefing_prompt
    # (Only used if old code path is triggered. Prefer the new prompt in alert_processor.py)
    from netbox_lookup import netbox_lookup, NbDevice
    nb_dev_compat = _raw_to_nbdev(nb_device) if nb_device.get("id") else None
    redundancy = _assess_redundancy(alerts, metrics, nb_dev_compat)
    components = []
    for a in alerts:
        if hasattr(a, "affected_component"):
            components.append(a.affected_component)
        elif isinstance(a, dict):
            comp = a.get("affected_component", a.get("component", ""))
            if comp:
                components.append(comp)

    ctx = EnrichmentContext(
        cluster_id=device_name,
        device_name=device_name,
        alert_types=list(alert_types),
        site=(nb_device.get("site", {}) or {}).get("display", (nb_device.get("site", {}) or {}).get("name", "Unknown")),
        device_role=(nb_device.get("role", {}) or {}).get("display", (nb_device.get("role", {}) or {}).get("name", "Unknown")),
        is_core="core_device" in [t.get("slug") or t.get("name", "").lower().replace(" ", "-") for t in nb_device.get("tags", [])],
        is_critical=bool([t for t in nb_device.get("tags", []) if (t.get("slug") or t.get("name", "").lower()) in {"core_device", "customer_critical_site"}]),
        alerts=[asdict(a) if hasattr(a, "__dataclass_fields__") else a for a in alerts],
        connected_devices=connected,
        cables=cables,
        metrics=metrics,
        redundancy_status=redundancy,
        affected_components=components,
        nb_tags=[t.get("slug") or t.get("name", "").lower().replace(" ", "-") for t in nb_device.get("tags", [])],
    )

    return {
        "context": ctx,
        "device": nb_device,
        "connected": connected,
        "cables": cables,
        "vm_hosts": vm_hosts,
        "severity": final_severity,
        "delivery": compute_delivery(final_severity),
    }


def _raw_to_nbdev(raw: dict) -> Optional["NbDevice"]:
    """Convert raw NetBox dict from alert_processor to NbDevice for legacy compat."""
    if not raw or not raw.get("id"):
        return None
    from netbox_lookup import NbDevice as NB
    role = raw.get("role", {}) or {}
    site = raw.get("site", {}) or {}
    return NB(
        id=raw.get("id", 0),
        name=raw.get("name", ""),
        status=(raw.get("status", {}) or {}).get("value", "unknown"),
        role=role.get("display", role.get("name", "")),
        role_slug=role.get("slug", ""),
        site=site.get("display", site.get("name", "")),
        site_slug=site.get("slug", ""),
        tags=[t.get("slug") or t.get("name", "").lower().replace(" ", "-") for t in raw.get("tags", [])],
        device_type=(raw.get("device_type", {}) or {}).get("display", ""),
        platform=(raw.get("platform", {}) or {}).get("slug", ""),
        serial=raw.get("serial", ""),
        ip_address=(raw.get("primary_ip", {}) or {}).get("address", ""),
    )


def _unknown_device(name: str) -> NbDevice:
    return NbDevice(
        id=0, name=name, status="unknown",
        role="Unknown", role_slug="", site="Unknown", site_slug="",
        tags=[], device_type="", platform="", serial="", ip_address=""
    )


# ────────────────────────────────────────────────────────────────
# MOCK LAB METRICS
# ────────────────────────────────────────────────────────────────

MOCK_LAB_BASE = os.environ.get("MOCK_LAB_BASE", "http://192.168.1.9:5000")

def _fetch_mock_metrics(device: str) -> dict:
    """Fetch live metrics from mock lab Zabbix endpoint."""
    try:
        import urllib.request
        url = f"{MOCK_LAB_BASE}/zabbix/host/{device}/metrics"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"[Metrics] Could not fetch metrics for {device}: {e}", flush=True)
        return {}


# ────────────────────────────────────────────────────────────────
# SEVERITY RANK
# ────────────────────────────────────────────────────────────────

SEVERITY_RANK = {"critical": 0, "high": 1, "average": 2, "warning": 3, "info": 4}

def _severity_rank(s: str) -> int:
    return SEVERITY_RANK.get(s.lower(), 5)


# ────────────────────────────────────────────────────────────────
# REDUNDANCY ASSESSMENT
# ────────────────────────────────────────────────────────────────

def _assess_redundancy(alerts: list, metrics: dict, device: Optional[NbDevice]) -> Optional[str]:
    """Infer redundancy status from alerts and metrics."""
    # Power supply alerts
    ps_alerts = [a for a in alerts if "power" in (a.get("alert_type", "") or "").lower()
                 or "ps" in (a.get("affected_component", "") or "").lower()]

    if ps_alerts:
        ps0 = metrics.get("ps0", "").lower()
        ps1 = metrics.get("ps1", "").lower()
        if "off" in ps0 and "off" in ps1:
            return "FAILED — both PS units down"
        if "off" in ps0 or "off" in ps1:
            return "DEGRADED — single PS active"
        return "DEGRADED — PS failure reported"

    # BGP alerts
    bgp_alerts = [a for a in alerts if "bgp" in (a.get("alert_type", "") or "").lower()]
    if bgp_alerts:
        return "BGP PEER DOWN — routing affected"

    # Interface alerts
    iface_alerts = [a for a in alerts if "interface" in (a.get("alert_type", "") or "").lower()
                    or "eth" in (a.get("affected_component", "") or "").lower()]
    if iface_alerts:
        return "INTERFACE DOWN — potential traffic loss"

    return None


# ────────────────────────────────────────────────────────────────
# MAIN PROCESS FUNCTION (called by cron)
# ────────────────────────────────────────────────────────────────

def process_alerts():
    """
    Main entry point. Call this every 5 minutes via cron.
    Pulls alerts from buffer, clusters by device, enriches, dispatches to Telegram.
    """
    global last_processed_at, is_processing

    if is_processing:
        print("[Processor] Already processing — skipping this run.", flush=True)
        return {"status": "skipped", "reason": "already_processing"}

    is_processing = True
    try:
        with buffer_lock:
            if not alert_buffer:
                print("[Processor] Buffer empty — nothing to process.", flush=True)
                return {"status": "ok", "dispatched": 0}
            alerts = list(alert_buffer)
            alert_buffer.clear()

        print(f"[Processor] Processing {len(alerts)} alert(s)...", flush=True)

        # Cluster by device
        clusters = cluster_by_device(alerts)
        print(f"[Processor] {len(clusters)} device cluster(s): {list(clusters.keys())}", flush=True)

        dispatched = 0
        for device_name, device_alerts in clusters.items():
            try:
                # Enrich cluster
                result = enrich_cluster(device_name, device_alerts)
                ctx: EnrichmentContext = result["context"]
                severity = result["severity"]
                delivery = result["delivery"]

                print(f"[Processor] [{device_name}] severity={severity} delivery={delivery}", flush=True)

                # Skip low-priority alerts that should go to digest only
                if delivery in ("jira_only", "digest_1h"):
                    print(f"[Processor] [{device_name}] Skipping — {delivery} delivery", flush=True)
                    continue

                # Build LLM briefing
                prompt = build_cluster_briefing_prompt(ctx)
                briefing = complete(prompt)

                # Send to Telegram
                sent = send_alert(briefing, severity=severity)
                if sent:
                    dispatched += 1
                    print(f"[Processor] [{device_name}] Telegram dispatched ✅", flush=True)
                else:
                    print(f"[Processor] [{device_name}] Telegram dispatch failed ❌", flush=True)

            except Exception as e:
                print(f"[Processor] [{device_name}] Error: {e}", flush=True)
                import traceback
                traceback.print_exc()

        last_processed_at = datetime.utcnow()
        return {"status": "ok", "dispatched": dispatched, "clusters": len(clusters)}

    finally:
        is_processing = False


# ────────────────────────────────────────────────────────────────
# TEST
# ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[Processor] Running test — processing buffer...")
    result = process_alerts()
    print(f"Result: {result}")
