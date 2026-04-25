"""
noc_md_processor.py
====================
On-demand alert context builder for the NOC MD pipeline.

Reads ~/noc/inbox/{alert_id}.json → resolves device info from NetBox
(via mcporter) → SSH router and collect diagnostics (for lab devices only)
→ calls MiniMax ONCE with a merged enrichment+diagnostics prompt → writes
~/noc/context/{alert_id}.md

The resulting .md file is consumed by the /noc <alert_id> skill on demand.

Usage (manual trigger):
    python3 noc_md_processor.py R1-lab-20250424-001

Usage (as import):
    from noc_md_processor import process_alert_from_inbox
    process_alert_from_inbox(Path("~/noc/inbox/R1-lab-20250424-001.json"))
"""

import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# PTY-based SSH runner for Cisco IOL
from router_diagnostics import run_router_diagnostics, format_diagnostic_report

# ─── Logging ─────────────────────────────────────────────────────────────────
logger = logging.getLogger("noc_md")
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# ─── Constants ───────────────────────────────────────────────────────────────
INBOX_DIR  = Path.home() / "noc" / "inbox"
CTX_DIR    = Path.home() / "noc" / "context"
LOG_DIR    = Path.home() / "noc" / "logs"

INBOX_DIR.mkdir(parents=True, exist_ok=True)
CTX_DIR.mkdir(parents=True,  exist_ok=True)
LOG_DIR.mkdir(parents=True,  exist_ok=True)

# MiniMax LLM config
MINIMAX_API_KEY    = os.environ.get("MINIMAX_API_KEY",    "")
MINIMAX_API_SECRET = os.environ.get("MINIMAX_API_SECRET", "")
MINIMAX_MODEL      = "MiniMax-M2.7"
MINIMAX_URL        = "https://api.minimax.io/anthropic/v1/messages"
MINIMAX_MAX_TOKENS = 8000

# ─── LAB enforcement ──────────────────────────────────────────────────────────
LAB_MGMT_IPS = {
    "192.168.1.210",  # R-HQ-LAB
    "192.168.1.201",  # R1-lab
    "192.168.1.202",  # R2-lab
    "192.168.1.203",  # R3-lab
    "192.168.1.204",  # R1-lab-switch
    "192.168.1.205",  # R2-lab-switch
    "192.168.1.206",  # R3-lab-switch
}


# ─── NetBox via mcporter ──────────────────────────────────────────────────────

def resolve_device_netbox(device_name: str) -> dict:
    """
    Resolve device name → NetBox device dict using mcporter MCP.

    Returns a dict with keys: name, status, site, ip, model, serial, os, role.
    Returns an empty dict if the device is not found or mcporter fails.
    """
    import subprocess

    device_name_safe = device_name.strip()
    logger.info(f"[NetBox] Resolving '{device_name_safe}' via mcporter...")

    # Use mcporter to call the NetBox devices API filter by name
    # mcporter is at ~/.local/bin/mcporter (installed via pip)
    cmd = [
        "mcporter",
        "--format", "json",
        "netbox",
        "devices",
        "list",
        "--name", device_name_safe,
    ]

    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=15)
        # mcporter returns whitespace-delimited JSON Lines sometimes; grab all lines
        lines = [l for l in raw.decode("utf-8", errors="replace").strip().splitlines() if l.strip()]
        if not lines:
            return {}
        # Try to parse each line as JSON object
        for line in reversed(lines):  # last line is most relevant
            try:
                records = json.loads(line)
                if isinstance(records, list) and records:
                    # Return first match
                    for rec in records:
                        if rec.get("name") == device_name_safe:
                            return rec
                    # Fallback: return first record if exact name match not found
                    return records[0]
                elif isinstance(records, dict):
                    return records
            except json.JSONDecodeError:
                continue
        return {}
    except subprocess.TimeoutExpired:
        logger.warning(f"[NetBox] mcporter timed out for '{device_name_safe}'")
        return {}
    except FileNotFoundError:
        logger.warning("[NetBox] mcporter not found in PATH — skipping NetBox enrichment")
        return {}
    except subprocess.CalledProcessError as e:
        logger.warning(f"[NetBox] mcporter returned {e.returncode}: {e.output[:200]}")
        return {}


def netbox_summary(nb: dict) -> str:
    """Render a NetBox device dict as a compact single-line string."""
    if not nb:
        return "_No NetBox data_"
    parts = []
    for key in ("name", "status", "site", "role", "model", "serial"):
        val = nb.get(key)
        if val:
            parts.append(f"{key}={val}")
    return " | ".join(parts) if parts else "_Minimal NetBox data_"


# ─── MiniMax LLM caller ──────────────────────────────────────────────────────

def _call_minimax(prompt: str, max_tokens: int = MINIMAX_MAX_TOKENS) -> str:
    """
    Send a prompt to MiniMax LLM and return the text response.

    Returns a string (the LLM response) or raises on failure.
    """
    import requests as _requests

    if not MINIMAX_API_KEY or not MINIMAX_API_SECRET:
        raise RuntimeError("MINIMAX_API_KEY / MINIMAX_API_SECRET environment variables not set")

    headers = {
        "Authorization": f"Bearer {MINIMAX_API_SECRET}",
        "x-api-key":     MINIMAX_API_KEY,
        "Content-Type":  "application/json",
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model":      MINIMAX_MODEL,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "thinking":    {"type": "off"},
        "messages":   [{"role": "user", "content": prompt}],
    }

    resp = _requests.post(MINIMAX_URL, json=payload, headers=headers, timeout=90)
    if resp.status_code == 200:
        data = resp.json()
        text_parts = [
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        ]
        content = "".join(text_parts)
        return content.strip() if content else "⚠️ LLM returned empty response."
    elif resp.status_code in (429, 503):
        raise RuntimeError(f"MiniMax rate-limited ({resp.status_code}) — retry later")
    else:
        raise RuntimeError(f"MiniMax API error {resp.status_code}: {resp.text[:300]}")


# ─── Markdown builder ─────────────────────────────────────────────────────────

def build_md_content(
    inbox_data: dict,
    netbox_info: dict,
    diag_results: Optional[dict],
    llm_enrichment: str,
) -> str:
    """
    Assemble the full ~/noc/context/{alert_id}.md document.

    Structure:
        # Alert: {alert_id}
        ## Alert Info
        ## Enrichment
        ## Router Diagnostics   ← only present if lab device
        ## Raw Alert
    """
    alert    = inbox_data.get("alert", {})
    source   = inbox_data.get("source", "unknown")
    received = inbox_data.get("received_at", "")

    alert_id  = alert.get("alert_id")  or alert.get("device") or "unknown"
    device    = alert.get("device",    "")
    severity  = alert.get("severity",  "unknown")
    message   = alert.get("message",   alert.get("description", ""))
    timestamp = alert.get("timestamp", received)

    diag_timestamp  = diag_results.get("timestamp", "") if diag_results else ""
    diag_summary    = diag_results.get("summary",  "") if diag_results else ""
    diag_commands    = diag_results.get("commands", []) if diag_results else []
    diag_errors      = diag_results.get("errors",   []) if diag_results else []
    diag_ok_count    = sum(1 for c in diag_commands if c.get("ok"))   if diag_commands else 0
    diag_fail_count = sum(1 for c in diag_commands if not c.get("ok")) if diag_commands else 0

    lines = [
        f"# Alert: {alert_id}",
        "",
        "## Alert Info",
        f"| Field | Value |",
        f"| --- | --- |",
        f"| Device | {device} |",
        f"| Severity | {severity} |",
        f"| Source | {source} |",
        f"| Time | {timestamp} |",
        f"| Received | {received} |",
        "",
        f"| NetBox | {netbox_summary(netbox_info)} |",
        "",
        "---",
        "",
        "## Enrichment",
        llm_enrichment,
        "",
    ]

    if diag_results:
        lines += [
            "---",
            "",
            "## Router Diagnostics",
            f"**Device:** `{netbox_info.get('name', device)}` @ `{diag_results.get('mgmt_ip', 'N/A')}`  |  "
            f"**Commands:** {len(diag_commands)} "
            f"({diag_ok_count} ✅ / {diag_fail_count} ❌)  |  "
            f"**Timestamp:** `{diag_timestamp}`",
            "",
        ]
        if diag_errors:
            lines.append(f"⚠️ **Errors:** `{'`  `'.join(diag_errors)}`")
            lines.append("")

        # LLM summary
        lines.append("### LLM Analysis")
        lines.append(diag_summary)
        lines.append("")

        # Per-command output (condensed)
        lines.append("### Command Output")
        for cmd_entry in diag_commands:
            cmd     = cmd_entry.get("command", "")
            ok      = cmd_entry.get("ok", False)
            out     = cmd_entry.get("output", "")
            err     = cmd_entry.get("error", "")
            status  = "✅" if ok else "❌"
            # Trim each output to 600 chars to keep the file manageable
            out_disp = out[:600] + "..." if len(out) > 600 else out
            lines.append(f"**{status} `{cmd}`**")
            if out_disp:
                lines.append(f"```\n{out_disp}\n```")
            if err:
                lines.append(f"_Error: {err}_")
            lines.append("")

    lines += [
        "---",
        "",
        "## Raw Alert",
        "```json",
        json.dumps(alert, indent=2, default=str),
        "```",
    ]

    return "\n".join(lines)


# ─── Per-alert processor ──────────────────────────────────────────────────────

def process_alert_from_inbox(inbox_path: Path) -> dict:
    """
    Read an inbox JSON file, run the full enrichment+diagnostics pipeline,
    and write the resulting .md to ~/noc/context/.

    Args:
        inbox_path: Path to a ~/noc/inbox/{alert_id}.json file.

    Returns:
        dict with keys: alert_id, md_path, had_diagnostics, errors
    """
    logger.info(f"[Processor] Reading inbox: {inbox_path}")

    # ── 1. Load inbox JSON ──────────────────────────────────────────────────
    try:
        inbox_data = json.loads(inbox_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to read/parse inbox file: {e}")

    alert     = inbox_data.get("alert", {})
    source    = inbox_data.get("source", "unknown")
    alert_id  = alert.get("alert_id") or alert.get("device") or inbox_path.stem
    device    = alert.get("device", "")
    severity  = alert.get("severity", "warning")
    message   = alert.get("message", alert.get("description", ""))

    logger.info(f"[Processor] Processing alert_id={alert_id} device={device} severity={severity}")

    # ── 2. Resolve NetBox info ──────────────────────────────────────────────
    nb_info = {}
    if device:
        nb_info = resolve_device_netbox(device)
        logger.info(f"[NetBox] {device} → status={nb_info.get('status','?')} site={nb_info.get('site','?')}")

    # ── 3. Resolve mgmt IP (NetBox → LAB_MGMT_IPS check) ───────────────────
    # mgmt_ip can come from: alert dict (static mapping), NetBox primary_ip, or NetBox custom field
    mgmt_ip: Optional[str] = alert.get("mgmt_ip") or nb_info.get("primary_ip") or ""
    # Strip vrf/rd prefix if present (e.g. "192.168.1.201" from "192.168.1.201 (vrf:Management)")
    if mgmt_ip:
        mgmt_ip = re.split(r"\s+\(", str(mgmt_ip))[0].strip()

    is_lab_device = mgmt_ip in LAB_MGMT_IPS if mgmt_ip else False
    logger.info(f"[Processor] mgmt_ip={mgmt_ip} is_lab={is_lab_device}")

    # ── 4. Run router diagnostics (lab devices only) ─────────────────────────
    diag_results: Optional[dict] = None
    if is_lab_device and device:
        diag_results = run_router_diagnostics(
            hostname     = nb_info.get("name", device),
            mgmt_ip      = mgmt_ip,
            severity     = severity,
            command_set  = "full" if severity != "critical" else "quick",
            alert_id     = alert_id,   # passed so raw output is saved to diagnostics_raw/
        )
        logger.info(
            f"[Diag] {device} ({mgmt_ip}): "
            f"{sum(1 for c in diag_results['commands'] if c['ok'])}/"
            f"{diag_results['commands_run']} commands OK"
        )

    # ── 5. Build merged LLM prompt (enrichment + diagnostics together) ─────
    nb_text    = netbox_summary(nb_info)
    diag_text  = ""
    if diag_results:
        diag_text = format_diagnostic_report(diag_results, alert_message=message, severity=severity)
    elif is_lab_device:
        diag_text = "_No diagnostics run (device not in LAB_MGMT_IPS or diagnostics failed)_"

    prompt = f"""You are a senior network NOC analyst. Produce a structured Telegram briefing for an on-call engineer.

ALERT DETAILS
  Alert ID:  {alert_id}
  Device:    {device}
  Severity:  {severity}
  Source:    {source}
  Message:   {message}

NETBOX DEVICE INFO
  {nb_text}

ROUTER DIAGNOSTICS (from Cisco IOS)
  {diag_text if diag_text else '(none — production device or diagnostics skipped)'}

YOUR TASK:
1. Assess the alert severity and likely root cause given the device info and diagnostics.
2. If lab device: cite SPECIFIC evidence from the diagnostic output (interface names, BGP neighbor states, line protocol, etc.).
3. If production device: use NetBox info + alert message to give the best possible assessment.
4. Give 2 concrete next steps (check X, verify Y, escalate to Z).
5. Flag anything that needs urgent attention tonight vs. can wait until business hours.

FORMAT: Telegram Markdown. Be direct. Keep it under 400 words. Start with the finding immediately — no preamble like "Here is your briefing".
"""

    # ── 6. Call MiniMax ONCE ─────────────────────────────────────────────────
    llm_enrichment = ""
    errors: list[str] = []
    try:
        llm_enrichment = _call_minimax(prompt)
        logger.info(f"[MiniMax] Got {len(llm_enrichment)} chars of enrichment")
    except Exception as e:
        logger.warning(f"[MiniMax] LLM call failed: {e}")
        errors.append(f"MiniMax: {e}")
        llm_enrichment = f"_⚠️ LLM enrichment failed: {e}_"

    # ── 7. Build and write .md file ─────────────────────────────────────────
    md_content = build_md_content(inbox_data, nb_info, diag_results, llm_enrichment)
    ctx_path = CTX_DIR / f"{alert_id}.md"

    # Idempotency: if diag section already exists, replace it cleanly
    if ctx_path.exists():
        existing = ctx_path.read_text(encoding="utf-8")
        # Remove any existing "## Router Diagnostics" section so we don't duplicate
        marker = "## Router Diagnostics"
        if marker in existing:
            parts = existing.split(marker)
            # Keep everything before the marker (Alert Info + Enrichment + Raw)
            existing = parts[0]
            # Strip trailing whitespace from preserved part
            md_content = existing.rstrip() + "\n\n---\n\n" + marker + "\n" + marker.join(parts[1:])
        else:
            md_content = existing  # don't overwrite if already complete

    ctx_path.write_text(md_content, encoding="utf-8")
    logger.info(f"[Processor] Wrote {ctx_path}  ({len(md_content)} bytes)")

    # ── 8. Write processing log ──────────────────────────────────────────────
    log_path = LOG_DIR / f"{alert_id}.log"
    log_path.write_text(
        f"Processed at: {datetime.now(timezone.utc).isoformat()}\n"
        f"alert_id:     {alert_id}\n"
        f"device:       {device}\n"
        f"severity:     {severity}\n"
        f"mgmt_ip:      {mgmt_ip}\n"
        f"is_lab:       {is_lab_device}\n"
        f"netbox:       {json.dumps(nb_info, default=str)}\n"
        f"errors:       {errors}\n"
        f"diag_commands_run: {diag_results.get('commands_run', 0) if diag_results else 0}\n",
        encoding="utf-8",
    )

    return {
        "alert_id":         alert_id,
        "md_path":         str(ctx_path),
        "had_diagnostics": is_lab_device and diag_results is not None,
        "errors":          errors,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NOC MD — process an alert from inbox → context .md")
    parser.add_argument("alert_id", help="Alert ID (inbox filename without .json)")
    args = parser.parse_args()

    inbox_path = INBOX_DIR / f"{args.alert_id}.json"
    if not inbox_path.exists():
        # Try as exact path
        inbox_path = Path(args.alert_id)

    if not inbox_path.exists():
        print(f"[ERROR] Inbox file not found: {inbox_path}", flush=True)
        sys.exit(1)

    print(f"[NOC MD] Processing {inbox_path} ...", flush=True)
    result = process_alert_from_inbox(inbox_path)
    print(f"[NOC MD] Done → {result['md_path']}", flush=True)
    if result["errors"]:
        print(f"[NOC MD] Errors: {result['errors']}", flush=True)
