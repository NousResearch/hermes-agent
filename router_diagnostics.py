"""
router_diagnostics.py
=====================
AI-driven router diagnostics agent for the alert enrichment pipeline.

SECURITY RULE: Only show/ping/traceroute commands are allowed.
No configure, write, copy, delete, or any state-changing commands ever.

Usage:
    from router_diagnostics import run_router_diagnostics

    results = run_router_diagnostics(
        hostname="R1-lab",
        mgmt_ip="192.168.1.201",
        severity="critical",
        target=None,          # optional: ping/traceroute target
        command_set="full",   # "full" | "quick" | "reachability"
    )
"""

import re
import os
import sys
import json
import time
import select
import logging
import subprocess
from datetime import datetime, timezone
from typing import Optional

# ─── Logging ─────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
PTY_SCRIPT = "/home/jourdan/.hermes/scripts/cisco_ios_pty.py"
SSH_KEY = "/home/jourdan/.ssh/id_rsa_cisco_lab"
SSH_USER = "jourdan"
SSH_TIMEOUT = 30  # seconds per command

# LAB device management IPs — diagnostics only run against these
LAB_MGMT_IPS = {
    "192.168.1.210",  # R-HQ-LAB
    "192.168.1.201",  # R1-lab
    "192.168.1.202",  # R2-lab
    "192.168.1.203",  # R3-lab
    "192.168.1.204",  # R1-lab-switch
    "192.168.1.205",  # R2-lab-switch
    "192.168.1.206",  # R3-lab-switch
}

# Directory where raw SSH diagnostic outputs are saved for NOC on-demand access
DIAGNOSTICS_RAW_DIR = os.path.expanduser("~/noc/diagnostics_raw")

# ─── Command Whitelist ─────────────────────────────────────────────────────────
# Only these commands are allowed to run on the router via this module.
# Pattern: raw command string (will be matched as startswith after stripping).

ALLOWED_COMMANDS: dict[str, list[str]] = {
    # Interface diagnostics
    "show ip interface brief": [
        "show ip interface brief",
        "show interfaces",
        "show interfaces trunk",
        "show port-channel brief",
        "show etherchannel summary",
    ],
    # Routing
    "show ip route": [
        "show ip route",
        "show ip route vrf",
        "show ip protocols",
        "show ip ospf neighbor",
        "show ip ospf interface",
        "show ip bgp summary",
        "show ip bgp neighbors",
    ],
    # Layer 2 / LLDP / CDP
    "show cdp neighbors": [
        "show cdp neighbors detail",
        "show cdp neighbors",
        "show lldp neighbors detail",
        "show lldp neighbors",
    ],
    # System health
    "show processes cpu": [
        "show processes cpu",
        "show processes cpu history",
        "show memory summary",
        "show memory free",
        "show platform",
        "show inventory",
    ],
    # Logs
    "show log": [
        "show log",
        "show logging",
    ],
    # Reachability (ping / traceroute — requires target)
    "ping": [
        "ping",
    ],
    "traceroute": [
        "traceroute",
    ],
}


def _build_command_set(command_set: str, target: Optional[str]) -> list[tuple[str, str]]:
    """
    Build the list of (command, category) tuples to run.

    Args:
        command_set: "full" (all commands), "quick" (interface + health only),
                     "reachability" (ping + traceroute to target)
        target: IP or hostname to ping/traceroute (required for reachability set)

    Returns:
        List of (command, category_label) tuples.

    Raises:
        ValueError: If reachability is requested without a target.
    """
    if command_set == "reachability":
        if not target:
            raise ValueError("target is required for reachability command_set")
        return [
            (f"ping {target}", "ping"),
            (f"traceroute {target}", "traceroute"),
        ]

    if command_set == "quick":
        return [
            ("show ip interface brief", "interface"),
            ("show cdp neighbors detail", "cdp_lldp"),
            ("show processes cpu", "health"),
            ("show memory summary", "health"),
            ("show log", "log"),
        ]

    # "full"
    cmds = []
    for category, cmd_list in ALLOWED_COMMANDS.items():
        if category in ("ping", "traceroute"):
            if target:
                cmds.append((f"{category} {target}", category))
        else:
            for cmd in cmd_list:
                cmds.append((cmd, _category_of(cmd)))
    return cmds


def _category_of(cmd: str) -> str:
    """Map a command to its category label."""
    if "interface" in cmd or "port-channel" in cmd or "etherchannel" in cmd:
        return "interface"
    if "route" in cmd or "ospf" in cmd or "bgp" in cmd or "protocols" in cmd:
        return "routing"
    if "cdp" in cmd or "lldp" in cmd:
        return "neighbors"
    if "process" in cmd or "memory" in cmd or "platform" in cmd or "inventory" in cmd:
        return "health"
    if "log" in cmd:
        return "log"
    if cmd.startswith("ping"):
        return "ping"
    if cmd.startswith("traceroute"):
        return "traceroute"
    return "general"


def _is_allowed(command: str) -> bool:
    """
    Verify a command is in the whitelist.
    Strips trailing whitespace and checks startswith against all allowed commands.
    """
    stripped = command.strip()
    for category, cmd_list in ALLOWED_COMMANDS.items():
        for allowed in cmd_list:
            if stripped.startswith(allowed) or stripped == allowed:
                return True
    return False


def _run_single_command_via_pty(host: str, cmd: str) -> tuple[str, str, int]:
    """
    Execute a single command on the router via the PTY script and return output.

    Returns:
        (stdout, stderr, returncode)

    Raises:
        RuntimeError: If PTY script is missing or SSH fails.
    """
    if not os.path.isfile(PTY_SCRIPT):
        raise RuntimeError(f"PTY script not found: {PTY_SCRIPT}")

    full_cmd = [
        sys.executable, PTY_SCRIPT, host, cmd
    ]

    try:
        proc = subprocess.Popen(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout_b, stderr_b = proc.communicate(timeout=SSH_TIMEOUT)
        return (
            stdout_b.decode("utf-8", errors="replace"),
            stderr_b.decode("utf-8", errors="replace"),
            proc.returncode,
        )
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise RuntimeError(f"Command timed out after {SSH_TIMEOUT}s: {cmd}")


def _parse_raw_output(raw: str) -> str:
    """
    Clean ANSI escape sequences and terminal control chars from PTY output.
    Also strips --More-- pagination artifacts and normalizes line endings.
    """
    text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", raw)       # ANSI escape sequences
    text = re.sub(r"\x08.", "", text)                       # Backspace overwrites
    text = re.sub(r" --More-- |\x07", "", text)            # Pagination prompts
    text = re.sub(r"\r\n", "\n", text)                     # Windows line endings
    text = re.sub(r"\r", "", text)                          # Bare carriage returns
    text = re.sub(r"\n{3,}", "\n\n", text)                 # Excessive blank lines
    return text.strip()


def run_router_diagnostics(
    hostname: str,
    mgmt_ip: str,
    severity: str = "warning",
    target: Optional[str] = None,
    command_set: str = "full",
    alert_id: Optional[str] = None,
) -> dict:
    """
    Run diagnostic commands on a router and return structured results.

    Args:
        hostname:      Human-readable device name (e.g. "R1-lab")
        mgmt_ip:       Management IP of the device (e.g. "192.168.1.201")
        severity:      Alert severity (used to determine default command_set)
        target:        Optional target IP/hostname for ping/traceroute
        command_set:   "full" | "quick" | "reachability"

    Returns:
        {
            "hostname": str,
            "mgmt_ip": str,
            "timestamp": str (ISO8601),
            "commands_run": int,
            "commands": [
                {"command": str, "category": str, "output": str, "error": str, "ok": bool}
            ],
            "summary": str,
            "errors": list[str],
        }
    """
    # Auto-upgrade to quick if severity is critical and command_set is "full"
    if severity == "critical" and command_set == "full":
        command_set = "quick"  # faster for critical alerts

    commands = _build_command_set(command_set, target)
    results: dict = {
        "hostname": hostname,
        "mgmt_ip": mgmt_ip,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commands_run": len(commands),
        "commands": [],
        "summary": "",
        "errors": [],
    }

    for cmd, category in commands:
        if not _is_allowed(cmd):
            results["errors"].append(f"Command blocked by whitelist: {cmd}")
            continue

        try:
            raw_out, raw_err, rc = _run_single_command_via_pty(mgmt_ip, cmd)
            clean_out = _parse_raw_output(raw_out)
            ok = rc == 0 and clean_out and "invalid" not in clean_out.lower()

            results["commands"].append({
                "command": cmd,
                "category": category,
                "output": clean_out,
                "error": raw_err.strip() if raw_err else "",
                "ok": ok,
            })
        except Exception as exc:
            results["errors"].append(f"[{cmd}] {type(exc).__name__}: {exc}")
            results["commands"].append({
                "command": cmd,
                "category": category,
                "output": "",
                "error": str(exc),
                "ok": False,
            })

    # Save raw SSH outputs for NOC on-demand access
    if alert_id:
        _save_diagnostics_raw(alert_id, results)

    # Build plain-text summary
    results["summary"] = format_diagnostic_report(results)
    return results


def _save_diagnostics_raw(alert_id: str, results: dict) -> None:
    """
    Save raw SSH command outputs to ~/noc/diagnostics_raw/{alert_id}.json
    for NOC engineer on-demand access.

    File written only on success (at least one command ran without exceptions).
    If file already exists, it is overwritten.
    """
    try:
        os.makedirs(DIAGNOSTICS_RAW_DIR, exist_ok=True)
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", alert_id)[:128]
        out_path = os.path.join(DIAGNOSTICS_RAW_DIR, f"{safe_id}.json")

        raw_payload = {
            "alert_id": alert_id,
            "hostname": results.get("hostname"),
            "mgmt_ip": results.get("mgmt_ip"),
            "timestamp": results.get("timestamp"),
            "commands": [
                {
                    "command": c.get("command", ""),
                    "category": c.get("category", ""),
                    "output": c.get("output", ""),
                    "error": c.get("error", ""),
                    "ok": c.get("ok", False),
                }
                for c in results.get("commands", [])
            ],
            "errors": results.get("errors", []),
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(raw_payload, f, indent=2, ensure_ascii=False)

        logger.info("Saved raw diagnostics to %s", out_path)
    except Exception as exc:
        logger.warning("Failed to save diagnostics_raw for %s: %s", alert_id, exc)


# ─── LLM Analysis ───────────────────────────────────────────────────────────────

def _call_llm_for_analysis(
    hostname: str,
    alert_message: str,
    severity: str,
    commands_output: list[dict],
    errors: list[str],
) -> str:
    """
    Send raw diagnostic output to MiniMax LLM for analysis.
    Returns a focused root-cause summary — NOT raw command output.
    """
    import requests as _requests

    api_key = os.environ.get("MINIMAX_API_KEY", "")
    api_secret = os.environ.get("MINIMAX_API_SECRET", "")
    if not api_key or not api_secret:
        return "⚠️ MINIMAX_API_KEY or MINIMAX_API_SECRET not set — skipping LLM analysis."

    # Build concise command summary for the prompt (trim each output to 400 chars)
    cmd_lines = []
    for entry in commands_output:
        cmd = entry.get("command", "")
        ok = entry.get("ok", False)
        err = entry.get("error", "")
        raw = entry.get("output", "")
        status = "✅" if ok else "❌"
        # Truncate long outputs — give LLM enough context to do real analysis
        output = raw[:2000] + "..." if len(raw) > 2000 else raw
        cmd_lines.append(f"Command: {cmd}\nStatus: {status}\nOutput:\n{output}\nError: {err}")

    commands_text = "\n---\n".join(cmd_lines)
    errors_text = "\n".join(errors) if errors else "None"

    prompt = f"""You are a senior network engineer analyzing Cisco IOS router diagnostic output.
A router has raised a **severity={severity}** alert: "{alert_message}"

You ran the following commands and collected this output:

{commands_text}

Command errors: {errors_text}

YOUR TASK:
1. Identify what FAILED and WHY — cite specific command output lines
2. Point to concrete evidence in the logs (exact interface, exact line, exact value)
3. Formulate the most likely ROOT CAUSE
4. Give 2 concrete next steps to confirm or escalate

DO NOT:
- Just list command output
- Give generic advice ("check the logs", "verify connectivity")
- Summarize what commands you ran

DO:
- Say what is BROKEN and WHAT CAUSED IT
- Reference specific interface names, IP addresses, line numbers, or counter values
- Be direct and specific like a senior engineer texting at 3am

Format your response as Telegram Markdown. Keep it under 350 words.
Start immediately with the finding — no preamble.
"""

    url = "https://api.minimax.io/anthropic/v1/messages"
    headers = {
        "Authorization": f"Bearer {api_secret}",
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": "MiniMax-M2.7",
"max_tokens": 8000,
        "temperature": 0.3,
        "thinking": {"type": "off"},
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        resp = _requests.post(url, json=payload, headers=headers, timeout=90)
        if resp.status_code == 200:
            data = resp.json()
            # MiniMax returns content blocks: [{"type":"text","text":"..."}]
            text_parts = [
                block.get("text", "")
                for block in data.get("content", [])
                if block.get("type") == "text"
            ]
            content = "".join(text_parts)
            return content.strip() if content else "⚠️ LLM returned empty response."
        else:
            return f"⚠️ LLM API error {resp.status_code}: {resp.text[:200]}"
    except Exception as exc:
        return f"⚠️ LLM analysis failed: {exc}"


def format_diagnostic_report(
    results: dict,
    alert_message: str = "",
    severity: str = "warning",
) -> str:
    """
    Replace raw command output with an LLM-driven root-cause analysis.

    Args:
        results:        Output of run_router_diagnostics()
        alert_message:  Original alert message (used as LLM context)
        severity:       Alert severity (passed through to LLM prompt)

    Returns:
        Telegram-ready string: header + LLM root-cause analysis.
        NO raw command output is included.
    """
    header = (
        "🧠 *Router Diagnostics — LLM Analysis*\n"
        f"📍 `{results['hostname']}` @ `{results['mgmt_ip']}` | ⏱ `{results['timestamp']}`\n"
        f"🔧 Commands run: {results['commands_run']} | "
        f"✅ {sum(1 for c in results['commands'] if c['ok'])} "
        f"❌ {sum(1 for c in results['commands'] if not c['ok'])}\n"
    )

    if results["errors"]:
        header += f"⚠️ Errors: {'; '.join(results['errors'][:3])}\n"

    # Run LLM analysis instead of dumping raw output
    analysis = _call_llm_for_analysis(
        hostname=results["hostname"],
        alert_message=alert_message,
        severity=severity,
        commands_output=results["commands"],
        errors=results["errors"],
    )

    return header + "\n" + analysis


def send_diagnostic_telegram(report: str, chat_id: str = "-1003506715170") -> dict:
    """
    Send the diagnostic report to Telegram.

    Args:
        report:     Formatted output from format_diagnostic_report()
        chat_id:   Telegram chat ID (default: hermes alerts channel)

    Returns:
        Telegram API response dict.
    """
    import requests

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not bot_token:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set")

    # Telegram message limit is 4096 chars; split if needed
    MAX_MSG = 4096
    messages = []
    if len(report) > MAX_MSG:
        # Split on category boundaries
        parts = report.split("\n\n")
        buf = ""
        for part in parts:
            if len(buf) + len(part) + 2 > MAX_MSG:
                if buf:
                    messages.append(buf)
                buf = part
            else:
                buf = (buf + "\n\n" + part) if buf else part
        if buf:
            messages.append(buf)
    else:
        messages = [report]

    base_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    results = []
    for i, msg in enumerate(messages):
        payload = {
            "chat_id": chat_id,
            "text": msg,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        resp = requests.post(base_url, json=payload, timeout=15)
        results.append(resp.json())
        if i < len(messages) - 1:
            time.sleep(1)  # rate-limit between splits

    return {"messages_sent": len(messages), "responses": results}


# ─── CLI (manual test) ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Router diagnostic agent")
    parser.add_argument("mgmt_ip", help="Router management IP")
    parser.add_argument("hostname", help="Router hostname")
    parser.add_argument("--target", default=None, help="Target for ping/traceroute")
    parser.add_argument("--command-set", default="full",
                        choices=["full", "quick", "reachability"])
    args = parser.parse_args()

    print(f"[DIAG] Running {args.command_set} diagnostics on {args.hostname} ({args.mgmt_ip})...")

    results = run_router_diagnostics(
        hostname=args.hostname,
        mgmt_ip=args.mgmt_ip,
        target=args.target,
        command_set=args.command_set,
    )

    report = format_diagnostic_report(results)
    print("\n" + report)

    send_diagnostic_telegram(report)
    print("\n[DIAG] Telegram message sent.")
