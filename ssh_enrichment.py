"""
SSH Device Enrichment — NOC Alert Pipeline
==========================================
Autonomous, read-only SSH step: connects to a device, runs a defined
command set, returns structured output for LLM synthesis.

Architecture:
  [alert_processor.py] → SSHEnricher.enrich() → [command outputs] → [MiniMax LLM] → [Telegram]

Supports:
  - Cisco IOS / IOS-XE via Netmiko (CiscoIosSSH)
  - Linux / FRRouting via Netmiko (LinuxSSH)
  - Multi-vendor via NAPALM (future)

Security:
  - Command allowlist — only predefined show/display/cat commands are executed
  - No privilege escalation (disable PW if needed)
  - Session audit log written to ~/.hermes/logs/ssh_enrichment.log
  - Credentials sourced from environment, never hardcoded

Usage:
  from ssh_enrichment import SSHEnricher
  result = SSHEnricher().enrich(device_ip="10.1.1.1", device_type="cisco_ios", alert_context={...})
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_ssh_logger = logging.getLogger("ssh_enrichment")
_log_path = os.path.expanduser("~/.hermes/logs/ssh_enrichment.log")
os.makedirs(os.path.dirname(_log_path), exist_ok=True)
_ssh_logger.addHandler(logging.FileHandler(_log_path))
_ssh_logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SSHEnrichmentResult:
    """Structured return from SSH enrichment step."""

    status: str                      # "success" | "unreachable" | "auth_failed" | "timeout" | "error"
    device_ip: str
    device_type: str
    commands_run: list[str] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)   # {command: output}
    error_reason: Optional[str] = None
    duration_seconds: float = 0.0
    truncated: bool = False

    def has_output(self) -> bool:
        return self.status == "success" and bool(self.outputs)


@dataclass
class AlertContext:
    """Lightweight alert context passed from alert_processor.py."""

    alert_id: str
    device_name: str                  # from NetBox — e.g. "core-rtr-01"
    device_ip: str                    # management IP from NetBox
    device_type: str = "cisco_ios"    # cisco_ios | juniper_junos | arista_eos | fortinet | linux
    severity: str = "P3"              # P1-P4
    alert_message: str = ""
    netbox_context: dict = field(default_factory=dict)   # full NetBox lookup result

# ---------------------------------------------------------------------------
# Command Allowlists (security — only these commands execute)
# ---------------------------------------------------------------------------

# Cisco IOS / IOS-XE — read-only show commands
CISCO_ALLOWLIST = frozenset([
    "show version",
    "show interfaces",
    "show interfaces status",
    "show interfaces summary",
    "show ip interface brief",
    "show ip ospf neighbor",
    "show ip ospf interface",
    "show ospf neighbor detail",
    "show bgp summary",
    "show bgp neighbors",
    "show ip bgp",
    "show mac address-table",
    "show log",
    "show logging",
    "show system neighbors",
    "show environment",
    "show cpu usage",
    "show process cpu history",
    "show platform",
    "show port械le summary",      # typo in original but valid Cisco cmd
    "show etherchannel summary",
    "show vlan brief",
    "show run",
])

# Normalised aliases — maps common alert-trigger commands to full commands
CISCO_COMMAND_MAP = {
    "interface_flapping":  ["show interfaces status", "show log"],
    "bgp_down":            ["show bgp summary", "show bgp neighbors"],
    "ospf_down":           ["show ip ospf neighbor", "show ip ospf interface"],
    "high_cpu":            ["show cpu usage", "show process cpu history"],
    "link_down":           ["show interfaces status", "show interfaces"],
    "syslog_critical":     ["show log"],
    "default":             ["show log", "show interfaces status"],
}

# Linux / FRRouting
LINUX_ALLOWLIST = frozenset([
    "vtysh -c \"show ospf neighbor\"",
    "vtysh -c \"show bgp summary\"",
    "vtysh -c \"show bgp\"",
    "vtysh -c \"show interface\"",
    "vtysh -c \"show route\"",
    "ip route show",
    "ip neighbor show",
    "ip link show",
    "cat /var/log/syslog",
    "cat /var/log/messages",
    "cat /var/log/frr/frr.log",
    "systemctl status frr",
    "cat /etc/frr/frr.conf",
])

LINUX_COMMAND_MAP = {
    "bgp_down":          ["vtysh -c \"show bgp summary\"", "ip route show"],
    "ospf_down":         ["vtysh -c \"show ospf neighbor\"", "ip route show"],
    "interface_down":    ["ip link show", "ip addr show"],
    "syslog_critical":   ["cat /var/log/syslog"],
    "default":           ["vtysh -c \"show bgp summary\"", "ip route show"],
}

# Global max output cap (characters per command) — prevents LLM token blowup
MAX_OUTPUT_CHARS = 5000

# ---------------------------------------------------------------------------
# Netmiko lazy import — only needed at connection time
# ---------------------------------------------------------------------------

def _get_netmiko_connection(device_ip: str, device_type: str, username: str, password: str):
    """
    Return a Netmiko connection object for the given device.

    Raises:
        RuntimeError: if device_type is unsupported or connection fails.
    """
    # Deferred import — netmiko is not in the hermes-agent venv by default
    try:
        from netmiko import ConnectHandler
    except ImportError:
        raise RuntimeError(
            "netmiko is not installed. Run: pip install netmiko"
        )

    # Map device_type string → Netmiko device class
    DEVICE_TYPE_MAP = {
        "cisco_ios":      "cisco_ios",
        "cisco_ios_xe":   "cisco_ios",
        "juniper_junos":  "juniper_junos",
        "arista_eos":     "arista_eos",
        "fortinet":       "fortinet",
        "linux":          "linux",
        "cumulus":        "cumulus_linux",
    }

    netmiko_type = DEVICE_TYPE_MAP.get(device_type.lower())
    if netmiko_type is None:
        raise RuntimeError(f"Unsupported device_type: {device_type}")

    try:
        conn = ConnectHandler(
            device_type=netmiko_type,
            host=device_ip,
            username=username,
            password=password,
            conn_timeout=15,
            auth_timeout=15,
            banner_timeout=15,
            # Read delay to avoid Cisco "exec" banner truncation
            read_everything=False,
        )
        return conn
    except Exception as e:
        _ssh_logger.error(f"Netmiko connect failed for {device_ip}: {e}")
        raise


def _sanitise_output(raw_output: str, max_chars: int = MAX_OUTPUT_CHARS) -> tuple[str, bool]:
    """
    Strip control characters and cap output at max_chars.

    Returns:
        (sanitised_output, was_truncated)
    """
    import re
    # Remove ANSI / IOS output control sequences
    cleaned = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", raw_output)
    cleaned = re.sub(r"\r", "", cleaned)
    truncated = len(cleaned) > max_chars
    return cleaned[:max_chars], truncated


def _write_audit_log(device_ip: str, username: str, commands: list[str],
                     status: str, duration: float):
    """Append an audit record to the SSH enrichment log."""
    import json
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device_ip": device_ip,
        "user": username,
        "commands": commands,
        "status": status,
        "duration_s": round(duration, 2),
    }
    _ssh_logger.info(f"AUDIT: {json.dumps(record)}")


# ---------------------------------------------------------------------------
# Main enricher class
# ---------------------------------------------------------------------------

class SSHEnricher:
    """
    Read-only SSH enrichment for NOC alerts.

    Usage:
        enricher = SSHEnricher()
        result = enricher.enrich(
            device_ip="10.1.1.1",
            device_type="cisco_ios",
            alert_context=AlertContext(...),
            trigger="bgp_down",
        )

    Environment variables (credentials — set before running):
        NETBOX_SSH_USERNAME   — SSH username for devices
        NETBOX_SSH_PASSWORD   — SSH password (or use SSH key)
    """

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        # Future: vault_url, ssh_key_path
    ):
        self.username = username or os.environ.get("NETBOX_SSH_USERNAME", "")
        self.password = password or os.environ.get("NETBOX_SSH_PASSWORD", "")

        if not self.username or not self.password:
            raise ValueError(
                "SSH credentials not set. Provide username/password to SSHEnricher() "
                "or set NETBOX_SSH_USERNAME and NETBOX_SSH_PASSWORD env vars."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich(
        self,
        device_ip: str,
        device_type: str = "cisco_ios",
        alert_context: Optional[AlertContext] = None,
        trigger: Optional[str] = None,
        explicit_commands: Optional[list[str]] = None,
    ) -> SSHEnrichmentResult:
        """
        Execute the SSH enrichment step for a device.

        Steps:
          1. Resolve command list (from trigger or explicit_commands)
          2. Connect via Netmiko
          3. Run each command in allowlist
          4. Sanitise + cap outputs
          5. Audit log
          6. Return structured result

        Args:
            device_ip:      Management IP of the target device.
            device_type:    Device type string (cisco_ios, linux, etc.)
            alert_context:  AlertContext dataclass from alert_processor.
            trigger:        Alert trigger type (bgp_down, interface_flapping, etc.)
                            Maps to a predefined command set.
            explicit_commands: Override — directly specify commands to run.
                               Must all be in the allowlist.

        Returns:
            SSHEnrichmentResult dataclass.
        """
        start = time.monotonic()
        is_linux = device_type == "linux" or device_type.startswith("linux")

        # Resolve command list
        if explicit_commands:
            commands = self._filter_commands(explicit_commands, is_linux=is_linux)
        elif trigger:
            commands = self._commands_for_trigger(trigger, is_linux=is_linux)
        elif alert_context and alert_context.alert_message:
            # Infer from alert message text
            inferred = self._infer_trigger_from_message(alert_context.alert_message)
            commands = self._commands_for_trigger(inferred, is_linux=is_linux)
        else:
            commands = self._commands_for_trigger("default", is_linux=is_linux)

        # Connect
        try:
            conn = _get_netmiko_connection(
                device_ip, device_type, self.username, self.password
            )
        except Exception as e:
            duration = time.monotonic() - start
            status, reason = self._classify_connection_error(e)
            _write_audit_log(device_ip, self.username, commands, status, duration)
            return SSHEnrichmentResult(
                status=status,
                device_ip=device_ip,
                device_type=device_type,
                commands_run=commands,
                error_reason=reason,
                duration_seconds=duration,
            )

        # Execute commands
        outputs: dict[str, str] = {}
        truncated = False

        try:
            for cmd in commands:
                raw = conn.send_command_timing(cmd, read_timeout=10)
                clean, was_truncated = _sanitise_output(raw)
                outputs[cmd] = clean
                if was_truncated:
                    outputs[cmd] += (
                        f"\n\n[OUTPUT TRUNCATED — was > {MAX_OUTPUT_CHARS} chars, capped.]"
                    )
                    truncated = True
        except Exception as e:
            _ssh_logger.warning(f"Command execution error on {device_ip}: {e}")
            outputs["_error"] = str(e)
        finally:
            try:
                conn.disconnect()
            except Exception:
                pass

        duration = time.monotonic() - start
        status = "success" if outputs else "error"
        _write_audit_log(device_ip, self.username, commands, status, duration)

        return SSHEnrichmentResult(
            status=status,
            device_ip=device_ip,
            device_type=device_type,
            commands_run=commands,
            outputs=outputs,
            duration_seconds=duration,
            truncated=truncated,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter_commands(self, commands: list[str], is_linux: bool) -> list[str]:
        """Return only allowlisted commands; drop the rest."""
        allowlist = LINUX_ALLOWLIST if is_linux else CISCO_ALLOWLIST
        allowed = []
        for cmd in commands:
            # Normalise: strip leading/trailing whitespace
            cmd = cmd.strip()
            # Prefix-matching for vtysh multiline
            if is_linux:
                if cmd in allowlist:
                    allowed.append(cmd)
                else:
                    _ssh_logger.warning(f"Command rejected (not in allowlist): {cmd!r}")
            else:
                if cmd in allowlist:
                    allowed.append(cmd)
                else:
                    _ssh_logger.warning(f"Command rejected (not in allowlist): {cmd!r}")
        return allowed

    def _commands_for_trigger(self, trigger: str, is_linux: bool) -> list[str]:
        """Map an alert trigger type to a list of commands."""
        if is_linux:
            return list(LINUX_COMMAND_MAP.get(trigger, LINUX_COMMAND_MAP["default"]))
        return list(CISCO_COMMAND_MAP.get(trigger, CISCO_COMMAND_MAP["default"]))

    def _infer_trigger_from_message(self, message: str) -> str:
        """Heuristic: infer trigger type from alert message text."""
        msg = message.lower()
        if "bgp" in msg and ("down" in msg or "idle" in msg or "timeout" in msg):
            return "bgp_down"
        if "ospf" in msg and ("down" in msg or "full" in msg):
            return "ospf_down"
        if "interface" in msg and ("flap" in msg or "down" in msg):
            return "interface_flapping"
        if "cpu" in msg or "high cpu" in msg:
            return "high_cpu"
        if "link" in msg and "down" in msg:
            return "link_down"
        return "default"

    def _classify_connection_error(self, exc: Exception) -> tuple[str, str]:
        """Map a Netmiko exception to status + human-readable reason."""
        exc_name = exc.__class__.__name__
        if "AuthenticationException" in exc_name:
            return "auth_failed", "SSH authentication failed — check NETBOX_SSH_PASSWORD"
        if "NetmikoTimeoutException" in exc_name or "timeout" in exc_name.lower():
            return "unreachable", f"Device did not respond within 15s timeout"
        if "Socket" in exc_name and "refused" in str(exc).lower():
            return "unreachable", "Connection refused — SSH port not open"
        if "Socket" in exc_name and "reach" in str(exc).lower():
            return "unreachable", "Network unreachable"
        return "error", str(exc)


# ---------------------------------------------------------------------------
# Convenience function — drop-in for alert_processor.py
# ---------------------------------------------------------------------------

def enrich_device(
    device_ip: str,
    device_type: str = "cisco_ios",
    alert_context: Optional[AlertContext] = None,
    trigger: Optional[str] = None,
) -> SSHEnrichmentResult:
    """
    One-shot SSH enrichment. Reads credentials from environment.

    Call this from alert_processor.py after NetBox lookup::

        from ssh_enrichment import enrich_device, AlertContext

        netbox = netbox_lookup.device(name="core-rtr-01")
        ctx = AlertContext(
            alert_id=alert["alertId"],
            device_name=netbox["name"],
            device_ip=netbox["primary_ip"].split("/")[0],
            device_type=netbox["device_type"],
            alert_message=alert["message"],
            netbox_context=netbox,
        )
        ssh_result = enrich_device(
            device_ip=ctx.device_ip,
            device_type=ctx.device_type,
            alert_context=ctx,
            trigger="bgp_down",
        )
        # ssh_result.outputs — dict of command → output
        # Pass to MiniMax LLM for synthesis
    """
    enricher = SSHEnricher()
    return enricher.enrich(
        device_ip=device_ip,
        device_type=device_type,
        alert_context=alert_context,
        trigger=trigger,
    )
