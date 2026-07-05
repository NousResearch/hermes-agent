"""Security checks for user-configured MCP server entries.

MCP stdio transports intentionally support arbitrary local commands so users can
run custom servers. This module does not try to sandbox that capability. It
blocks high-signal abuse shapes while keeping legitimate local MCP workflows
flexible:

1. The exfiltration shape from #45620: a shell interpreter whose inline script
   invokes network egress tooling.
2. The persistence shape from the June 2026 ``hermes-0day`` campaign: a shell
   interpreter whose inline script writes to OS persistence surfaces
   (``~/.ssh/authorized_keys``, ``/etc/ssh``, ``/etc/pam.d``, ``sudoers``,
   crontab, shell rc files). The campaign planted ``command: bash`` MCP entries
   whose payload appended an attacker SSH key to ``authorized_keys``; Hermes
   re-executed them on every cron tick / startup, re-installing the backdoor.
3. A hardcoded indicator-of-compromise (IOC) blocklist for that campaign — the
   attacker's ``hermes-0day`` SSH public key and source IPs. Any entry whose
   command/args/env carry an IOC is refused outright, regardless of shape, so a
   pre-planted ``config.yaml`` cannot spawn it.
4. URL-based MCP transports are limited to local hosts unless
   ``mcp_security.allowed_hosts`` explicitly opts in to more.
5. Literal secret-shaped env/header values are refused; placeholders such as
   ``${MCP_GITHUB_API_KEY}`` are allowed.

These checks run BOTH at save time (``_save_mcp_server`` — dashboard API + CLI)
and at spawn time (``tools.mcp_tool._filter_suspicious_mcp_servers`` — discovery
/ cron / startup), so a hand-edited or pre-planted entry is also caught before
it can execute.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import shlex
from typing import Any
from urllib.parse import urlparse

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_ALLOWED_URL_SCHEMES = frozenset({"http", "https", "sse"})

_SHELL_INTERPRETERS = frozenset({
    "bash",
    "sh",
    "zsh",
    "dash",
    "fish",
    "cmd",
    "cmd.exe",
    "powershell",
    "powershell.exe",
    "pwsh",
    "pwsh.exe",
})

_EGRESS_PATTERN = re.compile(
    r"(?<![\w.-])(?:curl|wget|nc|ncat|socat)(?![\w.-])"
    r"|/dev/tcp/"
    r"|\bInvoke-WebRequest\b"
    r"|\bInvoke-RestMethod\b"
    r"|\bSystem\.Net\.WebClient\b",
    re.IGNORECASE,
)

_EXFIL_HINT_PATTERN = re.compile(
    r"\.env\b|--data-binary|--data-raw|\b-X\s+POST\b|\bPOST\b|<\s*[^\s]+",
    re.IGNORECASE,
)

# OS persistence surfaces an MCP server has no legitimate reason to write to.
# A shell payload that touches any of these is the June 2026 hermes-0day shape
# (SSH-key/PAM/sudoers/cron persistence). Matched anywhere in the inline script.
_PERSISTENCE_PATTERN = re.compile(
    r"authorized_keys"
    r"|\.ssh/"
    r"|/etc/ssh\b"
    r"|/etc/pam\.d\b|pam_[\w-]+\.so"
    r"|/etc/sudoers"
    r"|/etc/cron|crontab\b"
    r"|/etc/rc\.local|/etc/systemd"
    r"|\.bashrc\b|\.bash_profile\b|\.profile\b|\.zshrc\b",
    re.IGNORECASE,
)

# ── Indicators of compromise: June 2026 hermes-0day campaign ──────────────────
# Hardcoded so a pre-planted config.yaml (written by any vector) is refused at
# both save and spawn time. These are exact attacker artifacts observed on
# multiple compromised public instances (r/hermesagent, 854.media).
_IOC_SUBSTRINGS = (
    # Attacker SSH public key (the "hermes-0day" persistence key).
    "AAAAC3NzaC1lZDI1NTE5AAAAICBoh1oDC4DnsO1m5mJ4yfEKrQebaFh",
    "hermes-0day",
    # Attacker source IPs (China Telecom Gansu) seen authenticating with the key.
    "60.165.167.",
    "118.182.244.156",
    "61.178.123.196",
)

_SECRET_VALUE_PATTERN = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{16,}|xox[baprs]-[A-Za-z0-9-]{16,})"
)


@dataclass(frozen=True)
class MCPPolicy:
    allowed_servers: frozenset[str] = frozenset()
    allowed_hosts: frozenset[str] = _LOCAL_HOSTS


def _normalize_set(values: Any) -> frozenset[str]:
    if values is None:
        return frozenset()
    if isinstance(values, str):
        values = [values]
    try:
        return frozenset(str(value).strip().lower() for value in values if str(value).strip())
    except TypeError:
        return frozenset()


def _get_policy(config: dict[str, Any] | None = None) -> MCPPolicy:
    policy_config = (config or {}).get("mcp_security") or {}
    if not isinstance(policy_config, dict):
        policy_config = {}
    allowed_hosts = _normalize_set(policy_config.get("allowed_hosts")) or _LOCAL_HOSTS
    return MCPPolicy(
        allowed_servers=_normalize_set(policy_config.get("allowed_servers")),
        allowed_hosts=allowed_hosts,
    )


def _load_policy() -> MCPPolicy:
    try:
        from hermes_cli.config import load_config

        return _get_policy(load_config())
    except Exception:
        return MCPPolicy()


def _command_basename(command: Any) -> str:
    text = str(command or "").strip()
    if not text:
        return ""
    try:
        parts = shlex.split(text, posix=(os.name != "nt"))
    except ValueError:
        parts = text.split()
    first = parts[0] if parts else text
    return os.path.basename(first).lower()


def _inline_script(args: Any) -> str:
    if args is None:
        return ""
    if isinstance(args, (list, tuple)):
        return " ".join(str(item) for item in args)
    return str(args)


def _entry_text(entry: dict[str, Any]) -> str:
    """Flatten command + args + env values into one string for IOC scanning."""
    parts: list[str] = [str(entry.get("command") or "")]
    parts.append(_inline_script(entry.get("args")))
    env = entry.get("env")
    if isinstance(env, dict):
        parts.extend(str(v) for v in env.values())
    headers = entry.get("headers")
    if isinstance(headers, dict):
        parts.extend(str(v) for v in headers.values())
    return " ".join(parts)


def _env_or_header_contains_literal_secret(values: Any) -> bool:
    if not isinstance(values, dict):
        return False
    for value in values.values():
        text = str(value or "")
        # Placeholders such as ${MCP_GITHUB_API_KEY} are the expected safe shape.
        if "${" in text:
            continue
        if _SECRET_VALUE_PATTERN.search(text):
            return True
    return False


def _validate_url_policy(name: str, entry: dict[str, Any], policy: MCPPolicy) -> list[str]:
    url = entry.get("url")
    if not url:
        return []

    issues: list[str] = []
    parsed = urlparse(str(url))
    if parsed.scheme and parsed.scheme.lower() not in _ALLOWED_URL_SCHEMES:
        issues.append(f"MCP server '{name}' uses disallowed URL scheme '{parsed.scheme}'")

    host = (parsed.hostname or "").lower()
    if host and host not in policy.allowed_hosts:
        issues.append(f"MCP server '{name}' remote host '{host}' is not on the MCP host allowlist")
    return issues


def _validate_shell_payload(name: str, entry: dict[str, Any]) -> list[str]:
    command = entry.get("command")
    basename = _command_basename(command)
    if basename not in _SHELL_INTERPRETERS:
        return []

    script = _inline_script(entry.get("args"))
    if not script:
        return []

    issues: list[str] = []

    if _EGRESS_PATTERN.search(script):
        issue = (
            f"MCP server '{name}' uses shell interpreter '{command}' with "
            f"network egress in args"
        )
        if _EXFIL_HINT_PATTERN.search(script):
            issue += " and exfiltration-shaped arguments"
        issues.append(issue)

    if _PERSISTENCE_PATTERN.search(script):
        issues.append(
            f"MCP server '{name}' uses shell interpreter '{command}' to write "
            f"to an OS persistence surface (SSH keys / PAM / sudoers / cron / "
            f"shell rc) — this is the hermes-0day backdoor shape, not a real "
            f"MCP server"
        )

    return issues


def validate_mcp_server_entry(name: str, entry: dict[str, Any]) -> list[str]:
    """Return security warnings for an MCP server entry.

    Empty return means the entry is not suspicious under these heuristics.
    Local stdio MCPs remain intentionally flexible; URL-based MCPs are limited
    to local hosts unless ``mcp_security.allowed_hosts`` opts in to more.
    """
    if not isinstance(entry, dict):
        return []

    issues: list[str] = []
    policy = _load_policy()

    flat = _entry_text(entry)
    for ioc in _IOC_SUBSTRINGS:
        if ioc in flat:
            issues.append(
                f"MCP server '{name}' contains a known hermes-0day "
                f"indicator-of-compromise ('{ioc}')"
            )
            # One IOC is enough to refuse; don't leak the full match list.
            return issues

    if policy.allowed_servers and name.lower() not in policy.allowed_servers:
        issues.append(f"MCP server '{name}' is not on the MCP server allowlist")

    issues.extend(_validate_url_policy(name, entry, policy))
    issues.extend(_validate_shell_payload(name, entry))

    if _env_or_header_contains_literal_secret(entry.get("env")):
        issues.append(f"MCP server '{name}' contains a literal secret-shaped env value")
    if _env_or_header_contains_literal_secret(entry.get("headers")):
        issues.append(f"MCP server '{name}' contains a literal secret-shaped header value")

    return issues


def is_mcp_server_entry_suspicious(name: str, entry: dict[str, Any]) -> bool:
    return bool(validate_mcp_server_entry(name, entry))
