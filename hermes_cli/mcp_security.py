"""Security checks for user-configured MCP server entries.

MCP stdio transports intentionally support arbitrary local commands so users can
run custom servers. This module does not try to sandbox that capability. It
blocks two high-signal abuse shapes seen in the wild:

1. The exfiltration shape from #45620: an interpreter (shell or a
   general-purpose language like python/node/perl/ruby/php) whose inline
   script invokes network egress.
2. The persistence shape from the June 2026 ``hermes-0day`` campaign: an
   interpreter whose inline script writes to OS persistence surfaces
   (``~/.ssh/authorized_keys``, ``/etc/ssh``, ``/etc/pam.d``, ``sudoers``,
   crontab, shell rc files). The campaign planted ``command: bash`` MCP entries
   whose payload appended an attacker SSH key to ``authorized_keys``; Hermes
   re-executed them on every cron tick / startup, re-installing the backdoor.
   The same shape works identically via ``command: python3`` (or any other
   inline-script interpreter) with no shell involved at all.

3. A hardcoded indicator-of-compromise (IOC) blocklist for that campaign — the
   attacker's ``hermes-0day`` SSH public key and source IPs. Any entry whose
   command/args/env carry an IOC is refused outright, regardless of shape, so a
   pre-planted ``config.yaml`` cannot spawn it.

These checks run BOTH at save time (``_save_mcp_server`` — dashboard API + CLI)
and at spawn time (``tools.mcp_tool._filter_suspicious_mcp_servers`` — discovery
/ cron / startup), so a hand-edited or pre-planted entry is also caught before
it can execute.
"""
from __future__ import annotations

import os
import re
import shlex
from typing import Any

# Matches interpreter basenames including versioned binaries (python3.11,
# python3.12, ruby3.2, perl5.36 — the norm under pyenv/homebrew/most distro
# packaging, where the unversioned name is often just a symlink). An earlier
# version of this check used an exact-name frozenset, which missed every
# versioned spelling — command: python3.11 skipped the egress/persistence
# checks below exactly like a bare, unrecognized command would, even though
# it runs the identical inline script a bare "python3" would.
_SCRIPT_INTERPRETER_PATTERN = re.compile(
    r'^(?:bash|sh|zsh|dash|fish|cmd(?:\.exe)?|powershell(?:\.exe)?|pwsh(?:\.exe)?'
    r'|python[23]?(?:\.\d+)*(?:\.exe)?|node(?:js)?(?:\.exe)?|deno'
    r'|perl[0-9]*(?:\.\d+)*(?:\.exe)?|ruby[0-9]*(?:\.\d+)*(?:\.exe)?|php(?:\.exe)?)$',
    re.IGNORECASE,
)

_EGRESS_PATTERN = re.compile(
    r"(?<![\w.-])(?:curl|wget|nc|ncat|socat)(?![\w.-])"
    r"|/dev/tcp/"
    r"|\bInvoke-WebRequest\b"
    r"|\bInvoke-RestMethod\b"
    r"|\bSystem\.Net\.WebClient\b"
    # Interpreter-native network egress, so the check above isn't only
    # effective against shell tooling now that _SCRIPT_INTERPRETERS
    # covers python/node/perl/ruby/php too.
    r"|\burllib\.request\b|\burlopen\("
    r"|\brequests\.(?:get|post|put|patch)\("
    r"|\bsocket\.(?:socket|create_connection)\("
    r"|\bhttp\.client\b|\bhttpx\b"
    r"|\brequire\(['\"]https?['\"]\)|\bfetch\(|\bXMLHttpRequest\b|\bnet\.connect\("
    r"|\bLWP::UserAgent\b|\bNet::HTTP\b|\bopen-uri\b"
    r"|\bcurl_init\(|\bfsockopen\(|\bfile_get_contents\(['\"]https?:",
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
    r"authorized_keys"               # SSH key persistence (the campaign's payload)
    r"|\.ssh/"                       # any write under ~/.ssh
    r"|/etc/ssh\b"                   # sshd_config / AuthorizedKeysCommand backdoor
    r"|/etc/pam\.d\b|pam_[\w-]+\.so" # PAM credential logger
    r"|/etc/sudoers"                 # sudoers escalation
    r"|/etc/cron|crontab\b"          # cron persistence
    r"|/etc/rc\.local|/etc/systemd"  # init / unit persistence
    r"|\.bashrc\b|\.bash_profile\b|\.profile\b|\.zshrc\b",  # shell rc backdoor
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
    return " ".join(parts)


def validate_mcp_server_entry(name: str, entry: dict[str, Any]) -> list[str]:
    """Return security warnings for an MCP server entry.

    Empty return means the entry is not suspicious. This is intentionally not a
    whitelist: legitimate local MCPs can still use custom commands, Python
    scripts, npx, uvx, etc. We block three narrow shapes only:

    * a known hermes-0day IOC anywhere in command/args/env (hardcoded blocklist);
    * an interpreter (shell, or python/node/perl/ruby/php run with an inline
      script) whose inline script invokes network egress (#45620);
    * an interpreter whose inline script writes to an OS persistence
      surface (June 2026 hermes-0day SSH/PAM/sudoers/cron shape).
    """
    if not isinstance(entry, dict):
        return []

    issues: list[str] = []

    # 1. Hardcoded IOC blocklist — applies regardless of command shape.
    flat = _entry_text(entry)
    for ioc in _IOC_SUBSTRINGS:
        if ioc in flat:
            issues.append(
                f"MCP server '{name}' contains a known hermes-0day "
                f"indicator-of-compromise ('{ioc}')"
            )
            # One IOC is enough to refuse; don't leak the full match list.
            return issues

    command = entry.get("command")
    basename = _command_basename(command)
    if not _SCRIPT_INTERPRETER_PATTERN.match(basename):
        return issues

    script = _inline_script(entry.get("args"))
    if not script:
        return issues

    # 2. Network exfiltration shape.
    if _EGRESS_PATTERN.search(script):
        issue = (
            f"MCP server '{name}' uses interpreter '{command}' with "
            f"network egress in args"
        )
        if _EXFIL_HINT_PATTERN.search(script):
            issue += " and exfiltration-shaped arguments"
        issues.append(issue)

    # 3. OS persistence shape (SSH key / PAM / sudoers / cron / rc files).
    if _PERSISTENCE_PATTERN.search(script):
        issues.append(
            f"MCP server '{name}' uses interpreter '{command}' to write "
            f"to an OS persistence surface (SSH keys / PAM / sudoers / cron / "
            f"shell rc) — this is the hermes-0day backdoor shape, not a real "
            f"MCP server"
        )

    return issues


def is_mcp_server_entry_suspicious(name: str, entry: dict[str, Any]) -> bool:
    return bool(validate_mcp_server_entry(name, entry))
