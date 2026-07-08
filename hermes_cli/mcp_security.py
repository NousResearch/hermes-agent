"""Security checks for user-configured MCP server entries.

MCP stdio transports intentionally support arbitrary local commands so users can
run custom servers. This module does not try to sandbox that capability. It only
blocks the high-signal exfiltration shape from #45620: a shell interpreter whose
inline script invokes network egress tooling.
"""
from __future__ import annotations

import os
import re
import shlex
from typing import Any

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


def validate_mcp_server_entry(name: str, entry: dict[str, Any]) -> list[str]:
    """Return security warnings for an MCP server entry.

    Empty return means the entry is not suspicious under the narrow #45620
    exfiltration heuristic. This is intentionally not a whitelist: legitimate
    local MCPs can still use custom commands, Python scripts, npx, uvx, etc.
    """
    if not isinstance(entry, dict):
        return []

    command = entry.get("command")
    basename = _command_basename(command)
    if basename not in _SHELL_INTERPRETERS:
        return []

    script = _inline_script(entry.get("args"))
    if not script:
        return []

    if not _EGRESS_PATTERN.search(script):
        return []

    issue = (
        f"MCP server '{name}' uses shell interpreter '{command}' with network "
        "egress in args"
    )
    if _EXFIL_HINT_PATTERN.search(script):
        issue += " and exfiltration-shaped arguments"
    return [issue]


def is_mcp_server_entry_suspicious(name: str, entry: dict[str, Any]) -> bool:
    return bool(validate_mcp_server_entry(name, entry))
