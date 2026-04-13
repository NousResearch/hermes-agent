"""Permission engine plugin for Hermes Agent.

Five-layer permission evaluation:
  1. Whitelist — always allow safe read-only tools
  2. Blacklist — always deny explicitly blocked tools
  3. Registry metadata — ask for high-risk or confirmation-gated tools
  4. Risk rules — pattern-match dangerous commands and high-risk paths
  5. Default allow — anything not caught above is permitted
"""

import os
import re


def _get_tool_metadata(tool_name: str) -> dict:
    try:
        from tools.registry import registry
        return registry.get_metadata(tool_name)
    except Exception:
        return {}

# ── Layer 1: Whitelist ──────────────────────────────────────────────────
ALWAYS_ALLOW = {
    "read_file",
    "search_files",
    "session_search",
    "skills_list",
    "skill_view",
    "tool_search",
}

# ── Layer 2: Blacklist ──────────────────────────────────────────────────
ALWAYS_DENY: set[str] = set()

# ── Layer 4: Risk patterns ──────────────────────────────────────────────
DANGEROUS_COMMAND_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r">\s*/dev/",
    r"\bchmod\s+777\b",
    r"\bshutdown\b",
    r"\breboot\b",
]

ASK_COMMAND_PATTERNS = [
    r"\bsudo\b",
]

HIGH_RISK_PATHS = [
    "/etc/",
    "/usr/",
    "/bin/",
    "/sbin/",
    "/System/",
    "/Library/",
    os.path.expanduser("~/.ssh/"),
    os.path.expanduser("~/.gnupg/"),
]


def evaluate_permission(tool_name: str, args: dict) -> dict | None:
    """Return None to allow, or a dict with 'action' and 'reason'."""

    # Layer 1: whitelist
    if tool_name in ALWAYS_ALLOW:
        return None

    # Layer 2: blacklist
    if tool_name in ALWAYS_DENY:
        return {"action": "deny", "reason": f"Tool '{tool_name}' is blocked"}

    # Layer 3: registry metadata rules
    metadata = _get_tool_metadata(tool_name)
    if metadata.get("risk_level") == "high":
        return {"action": "ask", "reason": "high risk tool"}
    if metadata.get("mutates_external_world") is True:
        return {"action": "ask", "reason": "mutates external world"}
    if metadata.get("requires_confirmation_default") is True:
        return {"action": "ask", "reason": "requires confirmation"}

    # Layer 4: risk rules — dangerous / ask commands
    if tool_name in ("terminal", "execute_code"):
        cmd = args.get("command", "")
        for pattern in DANGEROUS_COMMAND_PATTERNS:
            if re.search(pattern, cmd):
                return {
                    "action": "deny",
                    "reason": f"Dangerous command detected: {cmd[:80]}",
                }
        for pattern in ASK_COMMAND_PATTERNS:
            if re.search(pattern, cmd):
                return {
                    "action": "ask",
                    "reason": f"High-risk command requires confirmation: {cmd[:80]}",
                }

    # Layer 4: risk rules — high-risk write paths
    if tool_name in ("write_file", "patch"):
        path = args.get("file_path", args.get("path", ""))
        for risk_path in HIGH_RISK_PATHS:
            if path.startswith(risk_path):
                return {
                    "action": "ask",
                    "reason": f"Writing to high-risk path: {path}",
                }

    # Layer 5: default allow
    return None


# ── Hook entry point ────────────────────────────────────────────────────
def pre_tool_call(tool_name: str, args: dict, **kwargs) -> dict | None:
    """Called by the plugin system before each tool invocation."""
    return evaluate_permission(tool_name, args)
