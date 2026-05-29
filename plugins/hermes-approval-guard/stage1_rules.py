"""Stage 1 rule engine — HARDLINE fast block (<1ms, zero LLM cost).

Design:
  Only hard-protection blocks. SAFE_TOOLS bypass and terminal delegation
  are handled in guard.py.

Returns:
  None            — rules uncertain, need LLM judgment
  {"action":"block","message":"..."} — HARDLINE deny
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# ── Sensitive path blocks (write_file / patch only) ────────────────────
SENSITIVE_PATHS = (
    "/etc/",
    "/boot/",
    "/sys/",
    "/proc/",
    "/dev/",
    "~/.ssh/",
    "~/.gnupg/",
)

HARDLINE_FILE_NAMES = frozenset({
    ".env", "config.yaml", "id_rsa", "id_ed25519",
    "id_rsa.pub", "authorized_keys",
})


def _is_sensitive_path(path: str) -> bool:
    """Check if path is under a sensitive directory."""
    import os
    expanded = os.path.expanduser(path)
    for sp in SENSITIVE_PATHS:
        if expanded.startswith(os.path.expanduser(sp)):
            return True
    return False


def _check_write_targets(tool_name: str, args: Dict[str, Any],
                         cfg: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Check write_file/patch target paths."""
    path = args.get("path", "")
    if not path:
        return None

    if _is_sensitive_path(path):
        from .feedback import build_hardline_message
        return build_hardline_message(
            f"{tool_name}(path={path})",
            "sensitive_path_write",
            f"Writing to system-critical path {path} — protected, direct modification may cause service unavailability"
        )

    basename = path.split("/")[-1].lower()
    if basename in HARDLINE_FILE_NAMES:
        from .feedback import build_hardline_message
        return build_hardline_message(
            f"{tool_name}(path={path})",
            "sensitive_file_write",
            f"Writing to sensitive file {basename} — may leak keys or corrupt configuration"
        )

    return None


def _check_delegate_danger(goal: str) -> Optional[Dict[str, str]]:
    """Check delegate_task goal for explicitly destructive operations."""
    danger_keywords = (
        "delete all", "rm -rf /", "format disk",
        "wipe system", "destroy everything",
    )
    if goal and any(kw in goal.lower() for kw in danger_keywords):
        from .feedback import build_hardline_message
        return build_hardline_message(
            f"delegate_task(goal={goal[:80]})",
            "dangerous_delegation",
            "Sub-task goal contains explicitly destructive operation — delegation denied"
        )
    return None


def fast_path(tool_name: str, args: Dict[str, Any],
              cfg: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """HARDLINE fast block.

    Only handles unambiguous dangerous operations. SAFE_TOOLS already
    bypassed in guard.py.

    Returns:
        None — rules uncertain, need LLM
        {"action": "block", "message": "..."} — HARDLINE deny
    """
    # ── write_file / patch path check ──────────────────────────────
    if tool_name in {"write_file", "patch"}:
        result = _check_write_targets(tool_name, args, cfg)
        if result:
            return result

    # ── delegate_task goal check ────────────────────────────────────
    if tool_name == "delegate_task":
        result = _check_delegate_danger(args.get("goal", ""))
        if result:
            return result

    # ── Rules uncertain → let LLM decide ────────────────────────────
    return None
