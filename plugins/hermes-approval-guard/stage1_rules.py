"""Stage 1: Risk signal extraction — no denial, provides context for LLM prompts only.

Design principle:
  Stage 1 never makes hardcoded DENY decisions. All "suspicious" operations
  are passed to the LLM for semantic judgment. This module only extracts
  risk signals as reference context for the LLM, never decides for it.

For terminal commands:
  Imports system tools.approval HARDLINE / DANGEROUS regex detection,
  extracts matched rule descriptions + pattern_key (for approve_session pre-marking).

Returns:
  dict — context information injected into the LLM prompt
    {
      "signals": [str, ...],              # Risk signal descriptions (for LLM)
      "dangerous_pattern_keys": [str, ..],# Matched DANGEROUS pattern keys (excludes HARDLINE)
      "tool_name": str,
    }
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Sensitive paths (context hints only, no hardcoded denial) ─────
_SENSITIVE_PATHS = (
    "/etc/",
    "/boot/",
    "/sys/",
    "/proc/",
    "/dev/",
    "~/.ssh/",
    "~/.gnupg/",
)

_SENSITIVE_FILE_NAMES = frozenset({
    ".env", "config.yaml", "id_rsa", "id_ed25519",
    "id_rsa.pub", "authorized_keys",
})

# ── delegate_task danger keywords (context hints only) ────────────
_DANGER_KEYWORDS = (
    "delete all", "rm -rf /", "format disk",
    "wipe system", "destroy everything",
)


def _get_sensitive_signals_for_write(path: str) -> List[str]:
    """Extract sensitive signals for write operations (descriptions only, no denial)."""
    import os
    signals: List[str] = []
    expanded = os.path.expanduser(path)

    for sp in _SENSITIVE_PATHS:
        if expanded.startswith(os.path.expanduser(sp)):
            signals.append(f"Target path is in a system-critical directory ({sp}...)")
            break

    basename = path.split("/")[-1].lower()
    if basename in _SENSITIVE_FILE_NAMES:
        signals.append(f"Target file is sensitive ({basename}), may contain keys or config")

    return signals


def _get_terminal_risk_signals(command: str) -> Tuple[List[str], List[str]]:
    """Run dangerous/hardline regex + tirith detection on terminal commands.

    Imports system tools.approval detection functions + tirith security scanner,
    reusing 12 HARDLINE + 47 DANGEROUS regex patterns plus tirith's richer rule set
    (pipe-to-interpreter, sudo abuse, etc.).

    Returns:
        (signals, dangerous_pattern_keys)
          - signals: Human-readable risk descriptions (injected into LLM prompt)
          - dangerous_pattern_keys: Matched DANGEROUS pattern keys (for approve_session pre-marking)
            Note: HARDLINE keys are NOT included — HARDLINE should never be pre-approved
    """
    signals: List[str] = []
    pattern_keys: List[str] = []
    try:
        from tools.approval import (
            detect_dangerous_command,
            detect_hardline_command,
        )

        # Hardline detection (destructive commands — never pre-marked, always unconditionally blocked)
        is_hardline, hardline_desc = detect_hardline_command(command)
        if is_hardline:
            signals.append(f"WARNING: HARDLINE pattern triggered: {hardline_desc}")

        # Dangerous detection (approvable patterns — keep pattern_key for pre-marking)
        is_dangerous, pattern_key, description = detect_dangerous_command(command)
        if is_dangerous:
            signals.append(f"WARNING: Dangerous pattern triggered: {description}")
            pattern_keys.append(pattern_key)

        # --- Tirith security scan (richer rules: pipe-to-interpreter, sudo abuse, etc.) ---
        try:
            from tools.tirith_security import check_command_security
            tirith_result = check_command_security(command)
            if tirith_result.get("action") in {"warn", "block"}:
                findings = tirith_result.get("findings") or []
                for f in findings[:10]:  # Cap at 10 findings to avoid prompt bloat
                    desc = f.get("description") or f.get("message") or str(f)
                    rule_id = f.get("rule_id", "tirith")
                    signals.append(
                        f"WARNING: Tirith security scan: [{rule_id}] {desc}"
                    )
        except ImportError:
            pass  # tirith not installed — skip, regex patterns alone are sufficient
        except Exception as exc:
            logger.warning("Tirith scan failed: %s, falling back to regex only", exc)

        if not signals:
            signals.append("No known dangerous patterns matched (likely a safe routine operation)")

    except ImportError:
        signals.append("(Unable to load system danger command detection module, manual judgment required)")
    except Exception as exc:
        signals.append(f"(Danger command detection error: {exc}, manual judgment required)")

    return signals, pattern_keys


def _get_delegate_risk_signals(goal: str) -> List[str]:
    """Extract delegate_task danger signals."""
    signals: List[str] = []
    if goal and any(kw in goal.lower() for kw in _DANGER_KEYWORDS):
        signals.append("Sub-task goal contains explicit destructive operation keywords")
    return signals


def extract_context(
    tool_name: str,
    args: Dict[str, Any],
    cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Extract risk signal context for a tool call.

    Never makes hardcoded denial. All signals are reference info for the LLM prompt only.

    Returns:
        {
            "signals": [str, ...],                # Risk signal descriptions (for LLM)
            "dangerous_pattern_keys": [str, ...], # Terminal DANGEROUS pattern keys (for pre-marking)
            "tool_name": str,
        }
    """
    signals: List[str] = []
    dangerous_pattern_keys: List[str] = []

    if tool_name in {"write_file", "patch"}:
        path = args.get("path", "")
        if path:
            signals.extend(_get_sensitive_signals_for_write(path))

    elif tool_name == "terminal":
        command = args.get("command", "")
        if command:
            sigs, pks = _get_terminal_risk_signals(command)
            signals.extend(sigs)
            dangerous_pattern_keys = pks

    elif tool_name == "delegate_task":
        goal = args.get("goal", "")
        if goal:
            signals.extend(_get_delegate_risk_signals(goal))

    return {
        "signals": signals,
        "dangerous_pattern_keys": dangerous_pattern_keys,
        "tool_name": tool_name,
    }
