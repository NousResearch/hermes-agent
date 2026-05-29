"""Structured denial feedback templates — cause, alternatives, trust escalation path.

Three levels:
  HARDLINE  — unconditional block (system protection)
  DENY       — overridable block (user must acknowledge risk)
  ESCALATE   — requires manual user decision
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_deny_message(
    tool_name: str,
    args: Dict[str, Any],
    reason: str,
    verdict: str = "DENY",
) -> Dict[str, str]:
    """Build structured block message.

    The main agent can:
    1. Understand why it was blocked
    2. Know alternatives
    3. Choose to override (reply "OK" or specific phrase)
    """
    lines = ["=" * 60]
    lines.append("SAFETY REVIEW: HARD PROTECTION" if verdict == "HARDLINE"
                 else "SAFETY REVIEW: BLOCKED")
    lines.append("=" * 60)
    lines.append(f"Tool: {tool_name}")
    lines.append(f"Reason: {reason}")

    if verdict == "HARDLINE":
        lines.append("⚠️  HARD PROTECTION — action blocked unconditionally")
        lines.append("    System-critical path, cannot override.")
    else:
        approval_id = f"{tool_name}:{reason}"[:40]
        lines.append("💡 How to override:")
        lines.append(f'    Reply "OK allow {approval_id}" → approve this time')
        lines.append('    Reply "always allow this"     → permanently trust this pattern')

    # Add alternatives
    suggestions = _get_alternatives(tool_name, args)
    if suggestions:
        lines.append("🔀 Alternatives:")
        for s in suggestions:
            lines.append(f"    • {s}")

    lines.append("=" * 60)
    return {"action": "block", "message": "\n".join(lines)}


def build_hardline_message(
    call_signature: str,
    reason_key: str,
    explanation: str,
) -> Dict[str, str]:
    """Build hard-protection block message."""
    approval_id = f"{reason_key}:{hash(call_signature) % 10000}"
    return {
        "action": "block",
        "message": "\n".join([
            "=" * 60,
            "⚠️  SYSTEM HARD PROTECTION — cannot override",
            "=" * 60,
            f"Action: {call_signature}",
            f"Reason: {explanation}",
            "    These paths/operations are critical system infrastructure,",
            "    modification may cause data loss or system unavailability.",
            f"    ID: {approval_id}",
            "=" * 60,
        ]),
    }


def _get_alternatives(
    tool_name: str,
    args: Dict[str, Any],
) -> List[str]:
    """Generate alternatives based on tool type and reason."""
    suggestions: List[str] = []
    path = args.get("path", "")

    if tool_name in ("write_file", "patch") and "/etc/" in path:
        suggestions.append("Write to project dir → deploy to system path via terminal after review")
    if path and any(kw in path for kw in (".env", "config.yaml", "id_rsa")):
        suggestions.append("Only modify .env.example template files")
        suggestions.append("Configure sensitive keys via hermes config set, not direct .env edits")
    if tool_name == "delegate_task":
        suggestions.append("Run ls first to verify target path")
        suggestions.append("Use rm (without -f) for per-file deletion, easier rollback")
        suggestions.append("Use mv to /tmp instead of direct delete, verify before cleanup")
    if tool_name == "execute_code":
        suggestions.append("Split destructive operations into separate small tasks, approve individually")
        suggestions.append("Declare safety boundaries explicitly in sub-agent goal")

    return suggestions
