"""Structured denial feedback templates — includes reason, alternatives, trust escalation path.

Three levels:
  HARDLINE   — unconditional block (system protection)
  DENY       — overridable denial (requires user to acknowledge risk)
  ESCALATE   — requires manual user decision
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict


def build_deny_message(
    tool_name: str,
    args: Dict[str, Any],
    reason: str,
    verdict: str = "DENY",
) -> Dict[str, str]:
    """Build structured denial message.

    The main Agent receives this message and can:
    1. Understand why it was denied
    2. See alternative approaches
    3. Choose to override (reply with confirmation phrase)
    """
    approval_id = f"apr_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    lines = []
    lines.append("=" * 60)
    lines.append("SAFETY REVIEW: DENIED")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Tool:    {tool_name}")
    lines.append(f"Reason:  {reason}")
    lines.append(f"Ref:     {approval_id}")
    lines.append("")

    if verdict == "HARDLINE":
        lines.append("WARNING: HARD PROTECTION — this operation is unconditionally blocked")
        lines.append("    System-critical path protection, cannot be overridden.")
    else:
        lines.append("To override this block:")
        lines.append(f'    Reply "confirm allow {approval_id}"  -> allow this time')
        lines.append('    Reply "always allow this operation"   -> permanent trust for this pattern')

    suggestions = _get_suggestions(tool_name, args, reason)
    if suggestions:
        lines.append("")
        lines.append("Alternative approaches:")
        for s in suggestions:
            lines.append(f"    - {s}")

    lines.append("")
    return {"action": "block", "message": "\n".join(lines)}


def build_hardline_message(
    context: str,
    rule_id: str,
    description: str,
) -> Dict[str, str]:
    """Build hard-protection denial message."""
    approval_id = f"hardline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    lines = [
        "=" * 60,
        "SAFETY REVIEW: HARD PROTECTION — BLOCKED",
        "=" * 60,
        "",
        f"Operation: {context}",
        f"Rule:      {rule_id}",
        f"Reason:    {description}",
        "",
        "WARNING: This operation is blocked by system hard protection and cannot be overridden.",
        "    These paths/operations are system-critical infrastructure;",
        "    modifying them may cause data loss or system unavailability.",
        f"    Block ID: {approval_id}",
        "",
    ]
    return {"action": "block", "message": "\n".join(lines)}


def _get_suggestions(
    tool_name: str, args: Dict[str, Any], reason: str
) -> list[str]:
    """Generate alternative suggestions based on tool type and reason."""
    suggestions = []

    if tool_name in {"write_file", "patch"}:
        path = args.get("path", "")
        if "/etc/" in path:
            suggestions.append("Write to project directory -> review -> deploy to system path via terminal")
            suggestions.append(f"terminal: sudo cp <project_path> {path}")
        elif ".env" in path.split("/")[-1].lower():
            suggestions.append("Only modify .env.example template files")
            suggestions.append("Configure sensitive keys via hermes config set, not direct .env edits")

    elif tool_name == "terminal":
        command = args.get("command", "")
        if "rm" in command and "-rf" in command:
            suggestions.append("Run ls first to verify target path")
            suggestions.append("Use rm (without -f) to delete files one by one, easier to roll back")
            suggestions.append("Use mv to /tmp instead of rm, cleanup after confirmation")

    elif tool_name == "delegate_task":
        suggestions.append("Split destructive operations into independent small tasks, approve individually")
        suggestions.append("Declare safety boundaries explicitly in sub-agent goal")

    return suggestions
