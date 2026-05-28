"""Human-readable dangerous command approval summaries for gateway UIs."""

from __future__ import annotations

import re
import shlex
from typing import Any, Mapping


_SCRIPT_FLAG_RE = re.compile(
    r"(?:^|[;&|()\s])(?P<interp>python\d*(?:\.\d+)?|node|ruby|perl|php|bash|sh|zsh)\b[^\n]*\s-(?P<flag>[ce])(?:\s|=|$)",
    re.IGNORECASE,
)


def _first_nonempty(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _truncate(value: str, limit: int = 220) -> str:
    value = " ".join((value or "").split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


def command_category(command: str, description: str = "") -> str:
    """Return a concise category for an approval-triggering command."""
    command_l = (command or "").lower()
    desc_l = (description or "").lower()
    match = _SCRIPT_FLAG_RE.search(command or "")
    if match or "script execution via -c" in desc_l or "script execution via -e" in desc_l:
        interp = match.group("interp") if match else "interpreter"
        flag = match.group("flag") if match else ("e" if "-e" in desc_l else "c")
        return f"Inline script execution ({interp} -{flag})"
    if re.search(r"\brm\b|\bunlink\b|\bshred\b", command_l):
        return "File deletion/destructive filesystem operation"
    if re.search(r"\bmv\b|\bcp\b|\bchmod\b|\bchown\b|\btee\b|>|\bpatch\b", command_l):
        return "Filesystem modification"
    if re.search(r"\bcurl\b|\bwget\b|\bssh\b|\bscp\b|\brsync\b", command_l):
        return "Network or remote command"
    if re.search(r"\bsudo\b|/etc/|/private/etc/|\bsystemctl\b", command_l):
        return "Elevated/system configuration command"
    if re.search(r"\bgit\s+(reset|clean|checkout|restore|push|commit|rebase)\b", command_l):
        return "Git repository mutation"
    return "Shell command flagged by safety detector"


def _what_it_does(command: str, description: str, category: str) -> str:
    if category.startswith("Inline script execution"):
        return (
            "Runs code embedded directly on the command line; the exact inline script is in the command block above."
        )
    if "deletion" in category.lower():
        return "May remove files or directories named in the command."
    if "filesystem" in category.lower():
        return "May create, overwrite, move, or change permissions on files referenced by the command."
    if "network" in category.lower():
        return "May contact remote hosts or execute content obtained over the network."
    if "git" in category.lower():
        return "May change repository history, working tree files, or remote repository state."
    if description and description != "dangerous command":
        return f"Matches safety detector: {description}."
    try:
        argv = shlex.split(command or "")
    except ValueError:
        argv = []
    if argv:
        return f"Runs `{argv[0]}` with the arguments shown above."
    return "Runs the shell command shown above."


def _risks(description: str, category: str) -> str:
    desc = description.strip() if description else ""
    if category.startswith("Inline script execution"):
        base = "Inline code can read/write files, spawn processes, or use credentials available to this session."
    elif "deletion" in category.lower():
        base = "Data loss if the target path or glob is broader than intended."
    elif "filesystem" in category.lower():
        base = "Accidental overwrite or permission changes may be hard to undo."
    elif "network" in category.lower():
        base = "Remote content or hosts may be untrusted; secrets could be exposed."
    elif "git" in category.lower():
        base = "Repository changes may discard work or affect collaborators if pushed."
    else:
        base = "May have side effects beyond normal read-only inspection."
    if desc and desc not in base:
        return _truncate(f"{base} Detector reason: {desc}.", 260)
    return base


def build_exec_approval_brief(
    command: str,
    description: str = "dangerous command",
    approval_data: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Build a structured approval brief with sensible fallbacks.

    ``approval_data`` may contain richer fields produced by future tool callers
    (goal, command_category/category, what_it_does, scope, risks,
    stop_rollback_plan/rollback_plan). Missing fields are inferred from the
    command and detector description so messaging popups never show only an
    opaque detector label.
    """
    data = _as_mapping(approval_data)
    nested = _as_mapping(data.get("approval_brief") or data.get("brief"))
    category = _first_nonempty(
        nested.get("command_category"),
        nested.get("category"),
        data.get("command_category"),
        data.get("category"),
    ) or command_category(command, description)
    goal = _first_nonempty(nested.get("goal"), data.get("goal"))
    if not goal:
        if category.startswith("Inline script execution"):
            goal = "Decide whether Hermes should run the inline script command."
        else:
            goal = "Decide whether Hermes should run this flagged command."
    what = _first_nonempty(
        nested.get("what_it_does"),
        nested.get("what"),
        data.get("what_it_does"),
        data.get("what"),
    ) or _what_it_does(command, description, category)
    scope = _first_nonempty(nested.get("scope"), data.get("scope"))
    if not scope:
        scope = "Current execution environment and any files, services, or accounts referenced by the command."
    risks = _first_nonempty(nested.get("risks"), data.get("risks")) or _risks(description, category)
    rollback = _first_nonempty(
        nested.get("stop_rollback_plan"),
        nested.get("rollback_plan"),
        nested.get("stop_plan"),
        data.get("stop_rollback_plan"),
        data.get("rollback_plan"),
        data.get("stop_plan"),
    )
    if not rollback:
        rollback = "Deny to stop before execution; if approved and changes occur, use checkpoints/backups/git restore where applicable."

    return {
        "Goal": _truncate(goal),
        "Command category": _truncate(category),
        "What it does": _truncate(what, 260),
        "Scope": _truncate(scope, 240),
        "Risks": _truncate(risks, 280),
        "Stop/rollback plan": _truncate(rollback, 260),
    }


def format_exec_approval_brief_text(
    command: str,
    description: str = "dangerous command",
    approval_data: Mapping[str, Any] | None = None,
) -> str:
    """Format the approval brief as plain labeled lines for chat surfaces."""
    brief = build_exec_approval_brief(command, description, approval_data)
    return "\n".join(f"{label}: {value}" for label, value in brief.items())
