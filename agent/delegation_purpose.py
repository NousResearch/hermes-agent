"""Delegation purpose contract shared by task validation, child prompts and tool dispatch."""
from __future__ import annotations

from typing import Any

RESEARCH_EVIDENCE = "research_evidence"
BOUNDED_IMPLEMENTATION = "bounded_implementation"
ALLOWED_DELEGATION_PURPOSES = frozenset({RESEARCH_EVIDENCE, BOUNDED_IMPLEMENTATION})
DEFAULT_DELEGATION_PURPOSE = RESEARCH_EVIDENCE

# Direct mutation surfaces that a research-only child never needs. Terminal and
# execute_code remain available for probes/calculation; existing approvals and
# task scope still govern their commands.
_RESEARCH_BLOCKED_TOOLS = frozenset({
    "write_file", "patch", "skill_manage", "memory", "cronjob_manage", "send_message",
})


def normalize_delegation_purpose(value: Any, *, default_missing: bool = True) -> str:
    text = str(value or "").strip().lower()
    if not text and default_missing:
        return DEFAULT_DELEGATION_PURPOSE
    if text not in ALLOWED_DELEGATION_PURPOSES:
        allowed = ", ".join(sorted(ALLOWED_DELEGATION_PURPOSES))
        raise ValueError(f"Delegation purpose must be one of: {allowed}.")
    return text


def purpose_prompt_block(purpose: str) -> str:
    normalized = normalize_delegation_purpose(purpose)
    if normalized == RESEARCH_EVIDENCE:
        return (
            "DELEGATION PURPOSE: research_evidence. Gather and verify evidence only. "
            "Do not create, edit, or delete project artifacts; direct mutation tools are Runtime-blocked."
        )
    return (
        "DELEGATION PURPOSE: bounded_implementation. You may modify only the files and systems explicitly "
        "allowed by the delegated task; verify the implementation before reporting completion."
    )


def purpose_tool_block_message(agent: Any, tool_name: str) -> str | None:
    raw_purpose = getattr(agent, "_delegate_purpose", None)
    if raw_purpose is None:
        return None
    purpose = normalize_delegation_purpose(raw_purpose)
    if purpose == RESEARCH_EVIDENCE and str(tool_name or "") in _RESEARCH_BLOCKED_TOOLS:
        return (
            f"Tool '{tool_name}' is unavailable for delegation purpose '{RESEARCH_EVIDENCE}'. "
            f"Spawn a '{BOUNDED_IMPLEMENTATION}' child for authorized artifact changes."
        )
    return None
