"""Read-only standing-command router for gateway plain-text triggers.

This is deliberately tiny: exact phrases may bypass the LLM, adjacent
phrases only ask for confirmation, and slash commands stay in the slash
command path. The goal is router-before-action, not another prompt soup
machine wearing a vest.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class StandingCommandRoute:
    name: str
    mode: str  # "exact" | "confirm"
    prompt: str


_EXACT: dict[str, str] = {
    "whats next": "next",
    "what is next": "next",
    "where are we at": "next",
    "status": "status",
    "reflect": "reflect",
}

_NEXT_CONFIRM_PATTERNS = (
    "what should we adapt next",
    "what should we do next",
    "what do we adapt next",
)

_PROMPTS: dict[str, str] = {
    "next": "what's next / where-are-we-at pickup",
    "status": "status scorecard",
    "reflect": "reflection and durable-memory/skill extraction review",
}


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.strip().lower()
    if not text or text.startswith("/"):
        return ""
    text = text.replace("’", "'")
    text = re.sub(r"\bwhat's\b", "whats", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def resolve_standing_command(text: str) -> StandingCommandRoute | None:
    """Resolve a plain-text gateway message to a read-only standing command.

    Exact triggers can auto-fire. Adjacent natural language requires a
    confirmation/picker response. Slash commands return None so the normal
    command dispatcher keeps precedence.
    """

    normalized = _normalize(text)
    if not normalized:
        return None

    name = _EXACT.get(normalized)
    if name:
        return StandingCommandRoute(name=name, mode="exact", prompt=_PROMPTS[name])

    if normalized in _NEXT_CONFIRM_PATTERNS:
        return StandingCommandRoute(name="next", mode="confirm", prompt=_PROMPTS["next"])

    return None


def render_confirmation(route: StandingCommandRoute) -> str:
    """Return a compact picker for non-exact route matches."""

    if route.name == "next":
        return (
            "I can route that through the pickup command.\n\n"
            "Reply exactly: what’s next?\n\n"
            "That reads STATE + latest handoff first, then gives the next move."
        )
    return f"I can route that through {route.prompt}. Send the exact trigger to run it."


def build_agent_instruction(route: StandingCommandRoute, original_text: str) -> str:
    """Wrap exact routes that still need agent/tool work in a deterministic prompt."""

    if route.name == "next":
        return (
            "Standing command: what's next / where are we at.\n"
            "Follow ace-level-up exactly: read /Users/ace/.hermes/workspace/STATE.md, "
            "resolve Latest handoff if present, then return the next concrete move. "
            "Do not ask a clarifying question unless the state is unreadable.\n\n"
            f"User trigger: {original_text}"
        )
    if route.name == "reflect":
        return (
            "Standing command: reflect.\n"
            "Review the current session for durable behavior changes, reusable skill/process updates, "
            "and memory/wiki boundaries. Use ace-reflect or ace-memory-boundary if relevant. "
            "Do not save task progress to persistent memory. Return findings and any concrete writes made.\n\n"
            f"User trigger: {original_text}"
        )
    return original_text
