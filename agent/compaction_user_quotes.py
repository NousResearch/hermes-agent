"""Bounded user quotes for lean compaction, excluding model-facing scaffolding."""

from typing import Any

from agent.compression_marker import ELISION_MARKER_MAX_LEN, elide, elide_middle


def _user_quote_text(msg: dict[str, Any], content: str) -> str:
    """Recover typed skill invocations and steers before filtering synthetic rows."""
    from agent.context_compressor import _synthetic_user_row, _typed_user_text
    from agent.prompt_builder import STEER_DISPLAY_KIND, STEER_MARKER_CLOSE, STEER_MARKER_OPEN
    from agent.skill_commands import skills_loaded_by_scaffold

    if skills_loaded_by_scaffold(content):
        typed = _typed_user_text(content).strip()
        return "" if _synthetic_user_row(typed) else typed
    text = content.strip()
    if (msg.get("display_kind") == STEER_DISPLAY_KIND
            and text.startswith(STEER_MARKER_OPEN) and text.endswith(STEER_MARKER_CLOSE)):
        inner = text[len(STEER_MARKER_OPEN):-len(STEER_MARKER_CLOSE)].strip()
        # A nested frame would re-emit the trusted marker inside the quote; keep today's skip.
        if not inner or STEER_MARKER_OPEN in inner or STEER_MARKER_CLOSE in inner:
            return ""
        return "(mid-turn steer) " + inner
    if msg.get("display_kind") not in (None, "", STEER_DISPLAY_KIND):
        return ""  # model-only rows (notifications, heartbeats): mirrors _is_actionable_user_turn
    return "" if _synthetic_user_row(content) else text


def _bounded_user_quote(text: str, cap: int, message_cap: int) -> str:
    """Keep both ends of a long message; a budget straddler retains only its head."""
    if cap == message_cap and len(text) > cap:
        available = cap - ELISION_MARKER_MAX_LEN
        head = available * 5 // 8
        return elide_middle(text, head, available - head)
    return elide(text, cap)


def _build_verbatim_user_section(turns: list[dict[str, Any]]) -> str:
    """Quote distinct real user messages newest-first under the lean char budget."""
    # The compressor owns these shared budgets and content adapters. Late imports
    # preserve its existing import seam without a module-level cycle.
    from agent.context_compressor import (
        _LEAN_USER_MESSAGE_MAX_CHARS,
        _LEAN_USER_MESSAGES_BUDGET_CHARS,
        _LEAN_USER_MESSAGES_HEADING,
        _content_text_for_contains,
    )

    collected: list[str] = []
    seen: set[str] = set()
    used = 0
    for msg in reversed(turns):
        if msg.get("role") != "user":
            continue
        text = _user_quote_text(msg, _content_text_for_contains(msg.get("content")))
        if not text or text in seen:
            continue
        remaining = _LEAN_USER_MESSAGES_BUDGET_CHARS - used
        if remaining <= 0:
            break
        if len(text) > remaining and remaining <= ELISION_MARKER_MAX_LEN:
            break  # no room for a marker plus content
        seen.add(text)
        text = _bounded_user_quote(text, min(_LEAN_USER_MESSAGE_MAX_CHARS, remaining), _LEAN_USER_MESSAGE_MAX_CHARS)
        collected.append("> " + text.replace("\n", "\n> "))
        used += len(text)
    if not collected:
        return ""
    return (
        "\n\n" + _LEAN_USER_MESSAGES_HEADING + "\n"
        + "\n\n".join(collected)
        + "\n(Every real user message from the compacted region, quoted "
        "verbatim. These are the user's actual words and override any "
        "paraphrase of them above.)"
    )
