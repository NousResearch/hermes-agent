"""Bounded user quotes for lean compaction, excluding model-facing scaffolding."""

from itertools import chain
from typing import Any

from agent.compaction_lean_carry import previous_user_quotes
from agent.compression_marker import ELISION_MARKER_MAX_LEN, elide, elide_middle

USER_QUOTES_NOTE = (
    "(Every real user message from the compacted region, quoted "
    "verbatim. These are the user's actual words and override any "
    "paraphrase of them above.)"
)


def _user_quote_text(msg: dict[str, Any], content: str) -> str:
    """Recover typed skill invocations and steers before filtering synthetic rows."""
    from agent.context_compressor import _synthetic_user_row
    from agent.prompt_builder import STEER_DISPLAY_KIND, STEER_MARKER_CLOSE, STEER_MARKER_OPEN
    from agent.skill_commands import describe_skill_invocation

    invocation = describe_skill_invocation(content, separator=" ") if _is_skill_scaffold(content) else None
    if invocation is not None:
        return invocation.strip()
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


def _is_skill_scaffold(content: str) -> bool:
    """Only a complete slash-skill, bundle or auto-load scaffold is re-rendered; typed text that merely
    shares a prefix (``[IMPORTANT: The "config" file is broken]``) is quoted as written."""
    from agent import skill_commands as sc

    if content.startswith(sc.INITIATE_SETUP_HEADER):
        return True
    if content.startswith(sc._AUTO_LOAD_PREFIX):
        return sc._SKILL_DIR_NOTE_END in content
    return content.startswith(sc._SKILL_INVOCATION_PREFIX) and (
        sc._SINGLE_SKILL_MARKER in content or sc._BUNDLE_MARKER in content
    )


def _bounded_user_quote(text: str, cap: int, message_cap: int) -> str:
    """Keep both ends of a long message; a budget straddler retains only its head."""
    if cap == message_cap and len(text) > cap:
        available = cap - ELISION_MARKER_MAX_LEN
        head = available * 5 // 8
        return elide_middle(text, head, available - head)
    return elide(text, cap)


def _build_verbatim_user_section(turns: list[dict[str, Any]], previous_summary: str | None = "") -> str:
    """Quote distinct real user messages newest-first under the lean char budget, then the quotes
    the previous summary's section already carried, in their existing order."""
    # The compressor owns these shared budgets and content adapters. Late imports
    # preserve its existing import seam without a module-level cycle.
    from agent.context_compressor import (
        _LEAN_USER_MESSAGE_MAX_CHARS,
        _LEAN_USER_MESSAGES_BUDGET_CHARS,
        _LEAN_USER_MESSAGES_HEADING,
        _content_text_for_contains,
    )
    from agent.conversation_compression import _strip_stale_todo_snapshot

    collected: list[str] = []
    seen: set[str] = set()
    used = 0
    current = (
        _user_quote_text(msg, _content_text_for_contains(msg.get("content")))
        for msg in reversed(turns) if msg.get("role") == "user"
    )
    carried = previous_user_quotes(previous_summary or "", _LEAN_USER_MESSAGES_HEADING)
    # A todo snapshot folded into a user row is scaffolding: quoted, its header would make the next fold
    # take the summary for the live snapshot and cut the appendix after it. Keep only the user's words.
    for text in map(_strip_stale_todo_snapshot, chain(current, carried)):
        if not text or text in seen:
            continue
        remaining = _LEAN_USER_MESSAGES_BUDGET_CHARS - used
        if remaining <= 0:
            break
        if len(text) > remaining and remaining <= ELISION_MARKER_MAX_LEN:
            break  # no room for a marker plus content
        quote = _bounded_user_quote(text, min(_LEAN_USER_MESSAGE_MAX_CHARS, remaining), _LEAN_USER_MESSAGE_MAX_CHARS)
        if quote in seen:
            continue  # already quoted in its bounded form (a long message carried from the last pass)
        seen.update((text, quote))
        collected.append("> " + quote.replace("\n", "\n> "))
        used += len(quote)
    if not collected:
        return ""
    return "\n\n" + _LEAN_USER_MESSAGES_HEADING + "\n" + "\n\n".join(collected) + "\n" + USER_QUOTES_NOTE
