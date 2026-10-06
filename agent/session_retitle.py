"""Explicit session retitling from recent human-visible conversation context."""

from __future__ import annotations

from typing import Any, Iterable

from agent.aux_accounting import reset_accounting_context, set_accounting_context
from agent.context_compressor import is_compaction_summary_message, user_originated_turn_view
from agent.message_content import flatten_message_text
from agent.portal_tags import reset_conversation_context, set_conversation_context
from agent.title_generator import MAX_TITLE_INPUT_CHARS, generate_title, is_titleable_user_message


RECENT_RETITLE_MESSAGES = 4


class RetitleGenerationError(RuntimeError):
    """The title model failed while handling an explicit retitle request."""


class RetitlePersistenceError(RuntimeError):
    """A generated title could not be persisted or read back."""


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    text = content if isinstance(content, str) else flatten_message_text(content)
    return " ".join((text or "").split())


def _retitle_line(message: Any) -> str | None:
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    if role == "user":
        live_view = user_originated_turn_view(message)
        if live_view is None:
            return None
        text = _message_text(live_view)
        if not is_titleable_user_message(text):
            return None
        return f"User: {text}"
    if role == "assistant":
        text = _message_text(message)
        if (
            is_compaction_summary_message(message)
            or message.get("tool_calls")
            or message.get("display_kind")
            or not text
        ):
            return None
        return f"Assistant: {text}"
    return None


def _bounded_recent_lines(lines: Iterable[str]) -> str:
    """Keep the newest complete labelled lines within the title-model input budget."""
    kept: list[str] = []
    remaining = MAX_TITLE_INPUT_CHARS
    for line in reversed(list(lines)):
        separator = 1 if kept else 0
        if len(line) + separator <= remaining:
            kept.append(line)
            remaining -= len(line) + separator
            continue
        if not kept:
            label, separator_text, text = line.partition(": ")
            prefix = label + separator_text
            kept.append(prefix + text[: max(0, remaining - len(prefix))])
        break
    return "\n".join(reversed(kept))


def recent_retitle_context(history: Iterable[dict[str, Any]]) -> str:
    """Return the newest human-visible user/assistant messages suitable for retitling."""
    lines = [line for message in history if (line := _retitle_line(message)) is not None]
    return _bounded_recent_lines(lines[-RECENT_RETITLE_MESSAGES:])


def generate_retitle(context: str) -> str | None:
    """Generate a title while preserving the failure signal hidden by auto-title best effort."""
    failures: list[BaseException] = []
    title = generate_title(context, failure_callback=lambda _task, exc: failures.append(exc))
    if failures:
        raise failures[0]
    return title


def retitle_session(session_db, session_id: str, history: Iterable[dict[str, Any]]) -> str | None:
    """Generate, persist, and read back a title for an existing session."""
    context = recent_retitle_context(history)
    if not context:
        return None

    conversation_id = session_id
    try:
        conversation_id = session_db.get_conversation_root(session_id) or session_id
    except Exception:
        pass
    conversation_token = set_conversation_context(str(conversation_id))
    accounting_token = set_accounting_context(session_db, session_id)
    try:
        try:
            title = generate_retitle(context)
        except Exception as exc:
            raise RetitleGenerationError(str(exc)) from exc
    finally:
        reset_accounting_context(accounting_token)
        reset_conversation_context(conversation_token)

    if not title:
        return None
    try:
        if session_db.set_session_title(session_id, title) is False:
            raise RetitlePersistenceError(f"session {session_id} not found")
        persisted = session_db.get_session_title(session_id)
    except RetitlePersistenceError:
        raise
    except Exception as exc:
        raise RetitlePersistenceError(str(exc)) from exc
    if not persisted:
        raise RetitlePersistenceError(f"session {session_id} title was not persisted")
    return str(persisted)
