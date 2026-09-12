"""Manual session retitling from a small window of recent conversation."""

from __future__ import annotations

from contextlib import suppress
from typing import Any, Optional

from agent.auxiliary_client import call_llm
from agent.message_content import flatten_message_text
from agent.title_generator import (
    MAX_TITLE_INPUT_CHARS,
    _LANGUAGE_RULE_MATCH_USER,
    _LANGUAGE_RULE_PINNED,
    _TITLE_RESPONSE_FORMAT,
    _clean_title,
    _extract_title_text,
    _summarize_user_message,
    _title_language,
    is_titleable_user_message,
)

# Two recent exchanges are enough to disambiguate follow-ups such as "continue"
# without turning title generation into conversation summarization.
RECENT_RETITLE_MESSAGES = 4

_RETITLE_PROMPT_TEMPLATE = (
    "You name chat sessions. Given the recent conversation, write a title "
    "that lets the user find this conversation again in a list.\n\n"
    "Rules:\n"
    "- 3 to 7 words, sentence case (capitalize only the first word and proper nouns).\n"
    "- Name what the conversation is actually about or what the user wants DONE.\n"
    "- Keep technical terms, filenames, numbers, and error codes exact.\n"
    "- No trailing punctuation, no quotes, no tool names, no 'Title:' prefix.\n"
    "- Never summarize or answer the conversation. Name it.\n"
    "__LANGUAGE_RULE__\n"
    'Reply with JSON only: {"title": "..."}'
)


def _conversation_text(message: dict) -> str:
    content = message.get("content")
    return content if isinstance(content, str) else flatten_message_text(content)


def recent_retitle_context(history: Any, limit: int = RECENT_RETITLE_MESSAGES) -> str:
    """Return the last few real user/assistant messages, excluding tool/internal scaffolding."""
    if not history or limit <= 0:
        return ""

    lines: list[str] = []
    for message in reversed(history):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue

        text = _conversation_text(message).strip()
        if role == "user":
            if not is_titleable_user_message(text):
                continue
            text = _summarize_user_message(text).strip()
        if not text:
            continue

        lines.append(f"{role.capitalize()}: {text}")
        if len(lines) >= limit:
            break

    # Reuse the existing title-input safety budget; no per-message clipping or
    # additional extraction policy is needed for this small recent window.
    return "\n".join(reversed(lines))[-MAX_TITLE_INPUT_CHARS:]


def generate_retitle(context: str, timeout: Optional[float] = None) -> Optional[str]:
    """Generate one title from recent conversation context."""
    context = (context or "").strip()
    if not context:
        return None

    language = _title_language()
    language_rule = (
        _LANGUAGE_RULE_PINNED.format(language=language)
        if language
        else _LANGUAGE_RULE_MATCH_USER
    )
    prompt = _RETITLE_PROMPT_TEMPLATE.replace("__LANGUAGE_RULE__", language_rule)

    response = call_llm(
        task="title_generation",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": context},
        ],
        max_tokens=64,
        temperature=0.3,
        timeout=timeout,
        extra_body={"response_format": _TITLE_RESPONSE_FORMAT},
    )
    return _clean_title(_extract_title_text(response.choices[0].message.content or ""))


def retitle_session(session_db, session_id: str, history: Any) -> Optional[str]:
    """Explicitly regenerate and replace a session title.

    ``set_session_title`` is intentionally used instead of the automatic-title
    provenance path: the generated text comes from the title model, but this
    mutation was explicitly requested by the user and should replace the title
    they currently see.
    """
    if session_db is None or not session_id:
        return None

    context = recent_retitle_context(history)
    if not context:
        return None

    # Match automatic title calls' accounting/conversation attribution without
    # making accounting availability part of the retitle contract.
    with suppress(Exception):
        from agent.aux_accounting import set_accounting_context
        from agent.portal_tags import set_conversation_context

        conversation_id = session_db.get_conversation_root(session_id) or session_id
        set_conversation_context(conversation_id)
        set_accounting_context(session_db, session_id)

    title = generate_retitle(context)
    if not title:
        raise RuntimeError("title generation returned no title")
    if not session_db.set_session_title(session_id, title):
        raise RuntimeError(f"session {session_id} not found while storing title")
    return title
