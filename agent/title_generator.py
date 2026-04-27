"""Auto-generate short session titles from the first user/assistant exchange.

Runs asynchronously after the first response is delivered so it never
adds latency to the user-facing reply.
"""

import logging
import re
import threading
from typing import Optional

from agent.auxiliary_client import call_llm, extract_content_or_reasoning

logger = logging.getLogger(__name__)

_TITLE_PROMPT = (
    "Generate a short, descriptive title (3-7 words) for a conversation that starts with the "
    "following exchange. The title should capture the main topic or intent. "
    "Return ONLY the title text, nothing else. Use the same language as the user when possible. "
    "Do not mention 'user' or 'assistant'. Do not return reasoning, analysis, or prefixes like "
    "'The user is asking'. No quotes, no punctuation at the end, no prefixes."
)

_META_TITLE_PREFIXES = (
    "the user is asking",
    "the user wants",
    "the assistant is",
    "the conversation is about",
)


def _strip_wrapping_quotes(text: str) -> str:
    stripped = (text or "").strip()
    pairs = {
        ('"', '"'),
        ("'", "'"),
        ("“", "”"),
        ("‘", "’"),
    }
    while len(stripped) >= 2 and (stripped[0], stripped[-1]) in pairs:
        stripped = stripped[1:-1].strip()
    return stripped


def _clean_generated_title(raw_title: str) -> Optional[str]:
    if not raw_title:
        return None

    title = re.sub(r"<[^>]+>", " ", raw_title)
    title = re.sub(r"\s+", " ", title).strip()
    title = _strip_wrapping_quotes(title)

    if title.lower().startswith("title:"):
        title = _strip_wrapping_quotes(title[6:].strip())

    lowered = title.lower()
    if lowered.startswith(_META_TITLE_PREFIXES):
        quoted = re.search(r'["“\'‘](.+?)["”\'’]', title)
        if quoted:
            title = quoted.group(1).strip()
        else:
            for prefix in _META_TITLE_PREFIXES:
                if lowered.startswith(prefix):
                    title = title[len(prefix):].strip(" :.-")
                    break

    title = _strip_wrapping_quotes(title)
    if not title:
        return None

    if len(title.split()) > 12:
        return None

    if len(title) > 80:
        title = title[:77] + "..."

    return title or None


def generate_title(user_message: str, assistant_response: str, timeout: float = 30.0) -> Optional[str]:
    """Generate a session title from the first exchange.

    Uses the auxiliary LLM client (cheapest/fastest available model).
    Returns the title string or None on failure.
    """
    # Truncate long messages to keep the request small
    user_snippet = user_message[:500] if user_message else ""
    assistant_snippet = assistant_response[:500] if assistant_response else ""

    messages = [
        {"role": "system", "content": _TITLE_PROMPT},
        {"role": "user", "content": f"User: {user_snippet}\n\nAssistant: {assistant_snippet}"},
    ]

    try:
        response = call_llm(
            task="title_generation",
            messages=messages,
            max_tokens=500,
            temperature=0.3,
            timeout=timeout,
        )
        title = extract_content_or_reasoning(response).strip()
        return _clean_generated_title(title)
    except Exception as e:
        logger.debug("Title generation failed: %s", e)
        return None


def auto_title_session(
    session_db,
    session_id: str,
    user_message: str,
    assistant_response: str,
) -> None:
    """Generate and set a session title if one doesn't already exist.

    Called in a background thread after the first exchange completes.
    Silently skips if:
    - session_db is None
    - session already has a title (user-set or previously auto-generated)
    - title generation fails
    """
    if not session_db or not session_id:
        return

    # Check if title already exists (user may have set one via /title before first response)
    try:
        existing = session_db.get_session_title(session_id)
        if existing:
            return
    except Exception:
        return

    title = generate_title(user_message, assistant_response)
    if not title:
        return

    try:
        session_db.set_session_title(session_id, title)
        logger.debug("Auto-generated session title: %s", title)
    except Exception as e:
        logger.debug("Failed to set auto-generated title: %s", e)


def maybe_auto_title(
    session_db,
    session_id: str,
    user_message: str,
    assistant_response: str,
    conversation_history: list,
) -> None:
    """Fire-and-forget title generation after the first exchange.

    Only generates a title when:
    - This appears to be the first user→assistant exchange
    - No title is already set
    """
    if not session_db or not session_id or not user_message or not assistant_response:
        return

    # Count user messages in history to detect first exchange.
    # conversation_history includes the exchange that just happened,
    # so for a first exchange we expect exactly 1 user message
    # (or 2 counting system). Be generous: generate on first 2 exchanges.
    user_msg_count = sum(1 for m in (conversation_history or []) if m.get("role") == "user")
    if user_msg_count > 2:
        return

    thread = threading.Thread(
        target=auto_title_session,
        args=(session_db, session_id, user_message, assistant_response),
        daemon=True,
        name="auto-title",
    )
    thread.start()
