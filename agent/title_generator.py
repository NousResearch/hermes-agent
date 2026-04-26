"""Auto-generate short session titles from the first user/assistant exchange.

Runs asynchronously after the first response is delivered so it never
adds latency to the user-facing reply.
"""

import logging
import threading
from typing import Optional

from agent.auxiliary_client import call_llm

logger = logging.getLogger(__name__)

_TITLE_PROMPT = (
    "Generate a concise 3-word title for this conversation, with words separated by dashes. "
    "Examples: setup-docker-compose, fix-login-bug, hermes-title-feature, deploy-staging-server. "
    "Return ONLY the 3-word-dash-separated title, nothing else. No quotes, no prefixes, lowercase."
)


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
        title = (response.choices[0].message.content or "").strip()
        # Clean up: remove quotes, trailing punctuation, prefixes like "Title: "
        title = title.strip('"\'')
        if title.lower().startswith("title:"):
            title = title[6:].strip()
        # Normalize to lowercase-dashed format
        import re
        title = re.sub(r'[^a-zA-Z0-9\s-]', '', title)
        title = re.sub(r'\s+', '-', title.strip()).lower()
        # Enforce max 3 words (dashed segments)
        parts = [p for p in title.split('-') if p]
        if len(parts) > 3:
            parts = parts[:3]
        title = '-'.join(parts)
        # Enforce reasonable length
        if len(title) > 80:
            title = title[:77] + "..."
        return title if title else None
    except Exception as e:
        logger.warning("Title generation failed: %s: %s", type(e).__name__, e)
        return None


def generate_title_from_history(conversation_history: list, timeout: float = 30.0) -> Optional[str]:
    """Generate a session title from conversation history (any point in the session).

    Extracts the most recent user and assistant messages to build context,
    then asks the auxiliary LLM for a short title.  Used by ``/title`` when
    called without arguments mid-conversation.

    Falls back to a truncated first user message if the LLM call fails.
    """
    if not conversation_history:
        return None

    # Collect recent user and assistant messages (last 3 of each, for context)
    user_msgs = [m["content"] for m in conversation_history
                 if m.get("role") == "user" and m.get("content")]
    assistant_msgs = [m["content"] for m in conversation_history
                      if m.get("role") == "assistant" and m.get("content")]

    if not user_msgs:
        return None

    # Use last few messages to capture the session's topic
    user_snippet = "\n".join(msg[:200] for msg in user_msgs[-3:])[:500]
    assistant_snippet = "\n".join(msg[:200] for msg in assistant_msgs[-3:])[:500] if assistant_msgs else ""

    title = generate_title(user_snippet, assistant_snippet, timeout=timeout)

    # Fallback: truncate first user message if LLM title generation failed
    if not title and user_msgs:
        first_msg = user_msgs[0].strip()
        # Remove system prefixes like "[SYSTEM: ...]"
        if first_msg.startswith("[SYSTEM:"):
            # Try next user message
            for msg in user_msgs[1:]:
                if not msg.strip().startswith("[SYSTEM:"):
                    first_msg = msg.strip()
                    break
            else:
                first_msg = ""
        if first_msg:
            import re
            # Extract first 3 meaningful words, dashed
            words = re.sub(r'[^a-zA-Z0-9\s]', '', first_msg).split()
            words = [w.lower() for w in words[:3] if w]
            title = '-'.join(words) if words else None
            if title:
                logger.info("Title generation fell back to truncated user message")

    return title


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
