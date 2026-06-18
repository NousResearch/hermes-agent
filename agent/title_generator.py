"""Auto-generate short session titles from the first user/assistant exchange.

Runs asynchronously after the first response is delivered so it never
adds latency to the user-facing reply.
"""

import logging
import threading
import time
from typing import Callable, Optional

from agent.auxiliary_client import call_llm

logger = logging.getLogger(__name__)

# Callback signature: (task_name, exception) -> None. Used to surface
# auxiliary failures to the user through AIAgent._emit_auxiliary_failure
# so silent-drops (e.g. OpenRouter 402 exhausting the fallback chain)
# become visible instead of piling up as NULL session titles.
FailureCallback = Callable[[str, BaseException], None]
TitleCallback = Callable[[str], None]

_TITLE_PROMPT = (
    "Generate a short, descriptive title (3-7 words) for a conversation that starts with the "
    "following exchange. The title should capture the main topic or intent. "
    "Return ONLY the title text, nothing else. No quotes, no punctuation at the end, no prefixes."
)


_INTENT_TITLE_PROMPT = (
    "Generate a short, descriptive title (3-7 words) for the user's task. "
    "Predict the user's intent instead of copying the raw text or URL. "
    "Use the user's apparent preferred language: follow an explicit language request when present, "
    "otherwise match the language of the user's message. Do not force English or translate unnecessarily. "
    "If the user only pasted a link, infer a useful title from the URL, domain, path, or visible text. "
    "Return ONLY the title text, nothing else. No quotes, no punctuation at the end, no prefixes."
)


def _clean_title(title: str, max_len: int = 80) -> Optional[str]:
    cleaned = (title or "").strip().strip('"\'')
    if cleaned.lower().startswith("title:"):
        cleaned = cleaned[6:].strip()
    cleaned = cleaned.rstrip(".。!！?？")
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 3].rstrip() + "..."
    return cleaned or None


def generate_intent_title(
    user_message: str,
    timeout: float = 10.0,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
) -> Optional[str]:
    """Generate a fast predicted title from the user's first task only.

    Messaging gateways can call this before the agent turn finishes: Discord
    creates an auto-thread with a stable processing placeholder, then renames
    it as soon as this lightweight intent prediction returns.
    """
    user_snippet = user_message[:500] if user_message else ""
    if not user_snippet.strip():
        return None

    messages = [
        {"role": "system", "content": _INTENT_TITLE_PROMPT},
        {"role": "user", "content": user_snippet},
    ]

    started = time.monotonic()
    logger.info(
        "Intent title generation started: user_chars=%s timeout=%ss",
        len(user_snippet),
        timeout,
    )
    try:
        response = call_llm(
            task="title_generation",
            messages=messages,
            max_tokens=120,
            temperature=0.1,
            timeout=timeout,
            main_runtime=main_runtime,
        )
        title = _clean_title(response.choices[0].message.content or "")
        if not title:
            logger.warning(
                "Intent title generation returned empty content: elapsed=%.2fs",
                time.monotonic() - started,
            )
            return None
        logger.info(
            "Intent title generation completed: title=%r elapsed=%.2fs",
            title,
            time.monotonic() - started,
        )
        return title
    except Exception as e:
        logger.warning("Intent title generation failed: %s", e)
        logger.debug("Intent title generation traceback", exc_info=True)
        if failure_callback is not None:
            try:
                failure_callback("intent title generation", e)
            except Exception:
                logger.debug("Intent title failure_callback raised", exc_info=True)
        return None


def pre_title_session(
    session_db,
    session_id: str,
    user_message: str,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    title_callback: Optional[TitleCallback] = None,
) -> None:
    """Generate and set an intent title before the first agent turn completes."""
    if not session_db or not session_id or not user_message:
        return
    try:
        existing = session_db.get_session_title(session_id)
        if existing:
            return
    except Exception:
        logger.debug("Pre-title skipped: failed to read existing title", exc_info=True)
        return

    title = generate_intent_title(
        user_message,
        failure_callback=failure_callback,
        main_runtime=main_runtime,
    )
    if not title:
        return
    # Persisting the title to the session DB is best-effort: a uniqueness
    # conflict (same title already used by another session — e.g. "Friendly
    # Greeting" appearing in a 寒暄 thread a month apart) raises ValueError
    # from SessionDB.set_session_title. We must not let that abort the
    # side-effects that the caller (Discord thread rename) depends on —
    # the rename does not need the DB write to succeed. Split the two
    # concerns: log the DB failure, but always invoke the callback.
    if session_db is not None:
        try:
            session_db.set_session_title(session_id, title)
            logger.info("Pre-title session title set: session=%s title=%r", session_id, title)
        except Exception as e:
            logger.debug("Pre-title: set_session_title failed (%s: %s); continuing to invoke callback", type(e).__name__, e)
    if title_callback is not None:
        try:
            title_callback(title)
        except Exception:
            logger.debug("Pre-title callback failed", exc_info=True)


def maybe_pre_title(
    session_db,
    session_id: str,
    user_message: str,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    title_callback: Optional[TitleCallback] = None,
) -> None:
    """Fire-and-forget intent title generation before the agent turn finishes."""
    if not session_db or not session_id or not user_message:
        return
    thread = threading.Thread(
        target=pre_title_session,
        args=(session_db, session_id, user_message),
        kwargs={
            "failure_callback": failure_callback,
            "main_runtime": main_runtime,
            "title_callback": title_callback,
        },
        daemon=True,
        name="pre-title",
    )
    thread.start()


def generate_title(
    user_message: str,
    assistant_response: str,
    timeout: float = 30.0,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
) -> Optional[str]:
    """Generate a session title from the first exchange.

    Uses the main runtime's model when available, falling back to the
    auxiliary LLM client (cheapest/fastest available model).
    Returns the title string or None on failure.

    ``failure_callback`` is invoked with ``(task, exception)`` when the
    auxiliary call raises — the caller typically wires this to
    ``AIAgent._emit_auxiliary_failure`` so the user sees a warning instead
    of silently accumulating untitled sessions.
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
            main_runtime=main_runtime,
        )
        title = _clean_title(response.choices[0].message.content or "")
        return title if title else None
    except Exception as e:
        # Log at WARNING so this shows up in agent.log without debug mode.
        # Full detail at debug level for operators who need the stack.
        logger.warning("Title generation failed: %s", e)
        logger.debug("Title generation traceback", exc_info=True)
        if failure_callback is not None:
            try:
                failure_callback("title generation", e)
            except Exception:
                logger.debug("Title generation failure_callback raised", exc_info=True)
        return None


def auto_title_session(
    session_db,
    session_id: str,
    user_message: str,
    assistant_response: str,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    title_callback: Optional[TitleCallback] = None,
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

    title = generate_title(
        user_message, assistant_response, failure_callback=failure_callback, main_runtime=main_runtime
    )
    if not title:
        return

    try:
        session_db.set_session_title(session_id, title)
        logger.debug("Auto-generated session title: %s", title)
        if title_callback is not None:
            try:
                title_callback(title)
            except Exception:
                logger.debug("Auto-title callback failed", exc_info=True)
    except Exception as e:
        logger.debug("Failed to set auto-generated title: %s", e)


def maybe_auto_title(
    session_db,
    session_id: str,
    user_message: str,
    assistant_response: str,
    conversation_history: list,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    title_callback: Optional[TitleCallback] = None,
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
        kwargs={
            "failure_callback": failure_callback,
            "main_runtime": main_runtime,
            "title_callback": title_callback,
        },
        daemon=True,
        name="auto-title",
    )
    thread.start()
