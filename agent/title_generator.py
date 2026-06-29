"""Auto-generate short session titles from the first user/assistant exchange.

A synchronous heuristic pass (``agent.title_heuristic``) sets an instant
placeholder title before the LLM call runs in a background thread.  The
LLM result overwrites the heuristic title when it returns, producing
better titles for edge cases while guaranteeing every session has *some*
title immediately — even when the LLM call fails.

Runs asynchronously after the first response is delivered so the LLM
call never adds latency to the user-facing reply.
"""

import logging
import threading
from typing import Callable, Optional

from agent.auxiliary_client import call_llm
from agent.title_heuristic import extract_title

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
    "Write the title in the same language the user is writing in. "
    "Return ONLY the title text, nothing else. No quotes, no punctuation at the end, no prefixes."
)

_TITLE_PROMPT_PINNED_LANGUAGE = (
    "Generate a short, descriptive title (3-7 words) for a conversation that starts with the "
    "following exchange. The title should capture the main topic or intent. "
    "Write the title in {language}. "
    "Return ONLY the title text, nothing else. No quotes, no punctuation at the end, no prefixes."
)


def _title_language() -> str:
    """Return configured title language, or empty string to match the user."""
    try:
        from hermes_cli.config import load_config

        return str(
            ((load_config() or {}).get("auxiliary") or {})
            .get("title_generation", {})
            .get("language", "")
        ).strip()
    except Exception:
        return ""


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

    language = _title_language()
    prompt = _TITLE_PROMPT_PINNED_LANGUAGE.format(language=language) if language else _TITLE_PROMPT

    messages = [
        {"role": "system", "content": prompt},
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
        title = (response.choices[0].message.content or "").strip()
        # Clean up: remove quotes, trailing punctuation, prefixes like "Title: "
        title = title.strip('"\'')
        if title.lower().startswith("title:"):
            title = title[6:].strip()
        # Enforce reasonable length
        if len(title) > 80:
            title = title[:77] + "..."
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
    heuristic_placeholder: Optional[str] = None,
) -> None:
    """Generate and set a session title via LLM.

    Called in a background thread after the first exchange completes.

    When ``heuristic_placeholder`` is provided, the LLM title overwrites
    it only if the current title still matches the placeholder —
    protecting user-set titles that arrived in the interim.

    When ``heuristic_placeholder`` is ``None``, the original behavior is
    preserved: skip if any title already exists.

    Args:
        heuristic_placeholder: The title set by the heuristic in
            ``maybe_auto_title``.  Pass ``None`` to preserve legacy
            "skip if title exists" behavior.
    """
    if not session_db or not session_id:
        return

    try:
        existing = session_db.get_session_title(session_id)
        if existing:
            if heuristic_placeholder is not None:
                # Race-condition guard: only overwrite if the current title
                # still matches the heuristic placeholder.  Protects user-set
                # titles that arrived between the heuristic set and here.
                if existing != heuristic_placeholder:
                    logger.debug(
                        "Title changed from heuristic placeholder (%r -> %r), skipping LLM overwrite",
                        heuristic_placeholder, existing,
                    )
                    return
                # Title matches placeholder — proceed to overwrite with LLM
            else:
                # Legacy path: any existing title means skip
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

    Sets a heuristic placeholder title synchronously (instant UI feedback),
    then spawns a background thread for the LLM call which overwrites the
    placeholder when it returns.

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

    # ── Synchronous heuristic: instant placeholder title ──
    # Set the heuristic title immediately so the session is never untitled,
    # even if the LLM call fails or is slow.  The background LLM thread
    # below overwrites this with a (usually better) LLM-generated title.
    heuristic_title = None
    try:
        existing = session_db.get_session_title(session_id)
        if existing:
            return  # user or prior auto-title already set, nothing to do
        heuristic_title = extract_title(user_message)
        if heuristic_title:
            session_db.set_session_title(session_id, heuristic_title)
            logger.debug("Heuristic session title: %s", heuristic_title)
            if title_callback is not None:
                try:
                    title_callback(heuristic_title)
                except Exception:
                    logger.debug("Heuristic title_callback failed", exc_info=True)
    except Exception as e:
        logger.debug("Heuristic title failed: %s", e)

    thread = threading.Thread(
        target=auto_title_session,
        args=(session_db, session_id, user_message, assistant_response),
        kwargs={
            "failure_callback": failure_callback,
            "main_runtime": main_runtime,
            "title_callback": title_callback,
            "heuristic_placeholder": heuristic_title if heuristic_title else None,
        },
        daemon=True,
        name="auto-title",
    )
    thread.start()
