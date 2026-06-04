"""Auto-generate short session titles from the first user/assistant exchange.

Runs asynchronously after the first response is delivered so it never
adds latency to the user-facing reply.
"""

import logging
import threading
import re
from typing import Callable, Optional

from agent.auxiliary_client import call_llm

logger = logging.getLogger(__name__)

# XML tarzı açılış/kapanış etiketleri
_REASONING_TAG_RE = re.compile(
    r"<(think|thinking|thought|reasoning|chain-of-thought|inner-monologue)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE
)

_REASONING_TAG_SELF_CLOSE_RE = re.compile(
    r"<(think|thinking|thought|reasoning|chain-of-thought|inner-monologue)[^>]*/>",
    re.DOTALL | re.IGNORECASE
)

# Köşeli parantez tarzı açılış/kapanış etiketleri (GLM 5.1)
_REASONING_TAG_BRACKET_RE = re.compile(
    r"\[(think|thinking|thought|reasoning|chain-of-thought|inner-monologue)\][^\[]*\[/\1\]",
    re.DOTALL | re.IGNORECASE
)

_REASONING_TAG_BRACKET_SELF_CLOSE_RE = re.compile(
    r"\[(think|thinking|thought|reasoning|chain-of-thought|inner-monologue)/\]",
    re.DOTALL | re.IGNORECASE
)
# Markdown kod bloklarını temizle
_CODE_BLOCK_RE = re.compile(r"```[a-zA-Z]*\s*\n?", re.DOTALL)

FailureCallback = Callable[[str, BaseException], None]
TitleCallback = Callable[[str], None]

_TITLE_PROMPT = (
    "Generate a short, descriptive title (3-7 words) for a conversation that starts with the "
    "following exchange. The title should capture the main topic or intent. "
    "Return ONLY the title text, nothing else. No quotes, no punctuation at the end, no prefixes."
)


def _strip_reasoning_blocks(text: str) -> str:
    """Model yanıtındaki içsel düşünce (reasoning) bloklarını temizler.

    Desteklenen formatlar:
    - XML tarzı: <thinking>...</thinking> (standard),  (DeepSeek-R1)
    - Köşeli parantez tarzı: [thinking]...[/thinking], [THINKING]...[/THINKING] (GLM 5.1)
    - Self-closing: <thinking/>, [thinking/]
    İç içe geçmiş yapıları iteratif while döngüleri ile güvenle temizler.
    """
    if not text:
        return ""
    cleaned = text
    # önce parantez tarzını iteratif temizle (en içteki bloklar önce)
    while re.search(_REASONING_TAG_BRACKET_RE, cleaned) or re.search(_REASONING_TAG_BRACKET_SELF_CLOSE_RE, cleaned):
        if re.search(_REASONING_TAG_BRACKET_RE, cleaned):
            cleaned = _REASONING_TAG_BRACKET_RE.sub("", cleaned, count=1)
        if re.search(_REASONING_TAG_BRACKET_SELF_CLOSE_RE, cleaned):
            cleaned = _REASONING_TAG_BRACKET_SELF_CLOSE_RE.sub("", cleaned, count=1)
    # sonra XML tarzını iteratif temizle (en içteki bloklar önce)
    while re.search(_REASONING_TAG_RE, cleaned) or re.search(_REASONING_TAG_SELF_CLOSE_RE, cleaned):
        if re.search(_REASONING_TAG_RE, cleaned):
            cleaned = _REASONING_TAG_RE.sub("", cleaned, count=1)
        if re.search(_REASONING_TAG_SELF_CLOSE_RE, cleaned):
            cleaned = _REASONING_TAG_SELF_CLOSE_RE.sub("", cleaned, count=1)
    # Markdown kod bloklarını temizle
    cleaned = _CODE_BLOCK_RE.sub("", cleaned)
    return cleaned.strip()


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
    user_snippet = _strip_reasoning_blocks(user_message or "")[:500]
    assistant_snippet = _strip_reasoning_blocks(assistant_response or "")[:500]

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
