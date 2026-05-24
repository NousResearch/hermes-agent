"""Auto-generate short session titles from user messages.

Runs asynchronously after the first response is delivered so it never
adds latency to the user-facing reply.

Also supports backfill: reading the last N user messages from existing
sessions to generate titles for previously untitled conversations.
"""

import logging
import re
import threading
import time
from typing import Any, Callable, List, Optional

from agent.auxiliary_client import call_llm, extract_content_or_reasoning

logger = logging.getLogger(__name__)

# Callback signature: (task_name, exception) -> None. Used to surface
# auxiliary failures to the user through AIAgent._emit_auxiliary_failure
# so silent-drops (e.g. OpenRouter 402 exhausting the fallback chain)
# become visible instead of piling up as NULL session titles.
FailureCallback = Callable[[str, BaseException], None]
TitleCallback = Callable[[str], None]

_TITLE_PROMPT = (
    "You are titling CLI conversation sessions. Given the last N user messages, "
    "generate a short natural-language title (3-8 words, max 60 characters).\n\n"
    "FRAMING:\n"
    "- Start with a verb. Action first, subject second.\n"
    "- Write like a human would naturally describe what they did.\n"
    "- No prefixes like 'X: do Y'. Just 'Do Y' or 'Do Y in X'.\n"
    "- Include the tool/system name inline if it helps: 'Fix git branch', 'Clean up cron jobs'.\n"
    "- For repeated topics on the same subject, add #2, #3 etc.\n"
    "- Greetings or single vague messages → 'Chat' or 'Quick question'.\n\n"
    "EXAMPLES OF GOOD TITLES:\n"
    "  'Fix wrong git branch commit'\n"
    "  'Review and clean up dojo skills'\n"
    "  'Clean up orphan cron jobs'\n"
    "  'Debug failing MCP services'\n"
    "  'Create Finch skill from Dojo research'\n"
    "  'Set up LinkedIn MCP login'\n"
    "  'Analyze sales data from CSV'\n"
    "  'Format and validate JSON'\n\n"
    "OUTPUT: Return ONLY the title. No quotes, no 'Title:' prefix."
)

_NOISE_PATTERNS = [
    re.compile(r'^\[CONTEXT COMPACTION'),
    re.compile(r'^\[System note:'),
    re.compile(r'^\[IMPORTANT:'),
    re.compile(r'^\[Your active task list'),
    re.compile(r'^\[backfill\]'),
    re.compile(r'^Important (User|Running)'),
]


def _filter_noise(messages: List[str]) -> List[str]:
    """Remove system notes, compaction summaries, and skill invocation boilerplate."""
    filtered = []
    for msg in messages:
        text = msg.strip()
        if len(text) < 2:
            continue
        if any(p.match(text) for p in _NOISE_PATTERNS):
            continue
        filtered.append(text)
    return filtered


def _build_messages(user_messages: List[str], assistant_response: str = "") -> list:
    """Build the chat messages list for the LLM title request."""
    clean = _filter_noise(user_messages)
    if not clean:
        clean = [m.strip() for m in user_messages if m.strip()][:5]

    messages_text = "\n---\n".join(
        f"[{i+1}] {msg[:500]}" for i, msg in enumerate(clean)
    )

    user_prompt = (
        f"Here are the last {len(clean)} user messages from a conversation:\n\n"
        f"{messages_text}\n\n"
        "Generate a short, descriptive title for this conversation."
    )

    return [
        {"role": "system", "content": _TITLE_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def generate_title(
    user_messages: List[str],
    assistant_response: str = "",
    timeout: float = 30.0,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: Optional[dict] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    llm_client: Optional[Any] = None,
    retry_count: int = 3,
    retry_base_delay: float = 5.0,
) -> Optional[str]:
    """Generate a session title from user messages.

    When ``llm_client`` is provided (an OpenAI-compatible client instance),
    it is used directly — no provider resolution, no credential lookups,
    no auxiliary fallback chain. This is the preferred path: the caller
    passes the main agent's already-authenticated client, so title
    generation always uses the same provider/model/credentials that just
    successfully handled the conversation.

    Falls back to ``call_llm()`` (auxiliary client resolution) only when
    ``llm_client`` is None.

    ``failure_callback`` is invoked with ``(task, exception)`` when the
    call raises — the caller typically wires this to
    ``AIAgent._emit_auxiliary_failure`` so the user sees a warning instead
    of silently accumulating untitled sessions.

    ``user_messages`` should be a list of user message strings, ordered
    chronologically. The last N messages are used (up to 10).
    """
    if not user_messages:
        return None

    messages = _build_messages(user_messages, assistant_response)

    last_exc: Optional[BaseException] = None
    for attempt in range(retry_count):
        try:
            if llm_client is not None:
                # Use a higher max_tokens for reasoning models (DeepSeek-R1,
                # Qwen-QwQ, etc.) whose thinking tokens consume most of the
                # budget before the final answer. 100 is too low.
                kwargs = {"messages": messages, "max_tokens": 512, "temperature": 0.3}
                if final_model:
                    kwargs["model"] = final_model
                response = llm_client.chat.completions.create(**kwargs)
            else:
                # call_llm path — same reasoning applies.
                # Fallback: resolve provider via auxiliary client chain.
                resolved_provider: Optional[str] = provider
                resolved_model: Optional[str] = model
                resolved_base_url: Optional[str] = base_url
                resolved_api_key: Optional[str] = api_key

                if not resolved_provider and main_runtime:
                    resolved_provider = main_runtime.get("provider")
                    resolved_model = resolved_model or main_runtime.get("model")
                    resolved_base_url = resolved_base_url or main_runtime.get("base_url")
                    resolved_api_key = resolved_api_key or main_runtime.get("api_key")

                response = call_llm(
                    task="title_generation",
                    messages=messages,
                    max_tokens=100,
                    temperature=0.3,
                    timeout=timeout,
                    provider=resolved_provider,
                    model=resolved_model,
                    base_url=resolved_base_url,
                    api_key=resolved_api_key,
                    main_runtime=main_runtime,
                )
            title = (extract_content_or_reasoning(response) or "").strip()
            # Clean up: remove quotes, trailing punctuation, prefixes like "Title: "
            title = title.strip('"\'')
            if title.lower().startswith("title:"):
                title = title[6:].strip()
            # Enforce reasonable length
            if len(title) > 60:
                title = title[:57] + "..."
            return title if title else None
        except Exception as e:
            last_exc = e
            err_str = str(e)
            # Only retry on 429 (rate limit). Auth errors (401), payment
            # errors (402), and other non-transient failures won't resolve
            # by retrying the same provider — skip straight to failure.
            if "429" in err_str and attempt < retry_count - 1:
                wait = retry_base_delay * (attempt + 1)
                logger.warning(
                    "Title generation rate limited (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1, retry_count, wait, e,
                )
                time.sleep(wait)
                continue
            # Non-retryable error or last attempt — fall through to failure handling
            break

    # All attempts failed — this is cosmetic, log at debug level
    logger.debug("Title generation failed: %s", last_exc)
    logger.debug("Title generation traceback", exc_info=True)
    if failure_callback is not None and last_exc is not None:
        try:
            failure_callback("title generation", last_exc)
        except Exception:
            logger.debug("Title generation failure_callback raised", exc_info=True)
    return None


def auto_title_session(
    session_db,
    session_id: str,
    user_messages: List[str],
    assistant_response: str = "",
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: Optional[dict] = None,
    title_callback: Optional[TitleCallback] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    llm_client: Optional[Any] = None,
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
        user_messages=user_messages,
        assistant_response=assistant_response,
        failure_callback=failure_callback,
        main_runtime=main_runtime,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        llm_client=llm_client,
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
    main_runtime: Optional[dict] = None,
    title_callback: Optional[TitleCallback] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    llm_client: Optional[Any] = None,
) -> None:
    """Fire-and-forget title generation after the first exchange.

    Only generates a title when:
    - This appears to be the first user→assistant exchange
    - No title is already set

    Accumulates user messages from conversation_history so the title
    captures the actual topic even if it takes a few messages to emerge.
    """
    if not session_db or not session_id or not user_message or not assistant_response:
        return

    # Count user messages in history to detect first exchange.
    # conversation_history includes the exchange that just happened,
    # so for a first exchange we expect exactly 1 user message
    # (or 2 counting system). Be generous: generate on first 3 exchanges.
    user_msg_count = sum(1 for m in (conversation_history or []) if m.get("role") == "user")
    if user_msg_count > 3:
        return

    # Collect all user messages from history for better context
    user_messages = [
        m.get("content", "")
        for m in (conversation_history or [])
        if m.get("role") == "user" and m.get("content")
    ]
    # Ensure the current message is included
    if not user_messages or user_messages[-1] != user_message:
        user_messages.append(user_message)

    thread = threading.Thread(
        target=auto_title_session,
        args=(session_db, session_id, user_messages, assistant_response),
        kwargs={
            "failure_callback": failure_callback,
            "main_runtime": main_runtime,
            "title_callback": title_callback,
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "llm_client": llm_client,
        },
        daemon=True,
        name="auto-title",
    )
    thread.start()
