"""Auto-generate short session titles from the first user/assistant exchange.

Runs asynchronously after the first response is delivered so it never
adds latency to the user-facing reply.
"""

import logging
import re
import threading
from typing import Callable, Optional

from agent.auxiliary_client import call_llm, extract_content_or_reasoning

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


def _extract_title_from_reasoning(raw: str) -> str:
    """Extract a clean title from raw LLM output (may contain reasoning).

    For reasoning models (e.g. --reasoning-format deepseek), the output
    is mostly thinking process.  We look for the final selected title
    among the options.
    """
    if not raw:
        return raw

    # Step 1: Try to find the "best option" or final selection
    for pattern in [
        r"(?:Select|Choose|Best|Final)[\s:]*['\"]?(.+?)['\"]?\s*$",
        r"(?:Best option|Best choice|Final choice)[\s:]*['\"]?(.+?)['\"]?\s*$",
        r"(?:I recommend|I choose|I select)[\s:]*['\"]?(.+?)['\"]?\s*$",
    ]:
        match = re.search(pattern, raw, re.IGNORECASE | re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            candidate = re.sub(r'^[\s\*\-\u2022]+', '', candidate)
            candidate = re.sub(r'\s*\(.*?\)', '', candidate)
            candidate = candidate.strip(' \t\n\r.,!?;:"\'')
            if 3 <= len(candidate) <= 80:
                return candidate

    # Step 2: Look for quoted title candidates in the reasoning.
    # The model often lists options like "Python 快速排序算法" (Python Quick Sort)
    skip_patterns = re.compile(r'(?:words?|characters?|char\.?|characters\.?)$', re.IGNORECASE)
    quoted_titles = re.findall(r'["\u201c\u201d](.+?)["\u201c\u201d]', raw)
    for qt in quoted_titles:
        clean = qt.strip()
        if skip_patterns.search(clean):
            continue
        clean = re.sub(r'\s*\(.*?\)', '', clean)
        clean = clean.strip(' \t\n\r.,!?;:"\'')
        if 3 <= len(clean) <= 30 and re.search(r'[\u4e00-\u9fff]|[a-zA-Z]{2,}', clean):
            return clean

    # Step 3: If no quoted titles, try the last plausible phrase
    lines = [l.strip() for l in raw.split('\n') if l.strip()]
    for line in reversed(lines):
        if any(kw in line for kw in (
            "Thinking Process", "Analysis:", "Step ", "Draft",
            "Constraint", "Output Format", "Content:", "Topic:",
            "Specific", "Intent:", "Length:", "Select",
        )):
            continue
        clean = re.sub(r'^[\s\*\-\u2022]+', '', line)
        clean = re.sub(r'\s*\(.*?\)', '', clean)
        clean = re.sub(r'^\d+\.\s*', '', clean)
        clean = clean.strip(' \t\n\r.,!?;:"\'')
        if 3 <= len(clean) <= 80:
            return clean

    # Step 4: Fallback — don't truncate here; let generate_title() handle
    # the 80-char limit so it can append "..." consistently.
    return raw.strip()


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
        # Use extract_content_or_reasoning to handle reasoning models
        # (e.g., Qwen3.5-4B with --reasoning-format deepseek outputs to
        # reasoning_content instead of content).
        raw = extract_content_or_reasoning(response)
        title = _extract_title_from_reasoning(raw)
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
