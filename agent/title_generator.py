"""Auto-generate short session titles from the first user/assistant exchange.

Runs asynchronously after the first response is delivered so it never
adds latency to the user-facing reply.
"""

import logging
import re
import threading
from typing import Callable, Optional

from agent.auxiliary_client import call_llm

logger = logging.getLogger(__name__)

# Callback signature: (task_name, exception) -> None. Used to surface
# auxiliary failures to the user through AIAgent._emit_auxiliary_failure
# so silent-drops (e.g. OpenRouter 402 exhausting the fallback chain)
# become visible instead of piling up as NULL session titles.
FailureCallback = Callable[[str, BaseException], None]
TitleCallback = Callable[[str], None]

# Validation callback: () -> bool. Called right before the LLM request in
# generate_title(). Return False to skip — e.g. the user switched models
# after this background thread captured its runtime snapshot, and sending
# the request would reload a model the runtime already evicted (#19027).
RuntimeValidator = Callable[[], bool]

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

# Re-title prompt. Unlike the first-exchange prompts above, this one is shown
# the WHOLE conversation plus the current title, and is biased to KEEP the
# existing title unless the durable topic has clearly drifted. This is what
# stops a localized detour (e.g. "write the PDF" inside a long USCIS-RFE
# conversation) from clobbering a title that describes the real subject.
_RETITLE_PROMPT = (
    "You are maintaining the title of an ongoing conversation. Below is the conversation "
    "so far (it may be condensed to its opening and most recent turns) and its CURRENT title.\n\n"
    "Assess whether the current title still captures the conversation's main topic and intent "
    "considered as a WHOLE — not just the most recent message.\n\n"
    "Rules:\n"
    "- If the current title is still accurate, return it UNCHANGED, verbatim.\n"
    "- Strongly prefer keeping or only lightly adjusting the current title. Most of the time it "
    "does not need to change.\n"
    "- Only write a substantially different title if the conversation's main subject has clearly "
    "and durably moved away from what the current title describes.\n"
    "- Do NOT retitle based on the latest message alone; a brief detour or sub-task is not a "
    "topic change.\n"
    "- Keep it short (3-7 words). Write the title in the same language the user is writing in.\n"
    "Return ONLY the title text, nothing else. No quotes, no punctuation at the end, no prefixes."
)

_RETITLE_PROMPT_PINNED_LANGUAGE = (
    "You are maintaining the title of an ongoing conversation. Below is the conversation "
    "so far (it may be condensed to its opening and most recent turns) and its CURRENT title.\n\n"
    "Assess whether the current title still captures the conversation's main topic and intent "
    "considered as a WHOLE — not just the most recent message.\n\n"
    "Rules:\n"
    "- If the current title is still accurate, return it UNCHANGED, verbatim.\n"
    "- Strongly prefer keeping or only lightly adjusting the current title. Most of the time it "
    "does not need to change.\n"
    "- Only write a substantially different title if the conversation's main subject has clearly "
    "and durably moved away from what the current title describes.\n"
    "- Do NOT retitle based on the latest message alone; a brief detour or sub-task is not a "
    "topic change.\n"
    "- Keep it short (3-7 words). Write the title in {language}.\n"
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


def _auto_title_enabled() -> bool:
    """Return whether automatic session title generation is enabled."""
    try:
        # Lazy imports, matching _title_language(): title_generator is imported
        # from agent code paths where a module-level hermes_cli import risks
        # circularity, and the read-only loader avoids config-migration writes.
        from hermes_cli.config import load_config_readonly
        from utils import is_truthy_value

        config = load_config_readonly()
        title_config = (config.get("auxiliary") or {}).get("title_generation") or {}
        return is_truthy_value(title_config.get("enabled"), default=True)
    except Exception:
        logger.debug("Failed to read title_generation.enabled", exc_info=True)
        return True


def generate_title(
    user_message: str,
    assistant_response: str,
    timeout: Optional[float] = None,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    runtime_validator: Optional[RuntimeValidator] = None,
) -> Optional[str]:
    """Generate a session title from the first exchange.

    Uses the main runtime's model when available, falling back to the
    auxiliary LLM client (cheapest/fastest available model).
    Returns the title string or None on failure.

    ``failure_callback`` is invoked with ``(task, exception)`` when the
    auxiliary call raises — the caller typically wires this to
    ``AIAgent._emit_auxiliary_failure`` so the user sees a warning instead
    of silently accumulating untitled sessions.

    ``runtime_validator`` is called right before the LLM request. If it
    returns False (e.g. the user's model was switched since the background
    thread captured its runtime snapshot), the call is skipped silently —
    no request is sent, so a stale title request can't reload a model the
    runtime already unloaded (#19027).
    """
    if not _auto_title_enabled():
        logger.debug("Auto-title skipped: auxiliary.title_generation.enabled=false")
        return None

    if runtime_validator is not None:
        try:
            if not runtime_validator():
                logger.debug("Title generation skipped: runtime validator returned False")
                return None
        except Exception:
            # Fail open: a broken validator must not disable titling.
            logger.debug("Title runtime validator raised; proceeding", exc_info=True)

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
        content = response.choices[0].message.content or ""
        # Strip thinking/reasoning blocks that think-enabled models
        # (MiniMax M2.7, DeepSeek, etc.) emit even for simple prompts like
        # title generation. Without this the raw <think>...</think> XML
        # leaks into session titles. Reuses the canonical scrubber so all
        # tag variants (unterminated blocks, orphan closes, mixed case)
        # are handled, not just a single literal <think> pair.
        from agent.agent_runtime_helpers import strip_think_blocks
        title = strip_think_blocks(None, content).strip()
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


def _persist_session_title(session_db, session_id, title):
    """Persist a generated title, recovering from duplicate-title collisions.

    The write goes through ``set_auto_title_if_empty`` (predicate + write in
    one transaction) so a manual ``/title`` set while LLM generation was in
    flight is never overwritten — a plain ``set_session_title`` fallback keeps
    older stores working. ``set_session_title`` raises ValueError when the
    title would collide with another session (the unique-title index). Rather
    than swallow it and leave the session untitled (#50537), append a #N
    suffix via get_next_title_in_lineage() when the store supports lineage
    dedup; otherwise re-raise so the caller can decide.

    Returns the title actually persisted, or None when a concurrent manual
    title won the race (nothing was written).
    """
    atomic_fn = getattr(session_db, "set_auto_title_if_empty", None)

    def _set(t):
        if atomic_fn is not None:
            if not atomic_fn(session_id, t):
                # Predicate failed: a title appeared while generation was in
                # flight (manual /title wins), or the session vanished.
                logger.debug(
                    "Skipping auto-generated session title because a title "
                    "was set while generation was in flight"
                )
                return None
            return t
        ok = session_db.set_session_title(session_id, t)
        if ok is False:
            raise RuntimeError(
                f"session {session_id} not found when storing title"
            )
        return t

    try:
        return _set(title)
    except ValueError:
        next_title_fn = getattr(session_db, "get_next_title_in_lineage", None)
        if next_title_fn is None:
            raise
        deduped = next_title_fn(title)
        if not deduped or deduped == title:
            raise
        return _set(deduped)


def _condense_history(
    conversation_history: list,
    head_turns: int = 1,
    tail_turns: int = 3,
    per_message: int = 400,
) -> str:
    """Render a conversation into a compact transcript for whole-conversation
    title assessment.

    Keeps the OPENING ``head_turns`` user/assistant exchanges (they anchor the
    conversation's original intent — the thing a good title names) and the most
    recent ``tail_turns`` exchanges (they reveal genuine topic drift). Anything
    in between is elided with a ``[…]`` marker so the request stays cheap on a
    long conversation instead of dumping the entire transcript into the
    auxiliary model.

    Only ``user`` and ``assistant`` roles are included; system/tool messages
    are skipped. Each message is truncated to ``per_message`` chars.
    """
    msgs = [
        m for m in (conversation_history or [])
        if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()
    ]
    if not msgs:
        return ""

    # A "turn" here is a single message; head/tail are counted in messages so
    # the head captures the opening user+assistant pair at head_turns=1 -> 2.
    head_n = max(0, head_turns) * 2
    tail_n = max(0, tail_turns) * 2

    if len(msgs) <= head_n + tail_n:
        selected = list(msgs)
        elided_at = -1
    else:
        head = msgs[:head_n]
        tail = msgs[len(msgs) - tail_n:] if tail_n else []
        selected = head + tail
        elided_at = len(head)

    lines = []
    for i, m in enumerate(selected):
        if i == elided_at:
            lines.append("[… earlier turns omitted …]")
        role = "User" if m.get("role") == "user" else "Assistant"
        content = (m.get("content") or "").strip()
        if len(content) > per_message:
            content = content[:per_message] + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _looks_like_title(text: str) -> bool:
    """Return True if ``text`` is shaped like a real title, not prose.

    The retitle model is asked "should the title change?" and sometimes answers
    CONVERSATIONALLY ("The title remains accurate. The conversation is still
    about …") instead of returning a title. Without this guard that sentence is
    sanitized, truncated at 80 chars, and stored AS the title — then pushed to
    the Discord thread name. A genuine 3-7 word title never trips these signals;
    prose reliably does. Multi-signal reject (approved threshold):

    - >10 words         → prose, not a 3-7 word title
    - >80 chars         → longer than any legitimate short title
    - mid-sentence '. ' → a period followed by more words is a sentence, not a title
    - internal newline  → multi-line output is never a title
    """
    if not text:
        return False
    if len(text) > 80:
        return False
    if "\n" in text:
        return False
    if len(text.split()) > 10:
        return False
    # A sentence break — a lowercase word ending in '.' followed by a space and
    # more text — means prose ("… accurate. The conversation …"). Requiring a
    # LOWERCASE letter before the dot avoids false-flagging abbreviations whose
    # dotted component is uppercase or single-letter ("U.S. Visa Renewal",
    # "e.g. Docker", "Q3 Review"). A trailing period is stripped elsewhere.
    if re.search(r"[a-z]\.\s+\S", text):
        return False
    return True


def regenerate_title(
    conversation_history: list,
    current_title: str,
    timeout: float = 30.0,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
) -> Optional[str]:
    """Re-assess an existing session title against the WHOLE conversation.

    Unlike :func:`generate_title` (which only sees the first exchange), this
    reads a condensed view of the entire conversation plus the current title,
    and is prompted to KEEP the existing title unless the durable topic has
    clearly drifted. Returns the (possibly unchanged) title, or ``None`` on
    failure / empty transcript.
    """
    transcript = _condense_history(conversation_history)
    if not transcript:
        return None

    language = _title_language()
    prompt = (
        _RETITLE_PROMPT_PINNED_LANGUAGE.format(language=language)
        if language else _RETITLE_PROMPT
    )

    user_block = (
        f"CURRENT TITLE: {current_title or '(none)'}\n\n"
        f"CONVERSATION:\n{transcript}"
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_block},
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
        title = title.strip('"\'')
        if title.lower().startswith("title:"):
            title = title[6:].strip()
        # Reject conversational / prose output instead of truncating it into a
        # title. When the model answers "should this change?" in prose rather
        # than returning a title, treat it as "no usable new title" → None,
        # which the caller reads as "keep the current title, don't rename".
        # (Do NOT fall back to title[:77]+"..." here — that is exactly how a
        # 100-char sentence became a Discord thread name.)
        if not _looks_like_title(title):
            logger.debug("Retitle: rejected non-title output: %r", title[:120])
            return None
        return title if title else None
    except Exception as e:
        logger.warning("Title regeneration failed: %s", e)
        logger.debug("Title regeneration traceback", exc_info=True)
        if failure_callback is not None:
            try:
                failure_callback("title regeneration", e)
            except Exception:
                logger.debug("Title regeneration failure_callback raised", exc_info=True)
        return None


def auto_title_session(
    session_db,
    session_id: str,
    user_message: str,
    assistant_response: str,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    title_callback: Optional[TitleCallback] = None,
    runtime_validator: Optional[RuntimeValidator] = None,
) -> None:
    """Generate and set a session title if one doesn't already exist.

    Called in a background thread after the first exchange completes.
    Silently skips if:
    - session_db is None
    - session already has a title (user-set or previously auto-generated)
    - title generation fails
    - runtime_validator returns False (model was switched)

    Never lets an exception escape: this is a daemon-thread target, and an
    escaping exception would spray a raw traceback into the user's terminal
    via the default threading excepthook. The canonical trigger is the
    post-``hermes update`` stale-module window, where this function's lazy
    imports read NEW source from disk while already-cached modules
    (``agent.portal_tags`` etc.) are still the OLD version — the resulting
    ImportError repeats on every auto-title attempt until the long-running
    process restarts.
    """
    try:
        _auto_title_session(
            session_db,
            session_id,
            user_message,
            assistant_response,
            failure_callback=failure_callback,
            main_runtime=main_runtime,
            title_callback=title_callback,
            runtime_validator=runtime_validator,
        )
    except Exception as e:
        # WARNING (not debug) so operators see it in agent.log; the message
        # names the likely cause so "restart the process" is discoverable.
        logger.warning(
            "Auto-title failed (harmless; if this started after an update, "
            "restart the running Hermes process): %s",
            e,
        )
        logger.debug("Auto-title traceback", exc_info=True)
        if failure_callback is not None:
            try:
                failure_callback("title generation", e)
            except Exception:
                logger.debug("Auto-title failure_callback raised", exc_info=True)


def _auto_title_session(
    session_db,
    session_id: str,
    user_message: str,
    assistant_response: str,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    title_callback: Optional[TitleCallback] = None,
    runtime_validator: Optional[RuntimeValidator] = None,
) -> None:
    """Body of :func:`auto_title_session` — see its docstring."""
    if not session_db or not session_id:
        return

    # Check if title already exists (user may have set one via /title before first response)
    try:
        existing = session_db.get_session_title(session_id)
        if existing:
            return
    except Exception:
        return

    # This runs on a bare daemon thread spawned AFTER the turn's ambient
    # conversation context was reset, so publish it here from the session id
    # we already hold — the title-generation LLM call then carries the same
    # ``conversation=`` Portal tag as the turn it titles. Root-of-lineage for
    # consistency with the agent loop (a no-op on first exchange, where
    # titling happens, but correct if this ever runs on a continuation).
    from agent.aux_accounting import set_accounting_context
    from agent.portal_tags import set_conversation_context

    conversation_id = session_id
    try:
        conversation_id = session_db.get_conversation_root(session_id) or session_id
    except Exception:
        pass
    set_conversation_context(conversation_id)
    # Same for the accounting context, so the title call's token usage is
    # recorded against this session (task='title_generation', #23270).
    set_accounting_context(session_db, session_id)

    title = generate_title(
        user_message,
        assistant_response,
        failure_callback=failure_callback,
        main_runtime=main_runtime,
        runtime_validator=runtime_validator,
    )
    if not title:
        return

    try:
        persisted = _persist_session_title(session_db, session_id, title)
        if persisted is None:
            return
        logger.debug("Auto-generated session title: %s", persisted)
        if title_callback is not None:
            try:
                title_callback(persisted)
            except Exception:
                logger.debug("Auto-title callback failed", exc_info=True)
    except Exception as e:
        logger.debug("Failed to set auto-generated title: %s", e)


def maybe_retitle_session(
    session_db,
    session_id: str,
    user_message: str,
    assistant_response: str,
    conversation_history: list,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    title_callback: Optional[TitleCallback] = None,
    every_n_turns: int = 6,
) -> None:
    """Periodically re-evaluate a session's title to keep it relevant as the
    conversation evolves. Fires every ``every_n_turns`` user turns AFTER the
    initial auto-title (so first-turn handling stays exclusively with
    :func:`maybe_auto_title`).

    Cheap path:
    - Only runs every Nth turn.
    - Only acts once conversation_history has at least 3 user messages.
    - Assesses the WHOLE conversation (condensed) against the current title via
      :func:`regenerate_title`, which is biased to keep the existing title
      unless the durable topic has clearly drifted. Only saves + fires the
      callback (which drives the thread rename) when the title actually changes.
    """
    if not session_db or not session_id or not user_message or not assistant_response:
        return
    user_msg_count = sum(1 for m in (conversation_history or []) if m.get("role") == "user")
    # First-turn is handled by maybe_auto_title; only act on 3rd+ user turns.
    if user_msg_count < 3:
        return
    if every_n_turns <= 0 or (user_msg_count % every_n_turns) != 0:
        return

    # Snapshot history now — the background thread must assess the conversation
    # as it stands at this turn, not whatever it has mutated into later.
    history_snapshot = list(conversation_history or [])

    def _runner():
        try:
            existing = session_db.get_session_title(session_id) or ""
        except Exception:
            return
        new_title = regenerate_title(
            history_snapshot, existing,
            failure_callback=failure_callback, main_runtime=main_runtime,
        )
        if not new_title:
            return
        new_title = new_title.strip()
        # regenerate_title is prompted to return the current title verbatim when
        # nothing has drifted; treat an unchanged title as a no-op so we don't
        # churn the DB or spuriously rename the thread. Compare case- and
        # trailing-punctuation-insensitively so "USCIS RFE Response." doesn't
        # count as a change from "USCIS RFE Response".
        def _norm(t: str) -> str:
            return t.strip().rstrip(".!?,;: ").lower()
        if not new_title or _norm(new_title) == _norm(existing):
            return
        try:
            session_db.set_session_title(session_id, new_title)
        except Exception:
            return
        if title_callback is not None:
            try:
                title_callback(new_title)
            except Exception:
                logger.debug("Retitle callback failed", exc_info=True)

    thread = threading.Thread(target=_runner, daemon=True, name="retitle")
    thread.start()


def maybe_auto_title(
    session_db,
    session_id: str,
    user_message: str,
    assistant_response: str,
    conversation_history: list,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    title_callback: Optional[TitleCallback] = None,
    runtime_validator: Optional[RuntimeValidator] = None,
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

    # Config read comes after the cheap first-exchange guard so the file
    # isn't touched on every subsequent turn of a long session.
    if not _auto_title_enabled():
        logger.debug("Auto-title skipped: auxiliary.title_generation.enabled=false")
        return

    thread = threading.Thread(
        target=auto_title_session,
        args=(session_db, session_id, user_message, assistant_response),
        kwargs={
            "failure_callback": failure_callback,
            "main_runtime": main_runtime,
            "title_callback": title_callback,
            "runtime_validator": runtime_validator,
        },
        daemon=True,
        name="auto-title",
    )
    thread.start()
