"""Auto-generate short session titles from the first user/assistant exchange.

Runs asynchronously after the first response is delivered so it never
adds latency to the user-facing reply.
"""

import json
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

_DEFAULT_TITLE_MAX_WORDS = 3
_DEFAULT_TITLE_MAX_CHARACTERS = 40


def _coerce_bounded_int(value, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _title_preferences() -> tuple[int, int, dict[str, str]]:
    """Return compact-title preferences from config with safe bounds."""
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
        title_config = (config.get("auxiliary") or {}).get("title_generation") or {}
    except Exception:
        logger.debug("Failed to read title-generation preferences", exc_info=True)
        title_config = {}

    max_words = _coerce_bounded_int(
        title_config.get("max_words"),
        default=_DEFAULT_TITLE_MAX_WORDS,
        minimum=1,
        maximum=10,
    )
    max_characters = _coerce_bounded_int(
        title_config.get("max_characters"),
        default=_DEFAULT_TITLE_MAX_CHARACTERS,
        minimum=12,
        maximum=80,
    )

    aliases: dict[str, str] = {}
    raw_aliases = title_config.get("name_aliases")
    if isinstance(raw_aliases, dict):
        for raw_alias, raw_name in list(raw_aliases.items())[:50]:
            alias = str(raw_alias or "").strip()
            canonical = str(raw_name or "").strip()
            if alias and canonical and len(alias) <= 80 and len(canonical) <= 80:
                aliases[alias] = canonical
    return max_words, max_characters, aliases


def _canonical_name_for_exchange(
    user_message: str,
    assistant_response: str,
    name_aliases: dict[str, str],
) -> Optional[str]:
    """Return the canonical name for the first configured alias in an exchange.

    The prompt still encourages the model to use configured names, but this
    post-generation lookup makes the documented case-insensitive behavior
    deterministic. At the same position, the longest alias wins so a specific
    alias such as ``atlas app`` takes precedence over ``atlas``.
    """
    if not name_aliases:
        return None

    exchange = re.sub(
        r"\s+",
        " ",
        f"{str(user_message or '')}\n{str(assistant_response or '')}",
    ).casefold()
    matches: list[tuple[int, int, int, str]] = []
    for order, (raw_alias, raw_canonical) in enumerate(name_aliases.items()):
        alias = re.sub(r"\s+", " ", str(raw_alias or "").strip()).casefold()
        canonical = str(raw_canonical or "").strip()
        if not alias or not canonical:
            continue
        match = re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", exchange)
        if match:
            matches.append((match.start(), -len(alias), order, canonical))

    return min(matches)[3] if matches else None


def _build_title_prompt(
    *,
    language: str,
    max_words: int,
    max_characters: int,
    name_aliases: dict[str, str],
) -> str:
    language_rule = (
        f"Write the title in {language}."
        if language
        else "Write the title in the same language the user is writing in."
    )
    prompt = (
        "Generate a compact, descriptive title for a conversation that starts with the following exchange. "
        f"Prefer 1-{max_words} words and stay within {max_characters} characters. "
        "When the user explicitly names a project, product, organization, person, or place, normally use "
        "that named project or proper name alone and preserve its spelling and capitalization. "
        "Avoid generic action or filler words such as Fixing, Update, Setup, Analysis, Discussion, Help, "
        "Working on, and Request unless one is essential to distinguish the subject. Do not include emoji. "
        f"{language_rule} "
        "Return ONLY the title text, nothing else. No quotes, no punctuation at the end, no prefixes."
    )
    if name_aliases:
        prompt += (
            " Apply these case-insensitive canonical-name replacements whenever an alias appears in the "
            "exchange: "
            + json.dumps(name_aliases, ensure_ascii=False, sort_keys=True)
            + "."
        )
    return prompt


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
    max_words, max_characters, name_aliases = _title_preferences()
    prompt = _build_title_prompt(
        language=language,
        max_words=max_words,
        max_characters=max_characters,
        name_aliases=name_aliases,
    )

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
        canonical_name = _canonical_name_for_exchange(
            user_message,
            assistant_response,
            name_aliases,
        )
        if canonical_name:
            # Explicit operator vocabulary wins over the generic display budget;
            # configured canonical names are already bounded to 80 characters.
            title = canonical_name
        # Enforce the configured display budget when the model ignores it.
        elif len(title) > max_characters:
            title = title[: max_characters - 3].rstrip() + "..."
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


def choose_topic_icon(
    title: str,
    user_message: str,
    allowed_emojis: list[str],
    timeout: float = 30.0,
) -> Optional[str]:
    """Choose one semantic Telegram topic emoji from a live allowlist.

    This is a best-effort auxiliary call used only when a platform explicitly
    opts into automatic topic icons. The returned value is a normal emoji key;
    the Telegram adapter resolves it to the allowed sticker's custom_emoji_id.
    """
    allowed = list(
        dict.fromkeys(
            str(emoji).strip()
            for emoji in allowed_emojis
            if str(emoji).strip()
        )
    )
    if not allowed:
        return None

    prompt = (
        "Choose exactly one emoji for a conversation topic. Select only from this allowed list: "
        + json.dumps(allowed, ensure_ascii=False)
        + ". Match the topic's meaning rather than its mood. Return exactly one emoji and nothing else."
    )
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                f"Title: {str(title or '')[:120]}\n"
                f"Opening request: {str(user_message or '')[:500]}"
            ),
        },
    ]

    try:
        response = call_llm(
            task="title_generation",
            messages=messages,
            max_tokens=32,
            temperature=0.2,
            timeout=timeout,
        )
        content = response.choices[0].message.content or ""
        from agent.agent_runtime_helpers import strip_think_blocks

        choice = strip_think_blocks(None, content).strip().strip('"\'')
        if choice.lower().startswith("emoji:"):
            choice = choice[6:].strip()
        if choice in allowed:
            return choice

        def _emoji_key(value: str) -> str:
            return value.replace("\ufe0e", "").replace("\ufe0f", "")

        normalized_matches = [
            emoji for emoji in allowed if _emoji_key(emoji) == _emoji_key(choice)
        ]
        if len(normalized_matches) == 1:
            return normalized_matches[0]

        matches = [emoji for emoji in allowed if emoji in choice]
        # Variation-selector forms can contain their plain counterpart. Keep the
        # most specific match so e.g. "⚙️" does not look ambiguous with "⚙".
        matches = [
            emoji
            for emoji in matches
            if not any(emoji != other and emoji in other for other in matches)
        ]
        return matches[0] if len(matches) == 1 else None
    except Exception:
        logger.debug("Telegram topic icon selection failed", exc_info=True)
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
