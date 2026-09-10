"""Auto-generate short session titles from the user's opening message.

Two stages, both off the critical path: an **instant** deterministic title (written before the model
is called, cannot fail), then an **upgrade** from one small-model call (cheap tier, thinking off,
JSON-constrained). Storage enforces provenance ``derived < llm < user``: stage 2 only replaces stage 1
and neither replaces a name the user typed."""

import json
import logging
import re
import threading
from contextlib import suppress
from typing import Any, Callable, Optional

from agent.auxiliary_client import call_llm
from agent.context_compressor import LEGACY_SUMMARY_PREFIX
from agent.message_content import flatten_message_text

logger = logging.getLogger(__name__)

# (task_name, exception) -> None; surfaces auxiliary failures so silent drops don't pile up as NULL titles.
FailureCallback = Callable[[str, BaseException], None]
# (title, source) -> None; source is the persisted provenance (``derived`` / ``llm``). Consumers paying a
# rate-limited remote rename per title (Discord thread, Telegram topic) should act on ``llm`` only.
TitleCallback = Callable[[str, str], None]
# () -> bool, called right before the LLM request; False skips (e.g. the user switched models and
# the request would reload one the runtime already evicted).
# Validation callback: () -> bool. See #19027.
RuntimeValidator = Callable[[], bool]

# Text budget handed to the model (Claude Code / OpenClaw converged on 1000).
MAX_TITLE_INPUT_CHARS = 1000
# Cap on the instant derived title; a raw fragment reads worse the longer it runs.
MAX_DERIVED_TITLE_CHARS = 48
# Answer-shaped guard: a tiny model sometimes answers instead of titling; longer is rejected, not truncated.
# Upper bound on accepted title word count. Titling is a 3-7 word task; a small tiny-model sometimes ignores
# the task and answers the user's message instead — that answer must never become the session title (see the
# answer-shaped output guard in generate_title; port of can1357/oh-my-pi#7306). 12 leaves headroom for
# legitimate wordy titles while excluding full-sentence answers.
_MAX_TITLE_WORDS = 12

_TITLE_PROMPT_TEMPLATE = (
    "You name chat sessions. Given the user's opening message, write a title "
    "that lets them find this conversation again in a list.\n\n"
    "Rules:\n"
    "- 3 to 7 words, sentence case (capitalize only the first word and proper nouns).\n"
    "- Name what the user wants DONE, not that they asked a question.\n"
    "- Keep technical terms, filenames, numbers, and error codes exact.\n"
    "- Drop filler words: the, this, my, a, an.\n"
    "- No trailing punctuation, no quotes, no tool names, no 'Title:' prefix.\n"
    "- Never answer the message. Name it.\n"
    "- Always produce something, even for a bare greeting.\n"
    "__LANGUAGE_RULE__\n"
    'Good: {"title": "Fix login button on mobile"}\n'
    'Good: {"title": "Postgres connection pool exhaustion"}\n'
    'Good: {"title": "Friendly greeting"}\n'
    'Too vague: {"title": "Code changes"}\n'
    'Too long: {"title": "Investigate and fix the issue where the login button '
    'does not respond on mobile devices"}\n\n'
    'Reply with JSON only: {"title": "..."}'
)

# Prompt for regenerate_title() — feeds condensed conversation history instead of opening message.
_RETITLE_PROMPT = (
    "You name chat sessions. Given the conversation so far, write a title "
    "that lets the user find this conversation again in a list.\n\n"
    "Rules:\n"
    "- 3 to 7 words, sentence case (capitalize only the first word and proper nouns).\n"
    "- Name what the conversation is actually about or what the user wants DONE.\n"
    "- Keep technical terms, filenames, numbers, and error codes exact.\n"
    "- Drop filler words: the, this, my, a, an.\n"
    "- No trailing punctuation, no quotes, no tool names, no 'Title:' prefix.\n"
    "- Never summarize the conversation. Name it.\n"
    "- Always produce something, even if the conversation opened with a greeting.\n"
    "__LANGUAGE_RULE__\n"
    'Good: {"title": "Debugging Postgres connection pool"}\n'
    'Good: {"title": "Hermes session auto-titling explained"}\n'
    'Too vague: {"title": "Conversation about databases"}\n'
    'Reply with JSON only: {"title": "..."}'
)

_LANGUAGE_RULE_MATCH_USER = "- Write the title in the same language as the user's message."
_LANGUAGE_RULE_PINNED = "- Write the title in {language}."

# Constrains the response to a single title field ("model answered instead of titling" failure class).
_TITLE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "session_title", "strict": True, "schema": {
        "type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"], "additionalProperties": False}},
}

# Control-tag wrappers around machine-authored content inside a nominal "user" message (Codex CLI's
# RECOGNIZED_CONTROL_WRAPPERS): stripped, titling continues on what remains.
_CONTROL_WRAPPERS = tuple(
    (f"<{tag}>", f"</{tag}>")
    for tag in ("command-message", "command-name", "command-args", "local-command-caveat", "local-command-stderr",
                "local-command-stdout", "task-notification", "system-reminder", "ide_opened_file", "ide_selection")
)

# Hermes' own machine-authored openers: a compaction handoff or resumed session must not be titled after them.
_MACHINE_PREFIXES = (
    "[CONTEXT COMPACTION", LEGACY_SUMMARY_PREFIX, "[Runtime note:", "[System note:", "[SYSTEM]",
    # tui_gateway.server._MODEL_SWITCH_MARKER_PREFIX (keep in sync); persisted as role="user" because
    # strict providers reject a non-first system message.
    # Model-switch marker from tui_gateway.server._append_model_switch_marker. It is persisted with
    # role="user" (strict OpenAI-compatible providers reject a system message that is not first — #48338),
    # so without this entry it looks like a real opening turn: switching models before the first real
    # message titled the session "[System: The active model for this chat has…" instead of the user's actual
    # question.
    "[System: The active model for this chat has changed to ",
)


def _title_config() -> dict:
    """``auxiliary.title_generation`` (lazy read-only import: no hermes_cli cycle, no migration writes)."""
    from hermes_cli.config import load_config_readonly
    return ((load_config_readonly() or {}).get("auxiliary") or {}).get("title_generation") or {}


def _title_language() -> str:
    """Configured title language, or "" to match the user."""
    try:
        return str(_title_config().get("language", "")).strip()
    except Exception:
        return ""


def _auto_title_enabled() -> bool:
    try:
        from utils import is_truthy_value
        return is_truthy_value(_title_config().get("enabled"), default=True)
    except Exception:
        logger.debug("Failed to read title_generation.enabled", exc_info=True)
        return True


# Defaults for auxiliary.title_generation.retitle. Kept in sync with
# hermes_cli/config_defaults.py — the merge below preserves any key the user
# omitted and skips explicit ``None`` overrides so a YAML ``null`` cannot blank
# out a default. Explicit ``false`` / ``0`` / ``""`` DO override.
_RETITLE_DEFAULTS = {
    "enabled": True,
    "auto_at_turn": 10,
    "turns_window": 10,
    "slash_command": True,
    "cli_command": True,
    "touch_platform_names": False,
    "provider": "",
    "model": "",
    "base_url": "",
    "api_key": "",
    "timeout": 30,
    "max_concurrency": 2,
    "prefer_fast_model": None,
}


def _retitle_config() -> dict:
    """Return the merged ``auxiliary.title_generation.retitle`` config.

    Defaults from ``_RETITLE_DEFAULTS`` are merged with any user overrides.
    User keys whose value is ``None`` are treated as "use default" so a YAML
    ``null`` does not blank out a baked-in default. Returns ``{}`` on any
    config error (fail-open at call sites).
    """
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
        title_config = (config.get("auxiliary") or {}).get("title_generation") or {}
        user = title_config.get("retitle") or {}
        merged = dict(_RETITLE_DEFAULTS)
        if isinstance(user, dict):
            for key, value in user.items():
                if value is None:
                    continue
                merged[key] = value
        return merged
    except Exception:
        logger.debug("Failed to read title_generation.retitle", exc_info=True)
        return {}


def _retitle_enabled() -> bool:
    """Return whether the retitle feature is enabled (default True)."""
    try:
        from utils import is_truthy_value

        cfg = _retitle_config()
        return is_truthy_value(cfg.get("enabled"), default=True)
    except Exception:
        logger.debug("Failed to read title_generation.retitle.enabled", exc_info=True)
        return True


def strip_control_wrappers(text: str) -> str:
    """Remove leading control wrappers (nested too) so a slash-command turn reduces to the prose the user typed."""
    current = (text or "").strip()
    for _ in range(len(_CONTROL_WRAPPERS) * 2):  # bounded: each pass must remove a wrapper or we stop
        stripped = _strip_one_wrapper(current)
        if stripped == current:
            break
        current = stripped
    return current


def _strip_one_wrapper(text: str) -> str:
    lowered = text.lower()
    for open_tag, close_tag in _CONTROL_WRAPPERS:
        if not lowered.startswith(open_tag):
            continue
        end = lowered.find(close_tag)
        if end == -1:  # unterminated wrapper: drop the opening tag and keep the body
            return text[len(open_tag):].strip()
        # Prefer trailing prose; otherwise the wrapper body is all we have.
        return (text[end + len(close_tag):].strip() or text[len(open_tag):end].strip()).strip()
    return text


def _summarize_user_message(user_message: str) -> str:
    """Text worth titling: describe a ``/skill`` invocation (it embeds the whole skill body), then strip wrappers."""
    if not user_message:
        return ""
    described = None
    try:
        from agent.skill_commands import describe_skill_invocation
        described = describe_skill_invocation(user_message)
    except Exception:
        logger.debug("Skill-scaffolding summary failed; titling raw", exc_info=True)
    return strip_control_wrappers(user_message if described is None else described)


def is_titleable_user_message(user_message: str) -> bool:
    """False for machine-authored openers and turns that reduce to nothing once scaffolding is stripped."""
    return (isinstance(user_message, str) and bool(user_message.strip()) and not user_message.lstrip().startswith(_MACHINE_PREFIXES)
            and bool(_summarize_user_message(user_message).strip()))


def derive_title(user_message: str) -> Optional[str]:
    """Instant title: first meaningful line trimmed to a word boundary. No model, never fails."""
    line = " ".join(_first_line(_summarize_user_message(user_message)).split())
    if len(line) > MAX_DERIVED_TITLE_CHARS:
        cut = line[:MAX_DERIVED_TITLE_CHARS]
        space = cut.rfind(" ")
        line = (cut[:space] if space > MAX_DERIVED_TITLE_CHARS // 2 else cut).rstrip(" ,.;:—-") + "…"
    return line or None


def _strip_title_prefix(text: str) -> str:
    return text[6:].strip() if text.lower().startswith("title:") else text


def _first_line(text: str) -> str:
    return next((ln.strip() for ln in text.splitlines() if ln.strip()), "")


def _extract_title_text(content: str) -> str:
    """Strict JSON, then a loose JSON scan, then first-line prose (a provider ignoring ``response_format`` still titles)."""
    if not content:
        return ""
    raw = content.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("title"), str):
            return parsed["title"].strip()
    except (ValueError, TypeError):
        pass
    match = re.search(r'"title\"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if match:
        with suppress(ValueError):
            return json.loads(f'"{match.group(1)}"').strip()
        return match.group(1).strip()
    # Prose fallback: scrub <think> blocks so reasoning can't leak into a title.
    try:
        from agent.agent_runtime_helpers import strip_think_blocks
        raw = strip_think_blocks(None, raw).strip()
    except Exception:
        logger.debug("strip_think_blocks unavailable for title output", exc_info=True)
    return _strip_title_prefix(_first_line(raw)).strip("\"'").strip()


def _clean_title(text: str) -> Optional[str]:
    """Normalize a model-produced title, or None when nothing usable remains."""
    title = _strip_title_prefix(" ".join((text or "").split()).strip("\"'").strip()).rstrip(".!,;:")
    if len(title) > 80:
        title = title[:77].rstrip() + "..."
    return title or None


def _looks_like_title(text: Optional[str]) -> bool:
    """Quality gate: accept a concise 2..12 word phrase, reject verbose prose.

    Used by ``regenerate_title`` to filter chatty small-model output — when
    handed a whole conversation instead of a single opening message, a
    tiny-model tier will sometimes answer with a summary sentence ("Here is
    a summary of the conversation...") instead of a title. This guard drops
    those before they land as the session name.
    """
    if not text:
        return False
    words = text.split()
    if len(words) < 2 or len(words) > 12:
        return False
    lowered = text.lower()
    _BAD_PREFIXES = (
        "here",
        "this conversation",
        "the conversation",
        "about ",
    )
    for bad in _BAD_PREFIXES:
        if lowered.startswith(bad):
            return False
    return True


def _safe_callback(callback: Optional[Callable], args: tuple, log_fmt: str, label: str) -> None:
    """Invoke an optional consumer callback, never raising."""
    try:
        if callback is not None:
            callback(*args)
    except Exception:
        logger.debug(log_fmt, label, exc_info=True)


def _report_failure(failure_callback: Optional[FailureCallback], exc: BaseException, label: str) -> None:
    _safe_callback(failure_callback, ("title generation", exc), "%s failure_callback raised", label)


def _notify_title(title_callback: Optional[TitleCallback], title: str, source: str, label: str) -> None:
    _safe_callback(title_callback, (title, source), "%s callback failed", label)


def generate_title(
    user_message: str,
    timeout: Optional[float] = None,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    runtime_validator: Optional[RuntimeValidator] = None,
) -> Optional[str]:
    """Title from the opening message alone (waiting for the assistant made this slow and bought
    nothing). ``runtime_validator`` runs right before the request; False skips silently.

    If it returns False (e.g. the user's model was switched since the background thread captured its runtime
    snapshot), the call is skipped silently — no request is sent, so a stale title request can't reload a
    model the runtime already unloaded (#19027).
    """
    if not _auto_title_enabled():
        logger.debug("Auto-title skipped: auxiliary.title_generation.enabled=false")
        return None
    try:
        if runtime_validator is not None and not runtime_validator():
            logger.debug("Title generation skipped: runtime validator returned False")
            return None
    except Exception:  # fail open: a broken validator must not disable titling
        logger.debug("Title runtime validator raised; proceeding", exc_info=True)
    user_snippet = _summarize_user_message(user_message)[:MAX_TITLE_INPUT_CHARS]
    if not user_snippet.strip():
        return None
    language = _title_language()
    # str.replace, not str.format: the prompt embeds literal JSON braces.
    prompt = _TITLE_PROMPT_TEMPLATE.replace(
        "__LANGUAGE_RULE__", _LANGUAGE_RULE_PINNED.format(language=language) if language else _LANGUAGE_RULE_MATCH_USER,
    )
    try:
        response = call_llm(
            task="title_generation",
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_snippet}],
            # A title is a handful of tokens; a larger ceiling let chatty models burn seconds.
            max_tokens=64, temperature=0.3, timeout=timeout, main_runtime=main_runtime,
            extra_body={"response_format": _TITLE_RESPONSE_FORMAT},
        )
        title = _clean_title(_extract_title_text(response.choices[0].message.content or ""))
        # Answer-shaped output guard: titling is a 3-7 word task, so a title with many words is a model that
        # ignored the task and answered the user's message instead ("I don't have context on X — that's not
        # something I recognize..."). Truncating would store half an assistant blob as the session title,
        # which is still an assistant blob — reject instead so the caller retries on the next exchange
        # (maybe_auto_title fires for the first two exchanges). Port of can1357/oh-my-pi#7306.
        if title is not None and len(title.split()) > _MAX_TITLE_WORDS:
            # Answer-shaped output: reject (not truncate) so the caller retries next exchange.
            logger.debug("Rejecting answer-shaped title output (%d words > %d)", len(title.split()), _MAX_TITLE_WORDS)
            return None
        return title
    except Exception as e:
        # WARNING so it shows in agent.log without debug mode; stack at debug.
        logger.warning("Title generation failed: %s", e)
        logger.debug("Title generation traceback", exc_info=True)
        _report_failure(failure_callback, e, "Title generation")
        return None


def _has_upgraded_title(session_db, session_id: str) -> bool:
    """True when the session already carries an ``llm``/``user`` title (or the check fails)."""
    try:
        source_fn = getattr(session_db, "get_session_title_source", None)
        if source_fn is not None:
            return source_fn(session_id) not in (None, "derived")
        return bool(session_db.get_session_title(session_id))
    except Exception:
        return True


def regenerate_title(
    condensed: str,
    timeout: Optional[float] = None,
) -> Optional[str]:
    """Regenerate a session title from condensed conversation history.

    Structural mirror of :func:`generate_title`, but takes an already-condensed
    conversation string (see :func:`_condense_history`) instead of a single
    opening user message. Runs on the same ``title_generation`` auxiliary
    task, so it inherits the small/fast tier, concurrency cap, and timeout
    floor configured for auto-titling.

    Returns ``None`` when:
    - ``condensed`` is empty/whitespace,
    - the retitle feature is disabled,
    - the LLM call raises (logged at WARNING),
    - the response fails the ``_looks_like_title`` quality gate.

    Callers handle failure surfacing and persistence — this function only
    produces a title candidate, it does not invoke a failure callback and
    does not touch storage.
    """
    if not condensed or not condensed.strip():
        return None
    if not _retitle_enabled():
        logger.debug("Retitle skipped: auxiliary.title_generation.retitle.enabled=false")
        return None

    language = _title_language()
    language_rule = (
        _LANGUAGE_RULE_PINNED.format(language=language)
        if language
        else _LANGUAGE_RULE_MATCH_USER
    )
    # Placeholder substitution, not str.format: the prompt embeds literal JSON
    # braces as few-shot examples, which format() would try to interpolate.
    prompt = _RETITLE_PROMPT.replace("__LANGUAGE_RULE__", language_rule)

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": condensed[:MAX_TITLE_INPUT_CHARS]},
    ]

    try:
        response = call_llm(
            task="title_generation",
            messages=messages,
            max_tokens=64,
            temperature=0.3,
            timeout=timeout,
            extra_body={"response_format": _TITLE_RESPONSE_FORMAT},
        )
        content = response.choices[0].message.content or ""
        title = _clean_title(_extract_title_text(content))
        if not _looks_like_title(title):
            logger.debug("Retitle rejected by _looks_like_title: %r", title)
            return None
        return title
    except Exception as e:
        logger.warning("Retitle generation failed: %s", e)
        logger.debug("Retitle generation traceback", exc_info=True)
        return None


def retitle_session(
    session_db,
    session_id: str,
    conversation_history: Any,
    *,
    turns_window: int = 10,
    force: bool = False,
    touch_platform_names: bool = False,
) -> Optional[str]:
    """Regenerate a session's title from recent history and persist it.

    Shared sync worker for every retitle entry point: the ``maybe_auto_retitle``
    daemon thread, the ``/retitle`` slash handler (via ``asyncio.to_thread``),
    and the ``hermes sessions retitle`` CLI. Writes with ``source="llm"`` — the
    same tier as the initial auto-title upgrade — so the provenance ladder
    (``derived < llm < user``) at the storage layer still protects any name
    the user typed.

    Guards:

    - ``session_db`` missing or ``session_id`` empty → returns None.
    - Unless ``force=True``, a title of ``user`` provenance is left alone.
      The CLI's ``--force`` flag is the one place that opts in to overwriting.

    Returns the title persisted, or ``None`` if nothing was written (empty
    condensed history, model returned nothing, or a higher-authority title
    held the row). Never raises — this is a daemon-thread target when called
    from :func:`maybe_auto_retitle`, and an escaping exception would spray a
    raw traceback into the user's terminal via the default threading
    excepthook.

    ``touch_platform_names`` is accepted but not acted on here: the platform
    rename callback lives on the agent and isn't threaded through to this
    worker. Wire the callback when the platform-name integration lands.
    """
    if session_db is None or not session_id:
        return None

    try:
        # Provenance guard. A user-set title is the only tier we refuse to
        # overwrite silently — force=True (the CLI opt-in) skips this check.
        if not force:
            try:
                source_fn = getattr(session_db, "get_session_title_source", None)
                if source_fn is not None:
                    existing_source = source_fn(session_id)
                    if existing_source == "user":
                        logger.debug(
                            "Retitle skipped: session %s carries a user-set title",
                            session_id,
                        )
                        return None
            except Exception:
                # If we can't read provenance, fall through to the write; the
                # storage layer's own precedence check is the backstop.
                logger.debug(
                    "Retitle provenance check failed for %s",
                    session_id, exc_info=True,
                )

        # Publish accounting + Portal contexts from the session id we hold,
        # so the LLM call's token usage and conversation tag land on this
        # session. Imported lazily to match _auto_title_session's post-update
        # stale-module hygiene — a daemon thread reloading NEW source here
        # while agent.portal_tags is still OLD would ImportError forever.
        from agent.aux_accounting import set_accounting_context
        from agent.portal_tags import set_conversation_context

        try:
            conversation_id = session_db.get_conversation_root(session_id) or session_id
        except Exception:
            conversation_id = session_id
        set_conversation_context(conversation_id)
        set_accounting_context(session_db, session_id)

        condensed = _condense_history(conversation_history, turns_window=turns_window)
        if not condensed:
            return None

        title = regenerate_title(condensed)
        if title is None:
            return None

        # ``set_auto_title`` refuses equal-rank writes (llm -> llm returns 0)
        # to stop the first-turn titler renaming a session on every subsequent
        # turn — see hermes_state._set_session_title docstring. Retitle is an
        # EXPLICIT rewrite intent, not that accidental case, so demote the row
        # to ``derived`` first, then land the new llm title on top. Both writes
        # are single-row updates; the intermediate ``derived`` state is invisible
        # and — even if the second write raced a manual /title — the CAS in
        # set_session_title would still reject the demotion cleanly.
        #
        # ``None``-source rows come from legacy sessions predating provenance
        # tracking; ``_title_rank(None)`` returns the same rank as ``user`` as
        # a safety default so passive re-titling never clobbers them. Explicit
        # retitle is not passive — the user or an operator-configured auto-fire
        # asked for the rewrite — so we treat None the same as llm here.
        try:
            source_fn = getattr(session_db, "get_session_title_source", None)
            demote_fn = getattr(session_db, "set_session_title_source", None)
            if callable(source_fn) and callable(demote_fn):
                current_src = source_fn(session_id)
                if current_src in ("llm", None):
                    demote_fn(session_id, "derived")
        except Exception:
            logger.debug(
                "Retitle pre-write demotion failed for %s (proceeding)",
                session_id, exc_info=True,
            )

        persisted = _persist_session_title(
            session_db, session_id, title, source="llm"
        )
        if persisted is None:
            return None

        logger.debug("Retitled session %s: %s", session_id, persisted)
        return persisted
    except Exception as e:
        logger.warning("Retitle failed (harmless): %s", e)
        logger.debug("Retitle traceback", exc_info=True)
        return None


def _persist_session_title(session_db, session_id, title, *, source, dedupe=True):
    """Persist at *source* authority via ``set_auto_title`` (precedence check + write in one
    transaction, so a manual ``/title`` is never overwritten); None when a higher authority held the row.
    ``ValueError`` = unique-title index collision → append ``#N`` via ``get_next_title_in_lineage``;
    ``dedupe=False`` re-raises instead (the derived title is on the critical path, collides constantly
    on "hi", and the model replaces it a second later anyway).

    ``ValueError`` means the name is taken by an unrelated session (the unique-title index); rather than
    leave the session untitled (#50537), append a ``#N`` suffix via ``get_next_title_in_lineage``.
    """
    auto_fn = getattr(session_db, "set_auto_title", None)

    def _set(candidate):
        if auto_fn is not None:
            if auto_fn(session_id, candidate, source=source):
                return candidate
            logger.debug("Skipping %s title: a higher-authority title already holds session %s", source, session_id)
            return None
        legacy_fn = getattr(session_db, "set_auto_title_if_empty", None)  # older store without provenance
        if legacy_fn is not None:
            return candidate if legacy_fn(session_id, candidate) else None
        if session_db.set_session_title(session_id, candidate) is False:
            raise RuntimeError(f"session {session_id} not found when storing title")
        return candidate

    try:
        return _set(title)
    except ValueError:
        next_title_fn = getattr(session_db, "get_next_title_in_lineage", None)
        deduped = next_title_fn(title) if dedupe and next_title_fn is not None else None
        if not deduped or deduped == title:
            raise
        return _set(deduped)


def apply_instant_title(session_db, session_id: str, user_message: str, title_callback: Optional[TitleCallback] = None) -> Optional[str]:
    """Write the derived title inline. Returns it, or None (no usable text, or a ``derived``+ title exists). Never raises."""
    if not session_db or not session_id:
        return None
    try:
        title = derive_title(user_message) if is_titleable_user_message(user_message) else None
        persisted = _persist_session_title(session_db, session_id, title, source="derived", dedupe=False) if title else None
        if persisted:
            _notify_title(title_callback, persisted, "derived", "Instant-title")
        return persisted
    except Exception:
        logger.debug("Instant title failed", exc_info=True)
        return None


def auto_title_session(
    session_db,
    session_id: str,
    user_message: str,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    title_callback: Optional[TitleCallback] = None,
    runtime_validator: Optional[RuntimeValidator] = None,
) -> None:
    """Generate and store the model title (daemon-thread target); skips sessions already carrying an
    ``llm``/``user`` title (a ``derived`` one is expected — upgrading it is the point). Never lets an
    exception escape (the threading excepthook would spray a traceback into the terminal); the canonical
    trigger is the post-``hermes update`` window where lazy imports read NEW source against OLD modules."""
    try:
        if not session_db or not session_id or _has_upgraded_title(session_db, session_id):
            return
        # This thread starts AFTER the turn's ambient context was reset; republish it so the call carries
        # the same Portal ``conversation=`` tag (root-of-lineage) and bills usage to this session.
        from agent.aux_accounting import set_accounting_context
        from agent.portal_tags import set_conversation_context
        conversation_id = session_id
        with suppress(Exception):
            conversation_id = session_db.get_conversation_root(session_id) or session_id
        set_conversation_context(conversation_id)
        # Same for the accounting context, so the title call's token usage is recorded against this session
        # (task='title_generation', #23270).
        set_accounting_context(session_db, session_id)
        title, source = generate_title(
            user_message, failure_callback=failure_callback, main_runtime=main_runtime, runtime_validator=runtime_validator,
        ), "llm"
        if not title:  # the inline attempt declined collisions; off the critical path the lineage scan is affordable
            title, source = derive_title(user_message), "derived"
        if not title:
            return
        try:
            persisted = _persist_session_title(session_db, session_id, title, source=source)
        except Exception as e:
            logger.debug("Failed to set auto-generated title: %s", e)
            return
        if persisted is not None:
            logger.debug("Auto-generated session title: %s", persisted)
            _notify_title(title_callback, persisted, source, "Auto-title")
    except Exception as e:
        # WARNING so operators see it in agent.log; names the likely cause.
        logger.warning("Auto-title failed (harmless; if this started after an update, restart the running Hermes process): %s", e)
        logger.debug("Auto-title traceback", exc_info=True)
        _report_failure(failure_callback, e, "Auto-title")


def _is_real_user_turn(message: Any) -> bool:
    """A question a person actually asked (Hermes persists machinery under ``role="user"``)."""
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    content = message.get("content")
    return is_titleable_user_message(content if isinstance(content, str) else flatten_message_text(content))


def _condense_history(history: Any, turns_window: int = 10) -> str:
    """Flatten the last ``turns_window`` user+assistant exchanges into text.

    Produces the compact retitle-input blob handed to the auxiliary title
    model: recent user turns and non-empty assistant replies, in order, each
    prefixed with a role label. Machine-authored user turns (compaction
    handoffs, model-switch markers) are skipped via :func:`_is_real_user_turn`
    so a session isn't titled after Hermes' own scaffolding. Tool messages are
    excluded — the titler cares about intent, not tool-call plumbing.

    Individual message bodies are clipped at 200 characters so a pasted log
    can't dominate the blob, and the final result is tail-clipped at
    :data:`MAX_TITLE_INPUT_CHARS` because recent context matters more than
    old.
    """
    if not history:
        return ""

    slice_size = turns_window * 2
    recent = history[-slice_size:]

    lines = []
    for message in recent:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")

        if role == "user":
            if not _is_real_user_turn(message):
                continue
            text = content if isinstance(content, str) else flatten_message_text(content)
            text = " ".join((text or "").split())
            if not text:
                continue
            lines.append("User: " + text[:200])
        elif role == "assistant":
            text = content if isinstance(content, str) else flatten_message_text(content)
            text = " ".join((text or "").split())
            if not text:
                continue
            lines.append("Assistant: " + text[:200])
        # skip tool, system, everything else

    joined = "\n".join(lines)
    if len(joined) > MAX_TITLE_INPUT_CHARS:
        joined = joined[-MAX_TITLE_INPUT_CHARS:]
    return joined


def _session_is_untitled(session_db, session_id: str) -> bool:
    """No title of any provenance; False when it can't tell (no model call per turn for an unreadable title)."""
    getter = getattr(session_db, "get_session_title", None)
    try:
        return callable(getter) and not str(getter(session_id) or "").strip()
    except Exception:
        logger.debug("Untitled check failed for %s", session_id, exc_info=True)
        return False


def maybe_auto_title(
    session_db,
    session_id: str,
    user_message: str,
    conversation_history: Optional[list] = None,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    title_callback: Optional[TitleCallback] = None,
    runtime_validator: Optional[RuntimeValidator] = None,
) -> None:
    """Instant inline title, then a daemon-thread upgrade. Call at the START of a turn, before the model."""
    if not session_db or not session_id or not user_message:
        return
    # History may be pre- or post-message. Skip only when BOTH past the opening turn AND named: count alone
    # left a machinery-opened session nameless; title alone never titles on an old store.
    user_msg_count = sum(1 for m in (conversation_history or []) if _is_real_user_turn(m))
    if (user_msg_count > 1 and not _session_is_untitled(session_db, session_id)) or not is_titleable_user_message(user_message):
        return
    if not _auto_title_enabled():  # config read after the cheap guards so the file isn't touched every turn
        logger.debug("Auto-title skipped: auxiliary.title_generation.enabled=false")
        return
    apply_instant_title(session_db, session_id, user_message, title_callback)
    threading.Thread(
        target=auto_title_session,
        args=(session_db, session_id, user_message),
        kwargs=dict(failure_callback=failure_callback, main_runtime=main_runtime, title_callback=title_callback, runtime_validator=runtime_validator),
        daemon=True,
        name="auto-title",
    ).start()


def maybe_auto_retitle(
    session_db,
    session_id: str,
    conversation_history: Optional[list] = None,
    *,
    auto_at_turn: int = 10,
    turns_window: int = 10,
) -> None:
    """One-shot retitle trigger: fire exactly once at the auto_at_turn mark.

    Called from the per-turn prologue. Uses **exactly-N** semantics — fires
    when ``user_msg_count == auto_at_turn`` and never fires again for the
    same session. Reasoning: this is a one-shot checkpoint, not periodic
    drift. If we used ``>=`` we would re-fire every turn afterwards; if the
    auto-fire got skipped for any reason (user-set title, disabled surface),
    we accept "manual retitle via /retitle only" over slow-motion spam.

    Nothing runs on the caller's thread beyond a cheap message count and a
    daemon-thread spawn. ``auto_at_turn=0`` disables the auto-fire.
    """
    if session_db is None or not session_id:
        return

    # Cheap guards first — the config read (below) is deferred so a long
    # session doesn't touch the config file every turn.
    if not _retitle_enabled():
        return
    if auto_at_turn <= 0:
        return

    user_msg_count = sum(
        1 for m in (conversation_history or []) if _is_real_user_turn(m)
    )
    if user_msg_count != auto_at_turn:
        return

    cfg = _retitle_config()
    touch = bool(cfg.get("touch_platform_names", False))

    thread = threading.Thread(
        target=retitle_session,
        args=(session_db, session_id, conversation_history),
        kwargs={
            "turns_window": turns_window,
            "force": False,  # auto-fire never forces
            "touch_platform_names": touch,
        },
        daemon=True,
        name="auto-retitle",
    )
    thread.start()
