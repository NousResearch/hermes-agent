"""Auto-rename Hermes Discord threads with a short, conversation-style title.

When Hermes auto-creates a thread on @mention (or runs inside a thread it
already participated in), the thread name starts as the raw user prompt
truncated to 80 chars.  After the first assistant response, this module
generates a clean 3-8 word title (Claude/ChatGPT style) and renames the
thread once via Discord's PATCH /channels/{thread_id} endpoint.

Kept deliberately platform-local: the Discord adapter owns the trigger
and state, so no broader gateway/core change is required.  Title
generation reuses ``agent.title_generator.generate_title`` (cheap
auxiliary LLM path).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# Discord enforces a 100-char hard limit on channel/thread names.
DISCORD_MAX_THREAD_NAME = 100

# Default cap for the generated title — keeps thread lists readable.
DEFAULT_MAX_LENGTH = 60

# Modes for when to fire the rename.  Only one mode is supported today;
# the field exists so future delays (after_first_user_response, after_n
# exchanges, ...) slot in without breaking config consumers.
DELAY_AFTER_FIRST_ASSISTANT_RESPONSE = "after_first_assistant_response"

_VALID_DELAYS = {DELAY_AFTER_FIRST_ASSISTANT_RESPONSE}

# Characters that look noisy in a thread title.
_TRAILING_PUNCTUATION = ".!?,;:\u2026"

# Discord channel/thread names cannot contain control chars; collapse
# any whitespace run (including newlines, tabs) into a single space.
_WS_RUN = re.compile(r"\s+")


@dataclass(frozen=True)
class DiscordAutoRenameConfig:
    """Resolved config for the auto-rename feature."""

    enabled: bool = False
    max_length: int = DEFAULT_MAX_LENGTH
    delay: str = DELAY_AFTER_FIRST_ASSISTANT_RESPONSE

    @property
    def effective_max_length(self) -> int:
        """Clamp the user-provided cap to Discord's hard limit."""
        if self.max_length <= 0:
            return DEFAULT_MAX_LENGTH
        return min(self.max_length, DISCORD_MAX_THREAD_NAME)


def load_config_from_env(env: Optional[dict] = None) -> DiscordAutoRenameConfig:
    """Build a config from environment variables.

    Env vars (all optional):
      DISCORD_AUTO_RENAME_THREADS    bool, default ``true``
      DISCORD_AUTO_RENAME_MAX_LENGTH int,  default ``60``
      DISCORD_AUTO_RENAME_DELAY      str,  default ``after_first_assistant_response``
    """
    src = env if env is not None else os.environ

    raw_enabled = src.get("DISCORD_AUTO_RENAME_THREADS", "true")
    enabled = str(raw_enabled).strip().lower() in ("1", "true", "yes", "on")

    raw_len = src.get("DISCORD_AUTO_RENAME_MAX_LENGTH", "")
    try:
        max_length = int(raw_len) if str(raw_len).strip() else DEFAULT_MAX_LENGTH
    except ValueError:
        max_length = DEFAULT_MAX_LENGTH

    raw_delay = (src.get("DISCORD_AUTO_RENAME_DELAY") or DELAY_AFTER_FIRST_ASSISTANT_RESPONSE).strip()
    delay = raw_delay if raw_delay in _VALID_DELAYS else DELAY_AFTER_FIRST_ASSISTANT_RESPONSE

    return DiscordAutoRenameConfig(enabled=enabled, max_length=max_length, delay=delay)


def sanitize_title(raw: str, max_length: int = DEFAULT_MAX_LENGTH) -> str:
    """Clean an LLM-generated title down to a Discord-safe thread name.

    - Strips surrounding quotes/whitespace and a leading "Title:" prefix.
    - Collapses internal whitespace runs into single spaces.
    - Trims trailing punctuation.
    - Truncates to ``max_length`` (clamped to Discord's 100-char limit)
      on a word boundary when possible, with a trailing ellipsis.
    Returns ``""`` for empty / unsalvageable input — callers should treat
    that as "skip rename".
    """
    if not raw:
        return ""

    title = raw.strip().strip("\"'\u201c\u201d\u2018\u2019")
    if title.lower().startswith("title:"):
        title = title[6:].strip().strip("\"'\u201c\u201d\u2018\u2019")

    # Collapse all whitespace (incl. newlines) into single spaces.
    title = _WS_RUN.sub(" ", title).strip()

    # Strip trailing punctuation (a single pass is enough since we
    # already collapsed whitespace).
    title = title.rstrip(_TRAILING_PUNCTUATION).strip()

    if not title:
        return ""

    cap = max(1, min(max_length, DISCORD_MAX_THREAD_NAME))
    if len(title) <= cap:
        return title

    # Truncate; prefer a word boundary so we don't chop a word in half.
    cut = title[: cap - 1]
    space = cut.rfind(" ")
    if space >= max(10, cap // 2):
        cut = cut[:space]
    return cut.rstrip(_TRAILING_PUNCTUATION + " ") + "\u2026"


def looks_like_raw_prompt(current_name: str, original_prompt: str) -> bool:
    """Heuristic: does ``current_name`` still match the raw user prompt?

    The Discord adapter seeds new threads with the first ~80 chars of the
    user's mention-stripped message (see ``DiscordAdapter._auto_create_thread``).
    If the current name still matches that seed, no human/agent has
    renamed it yet — safe to overwrite.
    """
    if not current_name or not original_prompt:
        return False
    cur = _WS_RUN.sub(" ", current_name).strip()
    seed = _WS_RUN.sub(" ", original_prompt).strip()
    if not cur or not seed:
        return False
    if cur == seed:
        return True
    # The adapter may have appended an ellipsis when truncating to 80
    # chars (``thread_name[:77] + "..."``).  Normalise that off.
    cur_no_ell = cur.rstrip(".\u2026").rstrip()
    seed_prefix = seed[: len(cur_no_ell)]
    return bool(cur_no_ell) and cur_no_ell == seed_prefix.strip()


async def maybe_auto_rename_thread(
    thread: Any,
    user_message: str,
    assistant_response: str,
    config: DiscordAutoRenameConfig,
    *,
    is_renamed: Callable[[], bool] = lambda: False,
    on_renamed: Callable[[str], None] = lambda _title: None,
    title_generator: Optional[Callable[[str, str], Optional[str]]] = None,
    edit_runner: Optional[Callable[[Any, str], Awaitable[None]]] = None,
) -> Optional[str]:
    """Generate a title and rename a Hermes-owned Discord thread.

    Returns the applied title on success, ``None`` if skipped or failed.
    All failure paths log at WARNING/DEBUG and never raise — the caller
    (a fire-and-forget background task in the adapter) must keep the
    user-facing reply uninterrupted.

    Injection points (all optional, real defaults):
      title_generator(user, assistant) -> Optional[str]
        Defaults to ``agent.title_generator.generate_title``.
      edit_runner(thread, name) -> Awaitable[None]
        Defaults to ``await thread.edit(name=name, reason=...)``.
    """
    if not config.enabled:
        return None
    if config.delay != DELAY_AFTER_FIRST_ASSISTANT_RESPONSE:
        # Reserved for future strategies; treat unknown delays as "skip"
        # rather than silently firing on the wrong trigger.
        logger.debug("Discord auto-rename: unsupported delay %r, skipping", config.delay)
        return None
    if thread is None:
        return None
    if not user_message or not assistant_response:
        return None

    try:
        if is_renamed():
            return None
    except Exception:
        # is_renamed should never raise; if it does, fail open (skip)
        # rather than risk a duplicate rename.
        return None

    current_name = getattr(thread, "name", "") or ""
    if not looks_like_raw_prompt(current_name, user_message):
        # User already picked a real name OR a previous run already set
        # one.  Don't stomp on it.
        logger.debug(
            "Discord auto-rename: thread name %r no longer matches raw prompt — skipping",
            current_name,
        )
        return None

    if title_generator is None:
        from agent.title_generator import generate_title as title_generator  # noqa: PLC0415

    try:
        raw_title = title_generator(user_message, assistant_response)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Discord auto-rename: title generation raised %s", e)
        return None

    cleaned = sanitize_title(raw_title or "", config.effective_max_length)
    if not cleaned:
        logger.debug("Discord auto-rename: empty title after sanitization, skipping")
        return None
    if cleaned == current_name:
        return None

    if edit_runner is None:
        async def edit_runner(t: Any, name: str) -> None:
            await t.edit(name=name, reason="Hermes auto-rename")

    try:
        await edit_runner(thread, cleaned)
    except Exception as e:
        logger.warning(
            "Discord auto-rename: failed to rename thread %s to %r: %s",
            getattr(thread, "id", "?"), cleaned, e,
        )
        return None

    try:
        on_renamed(cleaned)
    except Exception:  # pragma: no cover - defensive
        pass
    logger.info(
        "Discord auto-rename: thread %s renamed to %r",
        getattr(thread, "id", "?"), cleaned,
    )
    return cleaned
