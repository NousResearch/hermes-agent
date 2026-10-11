"""Recognizers for Hermes' own machine-authored message wrappers.

Two build sites frame a message for a MODEL that is not the user: the cron run hint
(``cron.scheduler_prompt._CRON_HINT``, prepended to every cron job prompt) and the bot-chat
delivery banner (``cron.scheduler_delivery._deliver_to_bot_chat``, wrapped around mirrored cron
output). Both are load-bearing for the receiving agent — the banner is what tells it the turn is
scheduled output and not the user — so they stay exactly where they are.

Display surfaces (session previews, roster rows) must NOT paint that framing: it is system
scaffolding, and the row ends up showing delivery instructions instead of the content. This leaf
holds the recognizers so every shaper strips them the same way.

Dependency-free on purpose (like ``agent.compression_marker``): the prefixes are byte-identical to
the build sites and contract tests pin that, so a reworded wrapper cannot silently stop being
stripped.
"""

from __future__ import annotations

# Byte-prefix of ``cron.scheduler_prompt._CRON_HINT``.
CRON_HINT_PREFIX = "[IMPORTANT: You are running as a scheduled cron job."
# Byte-prefix of the bot-chat delivery message built by ``_deliver_to_bot_chat``.
BOT_CHAT_DELIVERY_PREFIX = '[Cronjob "'

SYSTEM_WRAPPER_PREFIXES = (CRON_HINT_PREFIX, BOT_CHAT_DELIVERY_PREFIX)

# A wrapper is a bracketed paragraph: it ends at the first blank line whose last character is the
# closing bracket. Cutting at the first ``]`` instead would land mid-wrapper — the cron hint quotes
# ``[SILENT]`` and ``[CRON_FAILURE]`` inside itself.
WRAPPER_END = "]\n\n"
# The same terminator once a listing query has flattened newlines to spaces (``_PREVIEW_CONTENT_SQL``).
WRAPPER_END_FLATTENED = "]  "


def is_system_wrapper(text: str) -> bool:
    """True when *text* opens with a machine-authored wrapper (unflattened or flattened form)."""
    stripped = (text or "").lstrip()
    return stripped.startswith(SYSTEM_WRAPPER_PREFIXES)


def strip_system_wrappers(text: str) -> str:
    """Cut leading machine-authored wrapper paragraphs, returning the content behind them.

    Nested/stacked wrappers are cut in sequence (a bot-chat row can carry a cron-hint-wrapped
    payload). A wrapper whose terminator is missing — a truncated excerpt, or a reworded build
    site the recognizers no longer match end-to-end — is left alone rather than guessed at:
    showing half a wrapper is wrong, but silently deleting the whole message is worse.
    """
    current = (text or "").strip()
    for _ in range(len(SYSTEM_WRAPPER_PREFIXES) * 4):  # bounded: each pass must remove one or stop
        stripped = _strip_one_wrapper(current)
        if stripped == current:
            break
        current = stripped
    return current


def _strip_one_wrapper(text: str) -> str:
    stripped = text.lstrip()
    if not stripped.startswith(SYSTEM_WRAPPER_PREFIXES):
        return text
    end = stripped.find(WRAPPER_END)
    if end == -1:
        end = stripped.find(WRAPPER_END_FLATTENED)
    if end == -1:
        return text
    remainder = stripped[end + len(WRAPPER_END):].strip()
    return remainder if remainder else text
