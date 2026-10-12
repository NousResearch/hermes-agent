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

# ``cron.scheduler_prompt._prepend_context_block`` frames a previous/upstream run's output as a
# fenced block that sits AHEAD of the job prompt (after the cron hint). Pinned to the build site
# by contract test; the block always opens with the self-context heading (upstream-context rows
# are cron-run rows that open with the hint and are cut by it first).
CONTEXT_BLOCK_HEADING = "## Your previous run's output"
# ``_prepend_context_block`` emits ``## heading\n{intro}\n\n```\n{body}\n```\n\n{prompt}``.
# Unflattened, the opening fence is the first ``\n\n```\n`` and the closing fence the next
# ``\n```\n\n``. Once a listing query flattens newlines to spaces the fences are indistinguishable
# by spacing alone, so the flattened cut finds the FIRST fence pair positionally: opening =
# first "```" after the heading, closing = first "```" after the block body, then skip the
# blank-line space pair that precedes the job prompt.
FENCE = "```"
FENCE_OPEN = "\n\n```\n"
FENCE_BLOCK_END = "\n```\n\n"

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
    """Cut leading machine-authored wrappers, returning the content behind them.

    Cuts in sequence: wrapper paragraphs (cron hint, bot-chat banner — they can stack) and the
    fenced previous-run context block (``_prepend_context_block``), which sits between the hint
    and the job prompt. A construct whose terminator is missing — a truncated excerpt, or a
    reworded build site the recognizers no longer match end-to-end — is left alone rather than
    guessed at: showing half a wrapper is wrong, but silently deleting the whole message is worse.
    """
    current = (text or "").strip()
    for _ in range(12):  # bounded: each pass must remove one construct or the loop stops
        cut = _strip_one_wrapper(current)
        if cut == current:
            cut = _strip_one_context_block(current)
        if cut == current:
            break
        current = cut
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


def _strip_one_context_block(text: str) -> str:
    stripped = text.lstrip()
    if not stripped.startswith(CONTEXT_BLOCK_HEADING):
        return text
    open_idx = stripped.find(FENCE_OPEN)
    if open_idx != -1:
        close = stripped.find(FENCE_BLOCK_END, open_idx + len(FENCE_OPEN))
        if close == -1:
            return text
        remainder = stripped[close + len(FENCE_BLOCK_END):].strip()
        return remainder if remainder else text
    # Flattened (listing queries): newlines are gone, so the two fences are only distinguishable
    # positionally — first is the opening, second the closing; the blank line after it is now
    # spaces, absorbed by lstrip.
    first = stripped.find(FENCE)
    if first == -1:
        return text
    second = stripped.find(FENCE, first + len(FENCE))
    if second == -1:
        return text
    remainder = stripped[second + len(FENCE):].lstrip()
    return remainder if remainder else text
