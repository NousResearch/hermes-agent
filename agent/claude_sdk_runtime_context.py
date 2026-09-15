"""Context-window accounting for the claude-agent-sdk runtime: usage-int coercion, prompt-token
derivation from SDK usage, the usable CLI window and the one-shot sync of the
compressor's context length from the CLI's ground truth. Extracted from
``claude_sdk_runtime.py``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("agent.claude_sdk_runtime")


def _coerce_usage_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str):
        try:
            return max(int(value), 0)
        except ValueError:
            return 0
    return 0


def _sdk_context_prompt_tokens(usage: dict, fallback: int) -> int:
    """Prompt size of the LAST SDK iteration — i.e. real context pressure.

    LOCAL DIVERGENCE (2026-08-14). ``ResultMessage.usage`` AGGREGATES across
    every internal iteration of a turn. Each iteration re-reads the whole
    prompt from cache, so on a tool-heavy turn the aggregate answers "how many
    tokens did this turn bill?" — NOT "how big is the prompt?".

    Feeding the aggregate to the context compressor over-reported this session
    by ~25x (3,601,066 reported vs ~145k real), which pinned the runtime footer
    to a clamped 100% and made gateway hygiene fire every ~5 minutes forever
    against a threshold it could never satisfy.

    The aggregate remains correct for BILLING and is still used for session
    totals / cost persistence; only context-pressure consumers use this value.
    Falls back to the aggregate when per-iteration data is absent, so a future
    SDK that drops ``iterations`` degrades to today's behaviour rather than
    reporting zero (which would disable compression entirely).
    """
    iterations = usage.get("iterations")
    if not isinstance(iterations, (list, tuple)) or not iterations:
        return fallback
    last = iterations[-1]
    if not isinstance(last, dict):
        return fallback
    total = (
        _coerce_usage_int(last.get("input_tokens"))
        + _coerce_usage_int(last.get("cache_read_input_tokens"))
        + _coerce_usage_int(last.get("cache_creation_input_tokens"))
    )
    return total or fallback


def _usable_cli_context_window(value: Any) -> Optional[int]:
    """Return a CLI window only when it is safe for Hermes tool workflows.

    ``context_usage()`` is an SDK message boundary, not model metadata. Keep
    malformed or sub-minimum reports from collapsing the parent's footer,
    compression, and hygiene sizing; the existing metadata value is safer.
    """
    from agent.model_metadata import MINIMUM_CONTEXT_LENGTH

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        candidate = value
    elif isinstance(value, str):
        text = value.strip()
        if not text or not text.isdecimal():
            return None
        candidate = int(text)
    else:
        # In particular, reject floats: silently truncating 63_999.9 turns an
        # invalid report into a plausible but unsafe context window.
        return None
    return candidate if candidate >= MINIMUM_CONTEXT_LENGTH else None


def _sync_context_length_from_cli(agent: Any) -> None:
    """Point the compressor's context_length at the CLI's REAL window.

    Hermes sizes context from model metadata -- 1,000,000 for claude-opus-5 --
    but on this lane the window is whatever the spawned CLI is actually running
    with, and ``agent.claude_agent_sdk.env.CLAUDE_CODE_AUTO_COMPACT_WINDOW`` can
    cut that to a fraction. Measured 2026-08-16 with the window at 300,000: the
    runtime footer read 16% while the CLI sat at 53% of its real window, one
    turn from autocompacting. The gauge could never have read above ~27%,
    because compaction fires at 267k of a denominator of 1,000,000 -- so the
    whole scale was squashed into its bottom quarter. Gateway session hygiene
    sizes off the same value.

    ``maxTokens`` is fixed when the CLI is spawned, so this is queried once per
    session and cached -- ``context_usage()`` is a real round-trip to the child.
    The attempt is cached on failure too: retrying every turn would spend up to
    the query timeout on the turn path to re-learn the same unavailability.
    """
    compressor = getattr(agent, "context_compressor", None)
    session = getattr(agent, "_claude_sdk_session", None)
    if compressor is None or session is None:
        return
    if getattr(agent, "_sdk_ctxlen_synced_for", None) is session:
        return
    agent._sdk_ctxlen_synced_for = session

    usage = None
    try:
        usage = session.context_usage()
    except Exception:
        logger.debug("claude-sdk context-length sync failed", exc_info=True)
    max_tokens = _usable_cli_context_window(
        usage.get("maxTokens") if isinstance(usage, dict) else None
    )
    if max_tokens is None:
        logger.debug(
            "claude-sdk context length: CLI did not report maxTokens; keeping "
            "the model-metadata value (%s)",
            getattr(compressor, "context_length", None),
        )
        return

    previous = getattr(compressor, "context_length", 0) or 0
    if previous == max_tokens:
        return
    try:
        compressor.context_length = max_tokens
    except Exception:
        logger.debug("claude-sdk context-length assignment failed", exc_info=True)
        return
    logger.info(
        "claude-sdk context length: %d (CLI maxTokens) replaces %d "
        "(model metadata) for footer and hygiene sizing",
        max_tokens, previous,
    )
