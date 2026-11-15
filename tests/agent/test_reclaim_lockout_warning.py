"""User-visible companion to the over-threshold reclamation no-op warning (#101889).

``_warn_reclamation_no_op`` logs once per dedup key when a session rides above
``threshold_tokens`` with every reclamation path declining. This suite pins the
chat-facing twin: it rides the SAME dedup key (never more often than the log),
is silent under the threshold (ordinary hysteresis), and falls back to the
log-only behavior when no callback is plumbed (standalone/eval constructions).

Mirrors the harness shapes of test_proactive_prune_rearm_threshold.py and the
gateway delivery shapes of tests/gateway/test_compression_progress_notices.py.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from agent.context_compressor import (
    RECLAMATION_LOCKOUT_WARNING,
    ContextCompressor,
)

LARGE_WINDOW = 1_000_000
CHAT_PLATFORMS = ["telegram", "discord", "slack", "whatsapp"]


def _compressor(**kw: Any) -> ContextCompressor:
    defaults = dict(
        model="test",
        quiet_mode=True,
        threshold_percent=0.50,
        protect_first_n=2,
        protect_last_n=4,
        proactive_prune_tokens=48_000,
        proactive_prune_min_result_chars=8_000,
    )
    defaults.update(kw)
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=LARGE_WINDOW,
    ):
        return ContextCompressor(**defaults)


def _history(n_pairs: int = 8, big: int = 9_000) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    for i in range(n_pairs):
        cid = f"call_{i}"
        msgs.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": cid,
                "type": "function",
                "function": {"name": "terminal", "arguments": '{"cmd":"ls"}'},
            }],
        })
        msgs.append({
            "role": "tool",
            "tool_call_id": cid,
            "content": chr(65 + i) * big if i < 3 else "ok",
        })
    return msgs


def _over_threshold_warnings(caplog) -> list:
    return [
        r for r in caplog.records
        if r.levelno >= logging.WARNING
        and "over the compression threshold" in r.getMessage()
    ]


def test_lockout_warning_fires_once_per_dedup_key(caplog) -> None:
    """The chat warning rides the log's own dedup key: once per distinct state,
    never once per tool iteration."""
    emitted: list[str] = []
    c = _compressor(
        proactive_prune_min_reclaim_tokens=10_000_000,
        reclaim_lockout_warning_callback=emitted.append,
    )
    msgs = _history()
    billed = c.threshold_tokens + 5_000

    with caplog.at_level(logging.WARNING, logger="agent.context_compressor"):
        result, pruned = c.prune_tool_results_only(msgs, current_tokens=billed)
    assert (result, pruned) == (msgs, 0)
    assert emitted == [RECLAMATION_LOCKOUT_WARNING]

    # Same state on the next tool iteration: the log is deduped and so is the chat warning.
    with caplog.at_level(logging.WARNING, logger="agent.context_compressor"):
        c.prune_tool_results_only(msgs, current_tokens=billed)
    assert emitted == [RECLAMATION_LOCKOUT_WARNING], "chat warning fired more often than the log"
    assert len(_over_threshold_warnings(caplog)) == 1


def test_lockout_warning_refires_after_rearm_reset() -> None:
    """A rearm boundary (compaction / session reset / model recalibration)
    releases the dedup key — an identical later lockout warns in chat again."""
    emitted: list[str] = []
    c = _compressor(
        proactive_prune_min_reclaim_tokens=10_000_000,
        reclaim_lockout_warning_callback=emitted.append,
    )
    msgs = _history()
    billed = c.threshold_tokens + 5_000

    c.prune_tool_results_only(msgs, current_tokens=billed)
    c.prune_tool_results_only(msgs, current_tokens=billed)  # deduped
    assert len(emitted) == 1

    c.on_session_reset()
    c.prune_tool_results_only(msgs, current_tokens=billed)
    assert len(emitted) == 2, "lockout after a rearm reset was deduped against a stale key"


def test_lockout_warning_suppressed_under_threshold(caplog) -> None:
    """Ordinary hysteresis below the threshold stays quiet on the chat rail too."""
    emitted: list[str] = []
    c = _compressor(
        proactive_prune_min_reclaim_tokens=10_000_000,
        reclaim_lockout_warning_callback=emitted.append,
    )
    msgs = _history()

    with caplog.at_level(logging.WARNING, logger="agent.context_compressor"):
        result, pruned = c.prune_tool_results_only(
            msgs, current_tokens=c.threshold_tokens - 1
        )

    assert (result, pruned) == (msgs, 0)
    assert emitted == []
    assert not _over_threshold_warnings(caplog)


def test_callback_none_is_log_only(caplog) -> None:
    """No callback plumbed (standalone/eval constructions): current behavior —
    the log fires, nothing raises, no chat delivery is attempted."""
    c = _compressor(proactive_prune_min_reclaim_tokens=10_000_000)
    assert c.reclaim_lockout_warning_callback is None
    msgs = _history()
    billed = c.threshold_tokens + 5_000

    with caplog.at_level(logging.WARNING, logger="agent.context_compressor"):
        result, pruned = c.prune_tool_results_only(msgs, current_tokens=billed)

    assert (result, pruned) == (msgs, 0)
    assert len(_over_threshold_warnings(caplog)) == 1


def test_callback_failure_does_not_break_prune(caplog) -> None:
    """A broken status rail must never break the prune path — the no-op
    contract still returns the INPUT object and the log still fires."""

    def _boom(_message: str) -> None:
        raise RuntimeError("status rail down")

    c = _compressor(
        proactive_prune_min_reclaim_tokens=10_000_000,
        reclaim_lockout_warning_callback=_boom,
    )
    msgs = _history()
    billed = c.threshold_tokens + 5_000

    with caplog.at_level(logging.WARNING, logger="agent.context_compressor"):
        result, pruned = c.prune_tool_results_only(msgs, current_tokens=billed)

    assert (result, pruned) == (msgs, 0)
    assert len(_over_threshold_warnings(caplog)) == 1


@pytest.mark.parametrize("platform", CHAT_PLATFORMS)
def test_lockout_warning_passes_gateway_chat_filter(platform) -> None:
    """The lockout warning is actionable operator-facing signal, not routine
    compression noise: it must survive the gateway status filter (the same
    rail the '⚠ Compression summary failed…' warnings ride) on every chat
    platform, in the default (silent-by-design) config.

    Mirrors the delivery assertions in tests/gateway/test_compression_progress_notices.py.
    """
    import gateway.run as gateway_run
    from gateway.run import _prepare_gateway_status_message

    with patch.object(gateway_run, "_load_gateway_config", dict):
        assert (
            _prepare_gateway_status_message(platform, "warn", RECLAMATION_LOCKOUT_WARNING)
            == RECLAMATION_LOCKOUT_WARNING
        )


def test_lockout_warning_never_matches_noisy_status_regex() -> None:
    """Guard against regex drift: if the noisy-status filter ever grows an
    alternative that swallows the lockout warning, chat platforms would
    silently drop it — fail here first."""
    import gateway.run as gateway_run

    assert not gateway_run._TELEGRAM_NOISY_STATUS_RE.search(RECLAMATION_LOCKOUT_WARNING)


def test_pre_compaction_opportunistic_prune_suppresses_lockout_warning(caplog) -> None:
    """When full LLM compaction is eligible and about to run right after the
    opportunistic pre-compaction prune (`warn_on_noop=False`), a no-op prune
    over threshold must NOT emit a false `Context lockout` warning or poison
    the dedup key before `agent._compress_context` runs."""
    from types import SimpleNamespace
    from agent.turn_preflight import _proactive_tool_result_prune

    emitted: list[str] = []
    c = _compressor(
        proactive_prune_min_reclaim_tokens=10_000_000,
        reclaim_lockout_warning_callback=emitted.append,
    )
    msgs = _history()
    billed = c.threshold_tokens + 5_000
    agent = SimpleNamespace(compression_checkpoint_required=False, session_id="s1")

    with caplog.at_level(logging.WARNING, logger="agent.context_compressor"):
        out = _proactive_tool_result_prune(
            agent, c, msgs, billed, warn_on_noop=False
        )
    assert out is msgs
    assert emitted == []
    assert not _over_threshold_warnings(caplog)

    # Because the dedup key was not poisoned by the suppressed pre-compaction
    # check, a later genuinely blocked turn (`warn_on_noop=True`) still warns.
    with caplog.at_level(logging.WARNING, logger="agent.context_compressor"):
        out2 = _proactive_tool_result_prune(
            agent, c, msgs, billed, warn_on_noop=True
        )
    assert out2 is msgs
    assert emitted == [RECLAMATION_LOCKOUT_WARNING]
    assert len(_over_threshold_warnings(caplog)) == 1


def test_over_threshold_pre_compaction_prune_uses_tail_token_budget() -> None:
    """When oversized tool results sit inside `protect_last_n` (e.g. last 20 messages)
    and push the request over `threshold_tokens`, default sub-threshold prune
    (`warn_on_noop=True`) protects them by message count, whereas pre-compaction
    opportunistic prune (`warn_on_noop=False`) falls back to `tail_token_budget`
    so Phase-1 pruning reclaims them without calling the LLM summarizer."""
    c = _compressor(
        protect_last_n=20,
        proactive_prune_min_reclaim_tokens=4_096,
    )
    c.tail_token_budget = 2_000
    # 8 tool pairs = 17 messages total (<= protect_last_n=20, > _MAX_TAIL_MESSAGE_FLOOR=8),
    # with 30k-char tool payloads in pairs 0..2.
    msgs = _history(n_pairs=8, big=30_000)
    billed = c.threshold_tokens + 5_000

    # Ordinary count-protected prune (`warn_on_noop=True`) declines (`prune:tail_only`).
    out_default, n_default = c.prune_tool_results_only(msgs, current_tokens=billed, warn_on_noop=True)
    assert (out_default, n_default) == (msgs, 0)

    # Pre-compaction opportunistic prune (`warn_on_noop=False`) falls back to
    # `tail_token_budget` and prunes the oversized tool outputs deterministically,
    # including when invoked via `_proactive_tool_result_prune(..., warn_on_noop=False)`.
    from types import SimpleNamespace
    from agent.turn_preflight import _proactive_tool_result_prune

    agent = SimpleNamespace(compression_checkpoint_required=False, session_id="s-tail")
    out_via_preflight = _proactive_tool_result_prune(
        agent, c, msgs, billed, warn_on_noop=False
    )
    assert out_via_preflight is not msgs


def test_over_threshold_pre_compaction_prune_lowers_min_chars_to_compaction_floor() -> None:
    """After earlier sub-threshold prunes have already demoted >4000-char tool results,
    crossing `threshold_tokens` with medium tool results (500..4000 chars) must fall back
    to `_PRUNE_MIN_CHARS` (500) during `_over_threshold_pre_compaction` so Phase-1
    pruning reclaims them before calling full LLM compaction."""
    from types import SimpleNamespace
    from agent.turn_preflight import _proactive_tool_result_prune

    c = _compressor(
        protect_last_n=12,
        proactive_prune_min_result_chars=4_000,
        proactive_prune_min_reclaim_tokens=4_096,
    )
    c.tail_token_budget = 2_000
    # 16 tool pairs with 2,500-char payloads (< 4,000 proactive floor, >= 500 compaction floor)
    msgs: list[dict] = [{"role": "user", "content": "start"}]
    for i in range(16):
        cid = f"c_{i}"
        msgs.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": cid,
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ],
            }
        )
        msgs.append(
            {
                "role": "tool",
                "tool_call_id": cid,
                "content": (f"line_{i} " * 360)[:2500],
            }
        )
    billed = c.threshold_tokens + 2_500
    # Ordinary sub-threshold prune (`warn_on_noop=True`) ignores <4000-char payloads.
    out_default, n_default = c.prune_tool_results_only(
        msgs, current_tokens=billed, warn_on_noop=True
    )
    assert (out_default, n_default) == (msgs, 0)

    # Pre-compaction prune (`warn_on_noop=False`) lowers `min_prune_chars` to 500
    # and reclaims the medium tool results deterministically.
    agent = SimpleNamespace(compression_checkpoint_required=False, session_id="s-med")
    out_pre = _proactive_tool_result_prune(
        agent, c, msgs, billed, warn_on_noop=False
    )
    assert out_pre is not msgs



