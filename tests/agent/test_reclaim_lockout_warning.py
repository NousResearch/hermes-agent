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


def _history(n_pairs: int = 8, big: int = 9_000) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": "sys"}]
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
    emitted: List[str] = []
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
    emitted: List[str] = []
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
    emitted: List[str] = []
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

    with patch.object(gateway_run, "_load_gateway_config", lambda: {}):
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
