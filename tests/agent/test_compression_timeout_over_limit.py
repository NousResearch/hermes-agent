"""Compression timeout must preserve the transcript and must not send
an over-window request as if compaction succeeded.

Field shape: no summary output for 120s, host timed out, \"no messages
dropped — continuing\", then the next Codex request still ~302K against a
272K window. The 300s auxiliary compression floor (#54915) is defeated
when the host inactivity budget is 120s and a reasoning summarizer has
not yet emitted a token.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.auxiliary_client import _COMPRESSION_TIMEOUT_FLOOR_SECONDS
from agent.conversation_compression import (
    compression_skipped_due_to_lock,
    preflight_compression_should_continue_turn,
    resolve_context_compression_timeouts,
)


def test_host_idle_is_not_shorter_than_historical_aux_floor():
    idle, ceiling = resolve_context_compression_timeouts({})
    assert idle >= _COMPRESSION_TIMEOUT_FLOOR_SECONDS
    assert ceiling >= idle


def test_explicit_zero_host_idle_still_disables_wrapper():
    idle, ceiling = resolve_context_compression_timeouts(
        {"context_timeout_seconds": 0}
    )
    assert idle == 0.0


def test_explicit_higher_host_idle_is_kept():
    idle, ceiling = resolve_context_compression_timeouts(
        {
            "context_timeout_seconds": 450,
            "context_total_ceiling_seconds": 900,
        }
    )
    assert idle == 450.0
    assert ceiling == 900.0


def test_timeout_noop_is_not_treated_as_successful_compaction():
    agent = SimpleNamespace(_compression_skipped_due_to_lock=None)
    original = [{"role": "user", "content": "keep"}]
    after = original
    assert compression_skipped_due_to_lock(agent) is False
    assert preflight_compression_should_continue_turn(
        agent,
        original_messages=original,
        compressed_messages=after,
        request_tokens=302_000,
        context_length=272_000,
    ) is False


def test_lock_skip_still_defers_without_failing_closed():
    agent = SimpleNamespace(_compression_skipped_due_to_lock="other-holder")
    original = [{"role": "user", "content": "keep"}]
    outcome = preflight_compression_should_continue_turn(
        agent,
        original_messages=original,
        compressed_messages=original,
        request_tokens=302_000,
        context_length=272_000,
    )
    assert outcome == "defer_lock"


def test_successful_shrink_may_continue():
    agent = SimpleNamespace(_compression_skipped_due_to_lock=None)
    original = [{"role": "user", "content": "big"}]
    shrunk = [{"role": "user", "content": "small"}]
    assert preflight_compression_should_continue_turn(
        agent,
        original_messages=original,
        compressed_messages=shrunk,
        request_tokens=80_000,
        context_length=272_000,
    ) is True


def test_timeout_under_window_may_continue_with_original():
    """Below the model window, a timeout no-op is safe to send (existing
    \"no messages dropped — continuing\" UX). Over-window is not."""
    agent = SimpleNamespace(_compression_skipped_due_to_lock=None)
    original = [{"role": "user", "content": "ok"}]
    assert preflight_compression_should_continue_turn(
        agent,
        original_messages=original,
        compressed_messages=original,
        request_tokens=100_000,
        context_length=272_000,
    ) is True
