"""Regression: stale session_id must not cause an infinite Resume loop.

Issue hermes-agent#22179: when the QQ server expires a session, accepting
a Resume but immediately closing the WebSocket, the adapter previously
kept retrying Resume with the same expired session_id forever. The fix
caps Resume attempts at MAX_RESUME_ATTEMPTS and then discards the stale
session, falling back to a fresh Identify.

Tests drive ``_dispatch_payload`` synchronously with op-10 Hello frames
(where the Resume vs Identify decision is made) and inspect adapter
state. ``_create_task`` no-ops when no event loop is running, so we don't
need to actually fire the network calls — we just verify the bookkeeping.
"""

from __future__ import annotations

import pytest

from gateway.config import PlatformConfig


def _make_config(**extra):
    return PlatformConfig(enabled=True, extra=extra)


def _make_adapter():
    from gateway.platforms.qqbot import QQAdapter

    return QQAdapter(_make_config(app_id="a", client_secret="b"))


def _hello(adapter):
    adapter._dispatch_payload({"op": 10, "d": {"heartbeat_interval": 30000}})


class TestResumeLoopFallback:
    def test_initial_attempt_counter_is_zero(self):
        adapter = _make_adapter()
        assert adapter._resume_attempts == 0

    def test_resume_increments_attempt_counter(self):
        adapter = _make_adapter()
        adapter._session_id = "stale"
        adapter._last_seq = 42
        _hello(adapter)
        assert adapter._resume_attempts == 1
        # Session not yet discarded — under the threshold.
        assert adapter._session_id == "stale"
        assert adapter._last_seq == 42

    def test_resume_loop_capped_then_falls_back_to_identify(self):
        from gateway.platforms.qqbot.constants import MAX_RESUME_ATTEMPTS

        adapter = _make_adapter()
        adapter._session_id = "stale"
        adapter._last_seq = 42
        # Drive Hello until the cap is reached: each Hello increments the
        # counter and re-tries Resume.  The (cap+1)-th Hello must trip the
        # guard and discard the stale session.
        for _ in range(MAX_RESUME_ATTEMPTS):
            _hello(adapter)
        assert adapter._resume_attempts == MAX_RESUME_ATTEMPTS
        assert adapter._session_id == "stale"

        _hello(adapter)
        assert adapter._session_id is None
        assert adapter._last_seq is None
        assert adapter._resume_attempts == 0

    def test_resumed_dispatch_resets_counter(self):
        adapter = _make_adapter()
        adapter._session_id = "live"
        adapter._last_seq = 7
        _hello(adapter)
        _hello(adapter)
        assert adapter._resume_attempts == 2

        adapter._dispatch_payload(
            {"op": 0, "t": "RESUMED", "s": 8, "d": {}}
        )
        assert adapter._resume_attempts == 0
        # Session must remain intact — RESUMED is the success signal.
        assert adapter._session_id == "live"

    def test_ready_dispatch_resets_counter(self):
        adapter = _make_adapter()
        adapter._session_id = "stale"
        adapter._last_seq = 1
        _hello(adapter)
        _hello(adapter)
        assert adapter._resume_attempts == 2

        adapter._dispatch_payload(
            {"op": 0, "t": "READY", "s": 1, "d": {"session_id": "fresh"}}
        )
        assert adapter._resume_attempts == 0
        assert adapter._session_id == "fresh"

    def test_no_session_picks_identify_without_consuming_attempts(self):
        adapter = _make_adapter()
        # No session_id / last_seq → Identify path, counter untouched.
        _hello(adapter)
        assert adapter._resume_attempts == 0

    def test_after_fallback_subsequent_resume_starts_fresh(self):
        from gateway.platforms.qqbot.constants import MAX_RESUME_ATTEMPTS

        adapter = _make_adapter()
        adapter._session_id = "stale"
        adapter._last_seq = 42
        for _ in range(MAX_RESUME_ATTEMPTS + 1):
            _hello(adapter)
        # Identify success arrives.
        adapter._dispatch_payload(
            {"op": 0, "t": "READY", "s": 1, "d": {"session_id": "new"}}
        )
        # New disconnect happens; Resume should now be tried again with
        # the fresh session, starting from zero attempts.
        adapter._last_seq = 1
        _hello(adapter)
        assert adapter._resume_attempts == 1
        assert adapter._session_id == "new"


def test_max_resume_attempts_constant_exists_and_is_positive():
    """Source-level guard: future refactors must keep the cap configurable."""
    from gateway.platforms.qqbot.constants import MAX_RESUME_ATTEMPTS

    assert isinstance(MAX_RESUME_ATTEMPTS, int)
    assert MAX_RESUME_ATTEMPTS >= 1
