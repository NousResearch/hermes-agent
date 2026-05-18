"""Integration tests: circuit breaker interaction with HonchoSessionManager.

These cover the AC from t_272aeaa1:

  - dialectic_query returns "" in <5s when Honcho is unreachable
  - 3 consecutive transport failures open the breaker
  - subsequent calls short-circuit (no network attempt, near-zero latency)
  - non-transport (4xx / application) errors do NOT open the breaker
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from plugins.memory.honcho import circuit_breaker as cb
from plugins.memory.honcho.session import (
    HonchoSession,
    HonchoSessionManager,
    _is_backend_unreachable,
)


@pytest.fixture(autouse=True)
def _reset_breaker():
    """Each test gets a fresh, low-threshold breaker so we don't drift state."""
    cb.reset_breaker()
    # Install a deterministic breaker: threshold 3, cooldown 60s, no snapshot.
    breaker = cb.HonchoCircuitBreaker(
        failure_threshold=3, cooldown_s=60.0, snapshot_path=None
    )
    cb._breaker = breaker  # type: ignore[attr-defined]
    yield
    cb.reset_breaker()


def _make_manager_with_failing_chat(exc_factory):
    """Build a HonchoSessionManager whose dialectic .chat() always raises ``exc_factory()``."""
    mgr = HonchoSessionManager()
    session = HonchoSession(
        key="cli:test",
        user_peer_id="user1",
        assistant_peer_id="hermes",
        honcho_session_id="cli-test",
    )
    mgr._cache[session.key] = session

    ai_peer = MagicMock()

    def _raise(*args, **kwargs):
        raise exc_factory()

    ai_peer.chat = MagicMock(side_effect=_raise)
    mgr._get_or_create_peer = MagicMock(return_value=ai_peer)
    return mgr, session, ai_peer


def test_dialectic_query_returns_empty_fast_on_connection_refused():
    """When the SDK raises ConnectionRefusedError, dialectic_query returns ""
    in well under 5s and does not propagate the exception."""
    def boom():
        return ConnectionRefusedError(61, "Connection refused")

    mgr, session, ai_peer = _make_manager_with_failing_chat(boom)

    t0 = time.monotonic()
    result = mgr.dialectic_query(session.key, "who am I?")
    elapsed = time.monotonic() - t0

    assert result == ""
    assert elapsed < 5.0, f"dialectic_query took {elapsed:.3f}s, must be <5s"
    ai_peer.chat.assert_called_once()


def test_three_consecutive_transport_failures_open_breaker():
    """Three ConnectionRefused failures should open the circuit; the 4th call
    short-circuits without ever invoking the SDK."""
    def boom():
        return ConnectionRefusedError(61, "Connection refused")

    mgr, session, ai_peer = _make_manager_with_failing_chat(boom)

    # First three calls hit the SDK and fail.
    for _ in range(3):
        assert mgr.dialectic_query(session.key, "who am I?") == ""

    assert ai_peer.chat.call_count == 3
    assert cb.get_breaker().state == cb.STATE_OPEN

    # Fourth call must short-circuit: no new SDK invocation, near-zero latency.
    t0 = time.monotonic()
    assert mgr.dialectic_query(session.key, "who am I?") == ""
    elapsed = time.monotonic() - t0

    assert ai_peer.chat.call_count == 3, "breaker should have short-circuited the call"
    assert elapsed < 0.1, f"short-circuit took {elapsed:.3f}s, must be near-zero"


def test_timeout_error_opens_breaker():
    """TimeoutError (the observed real-world failure mode per errors.log) also
    counts as a transport failure."""
    def boom():
        return TimeoutError("Request timed out after 30.0s")

    mgr, session, _ = _make_manager_with_failing_chat(boom)

    for _ in range(3):
        mgr.dialectic_query(session.key, "who am I?")

    assert cb.get_breaker().state == cb.STATE_OPEN


def test_application_error_does_not_open_breaker():
    """A ValueError (representing an application-level bug like bad schema)
    should NOT advance the breaker — those don't predict future connectivity."""
    def boom():
        return ValueError("invalid query shape")

    mgr, session, _ = _make_manager_with_failing_chat(boom)

    for _ in range(10):  # well past the threshold
        mgr.dialectic_query(session.key, "who am I?")

    assert cb.get_breaker().state == cb.STATE_CLOSED


def test_successful_call_resets_failure_streak():
    """A successful call between failures should reset the consecutive-failure
    counter so the breaker doesn't open prematurely."""
    state = {"calls": 0}

    def maybe_boom():
        state["calls"] += 1
        # Pattern: fail, fail, succeed, fail, fail → 4 transport failures
        # total but never 3-in-a-row, so the breaker stays closed.
        if state["calls"] in (1, 2, 4, 5):
            raise ConnectionRefusedError(61, "Connection refused")
        return "ok"

    mgr = HonchoSessionManager()
    session = HonchoSession(
        key="cli:test",
        user_peer_id="user1",
        assistant_peer_id="hermes",
        honcho_session_id="cli-test",
    )
    mgr._cache[session.key] = session
    ai_peer = MagicMock()
    ai_peer.chat = MagicMock(side_effect=maybe_boom)
    mgr._get_or_create_peer = MagicMock(return_value=ai_peer)

    for _ in range(5):
        mgr.dialectic_query(session.key, "who am I?")

    assert cb.get_breaker().state == cb.STATE_CLOSED


def test_is_backend_unreachable_classifies_common_failures():
    """Sanity coverage for the classifier — the breaker integration depends
    on it routing transport failures correctly."""
    assert _is_backend_unreachable(ConnectionRefusedError(61, "Connection refused"))
    assert _is_backend_unreachable(ConnectionError("dropped"))
    assert _is_backend_unreachable(TimeoutError("slow"))
    assert _is_backend_unreachable(OSError(61, "Connection refused"))

    # SDK-shaped errors by class name (no honcho-sdk import required)
    class APITimeoutError(Exception):
        pass

    class APIConnectionError(Exception):
        pass

    assert _is_backend_unreachable(APITimeoutError("timeout"))
    assert _is_backend_unreachable(APIConnectionError("disconnect"))

    # 5xx APIStatusError → transport-like
    class APIStatusError(Exception):
        status_code = 503

    assert _is_backend_unreachable(APIStatusError("upstream gone"))

    # 4xx APIStatusError → application error
    class BadRequest(Exception):
        # Name in {"APIStatusError", "APIError"} required for the branch
        pass
    # Rename via type() to satisfy the name check
    BadRequest.__name__ = "APIStatusError"

    err = BadRequest("malformed")
    err.status_code = 400
    assert not _is_backend_unreachable(err)

    # Generic ValueError → not transport
    assert not _is_backend_unreachable(ValueError("nope"))
