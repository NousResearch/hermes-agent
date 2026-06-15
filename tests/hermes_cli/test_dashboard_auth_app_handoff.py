"""Unit tests for the desktop system-browser session-handoff store.

The store mints single-use, short-TTL codes that ``/auth/callback`` binds to a
freshly authenticated session and the app trades at
``/api/auth/desktop-exchange``. These tests pin the round trip, single-use
semantics, expiry, and the log-truncation behaviour — mirroring the
``ws_tickets`` tests, which share the same shape.
"""
from __future__ import annotations

import pytest

from hermes_cli.dashboard_auth import app_handoff
from hermes_cli.dashboard_auth.app_handoff import (
    TTL_SECONDS,
    HandoffInvalid,
    consume_handoff,
    mint_handoff,
    _reset_for_tests,
)


@pytest.fixture(autouse=True)
def _clean_store():
    _reset_for_tests()
    yield
    _reset_for_tests()


def _mint(**over):
    kwargs = {
        "access_token": "at-value",
        "refresh_token": "rt-value",
        "expires_at": 2_000_000_000,
    }
    kwargs.update(over)
    return mint_handoff(**kwargs)


def test_round_trip_returns_cookie_material():
    code = _mint()
    material = consume_handoff(code)
    assert material["access_token"] == "at-value"
    assert material["refresh_token"] == "rt-value"
    assert material["expires_at"] == 2_000_000_000


def test_codes_are_unique_and_high_entropy():
    codes = {_mint() for _ in range(50)}
    assert len(codes) == 50
    # token_urlsafe(32) → 43 base64url chars.
    assert all(len(c) >= 43 for c in codes)


def test_second_consume_is_rejected():
    code = _mint()
    consume_handoff(code)
    with pytest.raises(HandoffInvalid):
        consume_handoff(code)


def test_unknown_code_rejected():
    with pytest.raises(HandoffInvalid):
        consume_handoff("not-a-real-code")


def test_empty_code_rejected():
    with pytest.raises(HandoffInvalid):
        consume_handoff("")


def test_ttl_is_two_minutes():
    # Pinned so a refactor that widened the live window surfaces here.
    assert TTL_SECONDS == 120


def test_expired_code_rejected(monkeypatch):
    # Mock time inside the module so mint and consume see a controlled clock;
    # ``time`` is module-level there, matching the ws_tickets test approach.
    clock = {"now": 1_000.0}
    monkeypatch.setattr(app_handoff.time, "time", lambda: clock["now"])
    code = _mint()
    clock["now"] += TTL_SECONDS + 1
    with pytest.raises(HandoffInvalid):
        consume_handoff(code)


def test_at_exact_ttl_boundary_still_valid(monkeypatch):
    clock = {"now": 1_000.0}
    monkeypatch.setattr(app_handoff.time, "time", lambda: clock["now"])
    code = _mint()
    # expires_at == mint_time + TTL; the check rejects only when strictly past.
    clock["now"] += TTL_SECONDS
    assert consume_handoff(code)["access_token"] == "at-value"


def test_unknown_code_error_truncates_value():
    # A misused/guessed code must never appear in full in the error text.
    secret = "S" * 40
    with pytest.raises(HandoffInvalid) as exc:
        consume_handoff(secret)
    assert secret not in str(exc.value)
    assert "S" * 8 in str(exc.value)
