"""Regression tests: _resolve_anthropic_pool_token must skip expired OAuth entries.

BUG LOCATION
    agent/anthropic_credentials.py::_resolve_anthropic_pool_token

ROOT CAUSE
    The read-only pool enumeration (``clear_expired=False, refresh=False`` —
    diagnostics never refresh by design) returns entries whose access token is
    already expired, and the resolver hands back the first OAuth token it finds.
    A stale ``hermes_pkce`` row therefore shadows a valid ``claude_code`` grant:
    every diagnostic caller (account_usage, ``hermes models``) sends a dead token
    and gets a guaranteed 401, surfacing as ``no-data`` instead of real quota.

IMPACT
    ``fetch_account_usage("anthropic")`` returns None (the 401 raises and the
    outer guard fails open) although a usable OAuth token exists in the pool.

FIX CONTRACT
    Entries with a known, past ``expires_at_ms`` are skipped (same 120s skew the
    pool's own ``_entry_needs_refresh`` uses). Entries with unknown expiry
    (None/0 — managed keys) stay eligible: fail-open beats hiding a usable token.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.credential_pool import AUTH_TYPE_OAUTH
from agent.anthropic_credentials import _resolve_anthropic_pool_token

_NOW_MS = 1_800_000_000_000  # fixed reference instant (2027-01-15 UTC)


def _entry(source, *, expires_at_ms, token="sk-ant-oat01-x"):
    return SimpleNamespace(
        id=f"id-{source}",
        source=source,
        auth_type=AUTH_TYPE_OAUTH,
        access_token=token,
        refresh_token="",
        expires_at_ms=expires_at_ms,
    )


class _FakePool:
    def __init__(self, entries):
        self._entries = entries

    def _available_entries(self, clear_expired=False, refresh=False):
        assert clear_expired is False and refresh is False
        return list(self._entries), []


@pytest.fixture(autouse=True)
def _pin_time(monkeypatch):
    monkeypatch.setattr("time.time", lambda: _NOW_MS / 1000)


def _resolve_with(entries, *, skip_borrowed=True):
    with patch("agent.credential_pool.load_pool", return_value=_FakePool(entries)):
        return _resolve_anthropic_pool_token(skip_borrowed=skip_borrowed)


class TestExpiredPoolTokenSkipped:
    def test_expired_entry_is_skipped(self):
        expired = _entry("hermes_pkce", expires_at_ms=_NOW_MS - 1_000)
        assert _resolve_with([expired]) is None

    def test_valid_entry_after_expired_one_wins(self):
        expired = _entry("hermes_pkce", expires_at_ms=_NOW_MS - 1_000)
        valid = _entry("dashboard_pkce", expires_at_ms=_NOW_MS + 3_600_000,
                       token="sk-ant-oat01-fresh")
        assert _resolve_with([expired, valid]) == "sk-ant-oat01-fresh"

    def test_entry_expiring_within_skew_is_skipped(self):
        soon = _entry("hermes_pkce", expires_at_ms=_NOW_MS + 60_000)
        assert _resolve_with([soon]) is None

    def test_unknown_expiry_stays_eligible(self):
        """Managed keys carry no expires_at_ms; hiding them would be fail-closed."""
        managed = _entry("hermes_pkce", expires_at_ms=None)
        assert _resolve_with([managed]) == "sk-ant-oat01-x"
        zero = _entry("hermes_pkce", expires_at_ms=0)
        assert _resolve_with([zero]) == "sk-ant-oat01-x"

    def test_valid_entry_still_returned(self):
        valid = _entry("hermes_pkce", expires_at_ms=_NOW_MS + 3_600_000)
        assert _resolve_with([valid]) == "sk-ant-oat01-x"

    def test_borrowed_claude_code_still_skipped_when_requested(self):
        borrowed = _entry("claude_code", expires_at_ms=_NOW_MS + 3_600_000)
        assert _resolve_with([borrowed], skip_borrowed=True) is None
        assert _resolve_with([borrowed], skip_borrowed=False) == "sk-ant-oat01-x"
