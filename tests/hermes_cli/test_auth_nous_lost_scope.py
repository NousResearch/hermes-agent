"""Lost-scope nous override resolution warns once per override name-set per process.

Regression context: a multiplex-gateway deployment without the dev/staging
override env vars got one WARNING per unbound call (~600/day) because the same
callers (cron tickers, adapters, aux calls) re-raise ``UnscopedSecretError`` on
every request. The condition is process-static, so the first lost-scope
resolution for a name-set keeps the WARNING and repeats drop to DEBUG.
"""

import logging

import pytest

import hermes_cli.auth_nous as auth_nous
from agent.secret_scope import set_multiplex_active


@pytest.fixture(autouse=True)
def _fresh_warn_state():
    """Each test sees an empty warn-once set (module state persists per process)."""
    saved = set(auth_nous._LOST_SCOPE_WARNED)
    auth_nous._LOST_SCOPE_WARNED.clear()
    yield
    auth_nous._LOST_SCOPE_WARNED.clear()
    auth_nous._LOST_SCOPE_WARNED.update(saved)


def test_lost_scope_override_warns_once_per_name_set(caplog, monkeypatch):
    monkeypatch.delenv("NOUS_INFERENCE_BASE_URL", raising=False)
    monkeypatch.delenv("HERMES_PORTAL_BASE_URL", raising=False)
    monkeypatch.delenv("NOUS_PORTAL_BASE_URL", raising=False)
    set_multiplex_active(True)
    try:
        with caplog.at_level(logging.DEBUG, logger="hermes_cli.auth"):
            auth_nous._scoped_operator_override("NOUS_INFERENCE_BASE_URL")
            auth_nous._scoped_operator_override("NOUS_INFERENCE_BASE_URL")
            auth_nous._scoped_operator_override("HERMES_PORTAL_BASE_URL", "NOUS_PORTAL_BASE_URL")
        warns = [r for r in caplog.records
                 if r.levelno == logging.WARNING and "unreadable" in r.getMessage()]
        debugs = [r for r in caplog.records
                  if r.levelno == logging.DEBUG and "unreadable" in r.getMessage()]
        assert len(warns) == 2, f"expected one WARNING per name-set, got {warns}"
        assert len(debugs) == 1, f"expected the repeat to log at DEBUG, got {debugs}"
    finally:
        set_multiplex_active(False)


def test_lost_scope_warn_once_set_is_per_name_set(caplog, monkeypatch):
    """A different name-set gets its own first WARNING (not silenced by the other)."""
    monkeypatch.delenv("NOUS_INFERENCE_BASE_URL", raising=False)
    set_multiplex_active(True)
    try:
        with caplog.at_level(logging.DEBUG, logger="hermes_cli.auth"):
            auth_nous._scoped_operator_override("HERMES_PORTAL_BASE_URL")
            auth_nous._scoped_operator_override("HERMES_PORTAL_BASE_URL", "NOUS_PORTAL_BASE_URL")
        warns = [r for r in caplog.records
                 if r.levelno == logging.WARNING and "unreadable" in r.getMessage()]
        assert len(warns) == 2, f"each distinct name-set warns once, got {warns}"
    finally:
        set_multiplex_active(False)
