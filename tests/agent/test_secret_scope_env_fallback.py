"""The EXEMPT (default/launch) secret scope keeps its os.environ fallback under multiplex.

Three get_secret modes when a key is missing from the .env overlay:
- multiplex + EXEMPT scope        -> falls through to os.environ (env-injected creds survive)
- multiplex + NON-exempt scope    -> returns default (fail closed, never a peer's env)
- multiplex + NO scope            -> raises UnscopedSecretError (unchanged)
"""
from __future__ import annotations

import pytest

from agent import secret_scope as ss


@pytest.fixture(autouse=True)
def _reset():
    was_active = ss.is_multiplex_active()
    yield
    ss.set_multiplex_active(was_active)


def test_exempt_scope_falls_through_to_environ_on_miss(monkeypatch):
    monkeypatch.setenv("MY_CRED", "from-environ")
    ss.set_multiplex_active(True)
    scope_tok = ss.set_secret_scope({"OTHER": "x"})  # MY_CRED absent from overlay
    fb_tok = ss.set_secret_scope_env_fallback(True)
    try:
        assert ss.get_secret("MY_CRED") == "from-environ"
    finally:
        ss.reset_secret_scope_env_fallback(fb_tok)
        ss.reset_secret_scope(scope_tok)


def test_non_exempt_scope_returns_default_on_miss(monkeypatch):
    monkeypatch.setenv("MY_CRED", "from-environ")
    ss.set_multiplex_active(True)
    scope_tok = ss.set_secret_scope({"OTHER": "x"})  # NO env_fallback flag -> fail closed
    try:
        assert ss.get_secret("MY_CRED", "the-default") == "the-default"
    finally:
        ss.reset_secret_scope(scope_tok)


def test_no_scope_still_raises(monkeypatch):
    monkeypatch.setenv("MY_CRED", "from-environ")
    ss.set_multiplex_active(True)
    ss.set_secret_scope(None)
    with pytest.raises(ss.UnscopedSecretError):
        ss.get_secret("MY_CRED")


def test_exempt_overlay_hit_still_wins_over_environ(monkeypatch):
    monkeypatch.setenv("MY_CRED", "from-environ")
    ss.set_multiplex_active(True)
    scope_tok = ss.set_secret_scope({"MY_CRED": "from-overlay"})
    fb_tok = ss.set_secret_scope_env_fallback(True)
    try:
        assert ss.get_secret("MY_CRED") == "from-overlay"
    finally:
        ss.reset_secret_scope_env_fallback(fb_tok)
        ss.reset_secret_scope(scope_tok)


def test_exempt_flag_ignored_when_multiplex_inactive(monkeypatch):
    # Legacy semantics byte-for-byte: no multiplex -> scoped miss reads os.environ regardless.
    monkeypatch.setenv("MY_CRED", "from-environ")
    ss.set_multiplex_active(False)
    scope_tok = ss.set_secret_scope({"OTHER": "x"})
    try:
        assert ss.get_secret("MY_CRED") == "from-environ"
    finally:
        ss.reset_secret_scope(scope_tok)
