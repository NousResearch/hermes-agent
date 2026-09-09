"""Trust-boundary env toggles must resolve per-profile, never cross-profile.

Two security-sensitive toggles were read with a plain os.getenv, so under
gateway multiplexing (one process, one shared os.environ) another profile's
value could flip them: HERMES_ACCEPT_HOOKS (auto-approve shell hooks) and
HERMES_ALLOW_PRIVATE_URLS (disable SSRF/private-IP blocking). Both now resolve
through agent.secret_scope.secret_or, which returns the bound profile's value or
a restrictive default under multiplex, never os.environ.
"""
from __future__ import annotations

import pytest

import agent.secret_scope as ss
from agent.secret_scope import secret_or
from agent.shell_hooks import _resolve_effective_accept
from tools.url_safety import _resolve_allow_private_urls


@pytest.fixture(autouse=True)
def _reset_scope():
    ss.set_secret_scope(None)
    ss.set_multiplex_active(False)
    try:
        yield
    finally:
        ss.set_secret_scope(None)
        ss.set_multiplex_active(False)


# --- secret_or helper ---

def test_secret_or_scope_hit_wins(monkeypatch):
    monkeypatch.setenv("HERMES_ACCEPT_HOOKS", "1")  # another profile's leaked value
    ss.set_multiplex_active(True)
    ss.set_secret_scope({"HERMES_ACCEPT_HOOKS": "0"})
    assert secret_or("HERMES_ACCEPT_HOOKS", "") == "0"


def test_secret_or_unscoped_multiplex_returns_default_not_raise(monkeypatch):
    monkeypatch.setenv("HERMES_ACCEPT_HOOKS", "1")
    ss.set_multiplex_active(True)
    ss.set_secret_scope(None)
    assert secret_or("HERMES_ACCEPT_HOOKS", "") == ""  # fail closed, no raise


def test_secret_or_scope_miss_under_multiplex_is_default(monkeypatch):
    monkeypatch.setenv("HERMES_ACCEPT_HOOKS", "1")
    ss.set_multiplex_active(True)
    ss.set_secret_scope({"OTHER": "x"})
    assert secret_or("HERMES_ACCEPT_HOOKS", "") == ""


# --- HERMES_ACCEPT_HOOKS: auto-approve trust boundary ---

def test_leaked_accept_hooks_does_not_auto_approve(monkeypatch):
    # os.environ carries another profile's opt-in; the bound profile did not opt in.
    monkeypatch.setenv("HERMES_ACCEPT_HOOKS", "1")
    ss.set_multiplex_active(True)
    ss.set_secret_scope({"SOME_OTHER": "x"})
    assert _resolve_effective_accept({}, accept_hooks_arg=False) is False


def test_bound_profile_accept_hooks_is_honored(monkeypatch):
    ss.set_multiplex_active(True)
    ss.set_secret_scope({"HERMES_ACCEPT_HOOKS": "true"})
    assert _resolve_effective_accept({}, accept_hooks_arg=False) is True


def test_explicit_arg_still_wins(monkeypatch):
    ss.set_multiplex_active(True)
    ss.set_secret_scope(None)
    assert _resolve_effective_accept({}, accept_hooks_arg=True) is True


# --- HERMES_ALLOW_PRIVATE_URLS: SSRF blocking toggle ---

def test_leaked_allow_private_urls_does_not_disable_blocking(monkeypatch):
    # Another profile allowed private URLs; the bound profile did not.
    monkeypatch.setenv("HERMES_ALLOW_PRIVATE_URLS", "true")
    ss.set_multiplex_active(True)
    ss.set_secret_scope({"SOME_OTHER": "x"})
    # No leaked "true" observed -> falls through to config default (blocking on).
    assert _resolve_allow_private_urls() is not True


def test_bound_profile_allow_private_urls_is_honored(monkeypatch):
    ss.set_multiplex_active(True)
    ss.set_secret_scope({"HERMES_ALLOW_PRIVATE_URLS": "true"})
    assert _resolve_allow_private_urls() is True
