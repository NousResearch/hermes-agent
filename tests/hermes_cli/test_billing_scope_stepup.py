"""Tests for the Phase 2b billing:manage scope step-up (auth.py)."""

from __future__ import annotations

import pytest

import hermes_cli.auth as auth
from hermes_cli.auth import (
    NOUS_BILLING_MANAGE_SCOPE,
    nous_token_has_billing_scope,
    step_up_nous_billing_scope,
)


# ---------------------------------------------------------------------------
# nous_token_has_billing_scope
# ---------------------------------------------------------------------------


def test_has_scope_true_when_present(monkeypatch):
    monkeypatch.setattr(
        auth,
        "get_provider_auth_state",
        lambda p: {"scope": "inference:invoke tool:invoke billing:manage"},
    )
    assert nous_token_has_billing_scope() is True


def test_has_scope_false_when_absent(monkeypatch):
    monkeypatch.setattr(
        auth, "get_provider_auth_state", lambda p: {"scope": "inference:invoke tool:invoke"}
    )
    assert nous_token_has_billing_scope() is False


def test_has_scope_false_when_no_state(monkeypatch):
    monkeypatch.setattr(auth, "get_provider_auth_state", lambda p: None)
    assert nous_token_has_billing_scope() is False


def test_has_scope_no_substring_false_positive(monkeypatch):
    # "billing:manage-lite" must NOT match billing:manage (split-based, not substring).
    monkeypatch.setattr(
        auth, "get_provider_auth_state", lambda p: {"scope": "billing:manage-lite"}
    )
    assert nous_token_has_billing_scope() is False


# ---------------------------------------------------------------------------
# step_up_nous_billing_scope
# ---------------------------------------------------------------------------


@pytest.fixture
def _stub_persist(monkeypatch):
    """Neutralize the persistence side-effects so step-up tests are pure."""
    monkeypatch.setattr(auth, "_auth_store_lock", lambda: _NullCtx())
    monkeypatch.setattr(auth, "_load_auth_store", lambda: {})
    monkeypatch.setattr(auth, "_save_provider_state", lambda *a, **kw: None)
    monkeypatch.setattr(auth, "_save_auth_store", lambda *a, **kw: "auth.json")
    monkeypatch.setattr(auth, "_write_shared_nous_state", lambda *a, **kw: None)
    monkeypatch.setattr(auth, "_sync_nous_pool_from_auth_store", lambda: None)


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_step_up_requests_billing_scope_and_reuses_prior_urls(monkeypatch, _stub_persist):
    monkeypatch.setattr(
        auth,
        "get_provider_auth_state",
        lambda p: {
            "scope": "inference:invoke tool:invoke",
            "portal_base_url": "https://preview.example.com",
            "inference_base_url": "https://inf.example.com",
            "client_id": "hermes-cli",
        },
    )
    captured = {}

    def _fake_login(**kw):
        captured.update(kw)
        # Simulate the admin ticking the box → token comes back WITH the scope.
        return {"scope": "inference:invoke tool:invoke billing:manage", "access_token": "t"}

    monkeypatch.setattr(auth, "_nous_device_code_login", _fake_login)

    granted = step_up_nous_billing_scope()
    assert granted is True
    # Requested scope must include billing:manage, preserving prior scopes.
    assert NOUS_BILLING_MANAGE_SCOPE in captured["scope"].split()
    assert "inference:invoke" in captured["scope"].split()
    # Reuses the prior credential's deployment URLs (so a preview stays a preview).
    assert captured["portal_base_url"] == "https://preview.example.com"
    assert captured["client_id"] == "hermes-cli"


def test_step_up_returns_false_when_downscoped(monkeypatch, _stub_persist):
    # Non-admin / unticked → the server silently downscopes; token comes back WITHOUT scope.
    monkeypatch.setattr(auth, "get_provider_auth_state", lambda p: {"scope": "inference:invoke"})
    monkeypatch.setattr(
        auth,
        "_nous_device_code_login",
        lambda **kw: {"scope": "inference:invoke", "access_token": "t"},
    )
    assert step_up_nous_billing_scope() is False


def test_step_up_falls_back_to_standard_scope_when_no_prior(monkeypatch, _stub_persist):
    monkeypatch.setattr(auth, "get_provider_auth_state", lambda p: {})
    captured = {}

    def _fake_login(**kw):
        captured.update(kw)
        return {"scope": "inference:invoke tool:invoke billing:manage"}

    monkeypatch.setattr(auth, "_nous_device_code_login", _fake_login)
    step_up_nous_billing_scope()
    requested = captured["scope"].split()
    assert "inference:invoke" in requested
    assert "tool:invoke" in requested
    assert NOUS_BILLING_MANAGE_SCOPE in requested


# ---------------------------------------------------------------------------
# on_verification callback plumbing (TUI surfaces the device-flow URL via this)
# ---------------------------------------------------------------------------


def test_step_up_forwards_on_verification_callback(monkeypatch, _stub_persist):
    monkeypatch.setattr(auth, "get_provider_auth_state", lambda p: {})
    captured = {}

    def _fake_login(**kw):
        captured.update(kw)
        return {"scope": "inference:invoke tool:invoke billing:manage"}

    monkeypatch.setattr(auth, "_nous_device_code_login", _fake_login)

    def _cb(url, code):
        pass

    step_up_nous_billing_scope(on_verification=_cb)
    # The callback must be threaded straight through to the device-code login.
    assert captured["on_verification"] is _cb


def test_device_login_fires_on_verification_before_polling(monkeypatch):
    """on_verification(url, code) must fire BEFORE _poll_for_token (so the TUI
    can render the link while the flow blocks waiting for approval)."""
    order: list[str] = []

    monkeypatch.setattr(
        auth,
        "_request_device_code",
        lambda **kw: {
            "verification_uri_complete": "https://portal.example/device?code=ABCD",
            "user_code": "ABCD-1234",
            "device_code": "dev",
            "expires_in": 600,
            "interval": 5,
        },
    )

    def _fake_poll(**kw):
        order.append("poll")
        return {"access_token": "t", "scope": "inference:invoke", "expires_in": 3600}

    monkeypatch.setattr(auth, "_poll_for_token", _fake_poll)

    seen = {}

    def _cb(url, code):
        order.append("verify")
        seen["url"] = url
        seen["code"] = code

    # We only assert the callback fires before polling. Post-poll token
    # validation (JWT usability checks) is out of scope and may raise on the
    # synthetic token — swallow it; the ordering assertion is what matters.
    try:
        auth._nous_device_code_login(open_browser=False, on_verification=_cb)
    except Exception:
        pass

    assert order[:2] == ["verify", "poll"], "callback must fire before polling"
    assert seen["url"] == "https://portal.example/device?code=ABCD"
    assert seen["code"] == "ABCD-1234"
