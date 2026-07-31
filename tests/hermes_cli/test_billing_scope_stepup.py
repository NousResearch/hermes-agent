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


# ---------------------------------------------------------------------------
# on_verification callback plumbing (TUI surfaces the device-flow URL via this)
# ---------------------------------------------------------------------------




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
