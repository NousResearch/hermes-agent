"""Every plain device login that saves a Nous account over the free tier settles the config itself.

The guest handoff only moves apps; moving a config left on the free tier's route (the welcome model
on the welcome host) to the account is each sign-in's own completion step, run once the auth-store
locks are released. Driven against the fake portal ``hermes auth upgrade`` is tested with.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from hermes_cli import anon_auth, auth_nous, guest_handoff
from tests.hermes_cli.test_anon_upgrade import (
    EMAIL, FREE_PICK, INFERENCE, PORTAL, REFRESH_TOKEN, WELCOME, _jwt, _model_config, _write_model_config,
    free_account, portal)

__all__ = ["free_account", "portal"]


def _account() -> dict:
    return {"access_token": _jwt(sub="nas_user:9", client_id="hermes-cli", account_tier="free", email=EMAIL),
            "refresh_token": REFRESH_TOKEN, "portal_base_url": PORTAL, "inference_base_url": INFERENCE,
            "client_id": "hermes-cli", "expires_in": 900,
            "expires_at": (datetime.now(UTC) + timedelta(minutes=15)).isoformat()}


def _terminal_login(monkeypatch):
    monkeypatch.setattr(auth_nous, "_offer_shared_nous_import", lambda timeout: None)
    monkeypatch.setattr(auth_nous, "_pick_nous_model_after_login", lambda *a: None)
    from hermes_cli.auth import PROVIDER_REGISTRY
    auth_nous._login_nous(argparse.Namespace(no_browser=True, timeout=15.0), PROVIDER_REGISTRY["nous"])


def _auth_add(monkeypatch):
    from hermes_cli import auth_commands
    auth_commands._add_nous_oauth_credential(SimpleNamespace(label=None, timeout=None, no_browser=True), "nous")


def _step_up(monkeypatch):
    auth_nous.step_up_nous_billing_scope(open_browser=False)


def _tool_login(monkeypatch):
    from hermes_cli import nous_subscription
    monkeypatch.setattr(nous_subscription, "_confirm", lambda prompt: True)
    assert nous_subscription._run_nous_portal_login_only(capability="Web search") is True


@pytest.mark.parametrize("surface", [_terminal_login, _auth_add, _step_up, _tool_login],
                         ids=["hermes model / login", "auth add nous", "billing step-up", "tool setup login"])
def test_the_sign_in_settles_the_config_once_after_its_locks_are_released(
        portal, free_account, monkeypatch, surface):
    from hermes_cli import auth
    anon_auth.ensure_portal_identity(explicit=True)
    _write_model_config({"provider": "nous", "default": anon_auth.GUEST_MODEL, "base_url": WELCOME})
    monkeypatch.setattr(auth, "_nous_device_code_login", lambda **kw: _account())
    monkeypatch.setattr(auth_nous, "_nous_device_code_login", lambda **kw: _account())
    settles = []
    real_settle = anon_auth.settle_after_upgrade

    def _recording(state):
        settles.append(guest_handoff._holds_auth_locks())
        return real_settle(state)
    monkeypatch.setattr(anon_auth, "settle_after_upgrade", _recording)

    surface(monkeypatch)

    assert settles == [False]
    model_cfg = _model_config()
    assert model_cfg["default"] == FREE_PICK
    assert model_cfg["base_url"] == INFERENCE.rstrip("/")


@pytest.mark.parametrize("surface", [_terminal_login, _auth_add, _tool_login],
                         ids=["hermes model / login", "auth add nous", "tool setup login"])
def test_the_terminal_sign_in_prints_what_moved(portal, free_account, monkeypatch, capsys, surface):
    from hermes_cli import auth
    anon_auth.ensure_portal_identity(explicit=True)
    portal.claim_reply = {**portal.claim_reply, "connectors": {
        "kind": "moved", "moved_toolkits": ["gmail"], "dropped_toolkits": ["slack"]}}
    monkeypatch.setattr(auth, "_nous_device_code_login", lambda **kw: _account())
    monkeypatch.setattr(auth_nous, "_nous_device_code_login", lambda **kw: _account())

    surface(monkeypatch)

    out = capsys.readouterr().out
    assert f"Your apps moved to {EMAIL}: gmail." in out
    assert f"Reconnect in {EMAIL}: slack." in out
