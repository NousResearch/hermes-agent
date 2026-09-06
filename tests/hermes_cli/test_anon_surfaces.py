"""Nous free tier on the read-only display surfaces and the keepalive.

Contract (R-USR-1): wherever a free-tier identity renders (``hermes auth status nous``,
``hermes auth list``, ``hermes status``, ``hermes portal info``) the user sees the free-tier label
plus the upgrade hint, and never the internal identity vocabulary. A real account keeps its normal
rendering. The keepalive has nothing to keep alive for the free tier and must not start a thread.
"""

from __future__ import annotations

import base64
import json
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import anon_auth, auth_commands, nous_account, nous_auth_keepalive, portal_cli, status_auth
from hermes_cli.auth import _load_auth_store  # noqa: F401  (store import name kept for parity with core tests)
from hermes_constants import get_hermes_home

WELCOME = "https://welcome-api.nousresearch.com/v1"
# Words that must never appear on a user-facing free-tier surface.
_FORBIDDEN = re.compile(r"guest|anonymous|user id|org id|nas_user|nas_organisation", re.IGNORECASE)


def _jwt(**claims) -> str:
    def seg(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()
    payload = {"sub": "nas_user:abc", "client_id": "nas-anonymous", "account_tier": "anonymous",
               "scope": "inference:invoke tool:invoke", "exp": int(time.time()) + 10 ** 8, **claims}
    return f"{seg({'alg': 'RS256'})}.{seg(payload)}.sig"


def _write_auth(nous_state: dict) -> None:
    home = Path(get_hermes_home())
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(json.dumps({"active_provider": "nous", "providers": {"nous": nous_state}}))


def _guest_state() -> dict:
    return {
        "auth_method": "anonymous", "account_tier": "anonymous", "anon_token": "anon_t",
        "access_token": _jwt(), "expires_at": "2030-01-01T00:00:00+00:00",
        "inference_base_url": WELCOME, "user_id": "nas_user:abc", "org_id": "nas_organisation:def",
    }


def _account_state() -> dict:
    return {
        "auth_method": "oauth_device_code", "client_id": "hermes-cli",
        "access_token": _jwt(sub="nas_user:real", client_id="hermes-cli", account_tier="standard", paid_access=True),
        "refresh_token": "rt_live", "expires_at": "2030-01-01T00:00:00+00:00",
        "portal_base_url": "https://portal.nousresearch.com", "inference_base_url": "https://inference-api.nousresearch.com/v1",
    }


@pytest.fixture
def isolated_store(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared-store"))
    monkeypatch.delenv("HERMES_FORCE_GUEST", raising=False)
    for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NOUS_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    # No network: the account lookup is derived from the JWT the store already holds.
    monkeypatch.setattr(nous_account, "_fetch_nous_account_info",
                        lambda *a, **k: pytest.fail("portal fetch must not happen on a read-only surface"))
    nous_account.reset_nous_portal_account_info_cache()
    import hermes_cli.auth as auth_mod
    auth_mod.invalidate_nous_auth_status_cache()
    yield
    nous_account.reset_nous_portal_account_info_cache()
    auth_mod.invalidate_nous_auth_status_cache()


def _render_all(capsys) -> dict[str, str]:
    out: dict[str, str] = {}
    auth_commands.auth_status_command(SimpleNamespace(provider="nous"))
    out["auth status"] = capsys.readouterr().out
    auth_commands.auth_list_command(SimpleNamespace(provider="nous"))
    out["auth list"] = capsys.readouterr().out
    ctx = SimpleNamespace(config={}, nous_logged_in=False, nous_inference_present=False, nous_account_info=None)
    status_auth._render_auth_providers(ctx)
    out["hermes status"] = capsys.readouterr().out
    portal_cli._cmd_status(SimpleNamespace())
    out["portal info"] = capsys.readouterr().out
    return out


def test_free_tier_renders_free_tier_copy_on_every_surface(isolated_store, capsys):
    _write_auth(_guest_state())
    rendered = _render_all(capsys)
    for surface, text in rendered.items():
        assert "free tier" in text.lower(), f"{surface} did not name the free tier:\n{text}"
        assert anon_auth.FREE_TIER_LABEL in text and anon_auth.GUEST_MODEL in text, surface
        assert anon_auth.UPGRADE_HINT in text, f"{surface} lacks the upgrade hint:\n{text}"
        leaked = _FORBIDDEN.search(text)
        assert leaked is None, f"{surface} leaked {leaked.group(0)!r}:\n{text}"
    # Billing / entitlement copy for the free tier points at the upgrade path, never at billing.
    info = nous_account.get_nous_portal_account_info()
    assert info.is_anonymous_tier
    message = nous_account.format_nous_portal_entitlement_message(info, capability="managed web tools")
    assert message == nous_account.FREE_TIER_NEEDS_ACCOUNT
    assert "billing" not in message.lower() and _FORBIDDEN.search(message) is None


def test_real_account_keeps_account_rendering(isolated_store, capsys):
    _write_auth(_account_state())
    rendered = _render_all(capsys)
    for surface, text in rendered.items():
        assert "free tier" not in text.lower(), f"{surface} mislabelled a real account:\n{text}"
        assert anon_auth.UPGRADE_HINT not in text, surface
    assert "logged in" in rendered["auth status"]
    assert "credentials" in rendered["auth list"]
    info = nous_account.get_nous_portal_account_info()
    assert not info.is_anonymous_tier
    assert nous_account.format_nous_portal_entitlement_message(info) is None  # paid_access claim entitles


def test_keepalive_does_not_start_for_free_tier(isolated_store, monkeypatch):
    started: list = []

    class _Thread:
        def __init__(self, *args, **kwargs):
            self._name = kwargs.get("name")
        def start(self):
            started.append(self._name)
        def is_alive(self):
            return True
        def join(self, timeout=None):
            pass
    monkeypatch.setattr(nous_auth_keepalive.threading, "Thread", _Thread)
    monkeypatch.setattr(nous_auth_keepalive, "_keepalive_thread", None)

    _write_auth(_guest_state())
    assert nous_auth_keepalive.start_nous_auth_keepalive(interval_seconds=900) is None
    assert started == []

    _write_auth(_account_state())
    thread = nous_auth_keepalive.start_nous_auth_keepalive(interval_seconds=900)
    assert thread is not None and started == ["nous-auth-keepalive"]
    monkeypatch.setattr(nous_auth_keepalive, "_keepalive_thread", None)
