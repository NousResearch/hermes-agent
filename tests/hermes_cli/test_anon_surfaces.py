"""Nous free tier on the read-only display surfaces and the keepalive.

Contract (R-USR-1): wherever a free-tier identity renders (``hermes auth status nous``,
``hermes auth list``, ``hermes status``, ``hermes portal info``) the user sees the free-tier label
plus the upgrade hint, and never the internal identity vocabulary. A real account keeps its normal
rendering. The keepalive has nothing to keep alive for the free tier and must not start a thread.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, call

from hermes_cli import (
    anon_auth,
    auth_commands,
    model_setup_flows,
    nous_account,
    nous_auth_keepalive,
    portal_cli,
    status_auth,
)
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
    monkeypatch.setenv("HERMES_GUEST_ONBOARDING", "1")
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


def test_every_in_chat_free_tier_string_names_the_slash_command(monkeypatch):
    from gateway.run_notifications import GatewayNotificationsMixin

    monkeypatch.setattr("hermes_cli.auth.resolve_provider", lambda _requested: "nous")
    monkeypatch.setattr(anon_auth, "guest_carries_inference", lambda: True)
    startup = GatewayNotificationsMixin._free_tier_startup_line(object())
    command_copy = (
        anon_auth.FREE_TIER_AVAILABLE_NOTICE,
        anon_auth.FREE_TIER_STATUS_LINE,
        anon_auth.UPGRADE_UNAVAILABLE_CHAT,
        anon_auth.FREE_TIER_RATE_LIMIT_CHAT,
        nous_account.FREE_TIER_NEEDS_ACCOUNT_CHAT,
        startup,
    )
    refusal_copy = (
        anon_auth.LOGIN_DM_ONLY,
        anon_auth.LOGIN_BUSY_ELSEWHERE,
        anon_auth.LOGIN_NOT_ALLOWED,
    )
    assert all("/login" in text for text in command_copy)
    for text in (*command_copy, *refusal_copy):
        # The ruled refusal uses Hermes as the grammatical subject; only that exact product-name
        # phrase is exempt from the broad top-level-command gate.
        assert "hermes " not in text.replace("this Hermes can", "this product can").lower()


def test_no_chat_copy_of_any_sign_in_state_leaks_a_terminal_verb_or_a_forbidden_word():
    placeholders = {
        "link": "https://example.test/sign-in",
        "code": "code-1",
        "expires_in": 900,
        "interval": 1,
        "email": "person@example.test",
        "model": "model-1",
        "model_changed": True,
        "reason": "unknown",
        "detail": "private detail",
    }
    forbidden = re.compile(r"claim|nous portal|anonymous|guest", re.IGNORECASE)
    terminal_or_url = re.compile(r"hermes |https?://", re.IGNORECASE)

    for state_type in anon_auth.SignInState.__subclasses__():
        kwargs = {field.name: placeholders[field.name] for field in dataclasses.fields(state_type)}
        copy = state_type(**kwargs).copy
        assert _FORBIDDEN.search(copy) is None, state_type.__name__
        assert forbidden.search(copy) is None, state_type.__name__
        assert terminal_or_url.search(copy) is None, state_type.__name__


def test_terminal_only_strings_keep_the_terminal_verb(monkeypatch, capsys):
    assert "hermes " in nous_account.FREE_TIER_NEEDS_ACCOUNT
    assert "hermes " in anon_auth.UPGRADE_UNAVAILABLE
    assert "hermes " in anon_auth.FREE_TIER_NOT_SIGNED_IN

    monkeypatch.setattr("hermes_cli.auth.get_provider_auth_state", lambda _provider: {"access_token": "token"})
    monkeypatch.setattr("hermes_cli.model_switch_providers._free_tier_nous_row", lambda _provider: None)
    model_setup_flows._model_flow_nous({})
    assert "hermes " in capsys.readouterr().out


def test_the_paid_tool_notice_switches_wording_inside_a_chat():
    info = nous_account.NousPortalAccountInfo(
        logged_in=True, source="token", fresh=True, account_tier="anonymous"
    )
    assert nous_account.format_nous_portal_entitlement_message(
        info, in_chat=True
    ) == nous_account.FREE_TIER_NEEDS_ACCOUNT_CHAT
    assert nous_account.format_nous_portal_entitlement_message(
        info, in_chat=False
    ) == nous_account.FREE_TIER_NEEDS_ACCOUNT


@pytest.mark.asyncio
async def test_the_wait_line_is_only_composed_on_the_state(monkeypatch):
    from gateway.slash_commands_login import GatewayLoginCommandsMixin

    state = anon_auth.Code(
        link="https://example.test/sign-in", code="code-1", expires_in=900, interval=1
    )
    assert state.copy == anon_auth.UPGRADE_DO_NOT_SHARE
    assert state.copy_with_wait == (
        f"{anon_auth.UPGRADE_DO_NOT_SHARE} {anon_auth.format_wait_line(state.expires_in)}"
    )

    monkeypatch.setattr(
        "hermes_cli.auth_device_flow._print_device_code_instructions", lambda *_args, **_kwargs: None
    )
    cli_copy = []
    anon_auth.render_sign_in_cli_code(state, printer=cli_copy.append)
    runner = SimpleNamespace(_push_login=AsyncMock())
    attempt = object()
    await GatewayLoginCommandsMixin._render_login_state(runner, attempt, state)

    assert cli_copy == [f"  {state.copy_with_wait}"]
    assert runner._push_login.await_args_list == [
        call(attempt, state.link),
        call(attempt, state.code),
        call(attempt, state.copy_with_wait),
    ]

    # The wait line is composed once, on the state. A renderer that called
    # format_wait_line itself would emit the same text and pass the assertions
    # above, so the renderers are checked by source instead.
    repo = Path(anon_auth.__file__).resolve().parents[1]
    for rel in ("gateway/slash_commands_login.py", "hermes_cli/cli_commands_mixin.py"):
        assert "format_wait_line" not in (repo / rel).read_text(encoding="utf-8"), rel


def test_cli_chat_status_names_the_free_tier(isolated_store):
    from hermes_cli.cli_session_mixin import CLISessionMixin

    _write_auth(_guest_state())
    rendered = []
    cli = SimpleNamespace(
        _session_db=None,
        session_id="cli-free-tier-status",
        session_start=datetime.now(),
        agent=SimpleNamespace(session_total_tokens=0, reasoning_config=None),
        provider="nous",
        model=anon_auth.GUEST_MODEL,
        _agent_running=False,
        reasoning_config=None,
        show_reasoning=None,
        session_key="cli:free-tier-status",
        _get_status_bar_snapshot=lambda: {},
        _console_print=lambda text, **_kwargs: rendered.append(text),
    )
    CLISessionMixin._show_session_status(cli)

    assert anon_auth.FREE_TIER_STATUS_LINE in rendered[0]
