"""Regression tests for #62549: dashboard auth identity must reach the agent.

Three-hop chain, verified at each boundary:

  Hop 1: ``_ws_auth_reason(ws)`` returns ``(reason, credential, info)`` —
         ``info`` carries the ``{user_id, provider}`` bound to the credential.
  Hop 2: ``_resolve_chat_argv(user_id=...)`` injects that identity into the
         PTY child's environment as ``HERMES_TUI_USER_ID``.
  Hop 3: ``tui_gateway.server._make_agent`` reads ``HERMES_TUI_USER_ID`` and
         passes it into ``AIAgent(user_id=...)`` so memory providers can
         scope per user.

The legacy ``?token=`` path and the server-spawned internal credential
path are also covered because both are exercised today and must keep
behaving correctly after the signature change.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli import web_server
from hermes_cli.dashboard_auth.ws_tickets import (
    INTERNAL_USER_ID,
    _reset_for_tests,
    internal_ws_credential,
    mint_ticket,
)


@pytest.fixture(autouse=True)
def _reset_tickets():
    _reset_for_tests()
    yield
    _reset_for_tests()


def _fake_ws(query: str = "") -> SimpleNamespace:
    """Minimal duck-typed WebSocket — only the attrs ``_ws_auth_reason`` reads."""
    from urllib.parse import parse_qs

    params = {k: v[0] for k, v in parse_qs(query, keep_blank_values=True).items()}

    def _get(key, default=""):
        return params.get(key, default)

    return SimpleNamespace(
        query_params=SimpleNamespace(get=_get),
        client=SimpleNamespace(host="127.0.0.1"),
        url=SimpleNamespace(path="/api/pty"),
    )


def _set_auth_required(monkeypatch, value: bool) -> None:
    monkeypatch.setattr(web_server.app.state, "auth_required", value, raising=False)


# -----------------------------------------------------------------------
# Hop 1: _ws_auth_reason returns (reason, credential, info)
# -----------------------------------------------------------------------


class TestWsAuthReasonReturnsIdentity:
    def test_returns_three_tuple_on_no_credential(self, monkeypatch):
        _set_auth_required(monkeypatch, True)
        reason, cred, info = web_server._ws_auth_reason(_fake_ws(""))
        assert reason == "no_credential"
        assert cred == "none"
        assert info is None

    def test_ticket_accepted_returns_user_info(self, monkeypatch):
        _set_auth_required(monkeypatch, True)
        ticket = mint_ticket(user_id="alice@example.com", provider="basic_auth")
        reason, cred, info = web_server._ws_auth_reason(_fake_ws(f"ticket={ticket}"))
        assert reason is None
        assert cred == "ticket"
        assert info is not None
        assert info["user_id"] == "alice@example.com"
        assert info["provider"] == "basic_auth"
        assert "minted_at" in info

    def test_internal_credential_returns_server_internal(self, monkeypatch):
        _set_auth_required(monkeypatch, True)
        cred = internal_ws_credential()
        reason, _cred, info = web_server._ws_auth_reason(_fake_ws(f"internal={cred}"))
        assert reason is None
        assert info["user_id"] == INTERNAL_USER_ID
        assert info["provider"] == INTERNAL_USER_ID

    def test_invalid_ticket_returns_none_info(self, monkeypatch):
        _set_auth_required(monkeypatch, True)
        reason, cred, info = web_server._ws_auth_reason(_fake_ws("ticket=nope"))
        assert reason == "ticket_invalid"
        assert cred == "ticket"
        assert info is None

    def test_loopback_token_path_has_no_identity(self, monkeypatch):
        _set_auth_required(monkeypatch, False)
        reason, cred, info = web_server._ws_auth_reason(_fake_ws("token=wrong"))
        assert reason == "token_mismatch"
        assert cred == "token"
        assert info is None

    def test_ws_auth_ok_signature_unchanged(self, monkeypatch):
        """_ws_auth_ok must keep its 2-tuple behavior so existing callers
        (test_pty_keepalive_ws.py, plugins/kanban/dashboard, web/src/lib/api.ts)
        are unaffected."""
        _set_auth_required(monkeypatch, True)
        assert web_server._ws_auth_ok(_fake_ws("")) is False
        ticket = mint_ticket(user_id="u1", provider="x")
        assert web_server._ws_auth_ok(_fake_ws(f"ticket={ticket}")) is True


# -----------------------------------------------------------------------
# Hop 2: _resolve_chat_argv injects HERMES_TUI_USER_ID
# -----------------------------------------------------------------------


class TestResolveChatArgvInjectsUserIdEnv:
    def test_user_id_sets_env_var(self, monkeypatch):
        # Stub the Node-bundle side effects so the test runs without a
        # built TUI bundle on disk. ``_make_tui_argv`` is lazy-imported from
        # ``hermes_cli.main`` inside ``_resolve_chat_argv``, so we patch
        # the source module.
        import hermes_cli.main

        monkeypatch.setattr(
            hermes_cli.main, "_make_tui_argv", lambda *_a, **_k: (["fake-tui"], "/tmp")
        )
        monkeypatch.setattr(web_server, "_build_gateway_ws_url", lambda: None)
        argv, cwd, env = web_server._resolve_chat_argv(user_id="alice@example.com")
        assert env["HERMES_TUI_USER_ID"] == "alice@example.com"

    def test_no_user_id_leaves_env_unset(self, monkeypatch):
        import hermes_cli.main

        monkeypatch.setattr(
            hermes_cli.main, "_make_tui_argv", lambda *_a, **_k: (["fake-tui"], "/tmp")
        )
        monkeypatch.setattr(web_server, "_build_gateway_ws_url", lambda: None)
        argv, cwd, env = web_server._resolve_chat_argv()
        assert "HERMES_TUI_USER_ID" not in env

    def test_existing_env_vars_preserved(self, monkeypatch):
        """HERMES_TUI_DASHBOARD and friends must still be set alongside the
        new user_id var — confirms the fix doesn't regress the rest of the
        argv resolution."""
        import hermes_cli.main

        monkeypatch.setattr(
            hermes_cli.main, "_make_tui_argv", lambda *_a, **_k: (["fake-tui"], "/tmp")
        )
        monkeypatch.setattr(web_server, "_build_gateway_ws_url", lambda: None)
        argv, cwd, env = web_server._resolve_chat_argv(user_id="alice@example.com")
        assert env["HERMES_TUI_DASHBOARD"] == "1"
        assert env["HERMES_TUI_USER_ID"] == "alice@example.com"


# -----------------------------------------------------------------------
# Hop 3: _make_agent reads HERMES_TUI_USER_ID and propagates to AIAgent
# -----------------------------------------------------------------------


class TestMakeAgentReadsDashboardUserId:
    """End-to-end via stubbed AIAgent: _make_agent must read
    HERMES_TUI_USER_ID and pass it as user_id to AIAgent(...). Verified
    without depending on agent_init's heavy init chain."""

    @pytest.fixture
    def stubbed_tgs(self, monkeypatch):
        from tui_gateway import server as tgs

        captured = {}

        class FakeAIAgent:
            def __init__(self, **kwargs):
                captured.clear()
                captured.update(kwargs)

        # ``_make_agent`` does ``from run_agent import AIAgent`` lazily inside
        # the function body, so patching ``run_agent.AIAgent`` directly would
        # force us to import run_agent (which pulls in the heavy terminal_tool
        # → managed_modal → requests chain). Instead we swap the resolved name
        # in ``tui_gateway.server``'s module namespace by patching the
        # ``run_agent`` import inside the function via a sentinel attribute
        # that the lazy import picks up. The cleanest cross-cutting fix is a
        # sys.modules stub: replace ``run_agent`` with a tiny module exposing
        # ``AIAgent`` before _make_agent runs.
        import sys
        import types

        stub = types.ModuleType("run_agent")
        stub.AIAgent = FakeAIAgent
        monkeypatch.setitem(sys.modules, "run_agent", stub)

        monkeypatch.setattr(tgs, "_load_cfg", lambda: {})
        monkeypatch.setattr(tgs, "_prompt_text", lambda x: "")
        monkeypatch.setattr(tgs, "_parse_tui_skills_env", lambda: [])
        monkeypatch.setattr(tgs, "_load_provider_routing", lambda: {})
        monkeypatch.setattr(tgs, "_load_reasoning_config", lambda: {})
        monkeypatch.setattr(tgs, "_load_service_tier", lambda: None)
        monkeypatch.setattr(tgs, "_load_enabled_toolsets", lambda: None)
        monkeypatch.setattr(tgs, "_cfg_max_turns", lambda *_a, **_k: 90)
        monkeypatch.setattr(
            tgs, "_resolve_startup_runtime", lambda: ("test-model", None)
        )
        monkeypatch.setattr(tgs, "_resolve_runtime_with_fallback", lambda _kw: {})
        monkeypatch.setattr(tgs, "_resolve_agent_platform", lambda x: x or "tui")
        monkeypatch.setattr(tgs, "_load_fallback_model", lambda: None)
        monkeypatch.setattr(tgs, "_agent_cbs", lambda sid: {})
        monkeypatch.setattr(tgs, "is_truthy_value", lambda x: False)
        # MCP discovery waits are wrapped in try/except inside _make_agent
        # so we don't need to stub them — any import or call failure is
        # silently absorbed (matches the production resilience contract).

        return tgs, captured

    def test_dashboard_user_id_propagates(self, stubbed_tgs, monkeypatch):
        tgs, captured = stubbed_tgs
        monkeypatch.setenv("HERMES_TUI_USER_ID", "carol@example.com")
        try:
            tgs._make_agent("sid-1", "key-1")
        finally:
            monkeypatch.delenv("HERMES_TUI_USER_ID", raising=False)
        assert captured.get("user_id") == "carol@example.com", (
            "user_id from HERMES_TUI_USER_ID env was not propagated to AIAgent"
        )

    def test_unset_env_yields_none_user_id(self, stubbed_tgs, monkeypatch):
        tgs, captured = stubbed_tgs
        monkeypatch.delenv("HERMES_TUI_USER_ID", raising=False)
        tgs._make_agent("sid-1", "key-1")
        assert captured.get("user_id") is None

    def test_empty_env_yields_none_user_id(self, stubbed_tgs, monkeypatch):
        """Empty string must be treated as unset — never silently override
        a real user_id with ""."""
        tgs, captured = stubbed_tgs
        monkeypatch.setenv("HERMES_TUI_USER_ID", "")
        try:
            tgs._make_agent("sid-1", "key-1")
        finally:
            monkeypatch.delenv("HERMES_TUI_USER_ID", raising=False)
        assert captured.get("user_id") is None
