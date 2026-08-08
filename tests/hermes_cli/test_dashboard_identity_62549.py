"""Dashboard ticket and capability authentication regressions for #62549."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

from hermes_cli import web_server
from hermes_cli.dashboard_auth.ws_tickets import (
    INTERNAL_USER_ID,
    TicketInvalid,
    _reset_for_tests,
    internal_ws_credential,
    mint_ticket,
)


@pytest.fixture(autouse=True)
def reset_tickets():
    _reset_for_tests()
    yield
    _reset_for_tests()


def fake_ws(query: str = "") -> SimpleNamespace:
    values = parse_qs(query, keep_blank_values=True)
    params = {key: items[0] for key, items in values.items()}
    return SimpleNamespace(
        query_params=SimpleNamespace(get=lambda key, default="": params.get(key, default)),
        client=SimpleNamespace(host="127.0.0.1"),
        url=SimpleNamespace(path="/api/ws"),
    )


def set_auth_required(monkeypatch, value: bool) -> None:
    monkeypatch.setattr(web_server.app.state, "auth_required", value, raising=False)


def test_ticket_identity_is_available_without_breaking_default_shape(monkeypatch):
    set_auth_required(monkeypatch, True)
    ticket = mint_ticket(user_id="alice", provider="oauth")

    reason, credential = web_server._ws_auth_reason(fake_ws(f"ticket={ticket}"))
    assert (reason, credential) == (None, "ticket")

    ticket = mint_ticket(user_id="alice", provider="oauth")
    reason, credential, info = web_server._ws_auth_reason(
        fake_ws(f"ticket={ticket}"), include_info=True
    )
    assert (reason, credential) == (None, "ticket")
    assert info["user_id"] == "alice"
    assert info["provider"] == "oauth"
    assert isinstance(info["minted_at"], int)


def test_rejected_and_loopback_credentials_have_no_identity(monkeypatch):
    set_auth_required(monkeypatch, True)
    assert web_server._ws_auth_reason(fake_ws("ticket=bad"), include_info=True) == (
        "ticket_invalid",
        "ticket",
        None,
    )

    set_auth_required(monkeypatch, False)
    assert web_server._ws_auth_reason(fake_ws("token=bad"), include_info=True) == (
        "token_mismatch",
        "token",
        None,
    )


def test_plain_internal_credential_is_server_owned_not_a_dashboard_user(monkeypatch):
    set_auth_required(monkeypatch, True)
    reason, credential, info = web_server._ws_auth_reason(
        fake_ws(f"internal={internal_ws_credential()}"), include_info=True
    )
    assert (reason, credential) == (None, "internal")
    assert info["user_id"] == INTERNAL_USER_ID


def test_internal_attach_capability_restores_ticket_identity(monkeypatch):
    from hermes_cli.dashboard_auth.ws_tickets import mint_principal_capability

    set_auth_required(monkeypatch, True)
    capability = mint_principal_capability(user_id="alice", provider="oauth")
    reason, credential, info = web_server._ws_auth_reason(
        fake_ws(f"internal={internal_ws_credential()}&principal={capability}"),
        include_info=True,
    )
    assert (reason, credential) == (None, "internal")
    assert info == {"user_id": "alice", "provider": "oauth"}


def test_internal_attach_ignores_caller_supplied_user_id(monkeypatch):
    set_auth_required(monkeypatch, True)
    reason, credential, info = web_server._ws_auth_reason(
        fake_ws(f"internal={internal_ws_credential()}&user_id=bob"), include_info=True
    )
    assert (reason, credential) == (None, "internal")
    assert info["user_id"] == INTERNAL_USER_ID


def test_principal_capability_is_opaque_and_forgery_fails():
    from hermes_cli.dashboard_auth.ws_tickets import (
        consume_principal_capability,
        mint_principal_capability,
    )

    capability = mint_principal_capability(user_id="alice", provider="oauth")
    assert "alice" not in capability
    assert consume_principal_capability(capability)["user_id"] == "alice"
    assert consume_principal_capability(capability)["user_id"] == "alice"

    with pytest.raises(TicketInvalid):
        consume_principal_capability(f"forged-{capability}")


def test_resolve_chat_argv_uses_capability_and_profile_environment(monkeypatch):
    import hermes_cli.main

    monkeypatch.setattr(
        hermes_cli.main,
        "_make_tui_argv",
        lambda *_args, **_kwargs: (["fake-tui"], "/tmp"),
    )
    monkeypatch.setattr(
        web_server,
        "_build_gateway_ws_url",
        lambda **kwargs: f"ws://gateway/?principal={kwargs['principal_capability']}",
    )

    _argv, _cwd, env = web_server._resolve_chat_argv(
        user_id="alice", provider="oauth", principal_capability="opaque-cap"
    )
    assert env["HERMES_TUI_USER_ID"] == "alice"
    assert env["HERMES_TUI_USER_PROVIDER"] == "oauth"
    assert "opaque-cap" in env["HERMES_TUI_GATEWAY_URL"]


def test_gateway_url_never_serializes_raw_user_id(monkeypatch):
    monkeypatch.setattr(web_server, "_resolve_client_ws_host", lambda: "127.0.0.1")
    monkeypatch.setattr(web_server.app.state, "bound_port", 9999, raising=False)
    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)
    url = web_server._build_gateway_ws_url(principal_capability="opaque-cap")
    assert url is not None
    query = parse_qs(urlparse(url).query)
    assert query["principal"] == ["opaque-cap"]
    assert "user_id" not in query


def test_gateway_attach_principal_overwrites_rpc_identity():
    from tui_gateway import ws

    request = {"id": 1, "method": "session.create", "params": {"pty_user_id": "bob"}}
    rewritten = ws._bind_connection_identity(request, {"user_id": "alice", "provider": "oauth"})
    assert rewritten["params"]["pty_user_id"] == "alice"
    assert rewritten["params"]["pty_provider"] == "oauth"


def test_profile_child_keeps_identity_without_in_process_gateway(monkeypatch, tmp_path):
    import hermes_cli.main

    monkeypatch.setattr(
        hermes_cli.main,
        "_make_tui_argv",
        lambda *_args, **_kwargs: (["fake-tui"], "/tmp"),
    )
    monkeypatch.setattr(web_server, "_resolve_profile_dir", lambda _name: tmp_path)
    monkeypatch.setattr(
        web_server,
        "_build_gateway_ws_url",
        lambda: pytest.fail("profile-scoped chat must spawn its own gateway"),
    )

    _argv, _cwd, env = web_server._resolve_chat_argv(
        profile="alice", user_id="alice", provider="oauth"
    )
    assert env["HERMES_HOME"] == str(tmp_path)
    assert env["HERMES_TUI_USER_ID"] == "alice"
    assert env["HERMES_TUI_USER_PROVIDER"] == "oauth"
    assert "HERMES_TUI_GATEWAY_URL" not in env


def test_connection_identity_is_not_injected_for_internal_child():
    from tui_gateway import ws

    request = {
        "id": 1,
        "method": "session.create",
        "params": {"pty_user_id": "bob", "pty_provider": "oauth"},
    }
    sanitized = ws._bind_connection_identity(request, {"user_id": "server-internal"})
    assert "pty_user_id" not in sanitized["params"]
    assert "pty_provider" not in sanitized["params"]


def test_make_agent_passes_dashboard_user_id(monkeypatch):
    from tui_gateway import server

    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    stub = types.ModuleType("run_agent")
    stub.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", stub)
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_prompt_text", lambda value: "")
    monkeypatch.setattr(server, "_parse_tui_skills_env", lambda: [])
    monkeypatch.setattr(server, "_load_provider_routing", lambda: {})
    monkeypatch.setattr(server, "_load_reasoning_config", lambda _model: {})
    monkeypatch.setattr(server, "_load_service_tier", lambda: None)
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_args: None)
    monkeypatch.setattr(server, "_cfg_max_turns", lambda *_args: 10)
    monkeypatch.setattr(server, "_resolve_startup_runtime", lambda: ("model", None))
    monkeypatch.setattr(
        server,
        "_resolve_runtime_with_fallback",
        lambda _kwargs: SimpleNamespace(runtime={}, used_fallback=False),
    )
    monkeypatch.setattr(server, "_resolve_agent_platform", lambda value: value or "tui")
    monkeypatch.setattr(server, "_load_fallback_model", lambda: None)
    monkeypatch.setattr(server, "_agent_cbs", lambda _sid: {})

    server._make_agent("sid", "key", pty_user_id="alice")
    assert captured["user_id"] == "alice"


def test_deferred_session_record_preserves_provider():
    from tui_gateway import server

    record = server._deferred_session_record(
        "sid",
        cols=80,
        cwd="/tmp",
        history=[],
        lease=None,
        pty_user_id="alice",
        pty_provider="oauth",
    )
    assert record["pty_user_id"] == "alice"
    assert record["pty_provider"] == "oauth"


def test_lazy_session_creation_persists_agent_identity(monkeypatch):
    import run_agent

    captured = {}

    class FakeDB:
        def create_session(self, **kwargs):
            captured.update(kwargs)

    agent = run_agent.AIAgent.__new__(run_agent.AIAgent)
    agent._persist_disabled = False
    agent._session_db_created = False
    agent._session_db = FakeDB()
    agent.platform = "tui"
    agent._session_init_model_config = {}
    agent.session_id = "sid"
    agent.model = "model"
    agent._cached_system_prompt = "system"
    agent._parent_session_id = None
    agent._user_id = "alice"
    monkeypatch.setattr(run_agent, "_session_source_for_agent", lambda _platform: "tui")
    monkeypatch.setattr(run_agent, "_launch_cwd_for_session", lambda _source: None)

    agent._ensure_db_session()
    assert captured["user_id"] == "alice"
