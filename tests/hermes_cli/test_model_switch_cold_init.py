"""Cold /model switches resolve their selected route and roll back on failure (#154391)."""

import pytest

from hermes_cli import gateway_tools, runtime_provider
from hermes_cli.auth import AuthError
from hermes_cli.gateway_client import GatewayClientError
from hermes_cli.model_switch import ModelSwitchResult
from tests.hermes_cli.test_cli_gateway_model_route import RecordingConnection
from tests.hermes_cli.test_cli_gateway_model_scope import _chat_cli, _submits


def _cold_cli(monkeypatch):
    conn = RecordingConnection({"model": "old/model", "provider": "openai-codex"})
    cli_mod, cli, _printed = _chat_cli(monkeypatch, conn)
    cli.agent = None
    cli._gateway_session_open = True
    cli.provider = cli.requested_provider = "openai-codex"
    for name in ("_ensure_runtime_credentials", "_resolve_fallback_runtime", "_normalize_model_for_provider"):
        setattr(cli, name, getattr(cli_mod.HermesCLI, name).__get__(cli))
    cli._credential_pool = object()
    cli._provider_source = "launch"
    cli._credentials_rate_limited = False
    cli._credentials_terminal = False
    cli._active_agent_route_signature = ("launch",)
    monkeypatch.setattr(cli_mod, "_active_agent_ref", None)
    monkeypatch.setattr(gateway_tools, "_view", None)
    gateway_tools.bind_session(cli, conn)
    return cli_mod, cli, conn


def _quota(provider):
    return AuthError("Fixture quota exhausted (429)", provider=provider,
                     code="codex_rate_limited", relogin_required=False)


def _pick():
    return ModelSwitchResult(success=True, new_model="deepseek-flash", target_provider="deepseek",
                             api_key="fixture-selected-key", base_url="https://api.deepseek.com/v1",
                             api_mode="chat_completions")


def _runtime_snapshot(cli):
    fields = ("model", "provider", "requested_provider", "api_key", "base_url", "api_mode",
              "_explicit_api_key", "_explicit_base_url", "reasoning_config", "_reasoning_user_set",
              "acp_command", "acp_args", "_credential_pool", "_provider_source",
              "_credentials_rate_limited", "_credentials_terminal", "_active_agent_route_signature")
    return {name: getattr(cli, name) for name in fields}


def test_cold_switch_leaves_an_exhausted_route_and_submits_the_selected_model(monkeypatch):
    """Resolve the target before initialization, then commit it on the session used by chat."""
    cli_mod, cli, conn = _cold_cli(monkeypatch)
    resolve = runtime_provider.resolve_runtime_provider
    seen = []

    def old_route_exhausted(**kwargs):
        seen.append(kwargs["requested"])
        if kwargs["requested"] == "openai-codex":
            raise _quota("openai-codex")
        return resolve(**kwargs)

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", old_route_exhausted)
    assert cli_mod.HermesCLI._stage_and_swap_model(cli, _pick(), cli.model) is True
    assert seen == ["deepseek"]
    assert (cli.model, cli.provider) == (cli.agent.model, cli.agent.provider) == (
        "deepseek-flash", "deepseek")
    assert conn.state("s").info["model"] == cli.model
    assert cli.chat("the first turn") == "ok"
    [submitted] = _submits(conn)
    assert "model_override" not in submitted


@pytest.mark.parametrize("failure", ["credentials", "session", "after_bind", "declined", "mutation", "normalized"])
def test_failed_cold_switch_restores_runtime_and_pending_reasoning(monkeypatch, failure):
    """A failed pick neither selects a fallback nor consumes a pending /reasoning choice."""
    cli_mod, cli, conn = _cold_cli(monkeypatch)
    cli_mod.HermesCLI._handle_reasoning_command(cli, "/reasoning high")
    before = _runtime_snapshot(cli)
    cli._fallback_model = [{"provider": "openrouter", "model": "fallback-model",
                            "api_key": "fixture-fallback-key"}]
    resolve = runtime_provider.resolve_runtime_provider
    seen = []

    def selected_runtime(**kwargs):
        seen.append(kwargs["requested"])
        if kwargs["requested"] == "openai-codex" or failure == "credentials" and kwargs["requested"] == "deepseek":
            raise _quota(kwargs["requested"])
        return resolve(**kwargs)

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", selected_runtime)
    bind, call = gateway_tools.bind, conn.call

    def normalize_to_another_model(_provider):
        cli.model = "different-model"
        return True

    def unavailable_session():
        raise GatewayClientError("fixture_session_unavailable")

    def fail_after_bind(agent):
        view = bind(agent)
        if agent is cli.agent:
            raise GatewayClientError("fixture_after_bind")
        return view

    def refuse_model(method, **kwargs):
        if method == "session.mutate" and kwargs.get("operation") == "model":
            if failure == "mutation":
                raise GatewayClientError("fixture_model_refused")
            return {"status": "confirmation_required", "confirm": "fixture-confirm",
                    "confirm_message": "Confirm this model switch."}
        return call(method, **kwargs)

    hooks = {
        "normalized": (cli, "_normalize_model_for_provider", normalize_to_another_model),
        "session": (cli, "_ensure_gateway_session", unavailable_session),
        "after_bind": (gateway_tools, "bind", fail_after_bind),
        "declined": (conn, "call", refuse_model),
        "mutation": (conn, "call", refuse_model),
    }
    if hook := hooks.get(failure):
        monkeypatch.setattr(*hook)

    assert cli_mod.HermesCLI._stage_and_swap_model(cli, _pick(), cli.model) is False
    assert seen == ["deepseek"]
    assert _runtime_snapshot(cli) == before
    assert cli.agent is None
    assert cli_mod._active_agent_ref is None
    assert conn.state("s").info["model"] == before["model"]
    view = gateway_tools.process_view()
    assert view.connection is conn and view.session_id == cli.session_id
    assert view.agent is not cli.agent


def test_ordinary_initialization_keeps_credential_fallback(monkeypatch):
    _cli_mod, cli, _conn = _cold_cli(monkeypatch)
    cli._fallback_model = [{"provider": "deepseek", "model": "fallback-model",
                            "api_key": "fixture-fallback-key"}]
    resolve = runtime_provider.resolve_runtime_provider
    seen = []

    def old_route_exhausted(**kwargs):
        seen.append(kwargs["requested"])
        if kwargs["requested"] == "openai-codex":
            raise _quota("openai-codex")
        return resolve(**kwargs)

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", old_route_exhausted)
    assert cli._init_agent() is True
    assert seen == ["openai-codex", "deepseek"]
    assert (cli.model, cli.requested_provider) == ("fallback-model", "deepseek")
