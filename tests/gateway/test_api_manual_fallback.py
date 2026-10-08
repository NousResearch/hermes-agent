"""The API constructor has no interactive fallback authority, including locked runtimes."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


@pytest.mark.parametrize("locked", [False, True])
@pytest.mark.parametrize("chain", [[], [{"provider": "anthropic", "model": "fallback"}]])
def test_api_agent_propagates_manual_policy_without_rehydrating_explicit_empty_chain(monkeypatch, locked, chain):
    factory = Mock(return_value=SimpleNamespace())
    monkeypatch.setattr("run_agent.AIAgent", factory)
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", lambda: {"provider": "openai", "api_key": "test"})
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "primary")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {
        "fallback_providers": chain, "fallback": {"auto_activate": False}})
    monkeypatch.setattr("gateway.run.GatewayRunner._load_reasoning_config", lambda *_: None)
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
    adapter._create_agent(session_id="sid", confirmed_runtime_lock=locked)
    kwargs = factory.call_args.kwargs
    assert kwargs["fallback_model"] == (None if locked else chain)
    assert kwargs["fallback_auto_activate"] is False
    assert kwargs["fallback_selection_interactive"] is False
    assert kwargs.get("clarify_callback") is None


def test_api_entrypoint_never_resolves_a_manual_bootstrap_fallback(tmp_path, monkeypatch):
    from hermes_cli.auth import AuthError
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "model:\n  provider: openai\n  default: primary\n"
        "fallback:\n  auto_activate: false\n"
        "fallback_providers:\n  - provider: anthropic\n    model: alternate\n")
    resolver = Mock(side_effect=AuthError("primary unavailable"))
    factory = Mock()
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolver)
    monkeypatch.setattr("run_agent.AIAgent", factory)
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    with pytest.raises(RuntimeError, match="primary unavailable"):
        adapter._create_agent(session_id="sid")
    resolver.assert_called_once()
    factory.assert_not_called()


def test_api_policy_change_during_bootstrap_cannot_adopt_fallback_as_primary(monkeypatch):
    from agent.manual_fallback import prepare_turn_runtime
    factory = Mock(return_value=SimpleNamespace())
    monkeypatch.setattr("run_agent.AIAgent", factory)
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", lambda: {
        "provider": "anthropic", "model": "alternate", "api_key": "test", "_fallback_notice": "bootstrap switch"})
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {"fallback": {"auto_activate": False}})
    monkeypatch.setattr("gateway.run.GatewayRunner._load_reasoning_config", lambda *_: None)
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
    agent = adapter._create_agent(session_id="sid")
    assert agent._fallback_bootstrap_active is True
    agent._fallback_auto_activate = factory.call_args.kwargs["fallback_auto_activate"]
    with pytest.raises(RuntimeError, match="primary was unavailable at startup"):
        prepare_turn_runtime(agent, Mock())
