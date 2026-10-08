"""Manual fallback policy follows each gateway turn and its owning profile."""
from types import SimpleNamespace
import time
from unittest.mock import Mock

import pytest

from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.session import Platform, SessionSource
from gateway.turn_context import TurnContext
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


CHAIN = [{"provider": "openrouter", "model": "fallback/model"}]


def _config(home, automatic, model="fallback/model"):
    (home / "config.yaml").write_text(
        f"fallback:\n  auto_activate: {str(automatic).lower()}\n"
        f"fallback_providers:\n  - provider: openrouter\n    model: {model}\n")


def _runner():
    runner = object.__new__(GatewayRunner)
    runner._fallback_model = None
    runner._prefill_messages = runner._session_db = runner._service_tier = None
    return runner


def test_gateway_refresh_is_atomic_and_profile_isolated_after_torn_read(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(a))
    monkeypatch.setattr("gateway.run._hermes_home", a)
    _config(a, False)
    _config(b, True, "other/model")
    runner = _runner()
    assert runner._refresh_fallback_settings() == (CHAIN, False)
    token = set_hermes_home_override(str(b))
    try:
        assert runner._refresh_fallback_settings() == ([{"provider": "openrouter", "model": "other/model"}], True)
    finally:
        reset_hermes_home_override(token)
    (a / "config.yaml").write_text("fallback: [\n")
    assert runner._refresh_fallback_settings() == (CHAIN, False)
    (a / "config.yaml").unlink()
    assert runner._refresh_fallback_settings() == (None, True)


def test_new_profile_with_torn_config_never_inherits_another_profile(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    runner = _runner()
    _config(tmp_path, True)
    runner._refresh_fallback_settings()
    other = tmp_path / "other"; other.mkdir()
    (other / "config.yaml").write_text("fallback: [\n")
    token = set_hermes_home_override(str(other))
    try:
        assert runner._refresh_fallback_settings() == (None, False)
    finally:
        reset_hermes_home_override(token)


def test_auto_to_manual_edit_updates_chain_despite_live_fallback_cooldown():
    agent = SimpleNamespace(_fallback_auto_activate=True, _fallback_chain=CHAIN,
                            _fallback_model=CHAIN[0], _fallback_index=1,
                            _fallback_activated=True, _rate_limited_until=time.monotonic() + 600)
    replacement = [{"provider": "nous", "model": "replacement"}]
    GatewayRunner._apply_fallback_chain_to_agent(agent, replacement, auto_activate=False)
    assert agent._fallback_auto_activate is False
    assert agent._fallback_chain == replacement
    assert agent._fallback_model == replacement[0]


def test_fresh_gateway_agent_receives_policy_and_interactive_capability(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    _config(tmp_path, False)
    factory = Mock(return_value=SimpleNamespace())
    ctx = TurnContext(source=SessionSource(platform=Platform.SLACK, chat_id="c", user_id="u"),
                      message="hello", history=[], session_id="sid", session_key="key", user_config={}, AIAgent=factory)
    TurnRunner(_runner(), ctx)._build_fresh_agent(
        {"model": "primary", "runtime": {}}, "slack", "", 10, None, {}, False)
    assert factory.call_args.kwargs["fallback_model"] == CHAIN
    assert factory.call_args.kwargs["fallback_auto_activate"] is False
    assert factory.call_args.kwargs["fallback_selection_interactive"] is True


def test_manual_session_override_failure_does_not_substitute_default(monkeypatch):
    runner = _runner()
    runner._resolve_session_key_or_none = lambda *_: "key"
    runner._rehydrate_session_model_override = lambda *_: None
    runner._peek_session_state = lambda *_: SimpleNamespace(
        conversation=SimpleNamespace(model_override={"provider": "openai", "model": "chosen"}))
    default = Mock(return_value={"provider": "nous"})
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", default)
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs_for_provider",
                        Mock(side_effect=RuntimeError("chosen unavailable")))
    with pytest.raises(RuntimeError, match="chosen unavailable"):
        runner._resolve_session_agent_runtime(session_key="key", user_config={"fallback": {"auto_activate": False}})
    default.assert_not_called()


def test_cached_gateway_turn_refreshes_policy_and_interactivity(tmp_path, monkeypatch):
    import threading
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    _config(tmp_path, False)
    agent = SimpleNamespace(_fallback_chain=CHAIN, _fallback_model=CHAIN[0], _fallback_index=1,
                            _fallback_auto_activate=True, _fallback_selection_interactive=False,
                            _fallback_activated=True, _rate_limited_until=time.monotonic() + 600)
    runner = _runner()
    runner._agent_cache_lock = threading.Lock()
    runner._agent_cache = {"key": (agent, "signature", None, "sid")}
    runner._agent_config_signature = lambda *a, **k: "signature"
    runner._extract_cache_busting_config = lambda *_: {}
    runner._init_cached_agent_for_turn = lambda *_: None
    ctx = TurnContext(source=SessionSource(platform=Platform.SLACK, chat_id="c", user_id="u"),
                      message="hello", history=[], session_id="sid", session_key="key", user_config={})
    turn = TurnRunner(runner, ctx)
    monkeypatch.setattr(turn, "_cached_sid_is_dead", lambda *_: ("sid", False))
    result, reused = turn._resolve_turn_agent({"model": "primary", "runtime": {}}, "slack", "", 10, None, {})
    assert reused and result is agent
    assert agent._fallback_auto_activate is False
    assert agent._fallback_selection_interactive is True


@pytest.mark.asyncio
async def test_gateway_background_agent_has_no_selection_authority(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock
    from tests.gateway.test_background_command import _make_runner
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    _config(tmp_path, False)
    runner = _make_runner()
    adapter = AsyncMock()
    adapter.extract_media = MagicMock(return_value=([], "ok"))
    adapter.extract_images = MagicMock(return_value=([], "ok"))
    runner.adapters[Platform.TELEGRAM] = adapter
    factory = MagicMock()
    factory.return_value.run_conversation.return_value = {"final_response": "ok", "messages": []}
    monkeypatch.setattr("run_agent.AIAgent", factory)
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", lambda: {"api_key": "test"})
    source = SessionSource(platform=Platform.TELEGRAM, user_id="u", chat_id="c")
    await runner._run_background_task("hello", source, "bg")
    kwargs = factory.call_args.kwargs
    assert kwargs["fallback_model"] == CHAIN
    assert kwargs["fallback_auto_activate"] is False
    assert kwargs["fallback_selection_interactive"] is False


def test_torn_turn_config_cannot_reenable_default_override_substitution(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    _config(tmp_path, False)
    runner = _runner()
    runner._refresh_fallback_settings()
    (tmp_path / "config.yaml").write_text("fallback: [\n")
    runner._resolve_session_key_or_none = lambda *_: "key"
    runner._rehydrate_session_model_override = lambda *_: None
    runner._peek_session_state = lambda *_: SimpleNamespace(
        conversation=SimpleNamespace(model_override={"provider": "openai", "model": "chosen"}))
    default = Mock(return_value={"provider": "nous"})
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", default)
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs_for_provider",
                        Mock(side_effect=RuntimeError("chosen unavailable")))
    # The general gateway loader is fail-open, so a torn config can hand this entrypoint {}.
    with pytest.raises(RuntimeError, match="chosen unavailable"):
        runner._resolve_session_agent_runtime(session_key="key", user_config={})
    default.assert_not_called()
