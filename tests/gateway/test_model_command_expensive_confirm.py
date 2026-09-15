"""Gateway typed ``/model <name>`` must route through the expensive-model
confirmation gate.

The pickers (Telegram/Discord inline keyboards, TUI, dashboard) confirm
expensive models via their own UI affordances; the typed text command
previously bypassed the guard entirely — a user typing
``/model openai/gpt-5.5-pro`` switched silently while the picker warned.
These tests pin the typed path:

- warning fires → handler returns the slash-confirm prompt, switch NOT applied
- confirm ("once") → switch applies (session override set)
- cancel → switch not applied, current model unchanged
- no warning (cheap model) → switch applies immediately, no prompt
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    runner._session_db = None
    runner.session_store = SimpleNamespace()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store, set_model_override=AsyncMock()
    )
    return runner


def _make_event(text):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


def _fake_switch_result():
    from hermes_cli.model_switch import ModelSwitchResult

    return ModelSwitchResult(
        success=True,
        new_model="openai/gpt-5.5-pro",
        target_provider="openrouter",
        provider_changed=False,
        api_key="sk-test",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        provider_label="OpenRouter",
    )


def _fake_warning():
    return SimpleNamespace(
        message=(
            "!!! EXPENSIVE MODEL WARNING !!!\n"
            "openai/gpt-5.5-pro has known pricing above Hermes' safety threshold.\n"
            "did you mean to select openai/gpt-5.5?"
        ),
    )


def _setup_isolated_home(tmp_path, monkeypatch, *, warn, current_provider="openrouter"):
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    cfg_path = hermes_home / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"model": {"default": "old-model", "provider": current_provider}, "providers": {}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    config = {"model": {"default": "old-model", "provider": current_provider}, "providers": {}}
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda **k: config)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr("hermes_cli.config.read_user_config_raw", lambda *a: config)
    monkeypatch.setattr("hermes_cli.config.save_config", Mock())
    monkeypatch.setattr("hermes_cli.model_switch.resolve_display_context_length_async", AsyncMock(return_value=None))
    monkeypatch.setattr("hermes_cli.context_switch_guard.enrich_model_switch_warnings_for_gateway", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_data_policy_guard.data_training_warning", lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kw: _fake_switch_result(),
    )
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(
        "hermes_cli.model_cost_guard.expensive_model_warning",
        (lambda *a, **kw: _fake_warning()) if warn else (lambda *a, **kw: None),
    )
    return cfg_path


@pytest.mark.asyncio
async def test_typed_model_expensive_confirm_once_applies_switch(tmp_path, monkeypatch):
    """Resolving the confirm with "once" applies the switch."""
    _setup_isolated_home(tmp_path, monkeypatch, warn=True)
    runner = _make_runner()
    runner._evict_cached_agent = lambda session_key: None

    captured = {}

    async def _fake_request_slash_confirm(**kwargs):
        captured.update(kwargs)
        return None  # buttons rendered

    runner._request_slash_confirm = _fake_request_slash_confirm

    await runner._handle_model_command(_make_event("/model openai/gpt-5.5-pro"))
    assert runner._session_model_overrides == {}

    reply = await captured["handler"]("once")

    assert "gpt-5.5-pro" in reply
    assert "PROVIDER AUTOMATICALLY CHANGED" not in reply
    overrides = list(runner._session_model_overrides.values())
    assert len(overrides) == 1
    assert overrides[0]["model"] == "openai/gpt-5.5-pro"


@pytest.mark.asyncio
async def test_failed_inplace_swap_aborts_commit(tmp_path, monkeypatch):
    """A failed in-place agent swap must be a no-op, not a dead session.

    Regression for #50163: the resolution pipeline succeeds (valid model name)
    but the cached agent's ``switch_model()`` raises mid-conversation (bad key /
    unreachable URL). The agent rolls itself back to the old working model; the
    gateway must NOT then commit the broken model as a session override or evict
    the working cached agent — otherwise the next message rebuilds a dead agent
    and the conversation is lost.
    """
    _setup_isolated_home(tmp_path, monkeypatch, warn=False)
    runner = _make_runner()

    # Working cached agent whose in-place swap fails (and rolls itself back).
    class _FailingAgent:
        def __init__(self):
            self.model = "old-model"
            self.provider = "openrouter"

        def switch_model(self, **kwargs):
            # Mirrors agent_runtime_helpers.switch_model: the real method
            # restores old state then re-raises. We keep model unchanged.
            raise RuntimeError("connection refused: bad base_url")

    import threading

    agent = _FailingAgent()
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    session_key = runner._session_key_for_source(_make_event("/model x").source)
    runner._agent_cache[session_key] = [agent, None]
    runner._session_db = None

    evicted = []
    runner._evict_cached_agent = lambda sk: evicted.append(sk)

    result = await runner._handle_model_command(_make_event("/model openai/gpt-5.5-pro"))

    # Error surfaced to the user, not a success confirmation.
    assert result is not None
    assert "failed" in result.lower()
    # The broken switch must NOT have been committed anywhere.
    assert runner._session_model_overrides == {}
    # The working cached agent must NOT have been evicted.
    assert evicted == []
    # The agent stayed on its old model (rolled back).
    assert agent.model == "old-model"


_PROVIDER_WARNING = (
    "PROVIDER AUTOMATICALLY CHANGED: openai-codex -> openrouter. "
    "No --provider argument was supplied. This change may incur additional costs."
)


def _provider_change_setup(tmp_path, monkeypatch, *, warn):
    _setup_isolated_home(tmp_path, monkeypatch, warn=warn, current_provider="openai-codex")
    runner = _make_runner()
    result = _fake_switch_result()
    result.provider_changed = True
    result.provider_switch_warning = _PROVIDER_WARNING
    result.warning_message = "Ordinary validation warning."
    result.runtime_capabilities = {"native_compaction": True}
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **k: result)
    runner._session_model_overrides["unused"] = {}  # independent session is preserved
    agent = SimpleNamespace(model="old-model", provider="openai-codex")

    def swap(**kwargs):
        agent.model, agent.provider = kwargs["new_model"], kwargs["new_provider"]

    agent.switch_model = Mock(side_effect=swap)
    runner._cached_agent_for = lambda key: agent
    runner._evict_cached_agent = Mock()
    return runner, agent, result


@pytest.mark.asyncio
async def test_provider_warning_waits_for_confirmation_then_applied_swap(tmp_path, monkeypatch):
    runner, agent, result = _provider_change_setup(tmp_path, monkeypatch, warn=True)
    captured = {}

    async def request_confirm(**kwargs):
        captured.update(kwargs)
        return kwargs["message"]

    runner._request_slash_confirm = request_confirm
    pending = await runner._handle_model_command(_make_event("/model openai/gpt-5.5-pro --session"))
    assert _PROVIDER_WARNING not in pending
    assert "EXPENSIVE MODEL WARNING" in pending
    agent.switch_model.assert_not_called()
    reply = await captured["handler"]("once")

    assert reply.startswith(f"🚨 **{_PROVIDER_WARNING}**\n")
    assert reply.index(_PROVIDER_WARNING) < reply.index("Model switched")
    assert result.warning_message in reply
    assert agent.model == result.new_model
    assert agent.switch_model.call_args.kwargs["capabilities"] == result.runtime_capabilities
    runner.async_session_store.set_model_override.assert_awaited_once()
    assert runner._session_model_overrides["unused"] == {}


@pytest.mark.asyncio
async def test_cancelled_confirmation_has_no_provider_warning(tmp_path, monkeypatch):
    runner, agent, result = _provider_change_setup(tmp_path, monkeypatch, warn=True)
    captured = {}

    async def request_confirm(**kwargs):
        captured.update(kwargs)
        return kwargs["message"]

    runner._request_slash_confirm = request_confirm
    pending = await runner._handle_model_command(_make_event("/model openai/gpt-5.5-pro --session"))
    reply = await captured["handler"]("cancel")
    assert _PROVIDER_WARNING not in pending + reply
    assert "cancelled" in reply
    agent.switch_model.assert_not_called()
    assert runner._session_model_overrides == {"unused": {}}
    runner.async_session_store.set_model_override.assert_not_awaited()
    runner._evict_cached_agent.assert_not_called()


@pytest.mark.asyncio
async def test_failed_cached_swap_has_no_provider_warning(tmp_path, monkeypatch):
    runner, agent, result = _provider_change_setup(tmp_path, monkeypatch, warn=False)
    agent.switch_model.side_effect = RuntimeError("offline swap failure")
    reply = await runner._handle_model_command(_make_event("/model openai/gpt-5.5-pro --session"))
    assert "failed" in reply
    assert _PROVIDER_WARNING not in reply
    assert "Model switched" not in reply
    assert agent.model == "old-model"
    assert runner._session_model_overrides == {"unused": {}}
    runner.async_session_store.set_model_override.assert_not_awaited()
    runner._evict_cached_agent.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["--session", "--once", "--global"])
async def test_immediate_applied_provider_warning_preserves_scope(tmp_path, monkeypatch, scope):
    runner, agent, result = _provider_change_setup(tmp_path, monkeypatch, warn=False)
    runner._request_slash_confirm = AsyncMock(side_effect=AssertionError("no new confirmation gate"))
    reply = await runner._handle_model_command(_make_event(f"/model openai/gpt-5.5-pro {scope}"))
    assert reply.startswith(f"🚨 **{_PROVIDER_WARNING}**\n")
    assert agent.model == result.new_model
    assert result.warning_message in reply
    if scope == "--once":
        runner.async_session_store.set_model_override.assert_not_awaited()
        assert runner._pending_one_turn_model_restores
    else:
        runner.async_session_store.set_model_override.assert_awaited_once()
    from hermes_cli.config import save_config
    assert save_config.called == (scope == "--global")
