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

import pytest
import yaml

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
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


def _setup_isolated_home(tmp_path, monkeypatch, *, warn):
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    cfg_path = hermes_home / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"model": {"default": "old-model", "provider": "openrouter"}, "providers": {}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
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
async def test_typed_model_expensive_prompts_instead_of_switching(tmp_path, monkeypatch):
    """Expensive model typed directly → confirm prompt, no switch applied."""
    _setup_isolated_home(tmp_path, monkeypatch, warn=True)
    runner = _make_runner()

    captured = {}

    async def _fake_request_slash_confirm(**kwargs):
        captured.update(kwargs)
        return kwargs["message"]

    runner._request_slash_confirm = _fake_request_slash_confirm

    result = await runner._handle_model_command(_make_event("/model openai/gpt-5.5-pro"))

    assert result is not None
    assert "EXPENSIVE MODEL WARNING" in result
    # The switch must NOT have been applied yet.
    assert runner._session_model_overrides == {}
    assert captured["command"] == "model"


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
    overrides = list(runner._session_model_overrides.values())
    assert len(overrides) == 1
    assert overrides[0]["model"] == "openai/gpt-5.5-pro"


@pytest.mark.asyncio
async def test_typed_model_expensive_cancel_keeps_current_model(tmp_path, monkeypatch):
    """Resolving the confirm with "cancel" leaves everything unchanged."""
    cfg_path = _setup_isolated_home(tmp_path, monkeypatch, warn=True)
    runner = _make_runner()

    captured = {}

    async def _fake_request_slash_confirm(**kwargs):
        captured.update(kwargs)
        return None

    runner._request_slash_confirm = _fake_request_slash_confirm

    await runner._handle_model_command(_make_event("/model openai/gpt-5.5-pro --global"))

    reply = await captured["handler"]("cancel")

    assert "cancelled" in reply.lower()
    assert runner._session_model_overrides == {}
    # --global must not have persisted the cancelled switch.
    written = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert written["model"]["default"] == "old-model"


@pytest.mark.asyncio
async def test_typed_model_cheap_switches_without_prompt(tmp_path, monkeypatch):
    """No warning → switch applies immediately; confirm primitive never invoked."""
    _setup_isolated_home(tmp_path, monkeypatch, warn=False)
    runner = _make_runner()
    runner._evict_cached_agent = lambda session_key: None

    async def _fail_request_slash_confirm(**kwargs):  # pragma: no cover
        raise AssertionError("confirm should not be requested for cheap models")

    runner._request_slash_confirm = _fail_request_slash_confirm

    result = await runner._handle_model_command(_make_event("/model openai/gpt-5.5-pro"))

    assert result is not None
    assert "gpt-5.5-pro" in result
    overrides = list(runner._session_model_overrides.values())
    assert len(overrides) == 1


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
