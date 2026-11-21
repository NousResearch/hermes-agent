"""``/model``'s displayed current model must follow the resident agent's live model.

Regression for #133993: after an in-turn fallback the cached agent serves the fallback
model while config/override still name the primary. ``/model``'s cancel notice (and the
picker's current-model tag) read config+override only, so they reported the stale
configured model — contradicting ``/status`` and ``/usage``, which read the agent first.
"""

import threading
from types import SimpleNamespace

import pytest
import hermes_yaml as yaml

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
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
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
        title="Expensive model",
        message="openai/gpt-5.5-pro has known pricing above Hermes' safety threshold.",
    )


def _setup_isolated_home(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"model": {"default": "old-model", "provider": "openrouter"}, "providers": {}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kw: _fake_switch_result())
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(
        "hermes_cli.model_cost_guard.expensive_model_warning", lambda *a, **kw: _fake_warning(),
    )


@pytest.mark.asyncio
async def test_cancel_notice_reports_agent_model_not_stale_override(tmp_path, monkeypatch):
    """Cancel must name the model the agent actually serves, not the stale override/config one."""
    _setup_isolated_home(tmp_path, monkeypatch)
    runner = _make_runner()

    event = _make_event("/model openai/gpt-5.5-pro")
    session_key = runner._session_key_for_source(event.source)

    # Session override names the primary (apodex) but the cached agent fell back to minimax —
    # the exact /status vs /usage vs /model contradiction from the report.
    runner._session_model_overrides[session_key] = {
        "model": "apodex/apodex-1.1-mini:free", "provider": "openrouter",
    }
    agent = SimpleNamespace(model="minimax/minimax-m3:free", provider="openrouter", base_url=None, api_key=None)
    runner._agent_cache[session_key] = [agent, None]

    captured = {}

    async def _fake_request_slash_confirm(**kwargs):
        captured.update(kwargs)
        return None  # buttons rendered

    runner._request_slash_confirm = _fake_request_slash_confirm

    await runner._handle_model_command(event)

    reply = await captured["handler"]("cancel")
    assert "minimax/minimax-m3:free" in reply
    assert "apodex" not in reply
    assert "Current model unchanged" in reply


@pytest.mark.asyncio
async def test_cancel_notice_falls_back_to_override_when_no_agent(tmp_path, monkeypatch):
    """No resident agent (evicted): the override's model remains the current-model source."""
    _setup_isolated_home(tmp_path, monkeypatch)
    runner = _make_runner()

    event = _make_event("/model openai/gpt-5.5-pro")
    session_key = runner._session_key_for_source(event.source)

    runner._session_model_overrides[session_key] = {
        "model": "apodex/apodex-1.1-mini:free", "provider": "openrouter",
    }

    captured = {}

    async def _fake_request_slash_confirm(**kwargs):
        captured.update(kwargs)
        return None

    runner._request_slash_confirm = _fake_request_slash_confirm

    await runner._handle_model_command(event)

    reply = await captured["handler"]("cancel")
    assert "apodex/apodex-1.1-mini:free" in reply


class TestAlignCurrentModelWithAgent:
    """Direct unit tests for the alignment method (None/empty agents keep the resolved value)."""

    def _ctx(self):
        from gateway.slash_commands_model import _ModelSwitchContext

        return _ModelSwitchContext(
            session_key="sk", source=None, config_path=None, persist_global=False,
            current_model="apodex/apodex-1.1-mini:free",
        )

    def test_agent_model_wins(self):
        ctx = self._ctx()
        ctx.align_current_model_with_agent(SimpleNamespace(model="minimax/minimax-m3:free"))
        assert ctx.current_model == "minimax/minimax-m3:free"

    def test_none_agent_keeps_value(self):
        ctx = self._ctx()
        ctx.align_current_model_with_agent(None)
        assert ctx.current_model == "apodex/apodex-1.1-mini:free"

    def test_empty_agent_model_keeps_value(self):
        ctx = self._ctx()
        ctx.align_current_model_with_agent(SimpleNamespace(model="  "))
        assert ctx.current_model == "apodex/apodex-1.1-mini:free"

    def test_route_fields_untouched(self):
        """The alignment is display-only: provider/base_url/api_key stay on the config chain."""
        import uuid

        config_key = uuid.uuid4().hex
        agent_key = uuid.uuid4().hex
        ctx = self._ctx()
        ctx.current_provider = "openrouter"
        ctx.current_base_url = "https://openrouter.ai/api/v1"
        ctx.current_api_key = config_key
        ctx.align_current_model_with_agent(
            SimpleNamespace(model="groq/llama-3.3", provider="groq", base_url="https://x", api_key=agent_key)
        )
        assert ctx.current_model == "groq/llama-3.3"
        assert ctx.current_provider == "openrouter"
        assert ctx.current_base_url == "https://openrouter.ai/api/v1"
        assert ctx.current_api_key == config_key
