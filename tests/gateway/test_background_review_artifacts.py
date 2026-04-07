"""Tests for gateway-side background review artifact visibility config."""

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.session import SessionSource


def _make_runner():
    """Create a minimal GatewayRunner without full initialization."""
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    return runner


class _CapturingAgent:
    """Fake agent that records post-construction state."""

    last_init = None
    last_instance = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        type(self).last_instance = self
        self.tools = []
        self.background_review_callback = "unset"
        self.show_background_review_artifacts = "unset"
        self.tool_progress_callback = None
        self.step_callback = None
        self.stream_delta_callback = None
        self.status_callback = None
        self.reasoning_config = None

    def run_conversation(self, user_message: str, conversation_history=None, task_id=None):
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
        }


def _patch_runtime(monkeypatch, hermes_home):
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(gateway_run, "_env_path", hermes_home / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_mode": "chat_completions",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "test-key",
        },
    )
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)


def _make_source():
    return SessionSource(
        platform=Platform.LOCAL,
        chat_id="cli",
        chat_name="CLI",
        chat_type="dm",
        user_id="user-1",
    )


class TestGatewayBackgroundReviewArtifacts:
    def test_run_agent_disables_background_review_artifacts_from_config(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "display:\n  background_review_artifacts: false\n",
            encoding="utf-8",
        )

        _patch_runtime(monkeypatch, hermes_home)
        _CapturingAgent.last_init = None
        _CapturingAgent.last_instance = None

        runner = _make_runner()
        result = asyncio.run(
            runner._run_agent(
                message="ping",
                context_prompt="",
                history=[],
                source=_make_source(),
                session_id="session-1",
                session_key="agent:main:local:dm",
            )
        )

        assert result["final_response"] == "ok"
        assert _CapturingAgent.last_instance is not None
        assert _CapturingAgent.last_instance.show_background_review_artifacts is False
        assert _CapturingAgent.last_instance.background_review_callback is None

    def test_run_agent_defaults_background_review_artifacts_to_enabled(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("display:\n  tool_progress: all\n", encoding="utf-8")

        _patch_runtime(monkeypatch, hermes_home)
        _CapturingAgent.last_init = None
        _CapturingAgent.last_instance = None

        runner = _make_runner()
        result = asyncio.run(
            runner._run_agent(
                message="ping",
                context_prompt="",
                history=[],
                source=_make_source(),
                session_id="session-1",
                session_key="agent:main:local:dm",
            )
        )

        assert result["final_response"] == "ok"
        assert _CapturingAgent.last_instance is not None
        assert _CapturingAgent.last_instance.show_background_review_artifacts is True
        assert callable(_CapturingAgent.last_instance.background_review_callback)
