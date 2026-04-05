"""Tests for gateway /reasoning command and hot reload behavior."""

import asyncio
import inspect
import sys
import threading
import types
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/reasoning", platform=Platform.TELEGRAM, user_id="12345", chat_id="67890"):
    """Build a MessageEvent for testing."""
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    """Create a bare GatewayRunner without calling __init__."""
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
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    return runner


class _CapturingAgent:
    """Fake agent that records init kwargs for assertions."""

    last_init = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []

    def run_conversation(self, user_message: str, conversation_history=None, task_id=None):
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
        }


class _CachingRequestOptionsAgent:
    created = []

    def __init__(self, *args, **kwargs):
        self.init_kwargs = dict(kwargs)
        self.request_options = dict(kwargs.get("request_options") or {})
        self.tools = []
        type(self).created.append(self)

    def run_conversation(self, user_message: str, conversation_history=None, task_id=None):
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
        }


class TestReasoningCommand:
    @pytest.mark.asyncio
    async def test_reasoning_in_help_output(self):
        runner = _make_runner()
        event = _make_event(text="/help")

        result = await runner._handle_help_command(event)

        assert "/reasoning [level|show|hide]" in result

    def test_reasoning_is_known_command(self):
        source = inspect.getsource(gateway_run.GatewayRunner._handle_message)
        assert '"reasoning"' in source

    @pytest.mark.asyncio
    async def test_reasoning_command_reloads_current_state_from_config(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "agent:\n  reasoning_effort: none\ndisplay:\n  show_reasoning: true\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        monkeypatch.delenv("HERMES_REASONING_EFFORT", raising=False)

        runner = _make_runner()
        runner._reasoning_config = {"enabled": True, "effort": "xhigh"}
        runner._show_reasoning = False

        result = await runner._handle_reasoning_command(_make_event("/reasoning"))

        assert "**Effort:** `none (disabled)`" in result
        assert "**Display:** on ✓" in result
        assert runner._reasoning_config == {"enabled": False}
        assert runner._show_reasoning is True

    @pytest.mark.asyncio
    async def test_handle_reasoning_command_updates_config_and_cache(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text("agent:\n  reasoning_effort: medium\n", encoding="utf-8")

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        monkeypatch.delenv("HERMES_REASONING_EFFORT", raising=False)

        runner = _make_runner()
        runner._reasoning_config = {"enabled": True, "effort": "medium"}

        result = await runner._handle_reasoning_command(_make_event("/reasoning low"))

        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert saved["agent"]["reasoning_effort"] == "low"
        assert runner._reasoning_config == {"enabled": True, "effort": "low"}
        assert "takes effect on next message" in result

    def test_run_agent_reloads_reasoning_config_per_message(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("agent:\n  reasoning_effort: low\n", encoding="utf-8")

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        monkeypatch.setattr(gateway_run, "_env_path", hermes_home / ".env")
        monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
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
        monkeypatch.delenv("HERMES_REASONING_EFFORT", raising=False)
        fake_run_agent = types.ModuleType("run_agent")
        fake_run_agent.AIAgent = _CapturingAgent
        monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

        _CapturingAgent.last_init = None
        runner = _make_runner()
        runner._reasoning_config = {"enabled": True, "effort": "xhigh"}

        source = SessionSource(
            platform=Platform.LOCAL,
            chat_id="cli",
            chat_name="CLI",
            chat_type="dm",
            user_id="user-1",
        )

        result = asyncio.run(
            runner._run_agent(
                message="ping",
                context_prompt="",
                history=[],
                source=source,
                session_id="session-1",
                session_key="agent:main:local:dm",
            )
        )

        assert result["final_response"] == "ok"
        assert _CapturingAgent.last_init is not None
        assert _CapturingAgent.last_init["reasoning_config"] == {"enabled": True, "effort": "low"}

    def test_run_agent_refreshes_request_options_on_cached_agent(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "model:\n"
            "  default: gpt-5.4\n"
            "  provider: custom\n"
            "  base_url: https://api.openai.com/v1\n"
            "  request_options:\n"
            "    service_tier: priority\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        monkeypatch.setattr(gateway_run, "_env_path", hermes_home / ".env")
        monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "custom",
                "api_mode": "codex_responses",
                "base_url": "https://api.openai.com/v1",
                "api_key": "sk-openai-direct",
            },
        )
        monkeypatch.delenv("HERMES_REASONING_EFFORT", raising=False)
        fake_run_agent = types.ModuleType("run_agent")
        fake_run_agent.AIAgent = _CachingRequestOptionsAgent
        monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

        _CachingRequestOptionsAgent.created = []
        runner = _make_runner()
        runner._agent_cache = {}
        runner._agent_cache_lock = threading.Lock()

        source = SessionSource(
            platform=Platform.LOCAL,
            chat_id="cli",
            chat_name="CLI",
            chat_type="dm",
            user_id="user-1",
        )

        first = asyncio.run(
            runner._run_agent(
                message="ping",
                context_prompt="",
                history=[],
                source=source,
                session_id="session-1",
                session_key="agent:main:local:dm",
            )
        )

        config_path.write_text(
            "model:\n"
            "  default: gpt-5.4\n"
            "  provider: custom\n"
            "  base_url: https://api.openai.com/v1\n"
            "  request_options:\n"
            "    service_tier: flex\n",
            encoding="utf-8",
        )

        second = asyncio.run(
            runner._run_agent(
                message="ping again",
                context_prompt="",
                history=[],
                source=source,
                session_id="session-1",
                session_key="agent:main:local:dm",
            )
        )

        assert first["final_response"] == "ok"
        assert second["final_response"] == "ok"
        assert len(_CachingRequestOptionsAgent.created) == 1
        cached_agent = runner._agent_cache["agent:main:local:dm"][0]
        assert cached_agent.request_options == {"service_tier": "flex"}

    def test_run_agent_prefers_config_over_stale_reasoning_env(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("agent:\n  reasoning_effort: none\n", encoding="utf-8")

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
        monkeypatch.setenv("HERMES_REASONING_EFFORT", "low")
        fake_run_agent = types.ModuleType("run_agent")
        fake_run_agent.AIAgent = _CapturingAgent
        monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

        _CapturingAgent.last_init = None
        runner = _make_runner()

        source = SessionSource(
            platform=Platform.LOCAL,
            chat_id="cli",
            chat_name="CLI",
            chat_type="dm",
            user_id="user-1",
        )

        result = asyncio.run(
            runner._run_agent(
                message="ping",
                context_prompt="",
                history=[],
                source=source,
                session_id="session-1",
                session_key="agent:main:local:dm",
            )
        )

        assert result["final_response"] == "ok"
        assert _CapturingAgent.last_init is not None
        assert _CapturingAgent.last_init["reasoning_config"] == {"enabled": False}

    def test_run_agent_includes_enabled_mcp_servers_in_gateway_toolsets(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "platform_toolsets:\n"
            "  cli: [web, memory]\n"
            "mcp_servers:\n"
            "  exa:\n"
            "    url: https://mcp.exa.ai/mcp\n"
            "  web-search-prime:\n"
            "    url: https://api.z.ai/api/mcp/web_search_prime/mcp\n",
            encoding="utf-8",
        )

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

        _CapturingAgent.last_init = None
        runner = _make_runner()

        source = SessionSource(
            platform=Platform.LOCAL,
            chat_id="cli",
            chat_name="CLI",
            chat_type="dm",
            user_id="user-1",
        )

        result = asyncio.run(
            runner._run_agent(
                message="ping",
                context_prompt="",
                history=[],
                source=source,
                session_id="session-1",
                session_key="agent:main:local:dm",
            )
        )

        assert result["final_response"] == "ok"
        assert _CapturingAgent.last_init is not None
        enabled_toolsets = set(_CapturingAgent.last_init["enabled_toolsets"])
        assert "web" in enabled_toolsets
        assert "memory" in enabled_toolsets
        assert "exa" in enabled_toolsets
        assert "web-search-prime" in enabled_toolsets

    def test_run_agent_homeassistant_uses_default_platform_toolset(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("", encoding="utf-8")

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

        _CapturingAgent.last_init = None
        runner = _make_runner()

        source = SessionSource(
            platform=Platform.HOMEASSISTANT,
            chat_id="ha",
            chat_name="Home Assistant",
            chat_type="dm",
            user_id="user-1",
        )

        result = asyncio.run(
            runner._run_agent(
                message="ping",
                context_prompt="",
                history=[],
                source=source,
                session_id="session-1",
                session_key="agent:main:homeassistant:dm",
            )
        )

        assert result["final_response"] == "ok"
        assert _CapturingAgent.last_init is not None
        assert "homeassistant" in set(_CapturingAgent.last_init["enabled_toolsets"])
