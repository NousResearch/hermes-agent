"""Tests for gateway /reasoning command and hot reload behavior."""

import asyncio
import inspect
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

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
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
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

        runner = _make_runner()
        runner._reasoning_config = {"enabled": True, "effort": "medium"}

        result = await runner._handle_reasoning_command(_make_event("/reasoning low --global"))

        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert saved["agent"]["reasoning_effort"] == "low"
        assert runner._reasoning_config == {"enabled": True, "effort": "low"}
        assert "saved to config" in result

    @pytest.mark.asyncio
    async def test_handle_reasoning_command_defaults_to_session_scope(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text("agent:\n  reasoning_effort: medium\n", encoding="utf-8")

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        runner._reasoning_config = {"enabled": True, "effort": "medium"}
        event = _make_event("/reasoning high")
        session_key = runner._session_key_for_source(event.source)

        result = await runner._handle_reasoning_command(event)

        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert saved["agent"]["reasoning_effort"] == "medium"
        assert runner._session_reasoning_overrides[session_key] == {
            "enabled": True,
            "effort": "high",
        }
        assert "session only" in result

    @pytest.mark.asyncio
    async def test_handle_reasoning_command_rejects_malformed_global_substrings(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text("agent:\n  reasoning_effort: medium\n", encoding="utf-8")

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        runner._reasoning_config = {"enabled": True, "effort": "medium"}
        event = _make_event("/reasoning high--global")
        session_key = runner._session_key_for_source(event.source)

        result = await runner._handle_reasoning_command(event)

        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert saved["agent"]["reasoning_effort"] == "medium"
        assert session_key not in runner._session_reasoning_overrides
        assert "Unknown argument" in result

    @pytest.mark.asyncio
    async def test_handle_reasoning_command_global_persists_with_flag(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text("agent:\n  reasoning_effort: medium\n", encoding="utf-8")

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        runner._reasoning_config = {"enabled": True, "effort": "medium"}
        event = _make_event("/reasoning high --global")
        session_key = runner._session_key_for_source(event.source)

        result = await runner._handle_reasoning_command(event)

        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert saved["agent"]["reasoning_effort"] == "high"
        assert session_key not in runner._session_reasoning_overrides
        assert "saved to config" in result

    @pytest.mark.asyncio
    async def test_handle_reasoning_command_global_falls_back_to_session_override_when_save_fails(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text("agent:\n  reasoning_effort: medium\n", encoding="utf-8")

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        monkeypatch.setattr(
            gateway_run,
            "atomic_yaml_write",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("disk full")),
        )

        runner = _make_runner()
        runner._reasoning_config = {"enabled": True, "effort": "medium"}
        event = _make_event("/reasoning high --global")
        session_key = runner._session_key_for_source(event.source)

        result = await runner._handle_reasoning_command(event)

        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert saved["agent"]["reasoning_effort"] == "medium"
        assert runner._reasoning_config == {"enabled": True, "effort": "medium"}
        assert runner._session_reasoning_overrides[session_key] == {
            "enabled": True,
            "effort": "high",
        }
        assert "session only" in result

    @pytest.mark.asyncio
    async def test_handle_background_command_passes_active_session_key(self):
        runner = _make_runner()
        event = _make_event("/background summarize this", platform=Platform.LOCAL, user_id="u1", chat_id="cli")
        runner._run_background_task = AsyncMock(return_value=None)

        result = await runner._handle_background_command(event)
        await asyncio.sleep(0)

        session_key = runner._session_key_for_source(event.source)
        runner._run_background_task.assert_awaited_once()
        assert runner._run_background_task.await_args.args[3] == session_key
        assert "Background task started" in result

    @pytest.mark.asyncio
    async def test_session_expiry_watcher_clears_reasoning_override_for_expired_session(self):
        key = "agent:main:local:dm"
        entry = types.SimpleNamespace(session_id="sess-1", memory_flushed=False)

        class _Store:
            def __init__(self):
                self._entries = {key: entry}
                self._lock = self

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def _ensure_loaded(self):
                return None

            def _is_session_expired(self, current):
                return current is entry

            def _save(self):
                return None

        async def _fake_sleep(_seconds):
            if getattr(_fake_sleep, "seen", 0) == 0:
                _fake_sleep.seen = 1
                return None
            runner._running = False
            return None

        runner = _make_runner()
        runner.session_store = _Store()
        runner._running = True
        runner._agent_cache = {}
        runner._session_reasoning_overrides[key] = {"enabled": True, "effort": "high"}
        runner._async_flush_memories = AsyncMock(return_value=None)
        runner._cleanup_agent_resources = lambda agent: None

        with patch("gateway.run.asyncio.sleep", side_effect=_fake_sleep):
            await runner._session_expiry_watcher(interval=1)

        assert key not in runner._session_reasoning_overrides
        assert entry.memory_flushed is True

    @pytest.mark.asyncio
    async def test_handle_resume_command_clears_session_reasoning_override(self):
        runner = _make_runner()
        event = _make_event("/resume Focus", platform=Platform.LOCAL, user_id="u1", chat_id="cli")
        session_key = runner._session_key_for_source(event.source)
        runner._session_reasoning_overrides[session_key] = {"enabled": True, "effort": "high"}

        current_entry = types.SimpleNamespace(session_id="current-session")
        target_entry = types.SimpleNamespace(session_id="target-session")

        class _Store:
            def get_or_create_session(self, source):
                return current_entry

            def switch_session(self, key, target_id):
                assert key == session_key
                assert target_id == "target-session"
                return target_entry

            def load_transcript(self, session_id):
                assert session_id == "target-session"
                return [{"role": "user", "content": "hello"}]

        class _Task:
            def add_done_callback(self, callback):
                return None

        def _fake_create_task(coro):
            coro.close()
            return _Task()

        runner.session_store = _Store()
        runner._session_db = types.SimpleNamespace(
            resolve_session_by_title=lambda name: "target-session",
            get_session_title=lambda session_id: "Focus",
        )
        runner._release_running_agent_state = MagicMock()
        runner._clear_session_boundary_security_state = MagicMock()
        runner._async_flush_memories = AsyncMock(return_value=None)

        with patch("gateway.run.asyncio.create_task", side_effect=_fake_create_task):
            result = await runner._handle_resume_command(event)

        assert session_key not in runner._session_reasoning_overrides
        runner._release_running_agent_state.assert_called_once_with(session_key)
        runner._clear_session_boundary_security_state.assert_called_once_with(session_key)
        assert "Resumed session **Focus**" in result

    def test_run_agent_prefers_session_reasoning_override_over_global_config(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("agent:\n  reasoning_effort: low\n", encoding="utf-8")

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
                "api_key": "***",
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
        session_key = "agent:main:local:dm"
        runner._session_reasoning_overrides[session_key] = {
            "enabled": True,
            "effort": "xhigh",
        }

        result = asyncio.run(
            runner._run_agent(
                message="ping",
                context_prompt="",
                history=[],
                source=source,
                session_id="session-1",
                session_key=session_key,
            )
        )

        assert result["final_response"] == "ok"
        assert _CapturingAgent.last_init is not None
        assert _CapturingAgent.last_init["reasoning_config"] == {
            "enabled": True,
            "effort": "xhigh",
        }

    def test_run_background_task_prefers_session_reasoning_override(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("agent:\n  reasoning_effort: low\n", encoding="utf-8")

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        monkeypatch.setattr(gateway_run, "_env_path", hermes_home / ".env")
        monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
        monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": "***",
            },
        )
        fake_run_agent = types.ModuleType("run_agent")
        fake_run_agent.AIAgent = _CapturingAgent
        monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

        class _Adapter:
            def extract_media(self, response):
                return [], response

            def extract_images(self, response):
                return [], response

            async def send(self, *args, **kwargs):
                return None

            async def send_image(self, *args, **kwargs):
                return None

            async def send_document(self, *args, **kwargs):
                return None

        _CapturingAgent.last_init = None
        runner = _make_runner()
        runner.adapters[Platform.LOCAL] = _Adapter()
        runner._run_in_executor_with_context = AsyncMock(side_effect=lambda fn: fn())
        runner._cleanup_agent_resources = lambda agent: None

        source = SessionSource(
            platform=Platform.LOCAL,
            chat_id="cli",
            chat_name="CLI",
            chat_type="dm",
            user_id="user-1",
        )
        session_key = "agent:main:local:dm"
        runner._session_reasoning_overrides[session_key] = {
            "enabled": True,
            "effort": "xhigh",
        }

        asyncio.run(runner._run_background_task("ping", source, "bg-1", session_key))

        assert _CapturingAgent.last_init is not None
        assert _CapturingAgent.last_init["reasoning_config"] == {
            "enabled": True,
            "effort": "xhigh",
        }

    def test_run_agent_reloads_reasoning_config_per_message(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("agent:\n  reasoning_effort: low\n", encoding="utf-8")

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
