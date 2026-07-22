"""Tests for Discord channel_cwds resolution and session wiring."""

import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _ensure_discord_mock():
    """Stub ``discord`` only when the real package is genuinely absent.

    Checking ``sys.modules`` alone is not enough: when discord is installed
    but simply not imported yet, the stub wins the ``setdefault`` race and
    every later test module that imports the real package silently skips.
    """
    try:
        import discord  # noqa: F401
        return
    except ImportError:
        pass
    discord_mod = types.ModuleType("discord")
    discord_mod.Intents = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.Interaction = object
    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType, resolve_channel_cwd
from gateway.session import SessionSource


class _CapturingAgent:
    last_init = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []

    def run_conversation(self, user_message, conversation_history=None, task_id=None, persist_user_message=None):
        return {
            "final_response": "ok",
            "messages": [],
            "api_calls": 1,
            "completed": True,
        }


def _install_fake_agent(monkeypatch):
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _CapturingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)


def _make_adapter():
    _ensure_discord_mock()
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)
    adapter.config = MagicMock()
    adapter.config.extra = {}
    return adapter


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = "Global prompt"
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._service_tier = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._pending_model_notes = {}
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(streaming=None)
    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda source: SimpleNamespace(session_id="session-1"),
        load_transcript=lambda session_id: [],
    )
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._enrich_message_with_vision = AsyncMock(return_value="ENRICHED")
    return runner


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="12345",
        chat_type="thread",
        user_id="user-1",
    )


class TestResolveChannelCwd:
    def test_no_cwd_returns_none(self):
        assert resolve_channel_cwd({}, "123") is None

    def test_match_by_channel_id(self, tmp_path):
        extra = {"channel_cwds": {"100": str(tmp_path)}}
        assert resolve_channel_cwd(extra, "100") == str(tmp_path)

    def test_match_by_parent_id(self, tmp_path):
        extra = {"channel_cwds": {"200": str(tmp_path)}}
        assert resolve_channel_cwd(extra, "999", parent_id="200") == str(tmp_path)

    def test_exact_channel_overrides_parent(self, tmp_path):
        thread_dir = tmp_path / "thread"
        forum_dir = tmp_path / "forum"
        thread_dir.mkdir()
        forum_dir.mkdir()
        extra = {
            "channel_cwds": {
                "999": str(thread_dir),
                "200": str(forum_dir),
            }
        }
        assert resolve_channel_cwd(extra, "999", parent_id="200") == str(thread_dir)

    def test_tilde_is_expanded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "project").mkdir()
        extra = {"channel_cwds": {"100": "~/project"}}
        assert resolve_channel_cwd(extra, "100") == str(tmp_path / "project")

    def test_missing_directory_is_ignored(self, tmp_path):
        extra = {"channel_cwds": {"100": str(tmp_path / "does-not-exist")}}
        assert resolve_channel_cwd(extra, "100") is None

    def test_invalid_child_falls_back_to_valid_parent(self, tmp_path):
        """A stale child override must not block inheriting a valid parent cwd."""
        parent_dir = tmp_path / "forum"
        parent_dir.mkdir()
        extra = {
            "channel_cwds": {
                "999": str(tmp_path / "missing-thread"),  # invalid child
                "200": str(parent_dir),  # valid parent
            }
        }
        assert resolve_channel_cwd(extra, "999", parent_id="200") == str(parent_dir)

    def test_blank_values_are_ignored(self):
        extra = {"channel_cwds": {"100": "   "}}
        assert resolve_channel_cwd(extra, "100") is None

    def test_non_dict_config_is_ignored(self):
        extra = {"channel_cwds": ["not", "a", "dict"]}
        assert resolve_channel_cwd(extra, "100") is None


class TestDiscordAdapterChannelCwd:
    def test_build_slash_event_sets_channel_cwd(self, tmp_path):
        adapter = _make_adapter()
        adapter.config.extra = {"channel_cwds": {"321": str(tmp_path)}}
        adapter.build_source = MagicMock(return_value=SimpleNamespace())

        interaction = SimpleNamespace(
            channel_id=321,
            channel=SimpleNamespace(name="general", guild=None, parent_id=None),
            user=SimpleNamespace(id=1, display_name="Brenner"),
        )
        adapter._get_effective_topic = MagicMock(return_value=None)

        event = adapter._build_slash_event(interaction, "/retry")

        assert event.channel_cwd == str(tmp_path)

    @pytest.mark.asyncio
    async def test_dispatch_thread_session_inherits_parent_channel_cwd(self, tmp_path):
        adapter = _make_adapter()
        adapter.config.extra = {"channel_cwds": {"200": str(tmp_path)}}
        adapter.build_source = MagicMock(return_value=SimpleNamespace())
        adapter._get_effective_topic = MagicMock(return_value=None)
        adapter.handle_message = AsyncMock()

        interaction = SimpleNamespace(
            guild=SimpleNamespace(name="Wetlands"),
            channel=SimpleNamespace(id=200, parent=None),
            user=SimpleNamespace(id=1, display_name="Brenner"),
        )

        await adapter._dispatch_thread_session(interaction, "999", "new-thread", "hello")

        dispatched_event = adapter.handle_message.await_args.args[0]
        assert dispatched_event.channel_cwd == str(tmp_path)

    def test_channel_cwd_coexists_with_channel_prompt(self, tmp_path):
        """A channel may carry both a prompt and a cwd — neither affects the other."""
        adapter = _make_adapter()
        adapter.config.extra = {
            "channel_prompts": {"321": "Research mode"},
            "channel_cwds": {"321": str(tmp_path)},
        }
        adapter.build_source = MagicMock(return_value=SimpleNamespace())

        interaction = SimpleNamespace(
            channel_id=321,
            channel=SimpleNamespace(name="general", guild=None, parent_id=None),
            user=SimpleNamespace(id=1, display_name="Brenner"),
        )
        adapter._get_effective_topic = MagicMock(return_value=None)

        event = adapter._build_slash_event(interaction, "/retry")

        assert event.channel_prompt == "Research mode"
        assert event.channel_cwd == str(tmp_path)

    def test_event_without_config_has_no_channel_cwd(self):
        adapter = _make_adapter()
        adapter.build_source = MagicMock(return_value=SimpleNamespace())

        interaction = SimpleNamespace(
            channel_id=321,
            channel=SimpleNamespace(name="general", guild=None, parent_id=None),
            user=SimpleNamespace(id=1, display_name="Brenner"),
        )
        adapter._get_effective_topic = MagicMock(return_value=None)

        event = adapter._build_slash_event(interaction, "/retry")

        assert event.channel_cwd is None


@pytest.mark.asyncio
async def test_run_agent_wires_channel_cwd_to_session_and_terminal(monkeypatch, tmp_path):
    _install_fake_agent(monkeypatch)
    runner = _make_runner()

    project_dir = tmp_path / "project"
    project_dir.mkdir()

    (tmp_path / "config.yaml").write_text("agent:\n  system_prompt: Global prompt\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
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

    import hermes_cli.tools_config as tools_config

    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})

    import tools.terminal_tool as terminal_tool

    registered = {}

    def _capture_override(task_id, overrides):
        registered[task_id] = overrides

    monkeypatch.setattr(terminal_tool, "register_task_env_overrides", _capture_override)

    _CapturingAgent.last_init = None
    result = await runner._run_agent(
        message="hi",
        context_prompt="Context prompt",
        history=[],
        source=_make_source(),
        session_id="session-1",
        session_key="agent:main:discord:thread:12345",
        channel_cwd=str(project_dir),
    )

    assert result["final_response"] == "ok"
    assert _CapturingAgent.last_init["session_cwd"] == str(project_dir)
    assert registered == {"session-1": {"cwd": str(project_dir)}}


@pytest.mark.asyncio
async def test_run_agent_without_channel_cwd_registers_nothing(monkeypatch, tmp_path):
    _install_fake_agent(monkeypatch)
    runner = _make_runner()

    (tmp_path / "config.yaml").write_text("agent:\n  system_prompt: Global prompt\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "gpt-5.4")
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

    import hermes_cli.tools_config as tools_config

    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})

    import tools.terminal_tool as terminal_tool

    registered = {}

    def _capture_override(task_id, overrides):
        registered[task_id] = overrides

    monkeypatch.setattr(terminal_tool, "register_task_env_overrides", _capture_override)

    _CapturingAgent.last_init = None
    result = await runner._run_agent(
        message="hi",
        context_prompt="Context prompt",
        history=[],
        source=_make_source(),
        session_id="session-1",
        session_key="agent:main:discord:thread:12345",
    )

    assert result["final_response"] == "ok"
    assert _CapturingAgent.last_init["session_cwd"] is None
    assert registered == {}


@pytest.mark.asyncio
async def test_retry_preserves_channel_cwd(tmp_path):
    """/retry must carry channel_cwd onto the rebuilt event, not just the prompt."""
    from gateway.slash_commands import GatewaySlashCommandsMixin

    handler = object.__new__(GatewaySlashCommandsMixin)
    session_entry = SimpleNamespace(session_id="s1", last_prompt_tokens=5)
    handler.async_session_store = SimpleNamespace(
        get_or_create_session=AsyncMock(return_value=session_entry),
        load_transcript=AsyncMock(return_value=[{"role": "user", "content": "hello"}]),
        rewrite_transcript=AsyncMock(),
    )

    captured = {}

    async def _capture(evt):
        captured["event"] = evt
        return "ok"

    handler._handle_message = _capture

    event = MessageEvent(
        text="/retry",
        message_type=MessageType.TEXT,
        source=_make_source(),
        channel_prompt="Research mode",
        channel_cwd=str(tmp_path),
    )

    result = await handler._handle_retry_command(event)

    assert result == "ok"
    assert captured["event"].channel_cwd == str(tmp_path)
    assert captured["event"].channel_prompt == "Research mode"
    assert captured["event"].text == "hello"
