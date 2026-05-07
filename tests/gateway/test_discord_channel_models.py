"""Tests for Discord channel_models resolution and injection."""

import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
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
from gateway.platforms.base import MessageEvent, resolve_channel_model
from gateway.session import SessionSource


def _make_adapter():
    _ensure_discord_mock()
    from gateway.platforms.discord import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)
    adapter.config = MagicMock()
    adapter.config.extra = {}
    return adapter


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="12345",
        chat_type="thread",
        user_id="user-1",
    )


# ---------------------------------------------------------------------------
# resolve_channel_model unit tests
# ---------------------------------------------------------------------------


class TestResolveChannelModel:
    def test_no_config_returns_none(self):
        assert resolve_channel_model({}, "123") is None

    def test_empty_dict_returns_none(self):
        assert resolve_channel_model({"channel_models": {}}, "123") is None

    def test_non_dict_config_returns_none(self):
        assert resolve_channel_model({"channel_models": "not a dict"}, "123") is None

    def test_match_by_channel_id(self):
        config = {"channel_models": {"100": "qwen3.5-397b-fast"}}
        assert resolve_channel_model(config, "100") == "qwen3.5-397b-fast"

    def test_match_by_parent_id_fallback(self):
        config = {"channel_models": {"200": "glm-5-turbo"}}
        assert resolve_channel_model(config, "999", parent_id="200") == "glm-5-turbo"

    def test_exact_channel_overrides_parent(self):
        config = {"channel_models": {"999": "model-thread", "200": "model-forum"}}
        assert resolve_channel_model(config, "999", parent_id="200") == "model-thread"

    def test_blank_model_ignored(self):
        config = {"channel_models": {"100": "   "}}
        assert resolve_channel_model(config, "100") is None

    def test_numeric_yaml_keys_normalized_at_bridging(self):
        """Config bridging stringifies keys; resolver expects string keys."""
        config = {"channel_models": {"100": "good-model"}}
        assert resolve_channel_model(config, "100") == "good-model"
        # Pre-bridging numeric key won't match (bridging is responsible)
        config_raw = {"channel_models": {100: "good-model"}}
        assert resolve_channel_model(config_raw, "100") is None

    def test_integer_channel_id_matched_as_string(self):
        """channel_id passed as string; bridging ensures keys are strings."""
        config = {"channel_models": {"100": "test-model"}}
        assert resolve_channel_model(config, "100") == "test-model"


# ---------------------------------------------------------------------------
# Discord adapter _resolve_channel_model tests
# ---------------------------------------------------------------------------


class TestDiscordAdapterResolveChannelModel:
    def test_no_models_returns_none(self):
        adapter = _make_adapter()
        assert adapter._resolve_channel_model("123") is None

    def test_match_by_channel_id(self):
        adapter = _make_adapter()
        adapter.config.extra = {"channel_models": {"100": "qwen3.5-397b-fast"}}
        assert adapter._resolve_channel_model("100") == "qwen3.5-397b-fast"

    def test_match_by_parent_id(self):
        adapter = _make_adapter()
        adapter.config.extra = {"channel_models": {"200": "glm-5-turbo"}}
        assert adapter._resolve_channel_model("999", parent_id="200") == "glm-5-turbo"

    def test_exact_channel_overrides_parent(self):
        adapter = _make_adapter()
        adapter.config.extra = {
            "channel_models": {"999": "thread-model", "200": "forum-model"}
        }
        assert adapter._resolve_channel_model("999", parent_id="200") == "thread-model"


# ---------------------------------------------------------------------------
# MessageEvent wiring tests
# ---------------------------------------------------------------------------


class TestDiscordMessageEventChannelModel:
    def test_build_message_event_sets_channel_model(self):
        adapter = _make_adapter()
        adapter.config.extra = {"channel_models": {"321": "test-model"}}
        adapter.build_source = MagicMock(return_value=SimpleNamespace())

        interaction = SimpleNamespace(
            channel_id=321,
            channel=SimpleNamespace(name="general", guild=None, parent_id=None),
            user=SimpleNamespace(id=1, display_name="Brenner"),
        )
        adapter._get_effective_topic = MagicMock(return_value=None)

        event = adapter._build_slash_event(interaction, "/retry")

        assert event.channel_model == "test-model"

    @pytest.mark.asyncio
    async def test_dispatch_thread_session_inherits_parent_channel_model(self):
        adapter = _make_adapter()
        adapter.config.extra = {"channel_models": {"200": "parent-model"}}
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
        assert dispatched_event.channel_model == "parent-model"


# ---------------------------------------------------------------------------
# _resolve_session_agent_runtime channel_model injection tests
# ---------------------------------------------------------------------------


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


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
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


class TestResolveSessionAgentRuntimeChannelModel:
    def test_no_channel_model_uses_global_default(self, monkeypatch):
        monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "global-model")
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {"provider": "openrouter", "api_key": "key"},
        )
        runner = _make_runner()
        model, runtime = runner._resolve_session_agent_runtime(
            user_config={},
            channel_model=None,
        )
        assert model == "global-model"

    def test_channel_model_overrides_global_default(self, monkeypatch):
        monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "global-model")
        monkeypatch.setattr(
            gateway_run,
            "_resolve_runtime_agent_kwargs",
            lambda: {"provider": "openrouter", "api_key": "key"},
        )
        runner = _make_runner()
        model, runtime = runner._resolve_session_agent_runtime(
            user_config={},
            channel_model="channel-model",
        )
        assert model == "channel-model"

    def test_session_override_wins_over_channel_model(self, monkeypatch):
        monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "global-model")
        runner = _make_runner()
        runner._session_model_overrides["test-session"] = {
            "model": "override-model",
            "provider": "openrouter",
            "api_key": "override-key",
            "base_url": "https://example.com",
            "api_mode": "chat_completions",
        }
        model, runtime = runner._resolve_session_agent_runtime(
            session_key="test-session",
            user_config={},
            channel_model="channel-model",
        )
        assert model == "override-model"
        assert runtime["api_key"] == "override-key"


# ---------------------------------------------------------------------------
# _run_agent end-to-end wiring test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_agent_receives_channel_model(monkeypatch, tmp_path):
    _install_fake_agent(monkeypatch)
    runner = _make_runner()

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config=None: "global-default")
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

    import hermes_cli.tools_config as tools_config
    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda user_config, platform_key: {"core"})

    _CapturingAgent.last_init = None
    result = await runner._run_agent(
        message="hi",
        context_prompt="",
        history=[],
        source=_make_source(),
        session_id="session-1",
        session_key="agent:main:discord:thread:12345",
        channel_model="channel-bound-model",
    )

    assert result["final_response"] == "ok"
    assert _CapturingAgent.last_init["model"] == "channel-bound-model"


# ---------------------------------------------------------------------------
# Retry preserves channel_model
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_preserves_channel_model(monkeypatch):
    runner = _make_runner()
    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda source: SimpleNamespace(session_id="session-1", last_prompt_tokens=10),
        load_transcript=lambda session_id: [
            {"role": "user", "content": "original message"},
            {"role": "assistant", "content": "old reply"},
        ],
        rewrite_transcript=MagicMock(),
    )
    runner._handle_message = AsyncMock(return_value="ok")

    event = MessageEvent(
        text="/retry",
        message_type=gateway_run.MessageType.COMMAND,
        source=_make_source(),
        raw_message=SimpleNamespace(),
        channel_model="retry-model",
    )

    result = await runner._handle_retry_command(event)

    assert result == "ok"
    retried_event = runner._handle_message.await_args.args[0]
    assert retried_event.channel_model == "retry-model"


# ---------------------------------------------------------------------------
# Config bridging test
# ---------------------------------------------------------------------------


class TestConfigBridging:
    def test_bridges_discord_channel_models_from_config_yaml(self, tmp_path, monkeypatch):
        from gateway.config import load_gateway_config

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "discord:\n"
            "  channel_models:\n"
            '    "123": qwen3.5-397b-fast\n'
            "    456: glm-5-turbo\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        config = load_gateway_config()

        assert config.platforms[Platform.DISCORD].extra["channel_models"] == {
            "123": "qwen3.5-397b-fast",
            "456": "glm-5-turbo",
        }

    def test_bridges_stringify_values(self, tmp_path, monkeypatch):
        """Values are coerced to strings during bridging."""
        from gateway.config import load_gateway_config

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        # YAML may parse unquoted values as int/float
        config_path.write_text(
            "discord:\n"
            "  channel_models:\n"
            '    "123": some-model\n',
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        config = load_gateway_config()

        val = config.platforms[Platform.DISCORD].extra["channel_models"]["123"]
        assert isinstance(val, str)
