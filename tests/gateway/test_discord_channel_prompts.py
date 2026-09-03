"""Tests for Discord channel_prompts resolution and injection."""

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
from gateway.platforms.base import MessageEvent
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


class TestResolveChannelPrompts:
    def test_no_prompt_returns_none(self):
        adapter = _make_adapter()
        assert adapter._resolve_channel_prompt("123") is None

    def test_match_by_channel_id(self):
        adapter = _make_adapter()
        adapter.config.extra = {"channel_prompts": {"100": "Research mode"}}
        assert adapter._resolve_channel_prompt("100") == "Research mode"

    def test_numeric_yaml_keys_normalized_at_config_load(self):
        """Numeric YAML keys are normalized to strings by config bridging.

        The resolver itself expects string keys (config.py handles normalization),
        so raw numeric keys will not match — this is intentional.
        """
        adapter = _make_adapter()
        # Simulates post-bridging state: keys are already strings
        adapter.config.extra = {"channel_prompts": {"100": "Research mode"}}
        assert adapter._resolve_channel_prompt("100") == "Research mode"
        # Pre-bridging numeric key would not match (bridging is responsible)
        adapter.config.extra = {"channel_prompts": {100: "Research mode"}}
        assert adapter._resolve_channel_prompt("100") is None


@pytest.mark.asyncio
async def test_retry_preserves_channel_prompt(monkeypatch):
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
        channel_prompt="Channel prompt",
    )

    result = await runner._handle_retry_command(event)

    assert result == "ok"
    retried_event = runner._handle_message.await_args.args[0]
    assert retried_event.channel_prompt == "Channel prompt"


