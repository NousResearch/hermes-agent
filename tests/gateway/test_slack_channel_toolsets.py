"""Tests for Slack channel_toolsets resolution and gateway override.

Mirrors the shape of test_slack_channel_skills.py / test_discord_channel_prompts.py.

A Slack workspace may run domain conversations across many channels while
needing to lock down dangerous tools (terminal, file-write, etc.) on
business-facing channels. ``slack.channel_toolsets`` provides a per-channel
hard gate that overrides ``platform_toolsets.slack`` for the matching channel.
"""

from __future__ import annotations

import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Resolver-only tests (no Slack SDK required).
# ---------------------------------------------------------------------------


def _make_adapter(extra=None):
    """Create a minimal SlackAdapter stub with the given ``config.extra``."""
    from gateway.platforms.slack import SlackAdapter

    adapter = object.__new__(SlackAdapter)
    adapter.config = MagicMock()
    adapter.config.extra = extra or {}
    return adapter


def _resolve(adapter, channel_id, parent_id=None):
    from gateway.platforms.base import resolve_channel_toolsets

    return resolve_channel_toolsets(adapter.config.extra, channel_id, parent_id)


class TestResolveChannelToolsets:
    def test_no_mapping_returns_none(self):
        adapter = _make_adapter()
        assert _resolve(adapter, "C01SAFE") is None

    def test_match_by_channel_id(self):
        adapter = _make_adapter({
            "channel_toolsets": {
                "C01SAFE": ["web", "skills", "todo"],
            }
        })
        assert _resolve(adapter, "C01SAFE") == ["web", "skills", "todo"]

    def test_no_match_returns_none(self):
        adapter = _make_adapter({
            "channel_toolsets": {
                "C01SAFE": ["web", "skills"],
            }
        })
        assert _resolve(adapter, "C99OTHER") is None

    def test_match_by_parent_id_for_thread(self):
        """Thread replies inherit the parent channel's binding."""
        adapter = _make_adapter({
            "channel_toolsets": {
                "C0PARENT": ["web", "skills"],
            }
        })
        assert _resolve(adapter, "thread-ts-123", parent_id="C0PARENT") == [
            "web", "skills",
        ]

    def test_exact_channel_overrides_parent(self):
        adapter = _make_adapter({
            "channel_toolsets": {
                "C0THREAD": ["web"],
                "C0PARENT": ["terminal", "file"],
            }
        })
        assert _resolve(adapter, "C0THREAD", parent_id="C0PARENT") == ["web"]

    def test_string_value_returns_single_element_list(self):
        adapter = _make_adapter({
            "channel_toolsets": {
                "C01SAFE": "skills",
            }
        })
        assert _resolve(adapter, "C01SAFE") == ["skills"]

    def test_dedup_preserves_order(self):
        adapter = _make_adapter({
            "channel_toolsets": {
                "C01SAFE": ["web", "skills", "web", "todo", "skills"],
            }
        })
        assert _resolve(adapter, "C01SAFE") == ["web", "skills", "todo"]

    def test_empty_list_returns_empty_override(self):
        adapter = _make_adapter({
            "channel_toolsets": {
                "C01SAFE": [],
            }
        })
        assert _resolve(adapter, "C01SAFE") == []

    def test_empty_string_returns_empty_override(self):
        adapter = _make_adapter({
            "channel_toolsets": {
                "C01SAFE": "",
            }
        })
        assert _resolve(adapter, "C01SAFE") == []

    def test_blank_entries_filtered(self):
        adapter = _make_adapter({
            "channel_toolsets": {
                "C01SAFE": ["web", "  ", "", "skills"],
            }
        })
        assert _resolve(adapter, "C01SAFE") == ["web", "skills"]

    def test_non_dict_mapping_returns_none(self):
        adapter = _make_adapter({"channel_toolsets": ["not-a-dict"]})
        assert _resolve(adapter, "C01SAFE") is None

    def test_non_string_entries_skipped(self):
        adapter = _make_adapter({
            "channel_toolsets": {
                "C01SAFE": ["web", 42, None, "skills"],
            }
        })
        assert _resolve(adapter, "C01SAFE") == ["web", "skills"]


class TestSlackMessageEventChannelToolsets:
    """MessageEvent carries an optional channel_toolsets override."""

    def test_message_event_carries_channel_toolsets(self):
        from gateway.platforms.base import (
            MessageEvent,
            MessageType,
            Platform,
            SessionSource,
            resolve_channel_toolsets,
        )

        config_extra = {
            "channel_toolsets": {
                "C01SAFE": ["web", "skills", "todo"],
            }
        }
        ts = resolve_channel_toolsets(config_extra, "C01SAFE")
        source = SessionSource(
            platform=Platform.SLACK,
            chat_id="C01SAFE",
            chat_name="biz",
            chat_type="group",
            user_id="U0ABC",
            user_name="Biz",
        )
        event = MessageEvent(
            text="status?",
            message_type=MessageType.TEXT,
            source=source,
            raw_message={},
            message_id="123.456",
            channel_toolsets=ts,
        )
        assert event.channel_toolsets == ["web", "skills", "todo"]

    def test_message_event_defaults_channel_toolsets_to_none(self):
        from gateway.platforms.base import MessageEvent, MessageType

        event = MessageEvent(text="hi", message_type=MessageType.TEXT)
        assert event.channel_toolsets is None


# ---------------------------------------------------------------------------
# Slack adapter integration: verify the adapter resolves and attaches.
# ---------------------------------------------------------------------------


class TestSlackAdapterAttachesChannelToolsets:
    """The Slack adapter resolves channel_toolsets when building MessageEvent."""

    def test_resolve_uses_config_extra(self):
        """Adapter-level resolution mirrors the auto_skill / channel_prompt path."""
        adapter = _make_adapter({
            "channel_toolsets": {
                "C0OPS": ["terminal", "file", "mcp", "skills"],
            }
        })
        # Adapter doesn't need a private resolver — the standalone helper is
        # the contract; the slack handler imports and uses it directly. We
        # assert the helper returns what the handler will attach.
        from gateway.platforms.base import resolve_channel_toolsets

        assert resolve_channel_toolsets(adapter.config.extra, "C0OPS") == [
            "terminal", "file", "mcp", "skills",
        ]
        assert resolve_channel_toolsets(adapter.config.extra, "C0SAFE") is None


    @pytest.mark.asyncio
    async def test_slash_question_attaches_channel_toolsets(self):
        """Legacy /hermes questions must not bypass the channel hard gate."""
        from gateway.platforms.base import MessageType, Platform

        adapter = _make_adapter({
            "channel_toolsets": {
                "C0SAFE": ["web", "skills"],
            }
        })
        adapter.platform = Platform.SLACK
        adapter._channel_team = {}
        adapter._slash_command_contexts = {}
        adapter.handle_message = AsyncMock()

        await adapter._handle_slash_command({
            "command": "/hermes",
            "text": "summarize this channel",
            "user_id": "U123",
            "channel_id": "C0SAFE",
            "team_id": "T123",
        })

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.message_type == MessageType.TEXT
        assert event.text == "summarize this channel"
        assert event.channel_toolsets == ["web", "skills"]


# ---------------------------------------------------------------------------
# Gateway run wiring: verify _run_agent honors event.channel_toolsets.
# ---------------------------------------------------------------------------


from gateway.config import Platform  # noqa: E402
from gateway.platforms.base import MessageEvent  # noqa: E402
from gateway.session import SessionSource  # noqa: E402


def _import_gateway_run(monkeypatch):
    """Import gateway.run without leaking real API-server env into later tests.

    gateway.run loads Hermes' .env at import time.  In the developer's real
    environment that can set API_SERVER_KEY/API_SERVER_HOST, and those process
    env vars would make unrelated gateway config tests think api_server is
    enabled.  Keep this file's gateway-run wiring tests hermetic.
    """
    for name in (
        "API_SERVER_ENABLED",
        "API_SERVER_KEY",
        "API_SERVER_CORS_ORIGINS",
        "API_SERVER_PORT",
        "API_SERVER_HOST",
        "API_SERVER_MODEL_NAME",
    ):
        monkeypatch.delenv(name, raising=False)
    import gateway.run as gateway_run

    return gateway_run


class _CapturingAgent:
    """Captures the kwargs the gateway passed to AIAgent for assertion."""

    last_init = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []

    def run_conversation(
        self,
        user_message,
        conversation_history=None,
        task_id=None,
        persist_user_message=None,
    ):
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


def _make_runner(gateway_run):
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


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.SLACK,
        chat_id="C01SAFE",
        chat_type="group",
        user_id="U0ABC",
    )


def test_runner_resolves_channel_toolsets_for_synthetic_slack_events(monkeypatch):
    """Runner-created Slack events must get the same hard gate as adapter events."""
    gateway_run = _import_gateway_run(monkeypatch)
    runner = _make_runner(gateway_run)
    runner.adapters = {
        Platform.SLACK: SimpleNamespace(
            config=SimpleNamespace(
                extra={
                    "channel_toolsets": {
                        "C01SAFE": [],
                        "C02OPS": ["terminal", "skills"],
                    }
                }
            )
        )
    }

    assert runner._resolve_channel_toolsets_for_source(_make_source()) == []
    assert runner._resolve_channel_toolsets_for_source(
        SessionSource(
            platform=Platform.SLACK,
            chat_id="C02OPS",
            chat_type="group",
            user_id="U0ABC",
        )
    ) == ["terminal", "skills"]
    assert runner._resolve_channel_toolsets_for_source(
        SessionSource(platform=Platform.DISCORD, chat_id="C01SAFE", chat_type="group")
    ) is None


@pytest.mark.asyncio
async def test_run_agent_applies_channel_toolsets_override(monkeypatch, tmp_path):
    """When event.channel_toolsets is set, AIAgent.enabled_toolsets reflects it.

    The override must replace the platform-level toolset list (sourced from
    ``_get_platform_tools(user_config, platform_key)``) so dangerous tools
    can be hard-gated per channel.
    """
    gateway_run = _import_gateway_run(monkeypatch)
    _install_fake_agent(monkeypatch)
    runner = _make_runner(gateway_run)

    (tmp_path / "config.yaml").write_text(
        "agent:\n  system_prompt: Global prompt\n",
        encoding="utf-8",
    )
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

    # Fake resolver: when an override is set, echo it; otherwise return the
    # platform default. This is the behavior _get_platform_tools delivers in
    # production — channel_toolsets goes into platform_toolsets[platform_key]
    # before resolution, and the resolver returns it.
    def fake_get_platform_tools(user_config, platform_key, **_kwargs):
        platform_toolsets = user_config.get("platform_toolsets") or {}
        toolset_names = platform_toolsets.get(platform_key)
        if toolset_names:
            return set(toolset_names)
        return {"terminal", "file", "core"}

    monkeypatch.setattr(tools_config, "_get_platform_tools", fake_get_platform_tools)

    _CapturingAgent.last_init = None
    event = MessageEvent(
        text="hi",
        source=_make_source(),
        message_id="m1",
        channel_toolsets=["web", "skills", "todo"],
    )

    result = await runner._run_agent(
        message="hi",
        context_prompt="Context prompt",
        history=[],
        source=_make_source(),
        session_id="session-1",
        session_key="agent:main:slack:group:C01SAFE",
        channel_toolsets=event.channel_toolsets,
    )

    assert result["final_response"] == "ok"
    assert _CapturingAgent.last_init is not None
    assert _CapturingAgent.last_init["enabled_toolsets"] == ["skills", "todo", "web"]


@pytest.mark.asyncio
async def test_run_agent_channel_toolsets_exclude_default_mcp_servers(monkeypatch, tmp_path):
    """A safe channel override must not inherit globally enabled MCP servers."""
    gateway_run = _import_gateway_run(monkeypatch)
    _install_fake_agent(monkeypatch)
    runner = _make_runner(gateway_run)

    (tmp_path / "config.yaml").write_text(
        "agent:\n  system_prompt: Global prompt\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "mcp_servers": {
                "ops-mcp": {"command": "python", "args": ["server.py"]},
            }
        },
    )
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

    _CapturingAgent.last_init = None
    result = await runner._run_agent(
        message="hi",
        context_prompt="Context prompt",
        history=[],
        source=_make_source(),
        session_id="session-1",
        session_key="agent:main:slack:group:C01SAFE",
        channel_toolsets=["web", "skills"],
    )

    assert result["final_response"] == "ok"
    assert _CapturingAgent.last_init is not None
    assert _CapturingAgent.last_init["enabled_toolsets"] == ["skills", "web"]
    assert "ops-mcp" not in _CapturingAgent.last_init["enabled_toolsets"]


@pytest.mark.asyncio
async def test_run_agent_empty_channel_toolsets_stays_empty(monkeypatch, tmp_path):
    """An explicit empty channel override means no tools, not platform defaults."""
    gateway_run = _import_gateway_run(monkeypatch)
    _install_fake_agent(monkeypatch)
    runner = _make_runner(gateway_run)

    (tmp_path / "config.yaml").write_text(
        "agent:\n  system_prompt: Global prompt\n",
        encoding="utf-8",
    )
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

    _CapturingAgent.last_init = None
    result = await runner._run_agent(
        message="hi",
        context_prompt="Context prompt",
        history=[],
        source=_make_source(),
        session_id="session-1",
        session_key="agent:main:slack:group:C01SAFE",
        channel_toolsets=[],
    )

    assert result["final_response"] == "ok"
    assert _CapturingAgent.last_init is not None
    assert _CapturingAgent.last_init["enabled_toolsets"] == []


@pytest.mark.asyncio
async def test_run_agent_passes_channel_toolsets_to_proxy(monkeypatch):
    """Proxy mode must forward the hard tool gate to the remote API server."""
    gateway_run = _import_gateway_run(monkeypatch)
    runner = _make_runner(gateway_run)
    runner._get_proxy_url = lambda: "http://127.0.0.1:12345"
    captured = {}

    async def fake_proxy(**kwargs):
        captured.update(kwargs)
        return {"final_response": "ok", "messages": [], "api_calls": 0}

    monkeypatch.setattr(runner, "_run_agent_via_proxy", fake_proxy)

    result = await runner._run_agent(
        message="hi",
        context_prompt="Context prompt",
        history=[],
        source=_make_source(),
        session_id="session-1",
        session_key="agent:main:slack:group:C01SAFE",
        channel_toolsets=["web", "skills"],
    )

    assert result["final_response"] == "ok"
    assert captured["channel_toolsets"] == ["web", "skills"]

    captured.clear()
    result = await runner._run_agent(
        message="hi",
        context_prompt="Context prompt",
        history=[],
        source=_make_source(),
        session_id="session-1",
        session_key="agent:main:slack:group:C01SAFE",
        channel_toolsets=[],
    )
    assert result["final_response"] == "ok"
    assert captured["channel_toolsets"] == []


@pytest.mark.asyncio
async def test_goal_continuation_preserves_slack_channel_toolsets(monkeypatch):
    """Automatic goal-continuation turns must keep the channel hard gate."""
    gateway_run = _import_gateway_run(monkeypatch)
    runner = _make_runner(gateway_run)
    source = _make_source()
    adapter = SimpleNamespace(
        _pending_messages={},
        config=SimpleNamespace(
            extra={"channel_toolsets": {"C01SAFE": ["web", "skills"]}}
        ),
    )
    runner.adapters = {Platform.SLACK: adapter}
    runner._session_key_for_source = lambda _source: "goal-key"
    runner._goal_max_turns_from_config = lambda: 5
    runner._defer_goal_status_notice_after_delivery = AsyncMock()

    fake_goals = types.ModuleType("hermes_cli.goals")

    class FakeGoalManager:
        def __init__(self, session_id, default_max_turns):
            self.session_id = session_id
            self.default_max_turns = default_max_turns

        def is_active(self):
            return True

        def evaluate_after_turn(self, final_response, user_initiated=True):
            return {
                "message": "continue",
                "should_continue": True,
                "continuation_prompt": "next step",
            }

    fake_goals.GoalManager = FakeGoalManager
    monkeypatch.setitem(sys.modules, "hermes_cli.goals", fake_goals)

    await runner._post_turn_goal_continuation(
        session_entry=SimpleNamespace(session_id="session-1"),
        source=source,
        final_response="partial answer",
    )

    pending = adapter._pending_messages["goal-key"]
    assert pending.text == "next step"
    assert pending.channel_toolsets == ["web", "skills"]


@pytest.mark.asyncio
async def test_run_agent_without_channel_toolsets_uses_platform_default(
    monkeypatch, tmp_path,
):
    """When no channel override is provided, the platform default is used."""
    gateway_run = _import_gateway_run(monkeypatch)
    _install_fake_agent(monkeypatch)
    runner = _make_runner(gateway_run)

    (tmp_path / "config.yaml").write_text(
        "agent:\n  system_prompt: Global prompt\n",
        encoding="utf-8",
    )
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

    def fake_get_platform_tools(user_config, platform_key, **_kwargs):
        platform_toolsets = user_config.get("platform_toolsets") or {}
        toolset_names = platform_toolsets.get(platform_key)
        if toolset_names:
            return set(toolset_names)
        return {"terminal", "file", "core"}

    monkeypatch.setattr(tools_config, "_get_platform_tools", fake_get_platform_tools)

    _CapturingAgent.last_init = None

    result = await runner._run_agent(
        message="hi",
        context_prompt="Context prompt",
        history=[],
        source=_make_source(),
        session_id="session-1",
        session_key="agent:main:slack:group:C01SAFE",
        channel_toolsets=None,
    )

    assert result["final_response"] == "ok"
    assert _CapturingAgent.last_init is not None
    assert _CapturingAgent.last_init["enabled_toolsets"] == ["core", "file", "terminal"]


# ---------------------------------------------------------------------------
# Config bridge: ensure slack.channel_toolsets reaches PlatformConfig.extra.
# ---------------------------------------------------------------------------


class TestSlackChannelToolsetsConfigBridge:
    """gateway/config.py must bridge slack.channel_toolsets into extra."""

    def test_bridges_slack_channel_toolsets_from_config_yaml(self, tmp_path, monkeypatch):
        from gateway.config import Platform, load_gateway_config

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "slack:\n"
            "  channel_toolsets:\n"
            '    "C01SAFE": [web, skills, todo]\n'
            '    "C02OPS": [terminal, file, mcp, skills]\n',
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert config.platforms[Platform.SLACK].extra["channel_toolsets"] == {
            "C01SAFE": ["web", "skills", "todo"],
            "C02OPS": ["terminal", "file", "mcp", "skills"],
        }

    def test_numeric_yaml_keys_normalized_to_string(self, tmp_path, monkeypatch):
        from gateway.config import Platform, load_gateway_config

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "slack:\n"
            "  channel_toolsets:\n"
            "    123: [web]\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert config.platforms[Platform.SLACK].extra["channel_toolsets"] == {
            "123": ["web"],
        }
