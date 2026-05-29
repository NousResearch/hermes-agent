import asyncio
import logging
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.relationship_profiles import canonical_identity_key
import gateway.run as gateway_run
from gateway.session import SessionSource


class _CapturingAgent:
    last_init = None

    def __init__(self, *args, **kwargs):
        type(self).last_init = dict(kwargs)
        self.tools = []

    def run_conversation(self, user_message: str, conversation_history=None, task_id=None):
        return {"final_response": "ok", "messages": [], "api_calls": 1}


def _bare_runner() -> gateway_run.GatewayRunner:
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._session_reasoning_overrides = {}
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=[])
    runner._session_db = None
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    return runner


def _patch_agent_runtime(monkeypatch, tmp_path, config_yaml: str) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(config_yaml, encoding="utf-8")
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


def test_calendar_home_source_uses_home_identity_and_thread():
    runner = _bare_runner()
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                home_channel=HomeChannel(
                    Platform.TELEGRAM,
                    "888933588",
                    "Gwenael",
                    thread_id="42",
                ),
            )
        }
    )
    adapter = SimpleNamespace()
    runner.adapters[Platform.TELEGRAM] = adapter

    resolved_adapter, source = runner._calendar_home_source()

    assert resolved_adapter is adapter
    assert source.chat_id == "888933588"
    assert source.thread_id == "42"
    assert source.user_id == "888933588"
    assert source.user_id != "system:calendar"
    assert source.internal_kind == "calendar_wakeup"
    assert canonical_identity_key(source) == "telegram:user:888933588"


def test_calendar_dispatch_routes_to_configured_chat_and_thread():
    runner = _bare_runner()
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                home_channel=HomeChannel(
                    Platform.TELEGRAM,
                    "888933588",
                    "Gwenael",
                    thread_id="42",
                ),
            )
        }
    )
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner.adapters[Platform.TELEGRAM] = adapter

    asyncio.run(runner._dispatch_calendar_wakeup({"id": 7, "title": "Wake"}))

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.internal is True
    assert event.source.chat_id == "888933588"
    assert event.source.thread_id == "42"
    assert canonical_identity_key(event.source) == "telegram:user:888933588"


def test_normal_telegram_message_keeps_platform_toolsets(tmp_path, monkeypatch):
    _patch_agent_runtime(
        monkeypatch,
        tmp_path,
        "platform_toolsets:\n  telegram: [web]\n",
    )
    runner = _bare_runner()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="888933588",
        chat_type="dm",
        user_id="888933588",
    )

    result = asyncio.run(
        runner._run_agent(
            message="ping",
            context_prompt="",
            history=[],
            source=source,
            session_id="session-1",
            session_key="agent:main:telegram:dm",
        )
    )

    assert result["final_response"] == "ok"
    enabled = set(_CapturingAgent.last_init["enabled_toolsets"])
    assert "web" in enabled
    assert "terminal" not in enabled
    assert "memory" not in enabled
    assert "session_search" not in enabled


def test_calendar_wakeup_gets_required_toolset_overlay(tmp_path, monkeypatch):
    _patch_agent_runtime(
        monkeypatch,
        tmp_path,
        "platform_toolsets:\n  telegram: [web]\n",
    )
    runner = _bare_runner()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="888933588",
        chat_type="dm",
        user_id="888933588",
        internal_kind="calendar_wakeup",
    )

    asyncio.run(
        runner._run_agent(
            message="ping",
            context_prompt="",
            history=[],
            source=source,
            session_id="session-1",
            session_key="agent:main:telegram:dm",
        )
    )

    enabled = set(_CapturingAgent.last_init["enabled_toolsets"])
    assert {
        "web",
        "vision",
        "file",
        "terminal",
        "memory",
        "calendar",
        "session_search",
    }.issubset(enabled)


def test_calendar_wakeup_overlay_honors_disabled_toolsets(tmp_path, monkeypatch):
    _patch_agent_runtime(
        monkeypatch,
        tmp_path,
        "platform_toolsets:\n"
        "  telegram: [web]\n"
        "agent:\n"
        "  disabled_toolsets: [terminal, memory]\n",
    )
    runner = _bare_runner()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="888933588",
        chat_type="dm",
        user_id="888933588",
        internal_kind="calendar_wakeup",
    )

    asyncio.run(
        runner._run_agent(
            message="ping",
            context_prompt="",
            history=[],
            source=source,
            session_id="session-1",
            session_key="agent:main:telegram:dm",
        )
    )

    enabled = set(_CapturingAgent.last_init["enabled_toolsets"])
    assert "calendar" in enabled
    assert "terminal" not in enabled
    assert "memory" not in enabled


def test_calendar_wakeup_config_can_narrow_overlay_but_calendar_is_forced():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="888933588",
        user_id="888933588",
        internal_kind="calendar_wakeup",
    )
    enabled = gateway_run._resolve_gateway_enabled_toolsets(
        {
            "platform_toolsets": {"telegram": ["web"]},
            "calendar": {"wakeup_toolsets": ["memory"]},
        },
        "telegram",
        source,
    )

    assert "memory" in enabled
    assert "calendar" in enabled
    assert "terminal" not in enabled


def test_calendar_wakeup_watcher_logs_tick_exceptions_at_warning(monkeypatch, caplog):
    runner = _bare_runner()
    runner._running = True
    monkeypatch.setattr(runner, "_calendar_wakeup_interval_seconds", lambda: 1)

    from hermes_cli import calendar_db

    monkeypatch.setattr(calendar_db, "requeue_stale_firing", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    sleep_calls = 0

    async def fake_sleep(_seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 1:
            raise asyncio.CancelledError

    monkeypatch.setattr(gateway_run.asyncio, "sleep", fake_sleep)

    with caplog.at_level(logging.WARNING, logger=gateway_run.logger.name):
        try:
            asyncio.run(runner._calendar_wakeup_watcher())
        except asyncio.CancelledError:
            pass

    assert "Calendar wakeup watcher tick error: boom" in caplog.text
