import asyncio
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


@dataclass
class _FakeSessionEntry:
    session_id: str = "sess-1"
    session_key: str = "discord:dm:u1"
    created_at: datetime = datetime(2024, 1, 1)
    updated_at: datetime = datetime(2024, 1, 1)
    was_auto_reset: bool = False


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id="chat-1",
        chat_type="dm",
        user_id="u1",
        user_name="tester",
    )


def test_load_auto_background_config_reads_config_and_env(monkeypatch, tmp_path):
    import gateway.run as gw

    (tmp_path / "config.yaml").write_text(
        "agent:\n"
        "  auto_background:\n"
        "    enabled: true\n"
        "    threshold_seconds: 15\n"
        "    min_words: 7\n"
        "    min_chars: 80\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gw, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_AUTO_BACKGROUND_MIN_CHARS", "42")

    cfg = GatewayRunner._load_auto_background_config()

    assert cfg == {
        "enabled": True,
        "threshold_seconds": 15.0,
        "min_words": 7,
        "min_chars": 42,
    }


def test_load_auto_background_config_defaults_to_enabled(monkeypatch, tmp_path):
    import gateway.run as gw

    monkeypatch.setattr(gw, "_hermes_home", tmp_path)

    cfg = GatewayRunner._load_auto_background_config()

    assert cfg == {
        "enabled": True,
        "threshold_seconds": 10.0,
        "min_words": 6,
        "min_chars": 120,
    }


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("你好", False),
        ("帮我调试这个 Python 报错，定位原因并运行相关测试", True),
        ("Please investigate this failing CI job, identify the regression, and run the relevant tests.", True),
    ],
)
def test_should_auto_background_message_uses_actionable_heuristics(text, expected):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._auto_background_config = {
        "enabled": True,
        "threshold_seconds": 10.0,
        "min_words": 6,
        "min_chars": 120,
    }

    assert runner._should_auto_background_message(text) is expected


@pytest.mark.asyncio
async def test_handle_message_with_agent_short_circuits_to_background(monkeypatch, tmp_path):
    import gateway.run as gw

    (tmp_path / "config.yaml").write_text(
        "agent:\n  auto_background:\n    enabled: true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gw, "_hermes_home", tmp_path)

    runner = GatewayRunner(GatewayConfig())
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner.session_store.get_or_create_session = Mock(return_value=_FakeSessionEntry())
    runner.session_store.load_transcript = Mock(return_value=[])
    runner._set_session_env = lambda _context: {}
    runner._clear_session_env = lambda _tokens: None
    runner._prepare_inbound_message_text = AsyncMock(
        return_value="帮我调试这个 Python 报错，定位原因并运行相关测试"
    )
    runner._start_background_task = AsyncMock(return_value="AUTO-BG")
    runner._run_agent = AsyncMock(side_effect=AssertionError("should not run inline agent"))

    monkeypatch.setattr(gw, "build_session_context", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(gw, "build_session_context_prompt", lambda *_args, **_kwargs: "ctx")

    event = MessageEvent(text="帮我调试这个 Python 报错", source=_make_source(), message_id="m1")
    result = await runner._handle_message_with_agent(event, event.source, "quick-1")

    assert result == "AUTO-BG"
    runner._start_background_task.assert_awaited_once()
    runner._run_agent.assert_not_called()
    runner.hooks.emit.assert_awaited_once_with(
        "session:start",
        {
            "platform": "discord",
            "user_id": "u1",
            "session_id": "sess-1",
            "session_key": "discord:dm:u1",
        },
    )
