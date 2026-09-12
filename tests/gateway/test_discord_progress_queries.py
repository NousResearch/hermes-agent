"""Discord pre-router contract for read-only project progress recall."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


@pytest.fixture
def adapter(monkeypatch):
    import plugins.platforms.discord.adapter as discord_platform
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setattr(discord_platform.discord, "DMChannel", type("DMChannel", (), {}), raising=False)
    config = PlatformConfig(
        enabled=True,
        token="fake-token",
        extra={
            "progress_queries": {
                "enabled": True,
                "board": "project-burndown",
            }
        },
    )
    value = DiscordAdapter(config)
    value._client = SimpleNamespace(user=SimpleNamespace(id=999))
    value.send = AsyncMock()
    return value


def _event(text="How did the burndown go and what else do we need to do?"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        message_id="message-1",
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="project-updates",
            chat_type="group",
            user_id="operator-1",
        ),
    )


def test_progress_pre_router_answers_before_model_specialist_classifier(adapter, monkeypatch):
    from gateway.progress_queries import ProgressQueryResult

    captured = {}

    def fake_resolve(request, *, source, board):
        captured.update(request=request, source=source, board=board)
        return ProgressQueryResult(True, "Burndown: 1 completed; next: acceptance.", "resolved")

    monkeypatch.setattr("gateway.progress_queries.resolve_progress_query", fake_resolve)
    handled = asyncio.run(adapter._maybe_answer_progress_event(_event()))

    assert handled is True
    assert captured["request"] == "How did the burndown go and what else do we need to do?"
    assert captured["source"].platform == "discord"
    assert captured["source"].chat_id == "project-updates"
    assert not hasattr(captured["source"], "session_id")
    assert captured["board"] == "project-burndown"
    adapter.send.assert_awaited_once_with(
        "project-updates", content="Burndown: 1 completed; next: acceptance.", reply_to="message-1"
    )


@pytest.mark.parametrize("reason", ["unavailable", "no_match"])
def test_unhandled_progress_lookup_falls_through_without_send_or_model(
    adapter, monkeypatch, reason
):
    from gateway.progress_queries import ProgressQueryResult

    monkeypatch.setattr(
        "gateway.progress_queries.resolve_progress_query",
        lambda *args, **kwargs: ProgressQueryResult(False, "", reason),
    )

    handled = asyncio.run(adapter._maybe_answer_progress_event(_event()))

    assert handled is False
    adapter.send.assert_not_awaited()


def test_progress_query_configuration_requires_nonempty_explicit_board(adapter):
    settings = adapter._progress_query_settings()

    assert settings["board"] == "project-burndown"

    adapter.config.extra["progress_queries"]["board"] = ""
    assert adapter._progress_query_settings()["enabled"] is False


def test_config_bridges_discord_progress_queries(monkeypatch, tmp_path):
    from gateway.config import load_gateway_config

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "discord:\n"
        "  progress_queries:\n"
        "    enabled: true\n"
        "    board: project-status\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    config = load_gateway_config()

    discord_extra = config.platforms[Platform.DISCORD].extra
    assert discord_extra["progress_queries"] == {
        "enabled": True,
        "board": "project-status",
    }
