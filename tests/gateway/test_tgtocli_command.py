"""Tests for /tgtocli gateway slash command."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/tgtocli", platform=Platform.TELEGRAM):
    source = SessionSource(
        platform=platform,
        user_id="12345",
        chat_id="67890",
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner(session_id="20260519_101112_abcdef12", title="Phone work"):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SimpleNamespace(
        session_id=session_id,
        session_key="agent:main:telegram:dm:67890",
    )
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = title
    runner._session_db._get_session_rich_row.return_value = {
        "preview": "repair MLS strike list upload",
    }
    return runner


@pytest.mark.asyncio
async def test_tgtocli_rejects_non_telegram_sources():
    runner = _make_runner()
    result = await runner._handle_tgtocli_command(_make_event(platform=Platform.DISCORD))

    assert "/tgtocli" in result
    assert "Telegram" in result
    runner.session_store.get_or_create_session.assert_not_called()


@pytest.mark.asyncio
async def test_tgtocli_lists_human_targets_with_ids(monkeypatch):
    runner = _make_runner(session_id="20260519_101112_abcdef12")
    monkeypatch.setattr(
        runner,
        "_tgtocli_discover_targets",
        lambda: [
            {
                "kind": "tmux",
                "name": "work-cli",
                "pane_id": "%1",
                "cwd": "/repo",
                "summary": "MemoryMunch build (`20260519_090000_deadbeef`)",
            }
        ],
    )

    result = await runner._handle_tgtocli_command(_make_event())

    assert "Pick CLI target" in result
    assert "TG: Phone work (`20260519_101112_abcdef12`)" in result
    assert "1. work-cli pane `%1`" in result
    assert "MemoryMunch build (`20260519_090000_deadbeef`)" in result
    assert "2. New tmux CLI" in result
    assert "Reply: `/tgtocli 1`" in result


@pytest.mark.asyncio
async def test_tgtocli_sends_resume_to_selected_tmux(monkeypatch):
    runner = _make_runner(session_id="20260519_101112_abcdef12")
    monkeypatch.setattr(
        runner,
        "_tgtocli_discover_targets",
        lambda: [
            {
                "kind": "tmux",
                "name": "work-cli",
                "pane_id": "%1",
                "cwd": "/repo",
                "summary": "MemoryMunch build (`20260519_090000_deadbeef`)",
            }
        ],
    )

    with patch("gateway.run.subprocess.run") as run:
        run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
        result = await runner._handle_tgtocli_command(_make_event("/tgtocli 1"))

    assert "DONE" in result
    assert "Sent to `work-cli`" in result
    assert run.call_args.args[0] == [
        "tmux",
        "send-keys",
        "-t",
        "%1",
        "/resume 20260519_101112_abcdef12",
        "Enter",
    ]


@pytest.mark.asyncio
async def test_tgtocli_can_start_new_tmux(monkeypatch):
    runner = _make_runner(session_id="20260519_101112_abcdef12")
    monkeypatch.setattr(runner, "_tgtocli_discover_targets", lambda: [])

    with patch("gateway.run.shutil.which", return_value="/usr/bin/tmux"), patch(
        "gateway.run.subprocess.run"
    ) as run:
        run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
        result = await runner._handle_tgtocli_command(_make_event("/tgtocli 1"))

    assert "DONE" in result
    assert "Started new CLI" in result
    assert run.call_args.args[0] == [
        "tmux",
        "new-session",
        "-d",
        "-s",
        "hermes-tgtocli-abcdef12",
        "hermes --resume 20260519_101112_abcdef12",
    ]
