"""Tests for gateway restart auto-resume checkpointing and replay."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_runner():
    """Create a bare GatewayRunner without calling __init__."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._restart_resume_inflight = {}
    runner._background_tasks = set()
    return runner


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-123",
        user_id="u-1",
        user_name="tester",
        chat_type="dm",
        thread_id="42",
    )


def test_persist_restart_resume_pending_writes_checkpoint(tmp_path):
    """In-flight payloads are persisted to the restart checkpoint file."""
    runner = _make_runner()
    source = _make_source()
    runner._mark_restart_resume_inflight("sess-1", source, "Continue this task")

    with patch("gateway.run._hermes_home", tmp_path):
        runner._persist_restart_resume_pending()

    pending_path = tmp_path / ".restart_resume_pending.json"
    assert pending_path.exists()
    payload = json.loads(pending_path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert len(payload["entries"]) == 1
    entry = payload["entries"][0]
    assert entry["session_key"] == "sess-1"
    assert entry["message_text"] == "Continue this task"
    assert entry["source"]["platform"] == "telegram"
    assert entry["source"]["thread_id"] == "42"


def test_persist_restart_resume_pending_skips_commands(tmp_path):
    """Slash commands are not persisted for auto-resume replay."""
    runner = _make_runner()
    source = _make_source()
    runner._mark_restart_resume_inflight("sess-1", source, "/status")

    with patch("gateway.run._hermes_home", tmp_path):
        runner._persist_restart_resume_pending()

    pending_path = tmp_path / ".restart_resume_pending.json"
    assert not pending_path.exists()


def test_restart_resume_entry_freshness_guard():
    """Entries older than max age are skipped."""
    runner = _make_runner()

    assert runner._is_restart_resume_entry_fresh({}, max_age_seconds=3600)
    assert runner._is_restart_resume_entry_fresh(
        {"captured_at": "2099-01-01T00:00:00"},
        max_age_seconds=1,
    )
    assert not runner._is_restart_resume_entry_fresh(
        {"captured_at": "2000-01-01T00:00:00"},
        max_age_seconds=3600,
    )


@pytest.mark.asyncio
async def test_run_restart_auto_resume_replays_message_and_cleans_marker(tmp_path):
    """Startup replay sends a notice, replays the message, and removes marker file."""
    runner = _make_runner()
    adapter = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._handle_message = AsyncMock(return_value="Resumed output")

    pending_payload = {
        "version": 1,
        "entries": [
            {
                "session_key": "sess-1",
                "source": _make_source().to_dict(),
                "message_text": "please continue",
            }
        ],
    }

    pending_path = tmp_path / ".restart_resume_pending.json"
    pending_path.write_text(json.dumps(pending_payload), encoding="utf-8")

    with patch("gateway.run._hermes_home", tmp_path), \
         patch("gateway.run.asyncio.sleep", new=AsyncMock()):
        await runner._run_restart_auto_resume()

    # First send is the resume notice, second is replay response.
    assert adapter.send.await_count == 2
    first_call = adapter.send.await_args_list[0]
    second_call = adapter.send.await_args_list[1]
    assert first_call.args[0] == "chat-123"
    assert "Resuming the task" in first_call.args[1]
    assert second_call.args[1] == "Resumed output"

    # Metadata includes thread routing for Telegram topics.
    assert first_call.kwargs["metadata"] == {"thread_id": "42"}
    assert second_call.kwargs["metadata"] == {"thread_id": "42"}

    replay_event = runner._handle_message.await_args.args[0]
    assert isinstance(replay_event, MessageEvent)
    assert replay_event.text == "please continue"

    assert not pending_path.exists()


@pytest.mark.asyncio
async def test_run_restart_auto_resume_skips_stale_entries(tmp_path):
    """Very old captured entries are ignored and marker is still cleaned up."""
    runner = _make_runner()
    adapter = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._handle_message = AsyncMock(return_value="should not run")

    pending_payload = {
        "version": 1,
        "entries": [
            {
                "session_key": "sess-1",
                "source": _make_source().to_dict(),
                "message_text": "please continue",
                "captured_at": "2000-01-01T00:00:00",
            }
        ],
    }

    pending_path = tmp_path / ".restart_resume_pending.json"
    pending_path.write_text(json.dumps(pending_payload), encoding="utf-8")

    with patch("gateway.run._hermes_home", tmp_path), \
         patch("gateway.run.asyncio.sleep", new=AsyncMock()):
        await runner._run_restart_auto_resume()

    adapter.send.assert_not_awaited()
    runner._handle_message.assert_not_awaited()
    assert not pending_path.exists()


def test_mark_restart_resume_inflight_ignores_slash_commands():
    """Slash commands should not overwrite resumable in-flight payloads."""
    runner = _make_runner()
    source = _make_source()

    runner._mark_restart_resume_inflight("sess-1", source, "/status")

    assert runner._restart_resume_inflight == {}


@pytest.mark.asyncio
async def test_run_restart_auto_resume_recovers_from_claimed_file(tmp_path):
    """A claimed marker from a crashed resume task is still replayed and cleaned."""
    runner = _make_runner()
    adapter = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._handle_message = AsyncMock(return_value="Resumed output")

    pending_payload = {
        "version": 1,
        "entries": [
            {
                "session_key": "sess-1",
                "source": _make_source().to_dict(),
                "message_text": "please continue",
            }
        ],
    }

    claimed_path = tmp_path / ".restart_resume_pending.claimed.json"
    claimed_path.write_text(json.dumps(pending_payload), encoding="utf-8")

    with patch("gateway.run._hermes_home", tmp_path), \
         patch("gateway.run.asyncio.sleep", new=AsyncMock()):
        await runner._run_restart_auto_resume()

    assert adapter.send.await_count == 2
    runner._handle_message.assert_awaited_once()
    assert not claimed_path.exists()
