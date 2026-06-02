"""Gateway-native context-fragmentation cleanup command tests."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.MATRIX,
        user_id="u1",
        chat_id="!room:matrix.org",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.MATRIX: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {Platform.MATRIX: MagicMock()}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), emit_collect=AsyncMock(return_value=[]), loaded_hooks=False)
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._session_reasoning_overrides = {}
    runner._pending_approvals = {}
    runner._queued_events = {}
    runner._agent_cache_lock = None
    runner._session_db = None
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_run_generation = {}
    runner._is_user_authorized = lambda _source: True
    runner._format_session_info = lambda: ""

    session_key = build_session_key(_make_source())
    old_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-old",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.MATRIX,
        chat_type="dm",
    )
    new_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-new",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.MATRIX,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store._entries = {session_key: old_entry}
    runner.session_store.reset_session.return_value = new_entry
    runner.session_store.get_or_create_session.return_value = new_entry
    return runner


@pytest.mark.asyncio
async def test_fragmentation_finalize_reset_runs_finalizer_then_resets(monkeypatch, tmp_path):
    """/fragmentation finalize-reset should finalize recovery before rotating the live session."""
    from hermes_cli.commands import resolve_command

    finalizer = tmp_path / "fragmentation_finalize.py"
    finalizer.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_FRAGMENTATION_FINALIZER", str(finalizer))

    runner = _make_runner()
    calls = []

    async def fake_run_finalizer(**kwargs):
        calls.append(("finalize", kwargs))
        return {
            "profile": "responsible_memory_rollout",
            "recovery_path": "/tmp/recovery.md",
            "recovery_sha256": "abc123",
            "pending_marker": "/tmp/pending.json",
            "reset_required": True,
        }

    original_reset = runner._handle_reset_command

    async def recording_reset(event):
        calls.append(("reset", {"text": event.text}))
        return await original_reset(event)

    monkeypatch.setattr(runner, "_run_fragmentation_finalizer", fake_run_finalizer)
    monkeypatch.setattr(runner, "_handle_reset_command", recording_reset)

    with patch("hermes_cli.profiles.get_active_profile_name", return_value="responsible_memory_rollout"):
        result = await runner._handle_fragmentation_command(_make_event("/fragmentation finalize-reset --note cleanup"))

    assert resolve_command("fragmentation").name == "fragmentation"
    assert calls[0][0] == "finalize"
    assert calls[0][1]["profile"] == "responsible_memory_rollout"
    assert calls[0][1]["old_session_id"] == "sess-old"
    assert calls[0][1]["note"] == "cleanup"
    assert calls[1] == ("reset", {"text": "/new"})
    assert "Fragmentation cleanup finalized" in str(result)
    assert "/tmp/recovery.md" in str(result)
    runner.session_store.reset_session.assert_called_once()


@pytest.mark.asyncio
async def test_fragmentation_finalize_reset_rejects_other_profile(monkeypatch):
    """A live gateway reset may only target the current profile/room session."""
    runner = _make_runner()
    monkeypatch.setattr(runner, "_run_fragmentation_finalizer", AsyncMock())

    with patch("hermes_cli.profiles.get_active_profile_name", return_value="responsible_memory_rollout"):
        result = await runner._handle_fragmentation_command(
            _make_event("/fragmentation finalize-reset --profile other_profile")
        )

    assert "current live profile" in result
    runner._run_fragmentation_finalizer.assert_not_called()
