"""Gateway-native context-fragmentation cleanup command tests."""
import hashlib
import json
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


def _write_verified_fragmentation_artifacts(tmp_path, old_session_id="sess-old"):
    recovery = tmp_path / "recovery.md"
    recovery.write_text("# Recovery\n\nContinue safely from compact state.\n", encoding="utf-8")
    digest = hashlib.sha256(recovery.read_bytes()).hexdigest()
    marker = tmp_path / "pending.json"
    marker.write_text(
        json.dumps(
            {
                "recovery_path": str(recovery),
                "recovery_size_bytes": recovery.stat().st_size,
                "recovery_sha256": digest,
                "old_session_id": old_session_id,
                "reset_required": True,
            }
        ),
        encoding="utf-8",
    )
    return {
        "profile": "responsible_memory_rollout",
        "recovery_path": str(recovery),
        "recovery_size_bytes": recovery.stat().st_size,
        "recovery_sha256": digest,
        "pending_marker": str(marker),
        "reset_required": True,
    }


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
        return _write_verified_fragmentation_artifacts(tmp_path)

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
    assert str(tmp_path / "recovery.md") in str(result)
    reset_session = getattr(runner.session_store, "reset_session")
    reset_session.assert_called_once()
    method_calls = getattr(runner.session_store, "method_calls")
    assert "delete_session" not in {call[0] for call in method_calls}


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


@pytest.mark.asyncio
async def test_fragmentation_finalize_reset_refuses_verification_failure_before_reset(tmp_path):
    """The live reset must not rotate sessions if recovery/marker verification fails."""
    runner = _make_runner()

    async def fake_run_finalizer(**_kwargs):
        return {
            "profile": "responsible_memory_rollout",
            "recovery_path": str(tmp_path / "missing-recovery.md"),
            "recovery_size_bytes": 0,
            "recovery_sha256": "bad",
            "pending_marker": str(tmp_path / "missing-pending.json"),
            "reset_required": True,
        }

    runner._run_fragmentation_finalizer = fake_run_finalizer

    with patch("hermes_cli.profiles.get_active_profile_name", return_value="responsible_memory_rollout"):
        result = await runner._handle_fragmentation_command(_make_event("/fragmentation finalize-reset"))

    assert "failed before reset" in result
    assert "recovery file does not exist" in result
    getattr(runner.session_store, "reset_session").assert_not_called()


@pytest.mark.asyncio
async def test_fragmentation_finalize_reset_requires_live_session_id(tmp_path):
    """Live finalize-reset is tied to the current gateway session, not an empty/offline target."""
    runner = _make_runner()
    runner.session_store._entries = {}
    runner._run_fragmentation_finalizer = AsyncMock(return_value=_write_verified_fragmentation_artifacts(tmp_path))

    with patch("hermes_cli.profiles.get_active_profile_name", return_value="responsible_memory_rollout"):
        result = await runner._handle_fragmentation_command(_make_event("/fragmentation finalize-reset"))

    assert "No live gateway session is bound" in result
    runner._run_fragmentation_finalizer.assert_not_called()
    getattr(runner.session_store, "reset_session").assert_not_called()


@pytest.mark.asyncio
async def test_fragmentation_finalize_reset_uses_reset_path_without_deleting_history(tmp_path):
    """The live path should rotate the current mapping via reset_session, never delete historical rows."""
    runner = _make_runner()
    runner._run_fragmentation_finalizer = AsyncMock(
        return_value=_write_verified_fragmentation_artifacts(tmp_path)
    )

    with patch("hermes_cli.profiles.get_active_profile_name", return_value="responsible_memory_rollout"):
        await runner._handle_fragmentation_command(_make_event("/fragmentation finalize-reset"))

    getattr(runner.session_store, "reset_session").assert_called_once()
    method_calls = getattr(runner.session_store, "method_calls")
    assert "delete_session" not in {call[0] for call in method_calls}
