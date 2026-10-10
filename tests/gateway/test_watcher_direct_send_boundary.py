"""A process watcher must not address a new Discord session with an old turn's status."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _Registry:
    def __init__(self, process):
        self.process = process

    def get(self, _session_id):
        process, self.process = self.process, None
        return process

    def is_completion_consumed(self, _session_id):
        return False


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["busy_receipt", "final", "running"])
@pytest.mark.parametrize("state", ["live", "reset", "compression", "child_handoff", "handoff_reset",
                                   "no_stamp", "moved_key"])
async def test_direct_discord_watcher_send_stays_with_spawning_session(
    monkeypatch, tmp_path, kind, state,
):
    import gateway.run as gateway_run
    import tools.process_registry as process_module

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(GatewayRunner, "_VOICE_MODE_PATH", tmp_path / "gateway_voice_mode.json")
    (tmp_path / "config.yaml").write_text(
        f"display:\n  background_process_notifications: {'all' if kind == 'running' else 'concise'}\n",
        encoding="utf-8",
    )
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    adapter = SimpleNamespace(send=AsyncMock(), handle_message=AsyncMock())
    runner.adapters[Platform.DISCORD] = adapter
    source = SessionSource(
        platform=Platform.DISCORD, chat_id="channel", chat_type="thread",
        thread_id="thread", user_id="person",
    )
    spawned_in = runner.session_store.get_or_create_session(source)
    watcher = {
        "session_id": "proc_old", "check_interval": 0,
        "session_key": spawned_in.session_key, "parent_session_id": spawned_in.session_id,
        "platform": "discord", "chat_type": "thread",
        "chat_id": "channel", "thread_id": "thread",
        "notify_on_complete": kind == "busy_receipt",
    }
    process = SimpleNamespace(
        output_buffer="done\n", exited=kind != "running", exit_code=0,
        command="echo done", started_at=None,
        session_key=spawned_in.session_key, parent_session_id=spawned_in.session_id,
        owner_task_id="owner", task_id="owner",
    )
    monkeypatch.setattr(process_module, "process_registry", _Registry(process))
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    if kind == "busy_receipt":
        monkeypatch.setattr(runner, "_launching_turn_active", AsyncMock(return_value=True))
        # A terminal preflight produces no model delivery, but the old watcher still sent a receipt.
        monkeypatch.setattr(runner, "_enqueue_process_completion_notification", AsyncMock(return_value=None))
    db = runner.session_store._db_for_key(spawned_in.session_key)
    if state == "reset":
        replaced = runner.session_store.reset_session(spawned_in.session_key)
        assert replaced.session_id != spawned_in.session_id
        assert db.get_session(spawned_in.session_id)["end_reason"] == "session_reset"
    if state == "compression":
        db.end_session(spawned_in.session_id, "compression")
        db.create_session("continuation", source="discord", parent_session_id=spawned_in.session_id,
                          session_key=spawned_in.session_key)
        assert runner.session_store.switch_session(
            spawned_in.session_key, "continuation", expected_session_id=spawned_in.session_id,
        ).session_id == "continuation"
        assert db.get_session(spawned_in.session_id)["end_reason"] == "compression"
    if state in {"child_handoff", "handoff_reset"}:
        db.create_session("delegate-child", source="subagent", parent_session_id=spawned_in.session_id)
        watcher["parent_session_id"] = process.parent_session_id = "delegate-child"
        process.session_key = spawned_in.session_id  # transfer_ownership uses the parent's session ID.
        if state == "handoff_reset":
            runner.session_store.reset_session(spawned_in.session_key)
    if state == "no_stamp":
        watcher.pop("parent_session_id")
        process.parent_session_id = ""
    if state == "moved_key":
        process.session_key = "agent:main:discord:thread:other"

    await runner._run_process_watcher(watcher)

    assert adapter.send.await_count == int(state in {"live", "compression", "child_handoff"})
