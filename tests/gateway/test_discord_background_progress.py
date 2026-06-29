"""Discord-specific background process progress delivery tests."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner


class _FakeRegistry:
    def __init__(self, sessions):
        self._sessions = list(sessions)
        self._completion_consumed: set[str] = set()

    def get(self, _session_id):
        if self._sessions:
            return self._sessions.pop(0)
        return None

    def is_completion_consumed(self, session_id):
        return session_id in self._completion_consumed


def _build_runner(monkeypatch, tmp_path, mode="all"):
    (tmp_path / "config.yaml").write_text(
        "display:\n"
        f"  background_process_notifications: {mode}\n"
        "  background_process_status_min_elapsed_seconds: 0\n",
        encoding="utf-8",
    )
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner = GatewayRunner(GatewayConfig())
    adapter = SimpleNamespace(
        send=AsyncMock(return_value=SendResult(success=True, message_id="status-1")),
        edit_message=AsyncMock(return_value=SendResult(success=True, message_id="status-1")),
        handle_message=AsyncMock(),
    )
    runner.adapters[Platform.DISCORD] = adapter
    return runner, adapter


def _watcher():
    return {
        "session_id": "proc_discord",
        "check_interval": 0,
        "platform": "discord",
        "chat_id": "parent-123",
        "thread_id": "thread-456",
        "session_key": "agent:main:discord:thread:parent-123:thread-456",
    }


@pytest.mark.asyncio
async def test_discord_running_updates_use_one_edited_status_bubble(monkeypatch, tmp_path):
    import tools.process_registry as pr_module

    sessions = [
        SimpleNamespace(output_buffer="secret raw line 1\n", exited=False, exit_code=None, command="build"),
        SimpleNamespace(output_buffer="secret raw line 1\nsecret raw line 2\n", exited=False, exit_code=None, command="build"),
        SimpleNamespace(output_buffer="final output\n", exited=True, exit_code=0, command="build"),
    ]
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    async def _instant_sleep(*_a, **_kw):
        pass

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    runner, adapter = _build_runner(monkeypatch, tmp_path, "all")

    await runner._run_process_watcher(_watcher())

    assert adapter.send.await_count == 2  # one status bubble + one fresh final message
    first_send = adapter.send.await_args_list[0]
    final_send = adapter.send.await_args_list[1]
    assert "Still running background process proc_discord" in first_send.args[1]
    assert "secret raw line" not in first_send.args[1]
    assert first_send.kwargs["metadata"] == {"thread_id": "thread-456"}

    assert adapter.edit_message.await_count >= 2  # second running update + terminal completion edit
    edited_payloads = [call.args[2] for call in adapter.edit_message.await_args_list]
    assert any("Still running background process proc_discord" in payload for payload in edited_payloads)
    assert any("final message sent" in payload.lower() for payload in edited_payloads)
    assert all("secret raw line" not in payload for payload in edited_payloads)
    assert all(call.kwargs["metadata"] == {"thread_id": "thread-456"} for call in adapter.edit_message.await_args_list)

    assert "finished with exit code 0" in final_send.args[1]
    assert "final output" in final_send.args[1]


@pytest.mark.asyncio
async def test_dynamic_mode_flip_suppresses_running_update_without_restart(monkeypatch, tmp_path):
    import tools.process_registry as pr_module

    sessions = [
        SimpleNamespace(output_buffer="building...\n", exited=False, exit_code=None, command="build"),
        SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0, command="build"),
    ]
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))
    runner, adapter = _build_runner(monkeypatch, tmp_path, "all")
    sleep_count = 0

    async def _instant_sleep(*_a, **_kw):
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count == 1:
            (tmp_path / "config.yaml").write_text(
                "display:\n"
                "  background_process_notifications: result\n"
                "  background_process_status_min_elapsed_seconds: 0\n",
                encoding="utf-8",
            )

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    await runner._run_process_watcher(_watcher())

    assert adapter.edit_message.await_count == 0
    adapter.send.assert_awaited_once()
    assert "finished with exit code 0" in adapter.send.await_args.args[1]
    assert "Still running" not in adapter.send.await_args.args[1]


@pytest.mark.asyncio
async def test_agent_notify_does_not_send_user_facing_discord_progress(monkeypatch, tmp_path):
    import tools.process_registry as pr_module

    sessions = [SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0, command="build")]
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    async def _instant_sleep(*_a, **_kw):
        pass

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    runner, adapter = _build_runner(monkeypatch, tmp_path, "all")
    watcher = _watcher()
    watcher["notify_on_complete"] = True

    await runner._run_process_watcher(watcher)

    adapter.handle_message.assert_awaited_once()
    adapter.send.assert_not_awaited()
    adapter.edit_message.assert_not_awaited()
