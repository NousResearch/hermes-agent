"""Tests for configurable background process notifications in the gateway."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner


class _FakeRegistry:
    def __init__(self, sessions):
        self._sessions = list(sessions)

    def get(self, session_id):
        if self._sessions:
            return self._sessions.pop(0)
        return None


def _write_config(tmp_path, mode: str) -> None:
    (tmp_path / "config.yaml").write_text(
        "display:\n"
        f"  background_process_notifications: {mode}\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "sessions", "expected_calls", "expected_fragment"),
    [
        (
            "all",
            [SimpleNamespace(output_buffer="diff --git a\n", exited=False, exit_code=None), None],
            1,
            "is still running",
        ),
        (
            "result",
            [SimpleNamespace(output_buffer="done\n", exited=False, exit_code=None), None],
            0,
            None,
        ),
        (
            "off",
            [SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0)],
            0,
            None,
        ),
        (
            "result",
            [SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0)],
            1,
            "finished with exit code 0",
        ),
        (
            "error",
            [SimpleNamespace(output_buffer="done\n", exited=True, exit_code=0)],
            0,
            None,
        ),
        (
            "error",
            [SimpleNamespace(output_buffer="traceback\n", exited=True, exit_code=1)],
            1,
            "finished with exit code 1",
        ),
    ],
)
async def test_run_process_watcher_respects_notification_mode(
    monkeypatch, tmp_path, mode, sessions, expected_calls, expected_fragment
):
    _write_config(tmp_path, mode)

    import gateway.run as gateway_run
    import tools.process_registry as process_registry_module

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        process_registry_module,
        "process_registry",
        _FakeRegistry(sessions),
    )

    async def _no_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    runner = GatewayRunner(GatewayConfig())
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters[Platform.TELEGRAM] = adapter

    await runner._run_process_watcher(
        {
            "session_id": "proc_test",
            "check_interval": 0,
            "platform": "telegram",
            "chat_id": "123",
        }
    )

    assert adapter.send.await_count == expected_calls
    if expected_fragment is not None:
        sent_message = adapter.send.await_args.args[1]
        assert expected_fragment in sent_message
