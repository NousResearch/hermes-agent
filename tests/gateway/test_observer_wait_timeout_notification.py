"""A supervisor's ``agent wait`` that only ran out of time is not a failed background task.

``herdr agent wait <agent> --timeout N`` exits 1 with the CLI's JSON-RPC envelope
``{"error":{"code":"timeout",...},"id":"cli:agent:wait"}`` when the wait deadline passes while the
watched agent keeps working. The chat must not read ``❌ Background task failed`` plus a rerun hint,
``error`` mode must stay quiet about it, and the agent's own wake must say the observer timed out —
while every genuine non-zero exit keeps its failure notice.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from tests.gateway.test_background_process_notifications import _FakeRegistry, _build_runner, _watcher_dict
from tools.process_registry_notifications import format_process_notification, process_completion_display_text

WAIT_CMD = "herdr agent wait scn-umi9-sheet-toggle --timeout 1800000"
WAIT_TIMEOUT_OUTPUT = (
    '{"error":{"code":"timeout","message":"timed out waiting for agent status"},"id":"cli:agent:wait"}\n')


async def _run_watcher(monkeypatch, tmp_path, mode, *, command, exit_code, output, agent_notify=False):
    import tools.process_registry as pr_module

    session = SimpleNamespace(output_buffer=output, exited=True, exit_code=exit_code, command=command,
                              started_at=None)
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry([session]))

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    runner = _build_runner(monkeypatch, tmp_path, mode)
    runner._enqueue_process_completion_notification = AsyncMock(return_value=True)
    watcher = {**_watcher_dict(), "session_key": "agent:main:telegram:dm:123", "notify_on_complete": agent_notify}
    await runner._run_process_watcher(watcher)
    return runner, [call.args[1] for call in runner.adapters[Platform.TELEGRAM].send.await_args_list]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,command,exit_code,output,expected", [
    ("concise", WAIT_CMD, 1, WAIT_TIMEOUT_OUTPUT, "⏳ Supervisor wait timed out"),
    ("all", WAIT_CMD, 1, WAIT_TIMEOUT_OUTPUT, "⏳ Supervisor wait timed out"),
    ("error", WAIT_CMD, 1, WAIT_TIMEOUT_OUTPUT, None),
    # A genuine non-zero exit — even a JSON envelope from another call or error code — stays a failure.
    ("error", "make build", 2, "linker error\n", "❌ Background task failed"),
    ("concise", "npm run dist", 1, "error: notarization failed\n", "❌ Background task failed"),
    ("error", WAIT_CMD, 1, '{"error":{"code":"not_found"},"id":"cli:agent:wait"}\n', "❌ Background task failed"),
    ("error", "herdr agent send x", 1, '{"error":{"code":"timeout"},"id":"cli:agent:send"}\n',
     "❌ Background task failed"),
])
async def test_watcher_notice_tells_wait_timeout_from_failure(
        monkeypatch, tmp_path, mode, command, exit_code, output, expected):
    _runner, sent = await _run_watcher(monkeypatch, tmp_path, mode, command=command, exit_code=exit_code,
                                       output=output)
    if expected is None:
        assert sent == []
        return
    assert len(sent) == 1 and sent[0].startswith(expected)
    if expected.startswith("⏳"):
        assert "❌" not in sent[0] and "rerun" not in sent[0] and "Check its status" in sent[0]
    else:
        assert "rerun" in sent[0]


@pytest.mark.asyncio
async def test_error_mode_still_wakes_the_agent_with_an_observer_timeout_note(monkeypatch, tmp_path):
    """``error`` mode keeps the chat quiet, but a ``notify_on_complete`` supervisor is still woken —
    told it was its own wait deadline, not the watched agent, that ran out."""
    runner, sent = await _run_watcher(monkeypatch, tmp_path, "error", command=WAIT_CMD, exit_code=1,
                                      output=WAIT_TIMEOUT_OUTPUT, agent_notify=True)
    assert sent == []
    runner._enqueue_process_completion_notification.assert_awaited_once()
    wake = runner._enqueue_process_completion_notification.await_args.args[0]
    assert "wait timeout" in wake and "NOT a failure of the watched agent" in wake
    assert "status, pane and worktree" in wake


def test_agent_wake_and_title_keep_genuine_failures():
    timeout_evt = {"type": "completion", "session_id": "proc_w", "command": WAIT_CMD, "exit_code": 1,
                   "completion_reason": "exited", "output": WAIT_TIMEOUT_OUTPUT}
    failed_evt = {**timeout_evt, "command": "make build", "output": "linker error\n"}
    assert process_completion_display_text([timeout_evt]).startswith("Background Process Wait Timed Out:")
    assert process_completion_display_text([failed_evt]).startswith("Background Process Failed (exit 1):")
    assert "NOT a failure" not in format_process_notification(failed_evt)
