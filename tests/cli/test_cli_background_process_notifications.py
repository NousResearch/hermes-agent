"""Regression tests for CLI background-process notification handling."""

from __future__ import annotations

import queue
from unittest.mock import patch

import cli


class _FakeProcessRegistry:
    def __init__(self, events=(), consumed=()):
        self.completion_queue = queue.Queue()
        for event in events:
            self.completion_queue.put(event)
        self._consumed = set(consumed)

    def is_completion_consumed(self, session_id: str) -> bool:
        return session_id in self._consumed


def test_cli_background_notifications_off_drains_without_display():
    """Config off suppresses notifications but still drains stale queue events."""
    registry = _FakeProcessRegistry([
        {
            "type": "completion",
            "session_id": "proc_done",
            "command": "make test",
            "exit_code": 0,
            "output": "ok",
        }
    ])

    emitted = []
    with patch("tools.process_registry.process_registry", registry):
        displayed = cli._drain_process_notifications_for_cli(
            emit=emitted.append,
            config={"display": {"background_process_notifications": "off"}},
        )

    assert displayed == 0
    assert emitted == []
    assert registry.completion_queue.empty()


def test_cli_background_notifications_result_displays_only_final():
    registry = _FakeProcessRegistry([
        {
            "type": "watch_match",
            "session_id": "proc_watch",
            "command": "server",
            "pattern": "READY",
            "output": "READY\n",
        },
        {
            "type": "completion",
            "session_id": "proc_done",
            "command": "make test",
            "exit_code": 0,
            "output": "ok",
        },
    ])

    emitted = []
    with patch("tools.process_registry.process_registry", registry):
        displayed = cli._drain_process_notifications_for_cli(
            emit=emitted.append,
            config={"display": {"background_process_notifications": "result"}},
        )

    assert displayed == 1
    assert len(emitted) == 1
    assert "proc_done completed" in emitted[0]
    assert "proc_watch" not in emitted[0]
    assert registry.completion_queue.empty()


def test_cli_background_notifications_error_displays_only_failed_final():
    registry = _FakeProcessRegistry([
        {
            "type": "completion",
            "session_id": "proc_ok",
            "command": "true",
            "exit_code": 0,
            "output": "",
        },
        {
            "type": "completion",
            "session_id": "proc_fail",
            "command": "false",
            "exit_code": 1,
            "output": "FAILED",
        },
    ])

    emitted = []
    with patch("tools.process_registry.process_registry", registry):
        displayed = cli._drain_process_notifications_for_cli(
            emit=emitted.append,
            config={"display": {"background_process_notifications": "error"}},
        )

    assert displayed == 1
    assert "proc_fail completed" in emitted[0]
    assert "proc_ok" not in emitted[0]


def test_cli_background_notifications_skip_consumed_completion():
    registry = _FakeProcessRegistry(
        [
            {
                "type": "completion",
                "session_id": "proc_consumed",
                "command": "build",
                "exit_code": 0,
                "output": "already returned by process.wait",
            }
        ],
        consumed={"proc_consumed"},
    )

    emitted = []
    with patch("tools.process_registry.process_registry", registry):
        displayed = cli._drain_process_notifications_for_cli(emit=emitted.append)

    assert displayed == 0
    assert emitted == []
    assert registry.completion_queue.empty()


def test_cli_process_notifications_do_not_depend_on_pending_input():
    """The drain helper emits to a display side channel, not the input queue."""
    registry = _FakeProcessRegistry([
        {
            "type": "completion",
            "session_id": "proc_done",
            "command": "make test",
            "exit_code": 0,
            "output": "ok",
        }
    ])
    pending_input = queue.Queue()
    emitted = []

    with patch("tools.process_registry.process_registry", registry):
        displayed = cli._drain_process_notifications_for_cli(emit=emitted.append)

    assert displayed == 1
    assert emitted
    assert pending_input.empty()
