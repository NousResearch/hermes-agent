"""A sub-agent approval still posts after the parent turn's generation has gone stale.

The turn runner refuses to render a prompt once ``check_approval_generation`` says the
turn is no longer running. That is right for the parent session's own waits. A delegated
child pinned past that turn still has to reach the same chat, and must not also try to
register a local-viewer prompt the stale generation would reject.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from hermes_state_runtime import RuntimeStoreError

SESSION = "agent:main:discord:dm:1"


class _ButtonAdapter:
    typed_command_prefix = "/"

    def __init__(self) -> None:
        self.approvals: list[dict] = []

    def pause_typing_for_chat(self, chat_id: str) -> None:
        return None

    async def send_exec_approval(self, *args: Any, **kwargs: Any):
        from gateway.platforms.base import SendResult
        self.approvals.append(kwargs)
        return SendResult(success=True, message_id="card-1")

    async def send(self, chat_id: str, message: str, **kwargs: Any):
        from gateway.platforms.base import SendResult
        return SendResult(success=True, message_id="m2")


class _Stale:
    def check_approval_generation(self, session_id, generation):
        raise RuntimeStoreError("stale_generation")

    def register_approval(self, *args, **kwargs):
        raise AssertionError("a stale turn must not register a viewer approval")


class _Live:
    def __init__(self) -> None:
        self.registered: list = []

    def check_approval_generation(self, session_id, generation):
        return None

    def register_approval(self, *args, **kwargs):
        self.registered.append(args)


def _runner(adapter, authority):
    from gateway.run_turn_runner import TurnRunner

    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        _status_adapter=adapter, _status_chat_id="C1", _status_thread_metadata={"thread_id": "t1"},
        session_key=SESSION, source=SimpleNamespace(chat_id="C1", platform="discord", session_key=SESSION),
    )

    class _Fut:
        def __init__(self, result):
            self._result = result

        def result(self, timeout=None):
            return self._result

    runner._schedule = lambda coro, _label: _Fut(asyncio.run(coro))
    runner._close_native_stream_boundary = lambda _why: None
    runner._approval_owner = (authority, "sid", 3)
    return runner


def test_stale_parent_turn_still_posts_a_subagent_approval():
    adapter = _ButtonAdapter()
    _runner(adapter, _Stale())._approval_notify_sync({
        "command": "sh -c 'echo hi'",
        "description": "sub-agent: echo hi — shell command via -c/-lc flag",
        "delegated_child": True,
    })
    assert len(adapter.approvals) == 1
    assert adapter.approvals[0]["session_key"] == SESSION
    assert adapter.approvals[0]["description"].startswith("sub-agent:")


def test_stale_parent_turn_does_not_post_its_own_approval():
    adapter = _ButtonAdapter()
    with pytest.raises(RuntimeStoreError) as exc:
        _runner(adapter, _Stale())._approval_notify_sync({
            "command": "rm -rf build", "description": "recursive delete",
        })
    assert exc.value.reason == "stale_generation"
    assert adapter.approvals == []


def test_live_turn_still_registers_a_subagent_approval():
    adapter = _ButtonAdapter()
    authority = _Live()
    _runner(adapter, authority)._approval_notify_sync({
        "command": "sh -c 'echo hi'", "description": "shell", "delegated_child": True,
    })
    assert len(adapter.approvals) == 1
    assert authority.registered
