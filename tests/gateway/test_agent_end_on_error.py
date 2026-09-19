"""Regression for #113057: an exception between ``agent:start`` and the normal
``agent:end`` must still produce a terminal event, or hook observers silently
under-count exactly the failed turns they most want to see."""

import asyncio
from types import SimpleNamespace

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _RecordingHooks:
    def __init__(self):
        self.events = []

    async def emit(self, event_type, context=None):
        self.events.append((event_type, context or {}))


def _runner_with_hooks():
    runner = object.__new__(GatewayRunner)
    hooks = _RecordingHooks()
    runner.hooks = hooks

    async def stop_typing(event, source):
        return None

    runner._hmwa_stop_typing_for_turn = stop_typing
    return runner, hooks


def _error_reply(runner, err, history=(), message_text=None):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", user_id="u1")
    event = MessageEvent(text="hello", source=source)
    session_entry = SimpleNamespace(session_id="sess-1")
    prepared = runner._PreparedTurn(list(history), "hello", message_text, None, None, None)
    return asyncio.run(runner._hmwa_agent_error_reply(
        err, event, source, session_entry, "k", prepared,
    ))


def _agent_ends(hooks):
    return [ctx for name, ctx in hooks.events if name == "agent:end"]


def test_exception_path_pairs_agent_start_with_error_end():
    """The reported invariant: agent:start fired, the turn raised, observers see one
    agent:end marked as failed — carrying the sanitized user-facing reply, never the
    raw exception."""
    runner, hooks = _runner_with_hooks()
    reply = _error_reply(runner, RuntimeError("boom"))

    ends = _agent_ends(hooks)
    assert len(ends) == 1
    end = ends[0]
    assert end["error"] is True
    assert end["session_id"] == "sess-1"
    assert end["platform"] == "telegram"
    assert end["response"] == str(reply)[:500]
    assert "boom" not in end["response"]
    assert end["model"] == "" and end["provider"] == ""


def test_overflow_early_return_also_pairs_agent_end():
    """The context-overflow early return must not skip the terminal event either."""
    runner, hooks = _runner_with_hooks()
    err = RuntimeError("payload too large")
    err.status_code = 400
    history = [{"role": "user", "content": "x"}] * 51
    reply = _error_reply(runner, err, history=history, message_text="x")

    ends = _agent_ends(hooks)
    assert len(ends) == 1
    assert ends[0]["error"] is True
    assert ends[0]["response"] == str(reply)[:500]
