"""Mid-tail failure in _finish_turn must not wedge the session running (#115588).

An exception in the _finish_turn tail (persist/provenance/final-text sends)
used to skip the bare is_running=False release, leaving the session wedged
running and queued prompts stranded. The tail must release is_running /
current_prompt_text in a finally and still drain the queue.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionState


class _BoomConn:
    """Fails the final-text session_update once (transient mid-tail failure)."""

    def __init__(self) -> None:
        self.calls = 0

    async def session_update(self, session_id, update):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("mid-tail boom")


async def _noop_coro(*args, **kwargs):
    return None


def _make_server():
    server = HermesACPAgent.__new__(HermesACPAgent)
    server.session_manager = SimpleNamespace(save_session=lambda sid: None)
    server._conn = None
    drained = []

    async def fake_prompt(*, prompt, session_id, **kwargs):
        from acp.schema import PromptResponse

        drained.append(([getattr(b, "text", None) for b in prompt], session_id))
        return PromptResponse(stop_reason="end_turn")

    server.prompt = fake_prompt  # type: ignore[method-assign]
    server._send_usage_update = _noop_coro  # type: ignore[method-assign]
    return server, drained


def _running_state() -> SessionState:
    return SessionState(
        session_id="s1",
        agent=SimpleNamespace(session_id="h1"),
        history=[],
        cancel_event=None,
        is_running=True,
        queued_prompts=["queued-one"],
        runtime_lock=threading.Lock(),
        current_prompt_text="hello",
    )


def test_finish_turn_mid_tail_failure_releases_and_drains():
    server, drained = _make_server()
    state = _running_state()
    conn = _BoomConn()
    result = {"final_response": "done", "messages": []}

    with pytest.raises(RuntimeError, match="mid-tail boom"):
        asyncio.run(server._finish_turn(state, "s1", conn, result, "h1", False))

    assert state.is_running is False
    assert state.current_prompt_text == ""
    assert state.queued_prompts == []
    assert drained == [(["queued-one"], "s1")]
