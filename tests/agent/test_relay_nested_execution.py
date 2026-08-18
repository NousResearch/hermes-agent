"""Regression tests for nested managed Relay execution (#77244).

The native Relay pipeline binds its Futures to the event loop that entered
``run_in_session_async``. While a managed tool callback is executing, that
loop is blocked until the callback returns — so any NESTED managed relay call
made from inside the callback (e.g. vision_analyze's auxiliary LLM call on a
worker-thread loop) awaits a Future that can never complete:
``RuntimeError: ... attached to a different loop``, or a deadlock, or
``Event loop is closed`` at shutdown.

The fix: ``relay_runtime.managed_callback_guard`` marks the callback's
context (a ContextVar, so it propagates through ``contextvars.copy_context()``
into tool worker threads); ``resolve_execution_context`` returns the
no-relay triple while the marker is set, so nested calls run unmanaged.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading

import pytest

pytest.importorskip("nemo_relay")

from agent import relay_llm, relay_runtime, relay_tools


@pytest.fixture()
def relay_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    relay_runtime._reset_for_tests()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(),
        session_id="session-1",
        platform="cli",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
        lease,
        turn_id="turn-1",
        task_id="task-1",
    )
    lease.host.retain_managed_execution("test.nested_relay")
    try:
        yield lease.host
    finally:
        lease.host.release_managed_execution("test.nested_relay")
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()


def _nested_aux_llm_call_from_worker_thread() -> dict:
    """Mimic vision_analyze: aux LLM call via a worker thread's own loop."""

    async def aux_call():
        async def provider(request):
            await asyncio.sleep(0)
            return {
                "id": "aux-1",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "nested"},
                        "finish_reason": "stop",
                    }
                ],
            }

        return await relay_llm.execute_current_async(
            {"messages": [{"role": "user", "content": "look"}], "model": "m"},
            provider,
            name="nested-prov",
            model_name="m",
            metadata={
                "api_mode": "chat_completions",
                "api_request_id": "req-nested",
                "call_role": "auxiliary:vision",
            },
        )

    holder: dict = {}

    def run() -> None:
        loop = asyncio.new_event_loop()
        try:
            holder["result"] = loop.run_until_complete(aux_call())
        except BaseException as exc:  # pragma: no cover - assertion payload
            holder["error"] = exc
        finally:
            loop.close()

    ctx = contextvars.copy_context()
    thread = threading.Thread(target=lambda: ctx.run(run))
    thread.start()
    thread.join(timeout=30)
    assert not thread.is_alive(), "nested aux call deadlocked (#77244)"
    if "error" in holder:
        raise holder["error"]
    return holder["result"]


def test_nested_aux_llm_call_inside_managed_tool_does_not_cross_loops(relay_turn):
    """The #77244 shape: managed tool -> worker-thread aux LLM call."""
    host = relay_turn
    managed_llm_names: list[str] = []
    original_execute = host.relay.llm.execute

    def counting_execute(name, *args, **kwargs):
        managed_llm_names.append(name)
        return original_execute(name, *args, **kwargs)

    host.relay.llm.execute = counting_execute
    try:
        def the_tool(args):
            result = _nested_aux_llm_call_from_worker_thread()
            return {"analysis": result["choices"][0]["message"]["content"]}

        result, _final_args = relay_tools.execute(
            "vision_analyze",
            {"image_url": "/tmp/x.png"},
            the_tool,
            session_id="session-1",
            metadata={"api_request_id": "req-tool"},
        )
    finally:
        host.relay.llm.execute = original_execute

    assert "nested" in str(result)
    # The nested call must have bypassed the managed pipeline entirely.
    assert "nested-prov" not in managed_llm_names


def test_main_turn_llm_call_stays_managed(relay_turn):
    """The guard must not disable relay for top-level (non-nested) calls."""
    host = relay_turn
    managed_llm_names: list[str] = []
    original_execute = host.relay.llm.execute

    def counting_execute(name, *args, **kwargs):
        managed_llm_names.append(name)
        return original_execute(name, *args, **kwargs)

    host.relay.llm.execute = counting_execute
    try:
        out = relay_llm.execute(
            {"messages": [{"role": "user", "content": "hi"}], "model": "m"},
            lambda request: {
                "id": "main-1",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "main"},
                        "finish_reason": "stop",
                    }
                ],
            },
            session_id="session-1",
            name="main-prov",
            model_name="m",
            metadata={
                "api_mode": "chat_completions",
                "api_request_id": "req-main",
                "call_role": "primary",
            },
        )
    finally:
        host.relay.llm.execute = original_execute

    assert out is not None
    assert "main-prov" in managed_llm_names


def test_guard_resets_after_managed_callback_returns(relay_turn):
    """After the tool returns, subsequent calls are managed again."""
    host = relay_turn
    managed_llm_names: list[str] = []
    original_execute = host.relay.llm.execute

    def counting_execute(name, *args, **kwargs):
        managed_llm_names.append(name)
        return original_execute(name, *args, **kwargs)

    host.relay.llm.execute = counting_execute
    try:
        relay_tools.execute(
            "noop_tool",
            {},
            lambda args: {"ok": True},
            session_id="session-1",
            metadata={"api_request_id": "req-tool2"},
        )
        assert relay_runtime._MANAGED_CALLBACK_DEPTH.get() == 0
        relay_llm.execute(
            {"messages": [{"role": "user", "content": "hi"}], "model": "m"},
            lambda request: {
                "id": "after-1",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "after"},
                        "finish_reason": "stop",
                    }
                ],
            },
            session_id="session-1",
            name="after-prov",
            model_name="m",
            metadata={
                "api_mode": "chat_completions",
                "api_request_id": "req-after",
                "call_role": "primary",
            },
        )
    finally:
        host.relay.llm.execute = original_execute

    assert "after-prov" in managed_llm_names


def test_resolve_execution_context_bypasses_inside_guard(relay_turn):
    with relay_runtime.managed_callback_guard():
        runtime, session, parent = relay_runtime.resolve_execution_context(
            "session-1"
        )
    assert runtime is None and session is None and parent is None
    # Outside the guard the context resolves normally again.
    runtime, session, _parent = relay_runtime.resolve_execution_context("session-1")
    assert runtime is not None and session is not None
