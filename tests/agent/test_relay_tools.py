"""Tests for the core Relay-managed Hermes tool adapter."""

from __future__ import annotations

import contextvars
import json

import pytest

pytest.importorskip("nemo_relay")

from agent import relay_runtime, relay_tools


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
    lease.host.retain_managed_execution("test.relay_tools")
    try:
        yield lease.host.relay
    finally:
        lease.host.release_managed_execution("test.relay_tools")
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()


def test_tool_adapter_bypasses_relay_without_an_active_consumer(
    relay_turn, monkeypatch
):
    relay = relay_turn
    runtime = relay_runtime.get_runtime()
    assert runtime is not None
    runtime.release_managed_execution("test.relay_tools")
    args = {"command": "pwd"}

    monkeypatch.setattr(
        relay.tools,
        "execute",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("inactive Relay must not manage the tool call")
        ),
    )

    result, final_args = relay_tools.execute(
        "terminal",
        args,
        lambda value: value,
        session_id="session-1",
    )

    assert result is args
    assert final_args is args


def test_worker_thread_bypasses_managed_execution_for_async_tools(
    relay_turn, monkeypatch
):
    """Async tools running on worker threads must bypass the managed Relay
    wrapping (#77244).

    NeMo Relay's native ``tools.execute`` returns a Future bound to a loop
    captured at scope-push time on the main thread. When an async tool bridges
    sync->async via ``model_tools._run_async`` (per-thread persistent loop),
    awaiting that foreign-loop Future raises
    "attached to a different loop" — so managed execution must be skipped on
    worker threads and the callback run directly.
    """
    relay = relay_turn
    runtime = relay_runtime.get_runtime()
    assert runtime is not None
    assert runtime.managed_execution_enabled()

    # Simulate a worker-thread tool call: the Relay wrapper must NOT run.
    def _worker_thread_call(_args):
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_call_relay_tools_from_worker, relay, _args).result()

    monkeypatch.setattr(
        relay.tools,
        "execute",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("worker-thread tool call must bypass managed Relay")
        ),
    )

    args = {"image_url": "http://x/img.png", "user_prompt": "describe"}
    result, final_args = _worker_thread_call(args)
    assert result is args
    assert final_args is args


def _call_relay_tools_from_worker(relay, args):
    return relay_tools.execute(
        "vision_analyze",
        args,
        lambda value: value,
        session_id="session-1",
        metadata={"tool_call_id": "call-1"},
    )




def test_request_rewrite_reaches_authorized_callback_once(relay_turn):
    relay = relay_turn
    callback_args = []

    def rewrite_request(_name, args):
        return {**args, "path": "/approved/path"}

    async def wrap_execution(_name, args, next_call):
        result = await next_call(args)
        return relay.ToolExecutionInterceptOutcome({**result, "wrapped": True})

    relay.intercepts.register_tool_request(
        "hermes-test-tool-request", 1, False, rewrite_request
    )
    relay.intercepts.register_tool_execution(
        "hermes-test-tool-execution", 1, wrap_execution
    )
    try:
        result, observed_args = relay_tools.execute(
            "write_file",
            {"path": "/original/path"},
            lambda args: callback_args.append(args) or {"ok": True},
            session_id="session-1",
            metadata={"tool_call_id": "call-1"},
        )
    finally:
        relay.intercepts.deregister_tool_execution("hermes-test-tool-execution")
        relay.intercepts.deregister_tool_request("hermes-test-tool-request")

    assert callback_args == [{"path": "/approved/path"}]
    assert observed_args == {"path": "/approved/path"}
    assert isinstance(result, str)
    assert json.loads(result) == {"ok": True, "wrapped": True}






def test_tool_error_is_preserved_from_relay_wrapper_suffix(relay_turn, monkeypatch):
    relay = relay_turn

    class ToolError(Exception):
        pass

    tool_error = ToolError("dispatch failed")

    async def wrapping_execute(_name, args, callback, **_kwargs):
        try:
            return callback(args)
        except Exception as exc:
            raise RuntimeError(
                f"internal error: {type(exc).__name__}: {exc} (worker trace)"
            ) from None

    monkeypatch.setattr(relay.tools, "execute", wrapping_execute)

    with pytest.raises(ToolError) as caught:
        relay_tools.execute(
            "terminal",
            {"command": "false"},
            lambda _args: (_ for _ in ()).throw(tool_error),
            session_id="session-1",
        )

    assert caught.value is tool_error






