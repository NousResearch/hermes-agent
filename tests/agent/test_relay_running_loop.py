"""Synchronous managed Relay calls reject event-loop threads without leaking coroutines."""

import asyncio
import gc
import warnings

import pytest

pytest.importorskip("nemo_relay")

from agent import relay_llm, relay_runtime, relay_tools


@pytest.fixture()
def relay_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    relay_runtime._reset_for_tests()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(), session_id="session-1", platform="cli",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(lease, turn_id="turn-1", task_id="task-1")
    lease.host.retain_managed_execution("test.relay_loop")
    try:
        yield
    finally:
        lease.host.release_managed_execution("test.relay_loop")
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["llm", "stream", "tool"])
async def test_managed_sync_calls_do_not_leak_coroutines_on_running_loop(relay_turn, kind):
    request = {"model": "test-model", "messages": []}
    common = {"session_id": "session-1", "name": "custom", "model_name": "test-model"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        with pytest.raises(RuntimeError, match="event.loop|running event loop"):
            if kind == "stream":
                relay_llm.stream(request, lambda _: iter(()), finalizer=dict, **common)
            elif kind == "tool":
                relay_tools.execute("example", {}, lambda _: {}, session_id="session-1")
            else:
                relay_llm.execute(request, lambda _: {}, **common)
        gc.collect()

    assert not [warning for warning in caught if "was never awaited" in str(warning.message)]


def test_managed_stream_iteration_on_running_loop_does_not_leak(relay_turn):
    stream = relay_llm.stream(
        {"model": "test-model", "messages": []}, lambda _: iter([{"delta": "ok"}]),
        session_id="session-1", name="custom", model_name="test-model", finalizer=dict,
    )

    async def consume():
        with pytest.raises(RuntimeError, match="event.loop|running event loop"):
            next(stream)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        try:
            asyncio.run(consume())
            gc.collect()
        finally:
            stream.close()
    assert not [warning for warning in caught if "was never awaited" in str(warning.message)]
