"""Tests for the core Relay-managed Hermes tool adapter."""

from __future__ import annotations

import pytest

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
    try:
        yield lease.host.relay
    finally:
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()


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
    assert result == {"ok": True, "wrapped": True}


def test_provider_error_identity_is_preserved(relay_turn):
    del relay_turn

    class ToolError(Exception):
        pass

    tool_error = ToolError("dispatch failed")

    def fail(_args):
        raise tool_error

    with pytest.raises(ToolError) as caught:
        relay_tools.execute(
            "terminal",
            {"command": "false"},
            fail,
            session_id="session-1",
        )

    assert caught.value is tool_error


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


def test_tool_adapter_does_not_mask_relay_error_after_callback_failure(
    relay_turn, monkeypatch
):
    relay = relay_turn
    tool_error = RuntimeError("dispatch failed")
    relay_error = RuntimeError("internal error: RelayPolicyError: policy blocked")

    async def translating_execute(_name, args, callback, **_kwargs):
        try:
            callback(args)
        except Exception:
            raise relay_error

    monkeypatch.setattr(relay.tools, "execute", translating_execute)

    with pytest.raises(RuntimeError) as caught:
        relay_tools.execute(
            "terminal",
            {"command": "false"},
            lambda _args: (_ for _ in ()).throw(tool_error),
            session_id="session-1",
        )

    assert caught.value is relay_error
