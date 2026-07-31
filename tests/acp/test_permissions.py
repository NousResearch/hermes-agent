"""Tests for acp_adapter.permissions."""

import asyncio
import inspect
from concurrent.futures import Future
from unittest.mock import AsyncMock, MagicMock, patch

from acp.schema import (
    AllowedOutcome,
    DeniedOutcome,
    RequestPermissionResponse,
)

from acp_adapter.permissions import make_approval_callback
from tools.approval import prompt_dangerous_approval


def _make_response(outcome):
    return RequestPermissionResponse(outcome=outcome)


def _invoke_callback(
    outcome,
    *,
    allow_permanent=True,
    smart_denied=False,
    timeout=60.0,
    use_prompt_path=False,
):
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    request_permission = AsyncMock(name="request_permission")
    future = MagicMock(spec=Future)
    future.result.return_value = _make_response(outcome)

    scheduled = {}

    def _schedule(coro, passed_loop):
        scheduled["coro"] = coro
        scheduled["loop"] = passed_loop
        return future

    with patch("agent.async_utils.asyncio.run_coroutine_threadsafe", side_effect=_schedule):
        cb = make_approval_callback(request_permission, loop, session_id="s1", timeout=timeout)
        if use_prompt_path:
            result = prompt_dangerous_approval(
                "rm -rf /",
                "dangerous command",
                allow_permanent=allow_permanent,
                smart_denied=smart_denied,
                approval_callback=cb,
            )
        else:
            result = cb(
                "rm -rf /",
                "dangerous command",
                allow_permanent=allow_permanent,
                smart_denied=smart_denied,
            )

    scheduled["coro"].close()
    _, kwargs = request_permission.call_args
    return result, kwargs, scheduled, future, loop


class TestApprovalBridge:
    def test_bridge_schedules_request_on_the_given_loop(self):
        result, kwargs, scheduled, _, loop = _invoke_callback(
            AllowedOutcome(option_id="allow_once", outcome="selected"),
        )

        tool_call = kwargs["tool_call"]
        option_ids = [option.option_id for option in kwargs["options"]]

        assert result == "once"
        assert scheduled["loop"] is loop
        assert inspect.iscoroutine(scheduled["coro"])
        assert kwargs["session_id"] == "s1"
        assert tool_call.session_update == "tool_call_update"
        assert tool_call.tool_call_id.startswith("perm-check-")
        assert tool_call.kind == "execute"
        assert tool_call.status == "pending"
        assert "dangerous command" in tool_call.title
        assert "rm -rf /" in tool_call.title
        content_text = tool_call.content[0].content.text
        assert "$ rm -rf /" in content_text
        assert "dangerous command" in content_text
        assert tool_call.raw_input == {
            "command": "rm -rf /",
            "description": "dangerous command",
        }
        assert option_ids == [
            "allow_once",
            "allow_session",
            "allow_always",
            "deny",
            "deny_always",
        ]

    def test_tool_call_ids_are_unique(self):
        _, first_kwargs, _, _, _ = _invoke_callback(
            AllowedOutcome(option_id="allow_once", outcome="selected"),
        )
        _, second_kwargs, _, _, _ = _invoke_callback(
            AllowedOutcome(option_id="allow_once", outcome="selected"),
        )

        assert first_kwargs["tool_call"].tool_call_id != second_kwargs["tool_call"].tool_call_id





    def test_allow_always_maps_correctly(self):
        result, _, _, _, _ = _invoke_callback(
            AllowedOutcome(option_id="allow_always", outcome="selected"),
            use_prompt_path=True,
        )

        assert result == "always"


    def test_timeout_returns_deny_and_cancels_future(self):
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        request_permission = AsyncMock(name="request_permission")
        future = MagicMock(spec=Future)
        future.result.side_effect = TimeoutError("timed out")

        scheduled = {}

        def _schedule(coro, passed_loop):
            scheduled["coro"] = coro
            scheduled["loop"] = passed_loop
            return future

        with patch("agent.async_utils.asyncio.run_coroutine_threadsafe", side_effect=_schedule):
            cb = make_approval_callback(request_permission, loop, session_id="s1", timeout=0.01)
            result = cb("rm -rf /", "dangerous command")

        scheduled["coro"].close()

        assert result == "deny"
        assert scheduled["loop"] is loop
        assert future.cancel.call_count == 1



# ---------------------------------------------------------------------------
# Scheduler-failure regression
# ---------------------------------------------------------------------------

import gc  # noqa: E402
import warnings  # noqa: E402


class TestSchedulerFailure:
    def test_scheduler_failure_closes_permission_coroutine(self):
        """If run_coroutine_threadsafe raises, the coro is closed and we return 'deny'."""
        loop = MagicMock(spec=asyncio.AbstractEventLoop)
        created = {"coro": None}

        async def _response_coro(**kwargs):
            return _make_response(AllowedOutcome(option_id="allow_once", outcome="selected"))

        def _request_permission(**kwargs):
            created["coro"] = _response_coro(**kwargs)
            return created["coro"]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with patch(
                "agent.async_utils.asyncio.run_coroutine_threadsafe",
                side_effect=RuntimeError("scheduler down"),
            ):
                cb = make_approval_callback(_request_permission, loop, session_id="s1", timeout=0.01)
                result = cb("rm -rf /", "dangerous")
            gc.collect()

        assert result == "deny"
        assert created["coro"] is not None
        assert created["coro"].cr_frame is None
        runtime_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning)
            and "was never awaited" in str(w.message)
            and "_response_coro" in str(w.message)
        ]
        assert runtime_warnings == []
