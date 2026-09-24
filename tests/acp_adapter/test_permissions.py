"""Tests for acp_adapter.permissions."""

import asyncio
import inspect
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
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
    allow_session=True,
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
                allow_session=allow_session,
                smart_denied=smart_denied,
                approval_callback=cb,
            )
        else:
            result = cb(
                "rm -rf /",
                "dangerous command",
                allow_permanent=allow_permanent,
                allow_session=allow_session,
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

    def test_session_less_gate_offers_only_once_and_deny(self):
        """allow_session=False collapses the editor menu to once/deny.

        Hermes discards any scope broader than one operation for the
        protected agent-instruction gate, so an editor that renders
        "Allow for session" would re-prompt on the next write (#81887).
        """
        _, kwargs, _, _, _ = _invoke_callback(
            AllowedOutcome(option_id="allow_once", outcome="selected"),
            allow_permanent=False,
            allow_session=False,
        )

        assert [option.option_id for option in kwargs["options"]] == ["allow_once", "deny"]

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


    def test_timeout_returns_timeout_and_cancels_future(self):
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

        # A no-response expiry is classified as "timeout" (still blocked,
        # fail-closed) so the agent isn't told the user explicitly refused.
        assert result == "timeout"
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


class TestPermissionRequestToolCallReachesATerminalStatus:
    """The ``perm-check-N`` / ``edit-approval-N`` ToolCallUpdate attached to ``request_permission``
    is materialised by clients as a pending bubble; once the user answers it must be closed."""

    @staticmethod
    def _run(factory, outcome, call):
        request_permission = AsyncMock(name="request_permission")
        future = MagicMock(spec=Future)
        future.result.return_value = _make_response(outcome)
        sent = []
        with patch("agent.async_utils.asyncio.run_coroutine_threadsafe", return_value=future):
            cb = factory(request_permission, MagicMock(spec=asyncio.AbstractEventLoop), "s1", send_update=sent.append)
            call(cb)
        requested = request_permission.call_args.kwargs["tool_call"]
        return requested, sent

    @pytest.mark.parametrize("outcome, status", [
        (AllowedOutcome(option_id="allow_once", outcome="selected"), "completed"),
        (DeniedOutcome(outcome="cancelled"), "failed"),
    ])
    def test_command_permission_request_is_closed_per_outcome(self, outcome, status):
        requested, sent = self._run(make_approval_callback, outcome, lambda cb: cb("rm -rf /", "dangerous"))
        assert [(u.tool_call_id, u.status) for u in sent] == [(requested.tool_call_id, status)]

    def test_denied_edit_approval_request_is_closed_as_failed(self):
        from acp_adapter.edit_approval import EditProposal, make_acp_edit_approval_requester

        proposal = EditProposal(tool_name="write_file", path="/tmp/x", old_text="", new_text="y", arguments={})
        requested, sent = self._run(
            make_acp_edit_approval_requester, DeniedOutcome(outcome="cancelled"), lambda cb: cb(proposal),
        )
        assert [(u.tool_call_id, u.status) for u in sent] == [(requested.tool_call_id, "failed")]

    def test_allowed_edit_approval_request_is_closed_as_completed_once(self):
        """Live regression: a client answering with a plain ``selected`` outcome (not the SDK
        ``AllowedOutcome`` class) had the edit applied but the bubble closed ``failed``."""
        from acp_adapter.edit_approval import EditProposal, make_acp_edit_approval_requester

        proposal = EditProposal(tool_name="write_file", path="/tmp/x", old_text="", new_text="y", arguments={})
        decisions = []
        response = SimpleNamespace(outcome=SimpleNamespace(outcome="selected", option_id="allow_once"))
        request_permission = AsyncMock(name="request_permission")
        future = MagicMock(spec=Future)
        future.result.return_value = response
        sent = []
        with patch("agent.async_utils.asyncio.run_coroutine_threadsafe", return_value=future):
            requester = make_acp_edit_approval_requester(
                request_permission, MagicMock(spec=asyncio.AbstractEventLoop), "s1", send_update=sent.append,
            )
            decisions.append(requester(proposal))
        requested = request_permission.call_args.kwargs["tool_call"]
        assert decisions == [True]
        assert [(u.tool_call_id, u.status) for u in sent] == [(requested.tool_call_id, "completed")]


def test_default_permission_timeout_follows_approvals_config(monkeypatch):
    """No explicit timeout → the ACP bridge waits ``approvals.timeout`` like every other
    surface, instead of a hardcoded 60 s the host cannot raise (#73403)."""
    from acp_adapter import permissions
    from tools import approval_context

    monkeypatch.setattr(approval_context, "_get_approval_config", lambda: {"timeout": 900})
    assert permissions.resolve_permission_timeout(None) == 900.0
    assert permissions.resolve_permission_timeout(0.5) == 0.5
