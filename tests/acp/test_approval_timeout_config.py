"""Test that ACP approval callbacks honor the approvals.timeout config."""
from unittest.mock import MagicMock, patch

import pytest


def test_acp_approval_callback_uses_config_timeout():
    """Test that make_approval_callback receives timeout from _get_approval_timeout."""
    from acp_adapter.permissions import make_approval_callback
    from acp.schema import AllowedOutcome

    loop = MagicMock()
    timeout_used = []

    # Mock the ACP permission request to succeed
    request_permission = MagicMock()

    def _schedule(coro, passed_loop, **kwargs):
        future = MagicMock()
        # Mock response object with outcome attribute
        response_mock = MagicMock()
        response_mock.outcome = AllowedOutcome(option_id="allow_once", outcome="selected")
        future.result.return_value = response_mock

        # Capture timeout argument when result() is called
        def _result_with_timeout(*args, **kwargs):
            timeout_used.append(kwargs.get("timeout"))
            return response_mock
        future.result = _result_with_timeout

        return future

    with patch(
        "agent.async_utils.safe_schedule_threadsafe", side_effect=_schedule
    ), patch("tools.approval._get_approval_timeout", return_value=42):
        from tools.approval import _get_approval_timeout

        approval_timeout = float(_get_approval_timeout())
        cb = make_approval_callback(request_permission, loop, session_id="s1", timeout=approval_timeout)

        result = cb("rm -rf /", "dangerous command")

        assert result == "once"
        assert timeout_used == [42.0]


def test_acp_edit_approval_requester_uses_config_timeout():
    """Test that make_acp_edit_approval_requester receives timeout from _get_approval_timeout."""
    from acp_adapter.edit_approval import EditProposal, make_acp_edit_approval_requester
    from acp.schema import AllowedOutcome

    loop = MagicMock()
    timeout_used = []

    # Mock the ACP permission request to succeed
    request_permission = MagicMock()

    def _schedule(coro, passed_loop, **kwargs):
        future = MagicMock()
        # Mock response object with outcome attribute
        response_mock = MagicMock()
        response_mock.outcome = AllowedOutcome(option_id="allow_once", outcome="selected")
        future.result.return_value = response_mock

        # Capture timeout argument when result() is called
        def _result_with_timeout(*args, **kwargs):
            timeout_used.append(kwargs.get("timeout"))
            return response_mock
        future.result = _result_with_timeout

        return future

    with patch(
        "agent.async_utils.safe_schedule_threadsafe", side_effect=_schedule
    ), patch("tools.approval._get_approval_timeout", return_value=42):
        from tools.approval import _get_approval_timeout

        approval_timeout = float(_get_approval_timeout())
        requester = make_acp_edit_approval_requester(
            request_permission,
            loop,
            session_id="s1",
            timeout=approval_timeout,
        )

        proposal = EditProposal(
            tool_name="write_file",
            path="/tmp/test.txt",
            old_text="old",
            new_text="new",
            arguments={"path": "/tmp/test.txt", "content": "new"}
        )
        result = requester(proposal)

        assert result is True
        assert timeout_used == [42.0]