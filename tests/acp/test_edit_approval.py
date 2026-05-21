"""Tests for ACP pre-edit approval gating."""

from __future__ import annotations

import asyncio
import json
import tempfile
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from acp.schema import AllowedOutcome, RequestPermissionResponse

from acp_adapter.edit_approval import (
    EditProposal,
    build_acp_edit_tool_call,
    clear_edit_approval_requester,
    make_acp_edit_approval_requester,
    set_edit_approval_requester,
    should_auto_approve_edit,
)
from model_tools import handle_function_call


def teardown_function() -> None:
    clear_edit_approval_requester()


def test_acp_permission_tool_call_uses_edit_kind_and_diff_content():
    proposal = EditProposal(
        tool_name="write_file",
        path="demo.txt",
        old_text="old\n",
        new_text="new\n",
        arguments={"path": "demo.txt", "content": "new\n"},
    )

    tool_call = build_acp_edit_tool_call(proposal)

    assert tool_call.kind == "edit"
    assert tool_call.status == "pending"
    assert tool_call.rawInput == {"tool": "write_file", "arguments": proposal.arguments}
    assert len(tool_call.content) == 1
    diff = tool_call.content[0]
    assert diff.path == "demo.txt"
    assert diff.oldText == "old\n"
    assert diff.newText == "new\n"


def test_write_file_rejection_does_not_mutate_existing_file(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("before\n", encoding="utf-8")

    set_edit_approval_requester(lambda _proposal: False)

    result = json.loads(
        handle_function_call(
            "write_file",
            {"path": str(target), "content": "after\n"},
            task_id="acp-edit-reject",
        )
    )

    assert "error" in result
    assert "Edit approval denied" in result["error"]
    assert target.read_text(encoding="utf-8") == "before\n"


def test_write_file_approval_mutates_and_request_includes_diff(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("before\n", encoding="utf-8")
    proposals = []

    def approve(proposal):
        proposals.append(proposal)
        return True

    set_edit_approval_requester(approve)

    result = json.loads(
        handle_function_call(
            "write_file",
            {"path": str(target), "content": "after\n"},
            task_id="acp-edit-approve",
        )
    )

    assert result.get("bytes_written") == len("after\n")
    assert target.read_text(encoding="utf-8") == "after\n"
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.tool_name == "write_file"
    assert proposal.path == str(target)
    assert proposal.old_text == "before\n"
    assert proposal.new_text == "after\n"


def test_write_file_new_file_request_has_empty_old_text(tmp_path):
    target = tmp_path / "new.txt"
    proposals = []

    set_edit_approval_requester(lambda proposal: proposals.append(proposal) or True)

    result = json.loads(
        handle_function_call(
            "write_file",
            {"path": str(target), "content": "created\n"},
            task_id="acp-edit-new-file",
        )
    )

    assert result.get("bytes_written") == len("created\n")
    assert target.read_text(encoding="utf-8") == "created\n"
    assert proposals[0].old_text is None
    assert proposals[0].new_text == "created\n"


def test_requester_exception_denies_and_does_not_mutate(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("before\n", encoding="utf-8")

    def boom(_proposal):
        raise RuntimeError("zed disconnected")

    set_edit_approval_requester(boom)

    result = json.loads(
        handle_function_call(
            "write_file",
            {"path": str(target), "content": "after\n"},
            task_id="acp-edit-exception",
        )
    )

    assert "error" in result
    assert "Edit approval denied" in result["error"]
    assert target.read_text(encoding="utf-8") == "before\n"


def test_patch_replace_rejection_does_not_mutate(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")

    set_edit_approval_requester(lambda _proposal: False)

    result = json.loads(
        handle_function_call(
            "patch",
            {
                "mode": "replace",
                "path": str(target),
                "old_string": "beta\n",
                "new_string": "gamma\n",
            },
            task_id="acp-patch-reject",
        )
    )

    assert "error" in result
    assert "Edit approval denied" in result["error"]
    assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_patch_replace_approval_request_includes_full_file_diff(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    proposals = []

    set_edit_approval_requester(lambda proposal: proposals.append(proposal) or True)

    result = json.loads(
        handle_function_call(
            "patch",
            {
                "mode": "replace",
                "path": str(target),
                "old_string": "beta\n",
                "new_string": "gamma\n",
            },
            task_id="acp-patch-approve",
        )
    )

    assert result.get("success") is True
    assert target.read_text(encoding="utf-8") == "alpha\ngamma\n"
    assert proposals[0].tool_name == "patch"
    assert proposals[0].old_text == "alpha\nbeta\n"
    assert proposals[0].new_text == "alpha\ngamma\n"


def _build_requester_with_responses(responses):
    """Build a requester whose underlying ACP permission call returns the
    given outcomes in order, one per call. Returns ``(requester, call_log)``.

    ``call_log`` is a list of the option-id lists offered to each request,
    so tests can assert that ``allow_session`` is surfaced and that
    subsequent edits skip the prompt entirely (call_log shorter than the
    number of edits proposed).
    """

    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    request_permission = AsyncMock(name="request_permission")
    call_log: list[list[str]] = []

    response_iter = iter(responses)

    def _schedule(coro, passed_loop):
        coro.close()
        fut = MagicMock(spec=Future)
        fut.result.return_value = RequestPermissionResponse(outcome=next(response_iter))
        return fut

    patcher = patch(
        "agent.async_utils.asyncio.run_coroutine_threadsafe",
        side_effect=_schedule,
    )
    patcher.start()
    try:
        requester = make_acp_edit_approval_requester(
            request_permission, loop, session_id="s1", timeout=1.0,
        )
    finally:
        # Keep the patch active for the lifetime of the requester so that
        # repeated calls within a test exercise the same _schedule shim.
        pass

    original = requester

    def wrapped(proposal):
        result = original(proposal)
        # Record which options were offered on this call (empty if the
        # session opt-in short-circuited before scheduling the permission
        # request).
        if request_permission.call_args is not None:
            kwargs = request_permission.call_args.kwargs
            call_log.append([opt.option_id for opt in kwargs.get("options", [])])
            request_permission.reset_mock()
        else:
            call_log.append([])
        return result

    wrapped._patcher = patcher  # so callers can stop it
    return wrapped, call_log


def test_edit_approval_options_include_allow_session(tmp_path):
    proposal = EditProposal("write_file", str(tmp_path / "x.py"), None, "x", {})
    requester, call_log = _build_requester_with_responses(
        [AllowedOutcome(option_id="allow_once", outcome="selected")]
    )
    try:
        assert requester(proposal) is True
        assert call_log == [["allow_once", "allow_session", "deny"]]
    finally:
        requester._patcher.stop()


def test_edit_approval_allow_session_skips_subsequent_prompts(tmp_path):
    first = EditProposal("write_file", str(tmp_path / "a.py"), None, "a", {})
    second = EditProposal("write_file", str(tmp_path / "b.py"), None, "b", {})
    third = EditProposal("write_file", str(tmp_path / "c.py"), None, "c", {})

    requester, call_log = _build_requester_with_responses(
        [AllowedOutcome(option_id="allow_session", outcome="selected")]
    )
    try:
        assert requester(first) is True
        assert requester(second) is True
        assert requester(third) is True
        # Only the first proposal prompted; the next two short-circuited.
        assert call_log == [["allow_once", "allow_session", "deny"], [], []]
    finally:
        requester._patcher.stop()


def test_edit_approval_allow_session_still_prompts_on_sensitive_paths(tmp_path):
    benign = EditProposal("write_file", str(tmp_path / "a.py"), None, "a", {})
    sensitive = EditProposal("write_file", str(tmp_path / ".env"), None, "SECRET=x", {})

    requester, call_log = _build_requester_with_responses(
        [
            AllowedOutcome(option_id="allow_session", outcome="selected"),
            AllowedOutcome(option_id="allow_once", outcome="selected"),
        ]
    )
    try:
        assert requester(benign) is True
        # Sensitive path must re-prompt even after session opt-in.
        assert requester(sensitive) is True
        assert call_log == [
            ["allow_once", "allow_session", "deny"],
            ["allow_once", "allow_session", "deny"],
        ]
    finally:
        requester._patcher.stop()


def test_edit_approval_allow_once_does_not_enable_session(tmp_path):
    first = EditProposal("write_file", str(tmp_path / "a.py"), None, "a", {})
    second = EditProposal("write_file", str(tmp_path / "b.py"), None, "b", {})

    requester, call_log = _build_requester_with_responses(
        [
            AllowedOutcome(option_id="allow_once", outcome="selected"),
            AllowedOutcome(option_id="allow_once", outcome="selected"),
        ]
    )
    try:
        assert requester(first) is True
        assert requester(second) is True
        # Each proposal prompts — allow_once does not persist across calls.
        assert call_log == [
            ["allow_once", "allow_session", "deny"],
            ["allow_once", "allow_session", "deny"],
        ]
    finally:
        requester._patcher.stop()


def test_workspace_auto_approval_allows_workspace_and_tmp_but_not_sensitive(tmp_path):
    workspace_file = tmp_path / "src.py"
    # Use tempfile.gettempdir() so this test exercises the same code path on
    # Linux (`/tmp`), macOS (`/private/var/folders/...`) and Windows
    # (`%LOCALAPPDATA%\Temp`). Before the fix this branch only worked on Linux.
    tmp_file = Path(tempfile.gettempdir()) / "hermes-acp-auto-approve-test.txt"
    env_file = tmp_path / ".env"

    assert should_auto_approve_edit(
        EditProposal("write_file", str(workspace_file), None, "x", {}),
        "workspace_session",
        str(tmp_path),
    )
    assert should_auto_approve_edit(
        EditProposal("write_file", str(tmp_file), None, "x", {}),
        "workspace_session",
        str(tmp_path),
    )
    assert not should_auto_approve_edit(
        EditProposal("write_file", str(env_file), None, "SECRET=x", {}),
        "session",
        str(tmp_path),
    )
