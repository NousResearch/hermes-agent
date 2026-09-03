"""Tests for ACP pre-edit approval gating."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from acp_adapter.edit_approval import (
    EditProposal,
    build_acp_edit_tool_call,
    clear_edit_approval_requester,
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


def test_already_applied_patch_skips_approval_and_reports_no_change(tmp_path):
    """tools.file_operations.patch_replace() rescues a re-send of an
    already-landed edit (identical old/new strings, or old_string gone
    while new_string is present) into a graceful success no-op instead of
    an error — the most common patch failure in production. The ACP
    proposal builder used to raise before that rescue was ever reached,
    turning it into a misleading "denied by ACP client" error the user was
    never actually asked about. The requester below always denies, so if
    approval were requested this test would see an "Edit approval denied"
    error instead of a no-op success.
    """
    target = tmp_path / "sample.txt"
    target_text = "this text is already updated and present in the file"
    target.write_text(f"alpha\n{target_text}\nbeta\n", encoding="utf-8")

    requester_calls = []
    set_edit_approval_requester(lambda proposal: requester_calls.append(proposal) or False)

    result = json.loads(
        handle_function_call(
            "patch",
            {
                "mode": "replace",
                "path": str(target),
                "old_string": target_text,
                "new_string": target_text,
            },
            task_id="acp-patch-already-applied",
        )
    )

    assert requester_calls == []
    assert result.get("success") is True
    assert result.get("no_change") is True
    assert target.read_text(encoding="utf-8") == f"alpha\n{target_text}\nbeta\n"


def test_genuinely_unmatched_patch_still_denies(tmp_path):
    """A real "old_string not found" failure must still be blocked — the
    already-applied rescue must not swallow genuine build failures."""
    target = tmp_path / "sample.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")

    set_edit_approval_requester(lambda _proposal: True)

    result = json.loads(
        handle_function_call(
            "patch",
            {
                "mode": "replace",
                "path": str(target),
                "old_string": "this text does not exist in the file at all",
                "new_string": "replacement text",
            },
            task_id="acp-patch-genuine-failure",
        )
    )

    assert "error" in result
    assert "Edit approval denied" in result["error"]
    assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"








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
