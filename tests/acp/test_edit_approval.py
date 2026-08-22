"""Tests for ACP pre-edit approval gating."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from acp_adapter.edit_approval import (
    EditProposal,
    build_acp_edit_tool_call,
    build_edit_proposal,
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


def test_v4a_proposal_shows_simulated_result_not_patch_body(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    proposal = build_edit_proposal(
        "patch",
        {
            "mode": "patch",
            "patch": (
                "*** Begin Patch\n"
                f"*** Update File: {target}\n"
                "@@\n"
                "-beta\n"
                "+BETA\n"
                "*** End Patch"
            ),
        },
    )

    assert proposal is not None
    assert proposal.path == str(target)
    assert proposal.old_text == "alpha\nbeta\ngamma\n"
    assert proposal.new_text == "alpha\nBETA\ngamma\n"
    assert "*** Begin Patch" not in proposal.new_text


def test_v4a_proposal_delete_shows_empty_result(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")

    proposal = build_edit_proposal(
        "patch",
        {
            "mode": "patch",
            "patch": (
                "*** Begin Patch\n"
                f"*** Delete File: {target}\n"
                "*** End Patch"
            ),
        },
    )

    assert proposal is not None
    assert proposal.old_text == "alpha\nbeta\n"
    assert proposal.new_text == ""


def test_v4a_proposal_multi_file_falls_back_to_patch_body(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("aaa\n", encoding="utf-8")
    b.write_text("bbb\n", encoding="utf-8")

    proposal = build_edit_proposal(
        "patch",
        {
            "mode": "patch",
            "patch": (
                "*** Begin Patch\n"
                f"*** Update File: {a}\n"
                "@@\n"
                "-aaa\n"
                "+AAA\n"
                f"*** Update File: {b}\n"
                "@@\n"
                "-bbb\n"
                "+BBB\n"
                "*** End Patch"
            ),
        },
    )

    # ACP carries a single diff payload; multi-file patches keep the raw body.
    assert proposal is not None
    assert proposal.new_text.startswith("*** Begin Patch")
    assert proposal.old_text is None


def test_v4a_proposal_invalid_hunk_falls_back_to_patch_body(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("alpha\n", encoding="utf-8")

    proposal = build_edit_proposal(
        "patch",
        {
            "mode": "patch",
            "patch": (
                "*** Begin Patch\n"
                f"*** Update File: {target}\n"
                "@@\n"
                "-not-in-file\n"
                "+replacement\n"
                "*** End Patch"
            ),
        },
    )

    # Simulation failed validation -> keep the previous behavior so the call
    # is still permissioned with the raw body (and would be rejected at
    # execution anyway).
    assert proposal is not None
    assert proposal.new_text.startswith("*** Begin Patch")


def test_v4a_rejection_does_not_mutate(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")

    set_edit_approval_requester(lambda _proposal: False)

    result = json.loads(
        handle_function_call(
            "patch",
            {
                "mode": "patch",
                "patch": (
                    "*** Begin Patch\n"
                    f"*** Update File: {target}\n"
                    "@@\n"
                    "-beta\n"
                    "+gamma\n"
                    "*** End Patch"
                ),
            },
            task_id="acp-v4a-reject",
        )
    )

    assert "error" in result
    assert "Edit approval denied" in result["error"]
    assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"
