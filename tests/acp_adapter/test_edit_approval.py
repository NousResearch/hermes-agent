"""Tests for ACP pre-edit approval gating."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from acp_adapter.edit_approval import (
    EditProposal,
    build_acp_edit_tool_call,
    build_edit_proposal,
    set_edit_approval_requester,
    should_auto_approve_edit,
)
from model_tools import handle_function_call


def teardown_function() -> None:
    set_edit_approval_requester(None)


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


def test_multifile_v4a_patch_checks_every_real_path_not_the_joined_display_string(tmp_path):
    """Multi-file V4A proposals carry a comma-joined ``path`` for the dialog; the auto-approve
    checks must run per real target so a sensitive or escaping file cannot hide in the join (#115213)."""
    hidden_env = "*** Update File: .env\n@@\n+SECRET=x\n*** Update File: src/ok.py\n@@\n+ok\n"
    proposal = build_edit_proposal("patch", {"mode": "patch", "patch": hidden_env})
    assert not should_auto_approve_edit(proposal, "session", str(tmp_path))
    assert not should_auto_approve_edit(proposal, "workspace_session", str(tmp_path))

    # Escape target sits outside BOTH allowed roots (tempdir + session cwd).
    escape = Path.home() / ".hermes-acp-escape-test.txt"
    hidden_escape = f"*** Update File: {tmp_path}/src/ok.py\n@@\n+ok\n*** Update File: {escape}\n@@\n+evil\n"
    proposal = build_edit_proposal("patch", {"mode": "patch", "patch": hidden_escape})
    assert not should_auto_approve_edit(proposal, "workspace_session", str(tmp_path))

    # Control: every target inside the workspace still auto-approves.
    all_safe = f"*** Update File: {tmp_path}/a.py\n@@\n+a\n*** Update File: {tmp_path}/b.py\n@@\n+b\n"
    proposal = build_edit_proposal("patch", {"mode": "patch", "patch": all_safe})
    assert should_auto_approve_edit(proposal, "workspace_session", str(tmp_path))


def test_multifile_v4a_env_write_reaches_permission_prompt_e2e(tmp_path):
    """Pre-fix, ``session`` policy auto-approved the ``.env`` hidden in the join."""
    env_target = tmp_path / ".env"
    ok_target = tmp_path / "ok.py"
    patch = (
        f"*** Update File: {env_target}\n@@\n+SECRET=x\n"
        f"*** Update File: {ok_target}\n@@\n+ok\n"
    )

    prompted = []

    def requester(proposal):
        if should_auto_approve_edit(proposal, "session", str(tmp_path)):
            return True
        prompted.append(proposal.path)
        return False

    set_edit_approval_requester(requester)
    result = json.loads(
        handle_function_call("patch", {"mode": "patch", "patch": patch}, task_id="acp-v4a-e2e")
    )

    assert prompted, "multi-file .env patch must reach the prompt, not auto-approve"
    assert "denied" in result["error"].lower()
    assert not env_target.exists()
    assert not ok_target.exists()
