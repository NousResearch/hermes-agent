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


def test_patch_v4a_rejection_does_not_mutate(tmp_path):
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
                    " alpha\n"
                    "-beta\n"
                    "+gamma\n"
                    "*** End Patch\n"
                ),
            },
            task_id="acp-patch-v4a-reject",
        )
    )

    assert "error" in result
    assert "Edit approval denied" in result["error"]
    assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_patch_v4a_approval_request_includes_patch_targets(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    proposals = []

    set_edit_approval_requester(lambda proposal: proposals.append(proposal) or False)

    json.loads(
        handle_function_call(
            "patch",
            {
                "mode": "patch",
                "patch": (
                    "*** Begin Patch\n"
                    f"*** Update File: {target}\n"
                    "@@\n"
                    " alpha\n"
                    "-beta\n"
                    "+gamma\n"
                    "*** End Patch\n"
                ),
            },
            task_id="acp-patch-v4a-proposal",
        )
    )

    assert len(proposals) == 1
    assert proposals[0].tool_name == "patch"
    assert proposals[0].path == str(target)
    assert str(target) in proposals[0].new_text


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


def _v4a_multifile_patch(*paths: str) -> str:
    body = ["*** Begin Patch"]
    for p in paths:
        body.append(f"*** Update File: {p}")
        body.append("@@")
        body.append("+x")
    body.append("*** End Patch")
    return "\n".join(body) + "\n"


def test_shell_and_config_rc_files_are_sensitive(tmp_path):
    # New protected names added by this PR: writing any of these runs code or
    # redirects trust at the next shell/login/tool invocation.
    for name in (
        ".bashrc",
        ".bash_profile",
        ".profile",
        ".zshrc",
        ".zshenv",
        ".gitconfig",
        ".npmrc",
        ".pypirc",
        ".netrc",
        ".envrc",
    ):
        target = tmp_path / name
        assert not should_auto_approve_edit(
            EditProposal("write_file", str(target), None, "x", {}),
            "session",
            str(tmp_path),
        ), f"{name} should be treated as sensitive"


def test_persistence_and_credential_dir_paths_are_sensitive(tmp_path):
    for rel in (
        ".config/autostart/evil.desktop",
        ".config/systemd/user/evil.service",
        ".ssh/authorized_keys",
        ".git/config",
    ):
        target = tmp_path / rel
        assert not should_auto_approve_edit(
            EditProposal("write_file", str(target), None, "x", {}),
            "session",
            str(tmp_path),
        ), f"{rel} should be treated as sensitive"


def test_v4a_proposal_preserves_individual_target_paths(tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "pkg" / "b.py"
    proposal = build_edit_proposal(
        "patch", {"mode": "patch", "patch": _v4a_multifile_patch(str(a), str(b))}
    )
    assert proposal is not None
    # Individual real paths are preserved for security checks...
    assert proposal.target_paths == (str(a), str(b))
    # ...while the display path stays the joined, human-readable string.
    assert proposal.path == f"{a}, {b}"


def test_multifile_v4a_patch_with_sensitive_target_not_last_is_not_auto_approved(tmp_path):
    # Regression for the joined-display-string bug: the sensitive file is NOT the
    # trailing path, so parsing ", ".join(paths) as one Path previously saw only
    # "app.py" and auto-approved -- editing .bashrc with no prompt.
    sensitive = tmp_path / ".bashrc"
    normal = tmp_path / "app.py"
    proposal = build_edit_proposal(
        "patch",
        {"mode": "patch", "patch": _v4a_multifile_patch(str(sensitive), str(normal))},
    )
    assert proposal is not None
    assert proposal.target_paths == (str(sensitive), str(normal))
    # Even the most permissive non-ask policies must now force the prompt.
    assert not should_auto_approve_edit(proposal, "session", str(tmp_path))
    assert not should_auto_approve_edit(proposal, "workspace_session", str(tmp_path))


def test_multifile_v4a_patch_with_out_of_workspace_target_is_not_auto_approved(tmp_path):
    # The same joined-string flaw defeated the workspace-containment check: a
    # patch touching a file outside the workspace could still auto-approve.
    inside = tmp_path / "a.py"
    outside = Path(tmp_path.anchor) / "hermes_escape_dir_xyz" / "b.py"
    proposal = build_edit_proposal(
        "patch",
        {"mode": "patch", "patch": _v4a_multifile_patch(str(inside), str(outside))},
    )
    assert proposal is not None
    # workspace policy: one out-of-scope target forces the prompt.
    assert not should_auto_approve_edit(proposal, "workspace_session", str(tmp_path))
    # session policy trusts any non-sensitive path, so it still auto-approves.
    assert should_auto_approve_edit(proposal, "session", str(tmp_path))


def test_multifile_v4a_patch_all_safe_workspace_paths_auto_approves(tmp_path):
    # Ensure the stricter per-target logic does not over-block legitimate
    # multi-file patches entirely inside the workspace.
    a = tmp_path / "a.py"
    b = tmp_path / "pkg" / "b.py"
    proposal = build_edit_proposal(
        "patch", {"mode": "patch", "patch": _v4a_multifile_patch(str(a), str(b))}
    )
    assert proposal is not None
    assert should_auto_approve_edit(proposal, "workspace_session", str(tmp_path))
    assert should_auto_approve_edit(proposal, "session", str(tmp_path))
