"""Tests for ACP pre-edit approval gating."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from acp_adapter.edit_approval import (
    EditProposal,
    _is_sensitive_auto_approve_path,
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


# ── Regression tests for symlink-based bypass of the sensitive-path guard
# (#55367). Without symlink resolution, an attacker could plant
# `project/notes.txt -> ~/.ssh/authorized_keys` and the literal-path
# check (`Path("project/notes.txt").parts` → `('project', 'notes.txt')`)
# would not catch it, leading to auto-approval and a write to the
# credential file.


def _make_symlink(src: Path, link_path: Path) -> Path:
    """Create `link_path` as a symlink to `src`. Skip on Windows / unsupported FS."""
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks are not supported on this platform")
    link_path.symlink_to(src)
    return link_path


def test_symlink_to_sensitive_file_is_treated_as_sensitive(tmp_path):
    """A symlink whose target is in ~/.ssh/ must be auto-approved=False."""
    sensitive = tmp_path / ".ssh" / "authorized_keys"
    sensitive.parent.mkdir()
    sensitive.write_text("original\n")
    link = tmp_path / "project" / "notes.txt"
    link.parent.mkdir()
    _make_symlink(sensitive, link)

    # `should_auto_approve_edit` with `session` policy normally returns True
    # for any non-sensitive path. With the symlink fix, it must NOT
    # auto-approve because the link target is sensitive.
    assert not should_auto_approve_edit(
        EditProposal("write_file", str(link), None, "ATTACKER", {}),
        "session",
        None,
    )


def test_symlink_to_id_rsa_is_treated_as_sensitive(tmp_path):
    """A symlink whose target name is in SENSITIVE_AUTO_APPROVE_NAMES."""
    secret = tmp_path / "stuff" / "id_rsa"
    secret.parent.mkdir()
    secret.write_text("PRIVATE KEY\n")
    link = tmp_path / "random_looking.md"
    _make_symlink(secret, link)

    assert not should_auto_approve_edit(
        EditProposal("write_file", str(link), None, "OVERWRITE", {}),
        "session",
        None,
    )


def test_symlink_to_innocuous_file_remains_auto_approvable(tmp_path):
    """Symlinks to non-sensitive targets must still be auto-approvable."""
    real = tmp_path / "real_notes.txt"
    real.write_text("hello\n")
    link = tmp_path / "alias.txt"
    _make_symlink(real, link)

    assert should_auto_approve_edit(
        EditProposal("write_file", str(link), None, "hi", {}),
        "session",
        None,
    )


def test_broken_symlink_falls_back_to_literal_path(tmp_path):
    """A symlink whose target doesn't exist must not crash and must
    respect the literal-path check."""
    # A literal name that is sensitive on the basename alone.
    link = tmp_path / "id_rsa"
    _make_symlink(tmp_path / "nonexistent_target", link)

    # Literal-path check fires on `id_rsa` basename even though the
    # symlink target doesn't exist.
    assert not should_auto_approve_edit(
        EditProposal("write_file", str(link), None, "x", {}),
        "session",
        None,
    )


def test_is_sensitive_directly():
    """Direct unit tests of the helper for the cases the policy test covers."""
    # Literal sensitive
    assert _is_sensitive_auto_approve_path("~/.ssh/id_rsa")
    assert _is_sensitive_auto_approve_path("project/.env")
    # Literal innocent
    assert not _is_sensitive_auto_approve_path("project/notes.txt")
    assert not _is_sensitive_auto_approve_path("foo/bar/baz.md")
