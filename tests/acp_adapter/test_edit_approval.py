"""Tests for ACP pre-edit approval gating."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from acp_adapter import edit_approval
from acp_adapter.edit_approval import (
    EditProposal,
    build_acp_edit_tool_call,
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


def test_edit_permission_timeout_latches_only_its_acp_session(monkeypatch):
    calls = []

    def timed_out(_request_permission, _loop, session_id, **_kwargs):
        calls.append(session_id)
        return None, True

    monkeypatch.setattr("acp_adapter.permissions.await_permission", timed_out)
    proposal = EditProposal("write_file", "demo.txt", None, "new\n", {})
    first_state = edit_approval.EditApprovalState()
    first_session = edit_approval.make_acp_edit_approval_requester(
        lambda **_kwargs: None, None, "s1", state=first_state
    )
    second_session = edit_approval.make_acp_edit_approval_requester(
        lambda **_kwargs: None, None, "s2", state=edit_approval.EditApprovalState()
    )

    assert first_session(proposal) is False
    first_session = edit_approval.make_acp_edit_approval_requester(
        lambda **_kwargs: None, None, "s1", state=first_state
    )
    assert first_session(proposal) is False
    assert second_session(proposal) is False
    assert calls == ["s1", "s2"]


def test_explicit_edit_denial_does_not_latch(monkeypatch):
    calls = []

    def denied(_request_permission, _loop, session_id, **_kwargs):
        calls.append(session_id)
        outcome = SimpleNamespace(outcome="selected", option_id="deny")
        return SimpleNamespace(outcome=outcome), False

    monkeypatch.setattr("acp_adapter.permissions.await_permission", denied)
    state = edit_approval.EditApprovalState()
    requester = edit_approval.make_acp_edit_approval_requester(
        lambda **_kwargs: None, None, "s1", state=state
    )
    proposal = EditProposal("write_file", "demo.txt", None, "new\n", {})

    assert requester(proposal) is False
    requester = edit_approval.make_acp_edit_approval_requester(
        lambda **_kwargs: None, None, "s1", state=state
    )
    assert requester(proposal) is False
    assert calls == ["s1", "s1"]








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
