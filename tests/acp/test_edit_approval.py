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


def test_patch_replace_missing_args_carries_tool_guidance():
    """The ACP proposal builder runs before patch_tool and must not dead-end.

    It previously raised a bare "old_string and new_string required". The caller
    wraps a build failure as "Edit approval denied: could not prepare diff (...)",
    so a malformed call was reported to the model as if the user had refused it,
    with no route back. It now raises the same guidance patch_tool returns.
    """
    import pytest

    from acp_adapter.edit_approval import _proposal_for_patch_replace
    from tools.file_tools import REPLACE_MODE_ARGS_HELP

    with pytest.raises(ValueError) as excinfo:
        _proposal_for_patch_replace({"path": "f.py", "new_string": "x"})

    message = str(excinfo.value)
    assert message == REPLACE_MODE_ARGS_HELP, "must stay in sync with the tool's error"
    assert "mode='patch'" in message
    assert "not rewrite" in message.lower()


def test_malformed_patch_call_is_not_reported_as_a_denial():
    """A malformed call must not be framed as the user refusing the edit.

    build_edit_proposal raises before patch_tool runs. That ValueError used to be
    wrapped as "Edit approval denied: could not prepare diff (...)", telling the
    model its edit was rejected when in fact its arguments were wrong — so it
    retries the same broken call or gives up rather than fixing them.
    """
    from acp_adapter.edit_approval import (
        maybe_require_edit_approval,
        set_edit_approval_requester,
    )

    set_edit_approval_requester(lambda proposal: True)
    raw = maybe_require_edit_approval(
        "patch", {"mode": "replace", "path": "f.py", "new_string": "x"}
    )

    result = json.loads(raw)
    assert not result["error"].startswith("Edit approval denied"), (
        "a malformed call must not be attributed to the user"
    )
    assert result["success"] is False
    assert "mode='patch'" in result["error"], "must still carry the recovery guidance"


def test_genuine_denial_still_reads_as_a_denial(tmp_path):
    """The narrowed ValueError branch must not swallow real refusals."""
    from acp_adapter.edit_approval import (
        maybe_require_edit_approval,
        set_edit_approval_requester,
    )

    target = tmp_path / "f.py"
    target.write_text("x = 1\n")

    set_edit_approval_requester(lambda proposal: False)
    raw = maybe_require_edit_approval(
        "patch",
        {"mode": "replace", "path": str(target), "old_string": "x = 1", "new_string": "x = 2"},
    )

    assert "Edit approval denied" in json.loads(raw)["error"]
