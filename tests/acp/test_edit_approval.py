"""Tests for ACP pre-edit approval gating."""

from __future__ import annotations

import json
import shlex
import sys
import tempfile
from pathlib import Path

from acp_adapter.edit_approval import (
    DeniedEditReattempt,
    EditProposal,
    build_acp_edit_tool_call,
    build_acp_write_reattempt_tool_call,
    clear_edit_approval_requester,
    reset_edit_approval_requester,
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


def test_acp_write_reattempt_tool_call_uses_execute_kind_and_denied_path_context():
    proposal = DeniedEditReattempt(
        tool_name="terminal",
        path="demo.txt",
        resolved_path="/tmp/demo.txt",
        denied_tool_name="write_file",
        arguments={"command": "python -c 'write demo.txt'"},
    )

    tool_call = build_acp_write_reattempt_tool_call(proposal)

    assert tool_call.kind == "execute"
    assert tool_call.status == "pending"
    assert tool_call.rawInput == {
        "tool": "terminal",
        "arguments": proposal.arguments,
        "denied_path": "demo.txt",
        "denied_tool": "write_file",
    }
    assert "previously denied" in tool_call.title
    assert "demo.txt" in tool_call.title
    assert len(tool_call.content) == 1
    assert "fresh approval" in tool_call.content[0].content.text


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


def test_denied_write_file_prompts_before_terminal_write_to_same_path(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("before\n", encoding="utf-8")
    requests = []

    def decide(proposal):
        requests.append(proposal)
        return len(requests) == 2

    set_edit_approval_requester(decide)

    denied = json.loads(
        handle_function_call(
            "write_file",
            {"path": str(target), "content": "after\n"},
            task_id="acp-edit-terminal-bypass",
        )
    )
    assert "Edit approval denied" in denied["error"]

    code = f"from pathlib import Path; Path({str(target)!r}).write_text('after\\n', encoding='utf-8')"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    result = json.loads(
        handle_function_call(
            "terminal",
            {"command": command},
            task_id="acp-edit-terminal-bypass",
        )
    )

    assert result.get("exit_code") == 0
    assert target.read_text(encoding="utf-8") == "after\n"
    assert len(requests) == 2
    assert isinstance(requests[1], DeniedEditReattempt)
    assert requests[1].tool_name == "terminal"
    assert requests[1].path == str(target)


def test_denied_write_file_prompts_before_execute_code_write_to_same_path(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("before\n", encoding="utf-8")
    requests = []

    def decide(proposal):
        requests.append(proposal)
        return len(requests) == 2

    set_edit_approval_requester(decide)

    denied = json.loads(
        handle_function_call(
            "write_file",
            {"path": str(target), "content": "after\n"},
            task_id="acp-edit-code-bypass",
        )
    )
    assert "Edit approval denied" in denied["error"]

    result = json.loads(
        handle_function_call(
            "execute_code",
            {
                "code": (
                    "from pathlib import Path\n"
                    f"Path({str(target)!r}).write_text('after\\n', encoding='utf-8')\n"
                    "print('wrote file')\n"
                )
            },
            task_id="acp-edit-code-bypass",
        )
    )

    assert result.get("status") == "success"
    assert target.read_text(encoding="utf-8") == "after\n"
    assert len(requests) == 2
    assert isinstance(requests[1], DeniedEditReattempt)
    assert requests[1].tool_name == "execute_code"
    assert requests[1].path == str(target)


def test_denied_write_file_blocks_terminal_reattempt_when_fresh_request_denied(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("before\n", encoding="utf-8")
    requests = []

    def deny(proposal):
        requests.append(proposal)
        return False

    set_edit_approval_requester(deny)

    denied = json.loads(
        handle_function_call(
            "write_file",
            {"path": str(target), "content": "after\n"},
            task_id="acp-edit-terminal-reattempt-deny",
        )
    )
    assert "Edit approval denied" in denied["error"]

    code = f"from pathlib import Path; Path({str(target)!r}).write_text('after\\n', encoding='utf-8')"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    result = json.loads(
        handle_function_call(
            "terminal",
            {"command": command},
            task_id="acp-edit-terminal-reattempt-deny",
        )
    )

    assert "Alternate write approval denied" in result["error"]
    assert target.read_text(encoding="utf-8") == "before\n"
    assert len(requests) == 2
    assert isinstance(requests[1], DeniedEditReattempt)


def test_denied_write_file_fails_closed_when_reattempt_requester_raises(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("before\n", encoding="utf-8")
    requests = []

    def disconnect_on_reattempt(proposal):
        requests.append(proposal)
        if len(requests) == 2:
            raise RuntimeError("zed disconnected")
        return False

    set_edit_approval_requester(disconnect_on_reattempt)

    denied = json.loads(
        handle_function_call(
            "write_file",
            {"path": str(target), "content": "after\n"},
            task_id="acp-edit-terminal-reattempt-exception",
        )
    )
    assert "Edit approval denied" in denied["error"]

    code = f"from pathlib import Path; Path({str(target)!r}).write_text('after\\n', encoding='utf-8')"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    result = json.loads(
        handle_function_call(
            "terminal",
            {"command": command},
            task_id="acp-edit-terminal-reattempt-exception",
        )
    )

    assert "Alternate write approval denied" in result["error"]
    assert target.read_text(encoding="utf-8") == "before\n"
    assert len(requests) == 2
    assert isinstance(requests[1], DeniedEditReattempt)


def test_active_acp_guard_exception_blocks_terminal_instead_of_fail_open(monkeypatch, tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("before\n", encoding="utf-8")
    set_edit_approval_requester(lambda _proposal: True)

    import acp_adapter.edit_approval as edit_approval

    def broken_guard(_tool_name, _arguments):
        raise RuntimeError("guard bug")

    monkeypatch.setattr(edit_approval, "maybe_require_edit_approval", broken_guard)

    code = f"from pathlib import Path; Path({str(target)!r}).write_text('after\\n', encoding='utf-8')"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    result = json.loads(
        handle_function_call(
            "terminal",
            {"command": command},
            task_id="acp-edit-terminal-guard-exception",
        )
    )

    assert "approval guard failed" in result["error"]
    assert target.read_text(encoding="utf-8") == "before\n"


def test_active_acp_guard_exception_blocks_execute_code_instead_of_fail_open(monkeypatch, tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("before\n", encoding="utf-8")
    set_edit_approval_requester(lambda _proposal: True)

    import acp_adapter.edit_approval as edit_approval

    def broken_guard(_tool_name, _arguments):
        raise RuntimeError("guard bug")

    monkeypatch.setattr(edit_approval, "maybe_require_edit_approval", broken_guard)

    result = json.loads(
        handle_function_call(
            "execute_code",
            {
                "code": (
                    "from pathlib import Path\n"
                    f"Path({str(target)!r}).write_text('after\\n', encoding='utf-8')\n"
                )
            },
            task_id="acp-edit-code-guard-exception",
        )
    )

    assert "approval guard failed" in result["error"]
    assert target.read_text(encoding="utf-8") == "before\n"


def test_inactive_acp_guard_exception_does_not_block_terminal_tool(monkeypatch, tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("before\n", encoding="utf-8")

    import acp_adapter.edit_approval as edit_approval

    def broken_guard(_tool_name, _arguments):
        raise RuntimeError("guard bug")

    monkeypatch.setattr(edit_approval, "maybe_require_edit_approval", broken_guard)

    code = f"from pathlib import Path; Path({str(target)!r}).write_text('after\\n', encoding='utf-8')"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    result = json.loads(
        handle_function_call(
            "terminal",
            {"command": command},
            task_id="non-acp-terminal-guard-exception",
        )
    )

    assert result.get("exit_code") == 0
    assert target.read_text(encoding="utf-8") == "after\n"


def test_inactive_acp_guard_exception_does_not_block_execute_code_tool(monkeypatch, tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("before\n", encoding="utf-8")

    import acp_adapter.edit_approval as edit_approval

    def broken_guard(_tool_name, _arguments):
        raise RuntimeError("guard bug")

    monkeypatch.setattr(edit_approval, "maybe_require_edit_approval", broken_guard)

    result = json.loads(
        handle_function_call(
            "execute_code",
            {
                "code": (
                    "from pathlib import Path\n"
                    f"Path({str(target)!r}).write_text('after\\n', encoding='utf-8')\n"
                )
            },
            task_id="non-acp-code-guard-exception",
        )
    )

    assert result.get("status") == "success"
    assert target.read_text(encoding="utf-8") == "after\n"


def test_reset_edit_approval_requester_clears_denied_reattempt_state(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("before\n", encoding="utf-8")
    outer_requests = []

    outer_token = set_edit_approval_requester(lambda proposal: outer_requests.append(proposal) or False)
    inner_token = set_edit_approval_requester(lambda _proposal: False)

    try:
        denied = json.loads(
            handle_function_call(
                "write_file",
                {"path": str(target), "content": "after direct\n"},
                task_id="acp-edit-reset-clears-denied-state",
            )
        )
        assert "Edit approval denied" in denied["error"]

        reset_edit_approval_requester(inner_token)

        code = f"from pathlib import Path; Path({str(target)!r}).write_text('after\\n', encoding='utf-8')"
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
        result = json.loads(
            handle_function_call(
                "terminal",
                {"command": command},
                task_id="acp-edit-reset-clears-denied-state",
            )
        )

        assert result.get("exit_code") == 0
        assert target.read_text(encoding="utf-8") == "after\n"
        assert outer_requests == []
    finally:
        reset_edit_approval_requester(outer_token)


def test_approved_direct_edit_forgets_prior_denial_for_same_path(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("before\n", encoding="utf-8")
    requests = []

    def decide(proposal):
        requests.append(proposal)
        return len(requests) == 2

    set_edit_approval_requester(decide)

    denied = json.loads(
        handle_function_call(
            "write_file",
            {"path": str(target), "content": "denied direct\n"},
            task_id="acp-edit-forget-approved-direct-edit",
        )
    )
    assert "Edit approval denied" in denied["error"]

    approved = json.loads(
        handle_function_call(
            "write_file",
            {"path": str(target), "content": "approved direct\n"},
            task_id="acp-edit-forget-approved-direct-edit",
        )
    )
    assert approved.get("bytes_written") == len("approved direct\n")
    assert target.read_text(encoding="utf-8") == "approved direct\n"
    assert len(requests) == 2

    code = f"from pathlib import Path; Path({str(target)!r}).write_text('after terminal\\n', encoding='utf-8')"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    result = json.loads(
        handle_function_call(
            "terminal",
            {"command": command},
            task_id="acp-edit-forget-approved-direct-edit",
        )
    )

    assert result.get("exit_code") == 0
    assert target.read_text(encoding="utf-8") == "after terminal\n"
    assert len(requests) == 2


def test_denied_write_file_prompts_for_shell_quoted_terminal_redirect_to_same_path(tmp_path):
    target = tmp_path / "quo'te.txt"
    target.write_text("before\n", encoding="utf-8")
    requests = []

    def decide(proposal):
        requests.append(proposal)
        return len(requests) == 2

    set_edit_approval_requester(decide)

    denied = json.loads(
        handle_function_call(
            "write_file",
            {"path": str(target), "content": "after direct\n"},
            task_id="acp-edit-terminal-shell-quoted-path",
        )
    )
    assert "Edit approval denied" in denied["error"]

    payload = "after via shell\n"
    command = f"printf %s {shlex.quote(payload)} > {shlex.quote(str(target))}"
    result = json.loads(
        handle_function_call(
            "terminal",
            {"command": command},
            task_id="acp-edit-terminal-shell-quoted-path",
        )
    )

    assert result.get("exit_code") == 0
    assert target.read_text(encoding="utf-8") == "after via shell\n"
    assert len(requests) == 2
    assert isinstance(requests[1], DeniedEditReattempt)
    assert requests[1].tool_name == "terminal"
    assert requests[1].path == str(target)


def test_denied_write_file_still_allows_terminal_read_of_same_path(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("before\n", encoding="utf-8")
    set_edit_approval_requester(lambda _proposal: False)

    denied = json.loads(
        handle_function_call(
            "write_file",
            {"path": str(target), "content": "after\n"},
            task_id="acp-edit-terminal-read",
        )
    )
    assert "Edit approval denied" in denied["error"]

    code = f"from pathlib import Path; print(Path({str(target)!r}).read_text(encoding='utf-8'), end='')"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    result = json.loads(
        handle_function_call(
            "terminal",
            {"command": command},
            task_id="acp-edit-terminal-read",
        )
    )

    assert result.get("exit_code") == 0
    assert result.get("output") == "before"
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
