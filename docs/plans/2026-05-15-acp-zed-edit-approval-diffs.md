# ACP Zed Pre-Edit Approval Diffs Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Gate file mutations in ACP/Zed behind explicit pre-edit approval with a structured diff, similar to Codex/Kimi edit review behavior.

**Architecture:** Hermes already renders edit diffs after tools run. This PR adds a pre-mutation permission gate for file mutation tools. Intercept `write_file`, `patch`, and eventually `skill_manage` before they mutate disk; compute proposed old/new content; send ACP `session/request_permission` with `kind="edit"` and diff content; only execute the mutation after approval. Rejections return a clear tool result and leave files unchanged.

**Tech Stack:** Python, ACP `request_permission`, `FileEditToolCallContent` / `acp.tool_diff_content`, Hermes file tools, pytest with temp files.

---

### Task 1: Confirm current ACP diff/permission schema

Run:

```bash
/home/nour/.hermes/hermes-agent/venv/bin/python - <<'PY'
from acp.schema import RequestPermissionRequest, ToolCallUpdate
import acp, inspect
print(RequestPermissionRequest.model_fields)
print(ToolCallUpdate.model_fields)
print(inspect.signature(acp.tool_diff_content))
PY
```

Record actual field names. Do not rely on stale examples.

### Task 2: Add denied-write test

**Objective:** A rejected `write_file` must not mutate disk.

**Files:**
- Create/modify: `tests/acp/test_edit_approval.py`

Test shape:

```python
def test_write_file_rejected_by_acp_permission_does_not_mutate(tmp_path):
    path = tmp_path / "demo.txt"
    path.write_text("old")

    # Install fake ACP edit approval callback returning reject_once.
    # Invoke the same interception function that the terminal/tool path will call.

    result = maybe_gate_file_edit(
        tool_name="write_file",
        args={"path": str(path), "content": "new"},
        approval_requester=fake_reject,
    )

    assert path.read_text() == "old"
    assert "rejected" in result.lower()
```

The exact function name will be created in Task 4.

### Task 3: Add approved-write test

**Objective:** Approved writes proceed and include diff content in permission request.

Assert:

- fake requester received tool call `kind == "edit"`
- content includes diff block for `demo.txt`
- after approval, file content is changed

### Task 4: Implement edit proposal computation

**Files:**
- Create: `acp_adapter/edit_approval.py`

Add pure helpers first:

```python
@dataclass
class EditProposal:
    path: str
    old_text: str | None
    new_text: str
    title: str


def proposal_for_write_file(args: dict[str, Any]) -> EditProposal:
    path = str(args["path"])
    old_text = Path(path).read_text(encoding="utf-8") if Path(path).exists() else None
    new_text = str(args.get("content", ""))
    return EditProposal(path=path, old_text=old_text, new_text=new_text, title=f"Edit {path}")
```

For `patch`, start with replace-mode only. V4A/multi-file patches can be a second task or second PR if too risky.

### Task 5: Implement ACP permission requester

**Files:**
- Modify: `acp_adapter/permissions.py` or new `acp_adapter/edit_approval.py`

Build request with:

```python
acp.tool_diff_content(path=proposal.path, old_text=proposal.old_text, new_text=proposal.new_text)
```

Options:

- allow once
- reject once
- optionally allow always/reject always only after policy storage exists

Default deny on exception/cancel/timeout.

### Task 6: Intercept file mutation tools before execution

**Objective:** Ensure mutation cannot happen before approval.

**Files:**
- Likely modify: `model_tools.py` or `acp_adapter/server.py` session-context tool wrapper

Do not bury this inside post-execution `acp_adapter/events.py`; that is too late.

Preferred design:

- set an ACP session contextvar around `agent.run_conversation(...)`
- in the central tool execution path, before dispatching `write_file`/`patch`, call the ACP edit approval gate if contextvar exists
- if rejected, return a normal tool result string like `{"success": false, "error": "Edit rejected by user"}`
- if approved, continue to original tool implementation

### Task 7: Expand patch coverage

Add tests for:

- `patch` replace mode approved/rejected
- creating a new file via `write_file`
- missing old string -> should fail before approval or return normal patch error, but must not mutate
- permission requester exception -> deny and no mutation

### Task 8: Verification

Run:

```bash
scripts/run_tests.sh tests/acp/test_edit_approval.py tests/acp/test_events.py tests/acp/test_tools.py -q
```

Then run manual Zed verification:

1. Ask Hermes ACP to edit a small file.
2. Confirm Zed shows a diff before mutation.
3. Reject and verify file unchanged.
4. Approve and verify file changed.

**Do not merge** without manual reject-path verification.
