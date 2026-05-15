# ACP Zed Editor Filesystem Dirty Buffers Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Let Hermes ACP read and write through Zed's ACP filesystem APIs where available, so file tools can see unsaved editor buffers and coordinate writes with the editor.

**Architecture:** Keep Hermes local filesystem tools as the default for CLI/gateway and as fallback for ACP clients without filesystem capability. For ACP/Zed sessions, add an ACP-aware file access layer that can call client `fs/read_text_file` / `fs/write_text_file` before or instead of local `read_file` / `write_file`. Do not attempt patch/edit approval in this PR; this is about editor-buffer correctness.

**Tech Stack:** Python, ACP Python SDK filesystem methods, `acp_adapter/server.py`, tool dispatch/session context, pytest.

---

### Task 1: Inspect the installed ACP filesystem API

**Objective:** Confirm exact method and capability names.

Run:

```bash
/home/nour/.hermes/hermes-agent/venv/bin/python - <<'PY'
import acp, inspect
from acp.schema import ClientCapabilities
print(ClientCapabilities.model_fields)
for name in dir(acp.Client):
    if 'file' in name.lower() or 'text' in name.lower():
        attr = getattr(acp.Client, name)
        if callable(attr):
            print(name, inspect.signature(attr))
PY
```

Record exact names before coding.

### Task 2: Store ACP client filesystem capabilities

**Objective:** Make session code able to decide whether editor FS is available.

**Files:**
- Modify: `acp_adapter/server.py`
- Test: `tests/acp/test_server.py`

In `initialize()`, store `client_capabilities` on `HermesACPAgent`, e.g.:

```python
self._client_capabilities = client_capabilities
```

Add helper:

```python
def _client_supports_fs_read(self) -> bool: ...
def _client_supports_fs_write(self) -> bool: ...
```

Tests should pass a fake `ClientCapabilities` object with fs read/write flags and assert helpers return true.

### Task 3: Create an ACP file access module

**Objective:** Keep async client-file logic isolated from normal file tools.

**Files:**
- Create: `acp_adapter/filesystem.py`

Add functions/classes like:

```python
@dataclass
class AcpFileAccess:
    conn: acp.Client
    session_id: str
    cwd: str

    async def read_text_file(self, path: str) -> str:
        abs_path = normalize_against_cwd(path, self.cwd)
        return await self.conn.read_text_file(session_id=self.session_id, path=abs_path)

    async def write_text_file(self, path: str, content: str) -> None:
        abs_path = normalize_against_cwd(path, self.cwd)
        await self.conn.write_text_file(session_id=self.session_id, path=abs_path, content=content)
```

Use exact SDK argument names from Task 1.

### Task 4: Decide and implement tool interception point

**Objective:** Route only ACP session file tools through editor FS without changing CLI behavior.

Preferred pattern:

1. `acp_adapter/server.py` sets an ACP session contextvar around `agent.run_conversation(...)`.
2. Central tool dispatch checks the contextvar for `read_file` and `write_file`.
3. If ACP FS read/write is supported, use the ACP file access implementation.
4. Else fall back to existing Hermes tools unchanged.

Likely files:

- `acp_adapter/server.py`
- `model_tools.py` or file tool wrapper path
- possibly `tools/file_tools.py` depending actual tool names

Do not add Zed-specific code to generic tools without a context guard.

### Task 5: Add read_file dirty-buffer tests

**Objective:** Prove ACP read path is used when available.

**Files:**
- Create/modify: `tests/acp/test_filesystem.py`

Test:

- local file contains `old`
- fake ACP client read returns `unsaved new buffer`
- invoking ACP-aware `read_file` returns `unsaved new buffer`
- fake client read was called with normalized absolute path

### Task 6: Add fallback tests

**Objective:** Preserve normal Hermes behavior when ACP FS is unavailable or fails.

Tests:

- no fs capability -> local read_file path
- ACP read raises -> either fallback local read with warning or return clear error; choose one and document it
- Windows/Zed cwd path normalization still uses existing `_translate_acp_cwd` behavior

### Task 7: Add write_file tests

**Objective:** Prove editor write is used when available.

Tests:

- ACP write called with path/content
- local disk is not mutated by duplicate execution if ACP write succeeds
- ACP write failure returns a clear tool error

This PR should not add approval prompts. If approval is required, layer it later via the pre-edit approval PR.

### Task 8: Verification

Run:

```bash
scripts/run_tests.sh tests/acp/test_filesystem.py tests/acp/test_server.py tests/acp/test_session.py -q
```

Manual Zed check:

1. Open a file in Zed and make unsaved changes.
2. Ask Hermes ACP to read that file.
3. Confirm the model sees unsaved content.
4. Ask Hermes to write a small test file and confirm Zed updates/reflects it cleanly.

**Do not merge** without dirty-buffer manual verification; unit tests cannot prove Zed buffer semantics.
