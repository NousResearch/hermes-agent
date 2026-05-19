# Phase 002-B: Runtime Session Isolation

## Goal
Give every conversation session an isolated ephemeral sandbox under `runtime/sessions/{session-id}/` so that parent agents, subagents, and parallel sessions cannot read or write each other's working state. On session close, promote outputs to `users/{id}/artifacts/sessions/` and destroy the sandbox.

## Context
Today `AIAgent` in `run_agent.py` already generates a `session_id` (line ~1889–1894) and exports it as `HERMES_SESSION_ID`. Session log files land in `logs_dir`. But there is no per-session workspace: tools that write files, subagents spawned by `delegate_task`, and the terminal tool all share the same filesystem with no namespacing. This phase introduces `SessionRuntime` — a thin class that creates the sandbox on `__init__`, enforces subagent workspace containment, and promotes outputs on close.

## Dependencies
- **Phase 002-A must be complete** — `get_runtime_root()` from `hermes_constants.py` is required
- **Plan 001-0 (HermesIdentity)** is a soft dependency — `HERMES_USER_ID` env var is used instead of a full identity object until that plan completes

## Scope

### Files to Create
- `agent/session_runtime.py` — `SessionRuntime` class (see Implementation Notes)

### Files to Modify
- `run_agent.py` — `AIAgent.__init__` (line ~1149+): instantiate `SessionRuntime`; call `session_runtime.close()` on session end
- `tools/delegate_task.py` — force subagent `workdir` to `runtime/sessions/{parent-id}/subagents/{sub-id}/workspace/`; enforce toolset intersection (drop extras silently)
- `tools/terminal_tool.py` — when `workdir` is not explicitly provided by the agent, default to `session_runtime.workspace` rather than the process cwd

### Explicitly Out of Scope
- Credential injection (that's Phase 002-B's CredentialResolver — wait, that's Phase C)
- MCP gateway (Phase C)
- Any changes to session storage/DB — `state.db`, `session_log_file` still live at current paths
- Changing `delegate_task`'s API or parameters visible to the LLM
- Enforcing disk quotas on sandboxes

## Implementation Notes

### `SessionRuntime` class (`agent/session_runtime.py`)

```python
"""SessionRuntime — per-session ephemeral workspace with output promotion."""
from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from hermes_constants import get_runtime_root, get_user_home

logger = logging.getLogger(__name__)

_OUTPUTS_DIR = "outputs"   # subdir within workspace that gets promoted


class SessionRuntime:
    """Manages an ephemeral sandbox for one session.

    Lifecycle:
        runtime = SessionRuntime(session_id="20250518_abc123", user_id="blake")
        # ... session runs ...
        runtime.close()   # promotes outputs, destroys sandbox

    Directory layout created:
        runtime/sessions/{session_id}/
            workspace/          ← default cwd for terminal tool
            workspace/outputs/  ← files here are promoted to artifacts on close
            subagents/          ← container for subagent workspaces (Phase B)
    """

    def __init__(self, session_id: str, user_id: Optional[str] = None):
        self.session_id = session_id
        self.user_id = user_id or os.environ.get("HERMES_USER_ID", "")
        self.root = get_runtime_root() / "sessions" / session_id
        self.workspace = self.root / "workspace"
        self.outputs = self.workspace / _OUTPUTS_DIR
        self.subagents_dir = self.root / "subagents"
        self._setup()

    def _setup(self) -> None:
        """Create sandbox directory tree."""
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.outputs.mkdir(parents=True, exist_ok=True)
        self.subagents_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("SessionRuntime: created sandbox at %s", self.root)

    def subagent_workspace(self, sub_id: str) -> Path:
        """Return (and create) an isolated workspace for a subagent."""
        path = self.subagents_dir / sub_id / "workspace"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def close(self) -> None:
        """Promote outputs to artifacts and destroy sandbox."""
        if not self.root.exists():
            return
        if self.user_id and any(self.outputs.iterdir() if self.outputs.exists() else []):
            self._promote_outputs()
        try:
            shutil.rmtree(self.root)
            logger.debug("SessionRuntime: destroyed sandbox at %s", self.root)
        except Exception as exc:
            logger.warning("SessionRuntime: failed to destroy sandbox %s: %s", self.root, exc)

    def _promote_outputs(self) -> None:
        """Move workspace/outputs/ → users/{id}/artifacts/sessions/{date}-{session_id}/outputs/"""
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        dest = (
            get_user_home(self.user_id)
            / "artifacts" / "sessions"
            / f"{date_str}-{self.session_id}"
            / "outputs"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(self.outputs), str(dest))
        logger.info("SessionRuntime: promoted outputs → %s", dest)
```

### Changes to `run_agent.py`

Find `AIAgent.__init__` (around line 1887 where `self.session_id` is set) and add SessionRuntime initialization immediately after:

```python
# After: os.environ["HERMES_SESSION_ID"] = self.session_id  (line ~1902)
from agent.session_runtime import SessionRuntime
self._session_runtime = SessionRuntime(
    session_id=self.session_id,
    user_id=os.environ.get("HERMES_USER_ID", ""),
)
```

Find the session close/cleanup path (search for `on_session_end` or wherever the agent tears down) and add:

```python
if hasattr(self, "_session_runtime"):
    self._session_runtime.close()
```

### Changes to `tools/delegate_task.py`

When building the subagent config, inject the workspace path and enforce toolset intersection. Look for where `workdir` is set on the subagent (or where the subprocess/agent is spawned) and add:

```python
# Inject isolated subagent workspace
parent_runtime_id = os.environ.get("HERMES_SESSION_ID", "")
if parent_runtime_id:
    from agent.session_runtime import SessionRuntime
    parent_runtime = SessionRuntime.__new__(SessionRuntime)
    parent_runtime.session_id = parent_runtime_id
    parent_runtime.root = get_runtime_root() / "sessions" / parent_runtime_id
    parent_runtime.subagents_dir = parent_runtime.root / "subagents"
    sub_workspace = parent_runtime.subagent_workspace(sub_agent_id)
    # Pass as workdir to subagent
    effective_workdir = sub_workspace

# Toolset intersection — drop any toolset not in parent's set
parent_toolsets = set(os.environ.get("HERMES_TOOLSETS", "").split(","))
if parent_toolsets and requested_toolsets:
    effective_toolsets = list(set(requested_toolsets) & parent_toolsets)
else:
    effective_toolsets = requested_toolsets
```

### Changes to `tools/terminal_tool.py`

When `workdir` is `None` (not explicitly set by the agent), use the session workspace instead of `os.getcwd()`:

```python
if workdir is None:
    session_id = os.environ.get("HERMES_SESSION_ID", "")
    if session_id:
        from hermes_constants import get_runtime_root
        candidate = get_runtime_root() / "sessions" / session_id / "workspace"
        if candidate.exists():
            workdir = str(candidate)
    if workdir is None:
        workdir = os.getcwd()
```

## Acceptance Criteria
- [ ] Each new `AIAgent` creates `runtime/sessions/{session-id}/workspace/` and `runtime/sessions/{session-id}/subagents/` on `__init__`
- [ ] `SessionRuntime.subagent_workspace(sub_id)` returns a distinct path per sub-id and creates the directory
- [ ] A file written to `workspace/outputs/` by a session is moved to `users/blake/artifacts/sessions/{date}-{id}/outputs/` after `session_runtime.close()`
- [ ] `runtime/sessions/{id}/` is deleted after `close()` completes
- [ ] A subagent spawned via `delegate_task` writes to `runtime/sessions/{parent-id}/subagents/{sub-id}/workspace/`, not the parent workspace
- [ ] A subagent cannot receive a toolset that is not a subset of the parent's toolset (extras are silently dropped)
- [ ] Terminal tool without explicit `workdir` defaults to `runtime/sessions/{id}/workspace/`
- [ ] After all sessions close, `runtime/sessions/` directory is empty
- [ ] `pytest tests/ -v` — zero regressions

## Verification Steps

```bash
# 1. Unit test SessionRuntime directly
cd ~/Documents/hermes-agent
python3 - <<'EOF'
import os, shutil, tempfile

# Override runtime root to a temp dir for testing
tmp = tempfile.mkdtemp()
os.environ["HERMES_RUNTIME_ROOT"] = tmp
os.environ["HERMES_USER_ID"] = "blake"

from agent.session_runtime import SessionRuntime

rt = SessionRuntime("test_session_001")
print("workspace:", rt.workspace)
assert rt.workspace.exists(), "workspace not created"
assert rt.outputs.exists(), "outputs not created"
assert rt.subagents_dir.exists(), "subagents dir not created"

# Write a file to outputs
(rt.outputs / "result.txt").write_text("hello world")

# Create a subagent workspace
sub_ws = rt.subagent_workspace("subagent_001")
assert sub_ws.exists(), "subagent workspace not created"
assert "subagents/subagent_001" in str(sub_ws)

# Close and verify promotion + cleanup
rt.close()
from hermes_constants import get_user_home
promoted = list((get_user_home("blake") / "artifacts/sessions").glob("**/result.txt"))
assert len(promoted) == 1, f"Expected 1 promoted file, got {promoted}"
print("promoted:", promoted[0])
assert not rt.root.exists(), "sandbox not cleaned up"
print("All SessionRuntime tests PASSED")

shutil.rmtree(tmp)
EOF

# 2. Run full test suite
pytest tests/ -v --tb=short 2>&1 | tail -30

# 3. Verify subagent isolation in integration (manual)
# Start hermes, run a delegate_task call, then check:
ls ~/.hermes/runtime/sessions/  # should show active session dirs while running
# After session ends:
ls ~/.hermes/runtime/sessions/  # should be empty (or only active sessions)
ls ~/.hermes/users/blake/artifacts/sessions/  # should show promoted outputs
```

## Status
Not started

## Bug Log
| # | Description | Status |
|---|-------------|--------|
