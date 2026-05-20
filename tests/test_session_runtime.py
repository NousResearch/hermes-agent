"""Tests for agent/session_runtime.py — Plan 002-B: Runtime Session Isolation.

Covers:
- SessionRuntime.__init__ creates the sandbox directory tree
- subagent_workspace() creates an isolated sub-directory per sub_id
- close() promotes workspace/outputs/ to artifacts when user_id is set
- close() destroys the sandbox after promotion
- close() is a no-op when the sandbox is already gone (idempotent)
- close() skips promotion when outputs/ is empty
- close() skips promotion when user_id is not set
- Sandbox root is under get_runtime_root() / "sessions" / session_id

All tests are hermetic: HERMES_RUNTIME_ROOT and HERMES_USERS_ROOT are
pointed at tmp_path so no real ~/.hermes state is touched.
"""

import os
import shutil
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_paths(tmp_path, monkeypatch):
    """Redirect runtime and user roots to tmp_path for full isolation."""
    runtime_root = tmp_path / "runtime"
    users_root = tmp_path / "users"
    runtime_root.mkdir()
    users_root.mkdir()
    monkeypatch.setenv("HERMES_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.setenv("HERMES_USERS_ROOT", str(users_root))
    monkeypatch.setenv("HERMES_USER_ID", "blake")
    # Ensure no stale module cache from other tests
    import importlib
    import agent.session_runtime as sr
    importlib.reload(sr)
    yield


def _make_runtime(session_id: str = "test_session_001", user_id: str = "blake"):
    from agent.session_runtime import SessionRuntime
    return SessionRuntime(session_id=session_id, user_id=user_id)


class TestSessionRuntimeInit:
    def test_workspace_created(self):
        rt = _make_runtime()
        assert rt.workspace.exists(), "workspace/ not created"

    def test_outputs_created(self):
        rt = _make_runtime()
        assert rt.outputs.exists(), "workspace/outputs/ not created"

    def test_subagents_dir_created(self):
        rt = _make_runtime()
        assert rt.subagents_dir.exists(), "subagents/ not created"

    def test_root_under_runtime_sessions(self, tmp_path):
        rt = _make_runtime(session_id="mysession")
        runtime_root = Path(os.environ["HERMES_RUNTIME_ROOT"])
        assert rt.root == runtime_root / "sessions" / "mysession"

    def test_user_id_from_env_when_not_passed(self, monkeypatch):
        monkeypatch.setenv("HERMES_USER_ID", "alice")
        import importlib, agent.session_runtime as sr
        importlib.reload(sr)
        rt = sr.SessionRuntime(session_id="s1")
        assert rt.user_id == "alice"


class TestSubagentWorkspace:
    def test_creates_distinct_path_per_sub_id(self):
        rt = _make_runtime()
        ws1 = rt.subagent_workspace("sa-0-aaa")
        ws2 = rt.subagent_workspace("sa-1-bbb")
        assert ws1 != ws2

    def test_workspace_under_subagents_dir(self):
        rt = _make_runtime()
        ws = rt.subagent_workspace("sa-0-abc")
        assert str(ws).startswith(str(rt.subagents_dir))
        assert ws.exists()

    def test_idempotent(self):
        rt = _make_runtime()
        ws1 = rt.subagent_workspace("sa-0-abc")
        ws2 = rt.subagent_workspace("sa-0-abc")
        assert ws1 == ws2
        assert ws1.exists()


class TestClose:
    def test_destroys_sandbox(self):
        rt = _make_runtime()
        root = rt.root
        rt.close()
        assert not root.exists(), "Sandbox not destroyed after close()"

    def test_promotes_outputs_when_present(self, tmp_path):
        rt = _make_runtime(session_id="sess_promote", user_id="blake")
        # Write a file to outputs/
        (rt.outputs / "result.txt").write_text("hello world")
        rt.close()
        # Verify promotion landed somewhere under users/blake/artifacts/sessions/
        users_root = Path(os.environ["HERMES_USERS_ROOT"])
        promoted = list((users_root / "blake" / "artifacts" / "sessions").glob("**/result.txt"))
        assert len(promoted) == 1, f"Expected 1 promoted file, got {promoted}"

    def test_promoted_path_contains_session_id(self, tmp_path):
        rt = _make_runtime(session_id="sess_xyz", user_id="blake")
        (rt.outputs / "out.txt").write_text("data")
        rt.close()
        users_root = Path(os.environ["HERMES_USERS_ROOT"])
        sessions_dir = users_root / "blake" / "artifacts" / "sessions"
        dirs = list(sessions_dir.iterdir()) if sessions_dir.exists() else []
        assert any("sess_xyz" in d.name for d in dirs), (
            f"Expected session_id in promoted dir name, got: {[d.name for d in dirs]}"
        )

    def test_no_promotion_when_outputs_empty(self, tmp_path):
        rt = _make_runtime(session_id="sess_empty", user_id="blake")
        # Leave outputs/ empty
        rt.close()
        users_root = Path(os.environ["HERMES_USERS_ROOT"])
        sessions_dir = users_root / "blake" / "artifacts" / "sessions"
        if sessions_dir.exists():
            entries = list(sessions_dir.iterdir())
            assert len(entries) == 0, f"Expected no promoted dirs, got {entries}"

    def test_no_promotion_when_user_id_missing(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_USER_ID", raising=False)
        import importlib, agent.session_runtime as sr
        importlib.reload(sr)
        rt = sr.SessionRuntime(session_id="sess_noid", user_id="")
        (rt.outputs / "file.txt").write_text("something")
        root = rt.root
        rt.close()
        assert not root.exists(), "Sandbox should still be destroyed"
        users_root = Path(os.environ["HERMES_USERS_ROOT"])
        sessions_dir = users_root / "artifacts"
        assert not sessions_dir.exists(), "Should not have promoted anything without user_id"

    def test_close_idempotent_when_sandbox_gone(self):
        rt = _make_runtime()
        rt.close()
        # Should not raise
        rt.close()

    def test_sandbox_gone_after_close(self):
        rt = _make_runtime(session_id="sess_cleanup")
        (rt.outputs / "f.txt").write_text("x")
        rt.close()
        assert not rt.root.exists()
