"""Tests for ``agent.workspace_awareness``."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from agent import workspace_awareness as wa


# ── helpers ──────────────────────────────────────────────────────────────

def _write_session_file(
    sessions_dir: Path,
    session_id: str,
    pid: int | None = None,
    profile: str = "test-profile",
    cwd: str = "/tmp/test-project",
    last_actions: list[dict] | None = None,
) -> Path:
    """Write a synthetic presence file and return its path."""
    if pid is None:
        pid = os.getpid()  # alive by default
    if last_actions is None:
        last_actions = [
            {"tool": "write_file", "target": "test.py", "ts": "2026-01-01T00:00:00Z"},
        ]
    record = {
        "session_id": session_id,
        "pid": pid,
        "profile": profile,
        "cwd": cwd,
        "task_id": "",
        "ts": "2026-01-01T00:00:00Z",
        "last_actions": last_actions,
    }
    path = sessions_dir / f"{session_id}.json"
    path.write_text(json.dumps(record))
    return path


# ── tests ────────────────────────────────────────────────────────────────

class TestUpdatePresence:
    """Tests for ``update_presence()``."""

    def test_writes_file(self, monkeypatch, tmp_path: Path):
        """update_presence writes a valid JSON file."""
        monkeypatch.setattr(wa, "_sessions_dir", lambda: tmp_path)
        monkeypatch.setattr(wa, "get_hermes_home", lambda: tmp_path)

        wa.update_presence(
            session_id="sess-abc",
            tool_name="write_file",
            args={"path": "/tmp/foo.py"},
            cwd="/tmp/project",
        )

        path = tmp_path / "sess-abc.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["session_id"] == "sess-abc"
        assert data["profile"] == "default"
        assert data["cwd"] == "/tmp/project"
        assert len(data["last_actions"]) == 1
        assert data["last_actions"][0]["tool"] == "write_file"
        assert data["last_actions"][0]["target"] == "/tmp/foo.py"

    def test_skips_readonly_tools(self, monkeypatch, tmp_path: Path):
        """Read-only tools are not logged."""
        monkeypatch.setattr(wa, "_sessions_dir", lambda: tmp_path)
        monkeypatch.setattr(wa, "get_hermes_home", lambda: tmp_path)

        wa.update_presence(
            session_id="sess-abc",
            tool_name="read_file",
            args={"path": "/tmp/foo.py"},
        )

        path = tmp_path / "sess-abc.json"
        assert not path.exists()

    def test_keeps_last_two_actions(self, monkeypatch, tmp_path: Path):
        """Only the last 2 actions are retained."""
        monkeypatch.setattr(wa, "_sessions_dir", lambda: tmp_path)
        monkeypatch.setattr(wa, "get_hermes_home", lambda: tmp_path)

        sid = "sess-abc"
        for i in range(3):
            wa.update_presence(
                session_id=sid,
                tool_name="write_file",
                args={"path": f"file_{i}.py"},
                cwd="/tmp/project",
            )

        data = json.loads((tmp_path / f"{sid}.json").read_text())
        assert len(data["last_actions"]) == 2
        assert data["last_actions"][0]["target"] == "file_1.py"
        assert data["last_actions"][1]["target"] == "file_2.py"

    def test_respects_disabled_env(self, monkeypatch, tmp_path: Path):
        """HERMES_WORKSPACE_AWARENESS=0 disables the feature."""
        monkeypatch.setenv("HERMES_WORKSPACE_AWARENESS", "0")
        monkeypatch.setattr(wa, "_sessions_dir", lambda: tmp_path)
        monkeypatch.setattr(wa, "get_hermes_home", lambda: tmp_path)

        wa.update_presence(
            session_id="sess-abc",
            tool_name="write_file",
            args={"path": "/tmp/foo.py"},
        )

        path = tmp_path / "sess-abc.json"
        assert not path.exists()

    def test_target_extraction_terminal(self, monkeypatch, tmp_path: Path):
        """Terminal commands are truncated to 80 chars in target."""
        monkeypatch.setattr(wa, "_sessions_dir", lambda: tmp_path)
        monkeypatch.setattr(wa, "get_hermes_home", lambda: tmp_path)

        long_cmd = "pip install " + "x" * 100
        wa.update_presence(
            session_id="sess-abc",
            tool_name="terminal",
            args={"command": long_cmd},
            cwd="/tmp/project",
        )

        data = json.loads((tmp_path / "sess-abc.json").read_text())
        target = data["last_actions"][0]["target"]
        assert len(target) <= 80


class TestGetCoworkers:
    """Tests for ``get_coworkers()``."""

    def test_returns_other_sessions_same_cwd(self, monkeypatch, tmp_path: Path):
        """Sessions in the same cwd are returned."""
        monkeypatch.setattr(wa, "_sessions_dir", lambda: tmp_path)
        monkeypatch.setattr(wa, "get_hermes_home", lambda: tmp_path)

        _write_session_file(tmp_path, "other-1", cwd="/tmp/project")
        _write_session_file(tmp_path, "other-2", cwd="/tmp/other-dir")

        coworkers = wa.get_coworkers("my-session", "/tmp/project")
        assert len(coworkers) == 1
        assert coworkers[0]["session_id"] == "other-1"

    def test_excludes_self(self, monkeypatch, tmp_path: Path):
        """The calling session is excluded from results."""
        monkeypatch.setattr(wa, "_sessions_dir", lambda: tmp_path)
        monkeypatch.setattr(wa, "get_hermes_home", lambda: tmp_path)

        _write_session_file(tmp_path, "my-session", cwd="/tmp/project")
        _write_session_file(tmp_path, "other-1", cwd="/tmp/project")

        coworkers = wa.get_coworkers("my-session", "/tmp/project")
        assert len(coworkers) == 1
        assert coworkers[0]["session_id"] == "other-1"

    def test_excludes_dead_pids(self, monkeypatch, tmp_path: Path):
        """Sessions with dead PIDs are excluded and cleaned up."""
        monkeypatch.setattr(wa, "_sessions_dir", lambda: tmp_path)
        monkeypatch.setattr(wa, "get_hermes_home", lambda: tmp_path)

        # Use PID 1 (always init, always alive but not ours) — but we mock
        # _pid_alive to control liveness.
        _write_session_file(tmp_path, "dead-session", pid=99999, cwd="/tmp/project")
        _write_session_file(tmp_path, "alive-session", pid=os.getpid(), cwd="/tmp/project")

        # Mock _pid_alive: only our own PID is alive.
        def _mock_pid_alive(pid: int) -> bool:
            return pid == os.getpid()

        with patch.object(wa, "_pid_alive", _mock_pid_alive):
            coworkers = wa.get_coworkers("my-session", "/tmp/project")

        assert len(coworkers) == 1
        assert coworkers[0]["session_id"] == "alive-session"

    def test_excludes_different_cwd(self, monkeypatch, tmp_path: Path):
        """Sessions in a different cwd are not returned."""
        monkeypatch.setattr(wa, "_sessions_dir", lambda: tmp_path)
        monkeypatch.setattr(wa, "get_hermes_home", lambda: tmp_path)

        _write_session_file(tmp_path, "other-1", cwd="/tmp/project-a")
        _write_session_file(tmp_path, "other-2", cwd="/tmp/project-b")

        coworkers = wa.get_coworkers("my-session", "/tmp/project-a")
        assert len(coworkers) == 1
        assert coworkers[0]["session_id"] == "other-1"


class TestBuildContextBlock:
    """Tests for ``build_context_block()``."""

    def test_returns_none_when_no_coworkers(self, monkeypatch, tmp_path: Path):
        """None returned when no other sessions are active."""
        monkeypatch.setattr(wa, "_sessions_dir", lambda: tmp_path)
        monkeypatch.setattr(wa, "get_hermes_home", lambda: tmp_path)

        result = wa.build_context_block("my-session", "/tmp/project")
        assert result is None

    def test_returns_context_with_one_coworker(self, monkeypatch, tmp_path: Path):
        """Context block with a single coworker."""
        monkeypatch.setattr(wa, "_sessions_dir", lambda: tmp_path)
        monkeypatch.setattr(wa, "get_hermes_home", lambda: tmp_path)

        _write_session_file(
            tmp_path,
            "other-1",
            pid=os.getpid(),
            profile="coder",
            cwd="/tmp/project",
            last_actions=[
                {"tool": "write_file", "target": "/tmp/project/main.py", "ts": "2026-01-01T00:00:00Z"},
            ],
        )

        result = wa.build_context_block("my-session", "/tmp/project")
        assert result is not None
        assert "══ WORKSPACE ══" in result
        assert "1 other hermes session" in result
        assert "coder" in result
        assert "write_file" in result
        assert "main.py" in result

    def test_shows_multiple_coworkers(self, monkeypatch, tmp_path: Path):
        """Context block with multiple coworkers."""
        monkeypatch.setattr(wa, "_sessions_dir", lambda: tmp_path)
        monkeypatch.setattr(wa, "get_hermes_home", lambda: tmp_path)

        _write_session_file(
            tmp_path, "other-1", pid=os.getpid(), profile="coder", cwd="/tmp/project",
            last_actions=[
                {"tool": "write_file", "target": "/tmp/project/a.py", "ts": "2026-01-01T00:00:00Z"},
            ],
        )
        _write_session_file(
            tmp_path, "other-2", pid=os.getpid(), profile="researcher", cwd="/tmp/project",
            last_actions=[
                {"tool": "web_search", "target": "qwen lora", "ts": "2026-01-01T00:00:00Z"},
            ],
        )

        result = wa.build_context_block("my-session", "/tmp/project")
        assert result is not None
        assert "2 other hermes sessions" in result
        assert "coder" in result
        assert "researcher" in result

    def test_shows_modified_files(self, monkeypatch, tmp_path: Path):
        """Context block includes files modified by other sessions."""
        monkeypatch.setattr(wa, "_sessions_dir", lambda: tmp_path)
        monkeypatch.setattr(wa, "get_hermes_home", lambda: tmp_path)

        _write_session_file(
            tmp_path, "other-1", pid=os.getpid(), profile="coder", cwd="/tmp/project",
            last_actions=[
                {"tool": "write_file", "target": "/tmp/project/a.py", "ts": "2026-01-01T00:00:00Z"},
                {"tool": "patch", "target": "/tmp/project/b.py", "ts": "2026-01-01T00:00:00Z"},
            ],
        )

        result = wa.build_context_block("my-session", "/tmp/project")
        assert result is not None
        assert "files modified by other sessions" in result
        assert "a.py" in result
        assert "b.py" in result

    def test_respects_disabled_env(self, monkeypatch, tmp_path: Path):
        """HERMES_WORKSPACE_AWARENESS=0 returns None."""
        monkeypatch.setenv("HERMES_WORKSPACE_AWARENESS", "0")
        monkeypatch.setattr(wa, "_sessions_dir", lambda: tmp_path)
        monkeypatch.setattr(wa, "get_hermes_home", lambda: tmp_path)

        _write_session_file(
            tmp_path, "other-1", pid=os.getpid(), profile="coder", cwd="/tmp/project",
        )

        result = wa.build_context_block("my-session", "/tmp/project")
        assert result is None
