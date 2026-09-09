"""`kanban_complete` must refuse a `dir` card with uncommitted TRACKED work.

2026-09-04 incident: `t_d354546b`, `t_f6325bc3`, `t_fe5cb6f5`, `t_1fbada04` were
built, verified live by Rodge and marked ``done`` — and the commit exists in no
branch and no reflog. All four used ``workspace_kind='dir'`` on the SHARED
BackupBrain working tree, so nothing was ever committed and the next card
clobbered the edits. 146 of 147 `dir` cards on the board share that exposure.

The charter wrote "done requires a commit" as prose. These tests pin it as
behaviour, and — as importantly — pin that the gate FAILS OPEN, because a gate
that can wrongly trap a card is worse than the leak it closes.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from tools import kanban_tools as kt


def _git(path, *args):
    subprocess.run(["git", "-C", str(path), *args], check=True,
                   capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "T")
    (r / "src.py").write_text("original\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "base")
    return r


# --- the detector -----------------------------------------------------------

def test_clean_repo_reports_nothing(repo):
    assert kt._dir_workspace_uncommitted(str(repo)) is None


def test_tracked_modification_is_reported(repo):
    (repo / "src.py").write_text("edited\n")
    dirty = kt._dir_workspace_uncommitted(str(repo))
    assert dirty == ["src.py"]


def test_untracked_scratch_is_ignored(repo):
    """A worker's scratch output must never trap its card."""
    (repo / "notes.txt").write_text("scratch\n")
    (repo / "probe.log").write_text("x\n")
    assert kt._dir_workspace_uncommitted(str(repo)) is None


def test_staged_but_uncommitted_is_reported(repo):
    (repo / "src.py").write_text("edited\n")
    _git(repo, "add", "-A")
    assert kt._dir_workspace_uncommitted(str(repo)) == ["src.py"]


def test_committed_work_is_clean(repo):
    (repo / "src.py").write_text("edited\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "done")
    assert kt._dir_workspace_uncommitted(str(repo)) is None


# --- fail-open: the gate must never be the reason a card cannot close --------

def test_non_repo_directory_fails_open(tmp_path):
    d = tmp_path / "plain"
    d.mkdir()
    (d / "f.txt").write_text("x")
    assert kt._dir_workspace_uncommitted(str(d)) is None


def test_missing_path_fails_open(tmp_path):
    assert kt._dir_workspace_uncommitted(str(tmp_path / "nope")) is None


def test_empty_path_fails_open():
    assert kt._dir_workspace_uncommitted("") is None


def test_git_failure_fails_open(repo, monkeypatch):
    def boom(*a, **k):
        raise OSError("no git")
    monkeypatch.setattr(subprocess, "run", boom)
    assert kt._dir_workspace_uncommitted(str(repo)) is None


# --- the gate, now a PLUGIN -------------------------------------------------
# Retired from core 2026-09-09 (manifest gate-core-retire-20260909). The control
# lives in the kanban-completion-gate plugin on upstream's pre_tool_call hook;
# these tests follow it there so the regression rig still covers it.

import importlib.util as _ilu
from pathlib import Path as _P


def _gate():
    home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    p = _P(home) / "plugins" / "kanban-completion-gate" / "__init__.py"
    if not p.exists():
        pytest.skip(f"kanban-completion-gate not installed at {p}")
    spec = _ilu.spec_from_file_location("_kcg_under_test", p)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    m._REFUSALS.clear()
    return m


def _fake_api(monkeypatch, m, kind, path):
    class _Task:
        workspace_kind = kind
        workspace_path = str(path)

    monkeypatch.setattr(m, "_kanban_api",
                        lambda: (lambda *a, **k: _FakeConn(),
                                 lambda conn, tid: _Task()))


def _call(m, tid, tool="kanban_complete"):
    return m.on_pre_tool_call(tool_name=tool, args={"task_id": tid}, task_id=tid)


def test_orchestrator_path_is_exempt(repo, monkeypatch):
    """No HERMES_KANBAN_TASK => CLI/orchestrator completion, never gated."""
    (repo / "src.py").write_text("edited\n")
    m = _gate()
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert _call(m, "t_whatever") is None


def test_gate_only_applies_to_dir_workspaces(repo, monkeypatch):
    """A worktree card has its own branch; a scratch card has no repo."""
    (repo / "src.py").write_text("edited\n")
    m = _gate()
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_x")
    _fake_api(monkeypatch, m, "worktree", repo)
    assert _call(m, "t_x") is None


def test_dirty_dir_workspace_is_blocked(repo, monkeypatch):
    (repo / "src.py").write_text("edited\n")
    m = _gate()
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_x")
    _fake_api(monkeypatch, m, "dir", repo)
    out = _call(m, "t_x")
    assert out is not None and out["action"] == "block"
    assert "src.py" in out["message"]


def test_clean_dir_workspace_is_allowed(repo, monkeypatch):
    m = _gate()
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_x")
    _fake_api(monkeypatch, m, "dir", repo)
    assert _call(m, "t_x") is None


def test_repeat_refusal_escalates(repo, monkeypatch):
    """The second refusal in one run points at kanban_block, not another retry."""
    (repo / "src.py").write_text("edited\n")
    m = _gate()
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_x")
    _fake_api(monkeypatch, m, "dir", repo)
    first = _call(m, "t_x")
    second = _call(m, "t_x")
    assert first["action"] == "block" and second["action"] == "block"
    assert "Do NOT keep retrying" not in first["message"]
    assert "Do NOT keep retrying" in second["message"]


def test_other_tools_pass_through(repo, monkeypatch):
    (repo / "src.py").write_text("edited\n")
    m = _gate()
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_x")
    _fake_api(monkeypatch, m, "dir", repo)
    assert _call(m, "t_x", tool="kanban_comment") is None


class _FakeConn:
    def execute(self, *a, **k):
        raise RuntimeError("not used")

    def close(self):
        pass
