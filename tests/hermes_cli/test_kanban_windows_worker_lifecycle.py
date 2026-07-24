"""Focused Windows worker reaping and exit classification regressions."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
    ):
        monkeypatch.delenv(var, raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb._worker_processes.clear()
    kb._recent_worker_exits.clear()
    kb._windows_direct_worker_exits.clear()
    return home


def _task() -> kb.Task:
    return kb.Task(
        id="t_windows_worker",
        title="worker test",
        body=None,
        assignee="teknium",
        status="ready",
        priority=0,
        created_by=None,
        created_at=0,
        started_at=None,
        completed_at=None,
        workspace_kind="scratch",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
    )


def test_default_spawn_retains_popen_handle_for_reaping(hermes_home, monkeypatch):
    class FakeProc:
        pid = 12346

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: FakeProc())

    kb._default_spawn(_task(), str(hermes_home / "ws"), board=None)

    assert kb._worker_processes[12346].pid == 12346


def test_windows_reaping_polls_only_exited_workers(monkeypatch):
    class FakeProc:
        def __init__(self, status):
            self.status = status

        def poll(self):
            return self.status

    monkeypatch.setattr(kb.os, "name", "nt")
    kb._worker_processes.clear()
    kb._recent_worker_exits.clear()
    kb._windows_direct_worker_exits.clear()
    kb._worker_processes.update({2001: FakeProc(None), 2002: FakeProc(75)})

    assert kb.reap_worker_zombies() == [2002]
    assert 2001 in kb._worker_processes
    assert kb._classify_worker_exit(2002) == ("rate_limited", 75)


@pytest.mark.parametrize(
    ("exit_code", "expected"),
    [
        (1, ("nonzero_exit", 1)),
        (75, ("rate_limited", 75)),
        (-9, ("signaled", 9)),
    ],
)
def test_windows_classifies_direct_popen_exit_codes(
    monkeypatch, exit_code, expected
):
    monkeypatch.setattr(kb.os, "name", "nt")
    kb._recent_worker_exits.clear()
    kb._windows_direct_worker_exits.clear()
    kb._record_worker_exit(3001, exit_code, windows_direct=True)

    assert kb._classify_worker_exit(3001) == expected


def test_exit_provenance_markers_are_removed_with_registry_eviction(monkeypatch):
    """The Windows-only side registry must remain bounded with its source."""
    kb._recent_worker_exits.clear()
    kb._windows_direct_worker_exits.clear()
    monkeypatch.setattr(kb.time, "time", lambda: 2_000.0)

    expired_pid = 4001
    kb._recent_worker_exits[expired_pid] = (1, 0.0)
    kb._windows_direct_worker_exits.add(expired_pid)
    for pid in range(4100, 4100 + kb._RECENT_WORKER_EXITS_MAX // 2):
        kb._record_worker_exit(pid, 1, windows_direct=True)

    assert expired_pid not in kb._recent_worker_exits
    assert expired_pid not in kb._windows_direct_worker_exits

    kb._recent_worker_exits.clear()
    kb._windows_direct_worker_exits.clear()
    oldest_pid = 5001
    kb._record_worker_exit(oldest_pid, 1, windows_direct=True)
    for pid in range(5100, 5100 + kb._RECENT_WORKER_EXITS_MAX):
        kb._record_worker_exit(pid, 1, windows_direct=True)

    assert oldest_pid not in kb._recent_worker_exits
    assert oldest_pid not in kb._windows_direct_worker_exits
