"""Tests: kanban dispatcher refuses docker workers whose workspace cannot be
host-backed (#91568 loud-failure slice).

With a remote ``DOCKER_HOST`` the Docker daemon runs on a different machine
from the Hermes dispatcher. A bind-mount source path is validated client-side
(``os.path.isdir`` in tools/environments/docker.py) but resolved on the daemon
host — which silently auto-creates missing directories. A docker-backend kanban
worker whose task workspace is a plain local path therefore sees an empty
sandbox instead of the real task directory: every commit it makes vanishes
when the container exits.

The dispatcher must fail that spawn loudly (dispatch attempt error + task
event) instead of losing work silently. Local daemons (unix socket / npipe /
loopback TCP) are untouched, and operators whose workspace genuinely lives on
storage shared with the daemon host (e.g. NFS at the same path on both sides)
can set ``kanban.docker_remote_workspace_force: true`` to downgrade the
refusal to a one-line warning.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Isolation fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_docker_env(monkeypatch):
    """Neutralize the developer's real DOCKER_HOST unless a test sets one."""
    monkeypatch.delenv("DOCKER_HOST", raising=False)


@pytest.fixture()
def kb_home(monkeypatch, tmp_path):
    """Fresh HERMES_HOME + kanban board, hermes_cli modules re-imported."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for mod in list(sys.modules.keys()):
        if (
            mod.startswith("hermes_cli")
            or mod.startswith("hermes_state")
            or mod == "hermes_constants"
        ):
            del sys.modules[mod]
    from hermes_cli import kanban_db

    kanban_db.create_board(slug="default", name="Test")
    yield kanban_db, home


def _write_profile_config(
    home: Path,
    name: str,
    *,
    backend: str,
    mount_cwd: bool,
    force: bool | None = None,
) -> None:
    """Materialize a profile whose config pins the terminal backend."""
    pdir = home / "profiles" / name
    pdir.mkdir(parents=True, exist_ok=True)
    lines = [
        "terminal:",
        f"  backend: {backend}",
        f"  docker_mount_cwd_to_workspace: {'true' if mount_cwd else 'false'}",
    ]
    if force is not None:
        lines += [
            "kanban:",
            f"  docker_remote_workspace_force: {'true' if force else 'false'}",
        ]
    (pdir / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _recording_spawn(calls: list):
    def _spawn(task, workspace, board=None):
        calls.append((task.id, str(workspace)))
        return 4242

    return _spawn


_REMOTE_HOST = "tcp://daemon.example.local:2375"


def _create_dir_task(kb, conn, ws: Path, assignee: str = "docky") -> str:
    return kb.create_task(
        conn,
        title="host-backed workspace please",
        assignee=assignee,
        workspace_kind="dir",
        workspace_path=str(ws),
    )


# ---------------------------------------------------------------------------
# Pure decision core
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "docker_host,expected_remote",
    [
        (None, False),
        ("", False),
        ("   ", False),
        ("unix:///var/run/docker.sock", False),
        ("npipe:////./pipe/docker_engine", False),
        ("/var/run/docker.sock", False),
        ("tcp://localhost:2375", False),
        ("tcp://127.0.0.1:2375", False),
        ("tcp://[::1]:2375", False),
        ("http://localhost:2375", False),
        ("tcp://192.168.1.50:2375", True),
        ("ssh://user@buildhost", True),
        ("https://daemon.corp.example:2376", True),
    ],
)
def test_docker_host_classification(kb_home, docker_host, expected_remote):
    kb, _home = kb_home
    assert kb._docker_host_is_remote(docker_host) is expected_remote


def test_unknown_docker_host_scheme_fails_closed(kb_home):
    kb, _home = kb_home
    # Unrecognizable non-empty values must be treated as remote.
    assert kb._docker_host_is_remote("weird://somehost") is True
    # Path-shaped scheme-less values are local socket paths.
    assert kb._docker_host_is_remote("\\\\.\\pipe\\docker_engine") is False


@pytest.mark.parametrize(
    "workspace,shared",
    [
        ("\\\\nas\\share\\tasks\\t_1", True),
        ("//nas/share/tasks/t_1", True),
        ("C:\\projects\\task-ws", False),
        ("/srv/workspaces/t_1", False),
        ("", False),
    ],
)
def test_network_shared_path_detection(kb_home, workspace, shared):
    kb, _home = kb_home
    assert kb._workspace_path_is_network_shared(workspace) is shared


def test_refusal_reason_matrix(kb_home):
    kb, _home = kb_home
    # Local host → never refuse.
    assert (
        kb._docker_remote_workspace_refusal_reason("C:\\ws", None) is None
    )
    assert (
        kb._docker_remote_workspace_refusal_reason("/srv/ws", "unix:///var/run/docker.sock")
        is None
    )
    # Remote host + UNC-style shared path → allowed.
    assert (
        kb._docker_remote_workspace_refusal_reason(
            "\\\\nas\\share\\ws", _REMOTE_HOST
        )
        is None
    )
    # Remote host + local paths → refusal naming the workspace AND the host.
    for workspace in ("C:\\projects\\task-ws", "/srv/workspaces/t_1"):
        reason = kb._docker_remote_workspace_refusal_reason(workspace, _REMOTE_HOST)
        assert reason is not None
        assert workspace in reason
        assert _REMOTE_HOST in reason
        assert "#91568" in reason
        assert "kanban.docker_remote_workspace_force" in reason
        assert "commit" in reason.lower()


def test_default_config_ships_fail_closed(kb_home):
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["kanban"]["docker_remote_workspace_force"] is False


# ---------------------------------------------------------------------------
# Dispatch-level behavior
# ---------------------------------------------------------------------------


def test_remote_host_refuses_spawn_and_records_event(kb_home, monkeypatch, caplog, tmp_path):
    """Remote DOCKER_HOST + local dir workspace → spawn refused, event recorded."""
    kb, home = kb_home
    _write_profile_config(home, "docky", backend="docker", mount_cwd=True)
    ws = tmp_path / "task_ws"
    ws.mkdir()
    monkeypatch.setenv("DOCKER_HOST", _REMOTE_HOST)

    calls: list = []
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="hermes_cli.kanban_db"):
        with kb.connect_closing() as conn:
            tid = _create_dir_task(kb, conn, ws)
            res = kb.dispatch_once(conn, spawn_fn=_recording_spawn(calls))

            # Nothing spawned; the claim was released back to ready.
            assert calls == []
            assert res.spawned == []
            row = conn.execute(
                "SELECT status, consecutive_failures, last_failure_error "
                "FROM tasks WHERE id = ?",
                (tid,),
            ).fetchone()
            assert row["status"] == "ready"
            assert row["consecutive_failures"] == 1
            error = row["last_failure_error"]
            assert "#91568" in error
            assert _REMOTE_HOST in error
            assert str(ws) in error
            # The dispatch attempt error is durably recorded as a task event
            # so `hermes kanban tail` shows why nothing is running.
            import json

            events = conn.execute(
                "SELECT kind, payload FROM task_events WHERE task_id = ? "
                "ORDER BY id DESC LIMIT 1",
                (tid,),
            ).fetchone()
            assert events["kind"] == "spawn_failed"
            event_payload = json.loads(events["payload"])
            assert str(ws) in event_payload["error"]
            assert _REMOTE_HOST in event_payload["error"]

    # The guard logs loudly too — an operator reading gateway/agent logs sees it.
    guard_errors = [
        r for r in caplog.records
        if r.levelno >= logging.ERROR and "#91568" in r.getMessage()
    ]
    assert guard_errors, [r.getMessage() for r in caplog.records]

    # The refusal is deterministic: the next tick refuses again and the
    # auto-block circuit breaker parks the task instead of looping forever.
    with kb.connect_closing() as conn:
        res2 = kb.dispatch_once(conn, spawn_fn=_recording_spawn(calls))
        assert calls == []
        assert tid in res2.auto_blocked
        row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        assert row["status"] == "blocked"
        gave_up = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        assert gave_up["kind"] == "gave_up"


def test_force_flag_downgrades_to_warning_and_spawns(kb_home, monkeypatch, caplog, tmp_path):
    """Profile-level kanban.docker_remote_workspace_force: true → warn + spawn."""
    kb, home = kb_home
    _write_profile_config(
        home, "docky", backend="docker", mount_cwd=True, force=True
    )
    ws = tmp_path / "nfs_ws"
    ws.mkdir()
    monkeypatch.setenv("DOCKER_HOST", _REMOTE_HOST)

    calls: list = []
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="hermes_cli.kanban_db"):
        with kb.connect_closing() as conn:
            tid = _create_dir_task(kb, conn, ws)
            res = kb.dispatch_once(conn, spawn_fn=_recording_spawn(calls))

            assert calls and calls[0][0] == tid
            assert [(t, a, w) for t, a, w in res.spawned] == [
                (tid, "docky", str(ws))
            ]
            row = conn.execute(
                "SELECT consecutive_failures FROM tasks WHERE id = ?", (tid,)
            ).fetchone()
            assert row["consecutive_failures"] == 0
            failure_events = conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ? AND kind IN "
                "('spawn_failed', 'gave_up')",
                (tid,),
            ).fetchall()
            assert failure_events == []

    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and "kanban.docker_remote_workspace_force" in r.getMessage()
        and _REMOTE_HOST in r.getMessage()
    ]
    assert len(warnings) == 1  # exactly the promised one-line warning
    errors = [
        r for r in caplog.records if r.levelno >= logging.ERROR
    ]
    assert errors == []


def test_force_flag_falls_back_to_dispatcher_config(kb_home, monkeypatch, caplog, tmp_path):
    """The dispatcher's own config.yaml force flag also unlocks the spawn."""
    kb, home = kb_home
    _write_profile_config(home, "docky", backend="docker", mount_cwd=True)
    (home / "config.yaml").write_text(
        "kanban:\n  docker_remote_workspace_force: true\n", encoding="utf-8"
    )
    ws = tmp_path / "shared_ws"
    ws.mkdir()
    monkeypatch.setenv("DOCKER_HOST", _REMOTE_HOST)

    calls: list = []
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="hermes_cli.kanban_db"):
        with kb.connect_closing() as conn:
            tid = _create_dir_task(kb, conn, ws)
            res = kb.dispatch_once(conn, spawn_fn=_recording_spawn(calls))

    assert calls and calls[0][0] == tid
    assert len(res.spawned) == 1
    assert any(
        r.levelno == logging.WARNING
        and "kanban.docker_remote_workspace_force" in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.parametrize(
    "local_docker_host",
    [None, "unix:///var/run/docker.sock", "npipe:////./pipe/docker_engine", "tcp://127.0.0.1:2375"],
)
def test_local_daemon_behavior_unchanged(kb_home, monkeypatch, caplog, tmp_path, local_docker_host):
    """Unix socket / npipe / loopback TCP → byte-identical happy path."""
    kb, home = kb_home
    _write_profile_config(home, "docky", backend="docker", mount_cwd=True)
    ws = tmp_path / "local_ws"
    ws.mkdir()
    if local_docker_host is not None:
        monkeypatch.setenv("DOCKER_HOST", local_docker_host)

    calls: list = []
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="hermes_cli.kanban_db"):
        with kb.connect_closing() as conn:
            tid = _create_dir_task(kb, conn, ws)
            res = kb.dispatch_once(conn, spawn_fn=_recording_spawn(calls))

    assert calls and calls[0][0] == tid
    assert len(res.spawned) == 1
    # No new checks fired: no refusals, no warnings, no failure bookkeeping.
    guard_logs = [
        r for r in caplog.records
        if "remote-workspace" in r.getMessage()
        or "kanban.docker_remote_workspace_force" in r.getMessage()
        or "#91568" in r.getMessage()
    ]
    assert guard_logs == []
    with kb.connect_closing() as conn:
        row = conn.execute(
            "SELECT consecutive_failures FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        assert row["consecutive_failures"] == 0
        failure_events = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? AND kind IN "
            "('spawn_failed', 'gave_up')",
            (tid,),
        ).fetchall()
        assert failure_events == []


def test_non_docker_backend_untouched_by_guard(kb_home, monkeypatch, caplog, tmp_path):
    """A local-backend profile ignores DOCKER_HOST entirely."""
    kb, home = kb_home
    _write_profile_config(home, "loky", backend="local", mount_cwd=False)
    ws = tmp_path / "plain_ws"
    ws.mkdir()
    monkeypatch.setenv("DOCKER_HOST", _REMOTE_HOST)

    calls: list = []
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="hermes_cli.kanban_db"):
        with kb.connect_closing() as conn:
            tid = _create_dir_task(kb, conn, ws, assignee="loky")
            res = kb.dispatch_once(conn, spawn_fn=_recording_spawn(calls))

    assert calls and calls[0][0] == tid
    assert len(res.spawned) == 1
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_mount_cwd_disabled_not_guarded(kb_home, monkeypatch, caplog, tmp_path):
    """Docker without cwd→/workspace mount runs an isolated sandbox BY DESIGN.

    The guard only fires on the bind-mount configuration that promises the
    real workspace inside the container; the tmpfs/sandbox fallback in
    tools/environments/docker.py stays untouched even against remote hosts.
    """
    kb, home = kb_home
    _write_profile_config(home, "docky", backend="docker", mount_cwd=False)
    ws = tmp_path / "tmpfs_lane_ws"
    ws.mkdir()
    monkeypatch.setenv("DOCKER_HOST", _REMOTE_HOST)

    calls: list = []
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="hermes_cli.kanban_db"):
        with kb.connect_closing() as conn:
            tid = _create_dir_task(kb, conn, ws)
            res = kb.dispatch_once(conn, spawn_fn=_recording_spawn(calls))

    assert calls and calls[0][0] == tid
    assert len(res.spawned) == 1
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_worktree_lane_refuses_too(kb_home, monkeypatch, caplog, tmp_path):
    """The worktree resolution lane gets the same gate before spawn."""
    kb, home = kb_home
    _write_profile_config(home, "docky", backend="docker", mount_cwd=True)
    wt = tmp_path / "repo-checkout"
    wt.mkdir()
    monkeypatch.setenv("DOCKER_HOST", _REMOTE_HOST)

    calls: list = []
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="hermes_cli.kanban_db"):
        with kb.connect_closing() as conn:
            tid = kb.create_task(
                conn,
                title="wt task",
                assignee="docky",
                workspace_kind="worktree",
                workspace_path=str(wt),
            )
            # Deterministic worktree resolution: pretend the path already is
            # a linked worktree checkout of this task's branch (no git
            # subprocesses involved).
            monkeypatch.setattr(kb, "_is_linked_worktree_checkout", lambda p: True)
            monkeypatch.setattr(
                kb, "_git_current_branch", lambda p: f"wt/{tid}"
            )
            res = kb.dispatch_once(conn, spawn_fn=_recording_spawn(calls))

    assert calls == []
    assert res.spawned == []
    with kb.connect_closing() as conn:
        row = conn.execute(
            "SELECT last_failure_error, consecutive_failures FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        assert row["consecutive_failures"] == 1
        assert "#91568" in row["last_failure_error"]
