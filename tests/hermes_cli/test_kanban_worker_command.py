"""Tests for direct-command workers (profile-level ``worker.command``)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


def _make_task(kb_mod, *, assignee: str, goal_mode: bool = False):
    return kb_mod.Task(
        id="t_worker_cmd",
        title="worker command",
        body=None,
        assignee=assignee,
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=7,
        goal_mode=goal_mode,
    )


def _write_profile(root, name: str, config_body: str) -> None:
    profile = root / "profiles" / name
    profile.mkdir(parents=True, exist_ok=True)
    profile.joinpath("config.yaml").write_text(config_body, encoding="utf-8")


COMMAND_PROFILE = "worker:\n  command:\n    - /usr/local/bin/lassdas-worker\n    - --stage-all\n"

SUPERVISOR = [sys.executable, "-P", "-m", "hermes_cli.kanban_command_worker"]


@pytest.fixture()
def spawn_env(monkeypatch, tmp_path):
    root = tmp_path / ".hermes"
    root.mkdir()
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])

    captured = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return root, str(workspace), captured


# ---------------------------------------------------------------------------
# Spawn: the supervisor replaces the agent argv
# ---------------------------------------------------------------------------


def test_worker_command_spawns_the_supervisor(spawn_env):
    """A profile-declared worker.command runs under the supervisor module,
    with the resolved argv frozen into the child environment and the full
    kanban worker environment still attached."""
    root, workspace, captured = spawn_env
    _write_profile(root, "engine", COMMAND_PROFILE)

    pid = kb._default_spawn(_make_task(kb, assignee="engine"), workspace)

    assert pid == 4242
    assert captured["cmd"] == SUPERVISOR
    assert json.loads(captured["env"]["HERMES_KANBAN_WORKER_COMMAND"]) == [
        "/usr/local/bin/lassdas-worker",
        "--stage-all",
    ]
    assert captured["env"]["HERMES_KANBAN_TASK"] == "t_worker_cmd"
    assert captured["env"]["HERMES_KANBAN_WORKSPACE"] == workspace
    assert captured["env"]["HERMES_KANBAN_RUN_ID"] == "7"
    assert captured["env"]["HERMES_KANBAN_CLAIM_LOCK"] == "lock"


def test_worker_command_ignores_agent_only_task_settings(spawn_env):
    """Goal mode is an agent-loop concern; a direct command must not grow
    agent flags out of it (a warning is logged instead)."""
    root, workspace, captured = spawn_env
    _write_profile(root, "engine", "worker:\n  command:\n    - /bin/lassdas-worker\n")

    kb._default_spawn(_make_task(kb, assignee="engine", goal_mode=True), workspace)

    assert captured["cmd"] == SUPERVISOR


def test_profile_without_worker_command_keeps_the_agent_argv(spawn_env):
    root, workspace, captured = spawn_env
    _write_profile(root, "elias", "toolsets:\n  - hermes-cli\n")

    kb._default_spawn(_make_task(kb, assignee="elias"), workspace)

    assert captured["cmd"][0] == "hermes"
    assert "chat" in captured["cmd"]


def test_env_references_expand_in_worker_command(spawn_env, monkeypatch):
    root, workspace, captured = spawn_env
    monkeypatch.setenv("LASSDAS_BIN", "/opt/lassdas/worker")
    _write_profile(
        root, "engine", "worker:\n  command:\n    - ${LASSDAS_BIN}\n    - run\n"
    )

    kb._default_spawn(_make_task(kb, assignee="engine"), workspace)

    assert json.loads(captured["env"]["HERMES_KANBAN_WORKER_COMMAND"]) == [
        "/opt/lassdas/worker",
        "run",
    ]


# ---------------------------------------------------------------------------
# Resolution: fail-loud contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "declared",
    [
        'worker:\n  command: "/bin/worker --flag"\n',
        "worker:\n  command: []\n",
        "worker:\n  command:\n    - /bin/worker\n    - 7\n",
        "worker:\n  command: {a: b}\n",
    ],
    ids=["string", "empty-list", "non-string-part", "dict"],
)
def test_invalid_worker_command_fails_loudly(spawn_env, declared):
    """A declared-but-invalid worker.command must raise, not silently fall
    back to the agent: running the wrong worker is worse than none."""
    root, workspace, captured = spawn_env
    _write_profile(root, "engine", declared)

    with pytest.raises(RuntimeError, match="worker.command"):
        kb._default_spawn(_make_task(kb, assignee="engine"), workspace)
    assert "cmd" not in captured


def test_unparsable_profile_config_fails_loudly(spawn_env):
    """The most ordinary way to break the declaration is a YAML indent typo.
    ``load_config()`` would swallow it and return defaults — the resolver
    must not, or the declared command silently becomes an agent run."""
    root, workspace, captured = spawn_env
    _write_profile(
        root,
        "engine",
        # One line indented by 3 spaces instead of 4 — unparsable YAML.
        "worker:\n  command:\n    - /bin/worker\n   - --flag\n",
    )

    with pytest.raises(RuntimeError, match="does not parse as YAML"):
        kb._default_spawn(_make_task(kb, assignee="engine"), workspace)
    assert "cmd" not in captured


def test_explicit_null_worker_command_means_agent(spawn_env):
    """``worker.command:`` with no value is an explicit non-declaration —
    the agent runs. Pinned as intended behaviour, not an accident."""
    root, workspace, captured = spawn_env
    _write_profile(root, "engine", "worker:\n  command:\n")

    kb._default_spawn(_make_task(kb, assignee="engine"), workspace)

    assert captured["cmd"][0] == "hermes"


def test_root_config_declaration_is_rejected(spawn_env):
    """``kanban.*`` keys are dispatcher scope; this key is assignee scope.
    The ``default`` assignee resolves to the root home, so a root-level
    declaration must fail instead of silently running for ``default`` and
    silently not running for everyone else."""
    root, workspace, captured = spawn_env
    root.joinpath("config.yaml").write_text(COMMAND_PROFILE, encoding="utf-8")

    with pytest.raises(RuntimeError, match="root config"):
        kb._default_spawn(_make_task(kb, assignee="default"), workspace)
    assert "cmd" not in captured


def test_root_guard_holds_when_dispatcher_runs_inside_a_profile_home(
    spawn_env, monkeypatch
):
    """``hermes -p engine kanban daemon`` sets the process HERMES_HOME to the
    profile directory. The scope guard must not compare against the process
    home: judged structurally (parent directory named ``profiles``), a named
    profile stays accepted and a root declaration for ``default`` stays
    rejected — in this constitution too."""
    root, workspace, captured = spawn_env
    _write_profile(root, "engine", COMMAND_PROFILE)
    root.joinpath("config.yaml").write_text(COMMAND_PROFILE, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "engine"))

    kb._default_spawn(_make_task(kb, assignee="engine"), workspace)
    assert captured["cmd"] == SUPERVISOR

    captured.clear()
    with pytest.raises(RuntimeError, match="root config"):
        kb._default_spawn(_make_task(kb, assignee="default"), workspace)
    assert "cmd" not in captured


def test_relative_argv0_is_rejected(spawn_env):
    """A relative argv[0] resolves against the per-task workspace, whose
    content the task's own branch controls — that is workspace-controlled
    code execution and must be refused."""
    root, workspace, captured = spawn_env
    _write_profile(root, "engine", "worker:\n  command:\n    - ./run.sh\n")

    with pytest.raises(RuntimeError, match="argv\\[0\\]"):
        kb._default_spawn(_make_task(kb, assignee="engine"), workspace)
    assert "cmd" not in captured


# ---------------------------------------------------------------------------
# The supervisor: rc is the completion report, delivered while alive
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _claim_with_run(conn, tid: str) -> int:
    host = kb._claimer_id().split(":", 1)[0]
    kb.claim_task(conn, tid, claimer=f"{host}:w0")
    row = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id=?", (tid,)
    ).fetchone()
    return row["current_run_id"]


def _run_supervisor(kanban_home, tid: str, run_id, argv: list) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ)
    env["HERMES_HOME"] = str(kanban_home)
    env["HERMES_KANBAN_DB"] = str(kb.kanban_db_path())
    env["HERMES_KANBAN_TASK"] = tid
    env["HERMES_KANBAN_RUN_ID"] = str(run_id) if run_id is not None else ""
    env["HERMES_KANBAN_WORKER_COMMAND"] = json.dumps(argv)
    return subprocess.run(
        SUPERVISOR, env=env, capture_output=True, text=True, timeout=60
    )


def test_supervisor_reports_rc0_as_complete(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ok", assignee="engine")
        run_id = _claim_with_run(conn, tid)
    proc = _run_supervisor(
        kanban_home, tid, run_id, [sys.executable, "-c", "pass"]
    )
    assert proc.returncode == 0, proc.stderr
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "done"


def test_supervisor_reports_nonzero_rc_as_block(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="fail", assignee="engine")
        run_id = _claim_with_run(conn, tid)
    proc = _run_supervisor(
        kanban_home, tid, run_id, [sys.executable, "-c", "import sys; sys.exit(3)"]
    )
    assert proc.returncode == 0, proc.stderr
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "blocked"
    with kb.connect() as conn:
        payloads = [
            r["payload"]
            for r in conn.execute(
                "SELECT payload FROM task_events WHERE task_id=? AND kind='blocked'",
                (tid,),
            ).fetchall()
        ]
    assert any("code 3" in (p or "") for p in payloads)


def test_supervisor_missing_executable_exits_nonzero_and_reports_nothing(
    kanban_home, tmp_path
):
    """An unstartable command is the supervisor's own failure: it exits
    non-zero without transitioning the task, and the ordinary crash path
    (retry + breaker) covers the card."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="missing", assignee="engine")
        run_id = _claim_with_run(conn, tid)
    proc = _run_supervisor(
        kanban_home, tid, run_id, [str(tmp_path / "definitely-not-here")]
    )
    assert proc.returncode != 0
    assert "not found" in proc.stderr
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "running"


def test_supervisor_respects_a_transition_the_command_already_made(kanban_home):
    """A command that already moved its task through a canonical channel is
    left alone: the command's own word stands, rc translation is a no-op."""
    script = (
        "import os, sqlite3\n"
        "from hermes_cli import kanban_db as kb\n"
        "conn = kb.connect().__enter__()\n"
        "kb.block_task(conn, os.environ['HERMES_KANBAN_TASK'],"
        " reason='needs an answer', kind='needs_input')\n"
    )
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="self", assignee="engine")
        run_id = _claim_with_run(conn, tid)
    proc = _run_supervisor(kanban_home, tid, run_id, [sys.executable, "-c", script])
    assert proc.returncode == 0, proc.stderr
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "blocked"
    with kb.connect() as conn:
        payloads = [
            r["payload"]
            for r in conn.execute(
                "SELECT payload FROM task_events WHERE task_id=? AND kind='blocked'",
                (tid,),
            ).fetchall()
        ]
    assert any("needs an answer" in (p or "") for p in payloads)
    # The command's own transition stands: no second 'blocked' event from
    # the supervisor's rc translation.
    assert len(payloads) == 1


# ---------------------------------------------------------------------------
# Integration: a real dispatch tick end-to-end (no mocks)
# ---------------------------------------------------------------------------


def test_dispatch_tick_runs_a_command_worker_to_done(kanban_home, monkeypatch):
    """The full path: create -> dispatch_once (real spawn) -> supervisor ->
    complete. Covers the ordering that unit-level reap tests cannot: the
    completion lands while the worker is alive, so no later reclaim pass
    can ever see a finished-but-running card."""
    _write_profile(
        kanban_home,
        "engine",
        f"worker:\n  command:\n    - {sys.executable}\n    - -c\n    - pass\n",
    )
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="e2e", assignee="engine")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        conn.commit()

        result = kb.dispatch_once(conn, max_in_progress=1)
        assert any(t[0] == tid for t in result.spawned), result

        deadline = time.time() + 30
        status = None
        while time.time() < deadline:
            task = kb.get_task(conn, tid)
            status = task.status
            if status == "done":
                break
            time.sleep(0.3)
        assert status == "done", f"worker never completed the card: {status}"


# ---------------------------------------------------------------------------
# The supervisor as a hostile-workspace / termination target (review round 3)
# ---------------------------------------------------------------------------


def test_workspace_cannot_shadow_the_supervisor_module(kanban_home, tmp_path):
    """The supervisor runs with the task workspace as cwd. Without -P,
    Python would put that cwd first on sys.path and a planted
    hermes_cli/kanban_command_worker.py from the workspace — a git branch
    under worktree, a previous worker's leftovers under scratch — would run
    instead of the real module."""
    import os

    workspace = tmp_path / "ws"
    planted = workspace / "hermes_cli"
    planted.mkdir(parents=True)
    planted.joinpath("__init__.py").write_text("", encoding="utf-8")
    planted.joinpath("kanban_command_worker.py").write_text(
        "print('PLANTED MODULE EXECUTED')\nraise SystemExit(0)\n",
        encoding="utf-8",
    )

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="shadow", assignee="engine")
        run_id = _claim_with_run(conn, tid)

    env = dict(os.environ)
    env["HERMES_HOME"] = str(kanban_home)
    env["HERMES_KANBAN_DB"] = str(kb.kanban_db_path())
    env["HERMES_KANBAN_TASK"] = tid
    env["HERMES_KANBAN_RUN_ID"] = str(run_id)
    env["HERMES_KANBAN_WORKER_COMMAND"] = json.dumps([sys.executable, "-c", "pass"])
    proc = subprocess.run(
        SUPERVISOR, env=env, cwd=str(workspace),
        capture_output=True, text=True, timeout=60,
    )
    assert "PLANTED" not in proc.stdout
    assert proc.returncode == 0, proc.stderr
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "done"


def test_sigterm_forwards_to_the_child_group_within_the_grace(kanban_home):
    """A SIGTERM to the supervisor must (1) stop the child, (2) finish and
    report well inside enforce_max_runtime's ~5s term-to-kill window, and
    (3) leave the card blocked with the signal named. The handler only
    forwards; the grace is enforced by the main wait loop — a child that
    ignores SIGTERM is killed via its process group after the grace."""
    import os

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="term", assignee="engine")
        run_id = _claim_with_run(conn, tid)

    env = dict(os.environ)
    env["HERMES_HOME"] = str(kanban_home)
    env["HERMES_KANBAN_DB"] = str(kb.kanban_db_path())
    env["HERMES_KANBAN_TASK"] = tid
    env["HERMES_KANBAN_RUN_ID"] = str(run_id)
    env["HERMES_KANBAN_COMMAND_TERM_GRACE"] = "1.0"
    # The child ignores SIGTERM, so only the group SIGKILL after the grace
    # can end it — the worst case for orphaning.
    stubborn = (
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "time.sleep(120)\n"
    )
    env["HERMES_KANBAN_WORKER_COMMAND"] = json.dumps(
        [sys.executable, "-c", stubborn]
    )
    proc = subprocess.Popen(SUPERVISOR, env=env)
    time.sleep(1.0)  # let the child start
    proc.send_signal(15)
    started = time.time()
    rc = proc.wait(timeout=10)
    elapsed = time.time() - started
    assert rc == 0
    assert elapsed < 4.0, f"supervisor took {elapsed:.1f}s — exceeds the kill window"
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "blocked"
    with kb.connect() as conn:
        payloads = [
            r["payload"]
            for r in conn.execute(
                "SELECT payload FROM task_events WHERE task_id=? AND kind='blocked'",
                (tid,),
            ).fetchall()
        ]
    assert any("signal" in (p or "") for p in payloads)
