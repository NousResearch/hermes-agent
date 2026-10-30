"""Host integration proof for the disposable systemd + Bubblewrap supervisor."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import uuid

import pytest


def _isolation_root() -> Path:
    value = os.environ.get("HERMES_TEST_SANDBOX_HOME") or os.environ.get("HERMES_HOME")
    if not value:
        pytest.skip("requires the explicit isolated Phase 1 test root")
    root = Path(value).resolve()
    if not root.is_dir():
        pytest.skip("isolated Phase 1 test root does not exist")
    if shutil.which("bwrap") is None or shutil.which("systemd-run") is None:
        pytest.skip("requires Bubblewrap and systemd-run on Linux")
    env = dict(os.environ)
    env["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"
    manager = subprocess.run(["systemctl", "--user", "show", "--property=Version"],
                             capture_output=True, text=True, timeout=5, env=env, check=False)
    if manager.returncode:
        pytest.skip("requires an available systemd user manager")
    return root


def _new_root() -> Path:
    root = _isolation_root() / "phase1-supervisor-poc" / uuid.uuid4().hex
    root.mkdir(parents=True)
    return root


def _env(root: Path) -> dict[str, str]:
    project = Path(__file__).resolve().parents[2]
    for name in ("home", "hermes/profiles/fred", "tmp", "xdg/config", "xdg/cache", "xdg/data",
                 "xdg/state", "xdg/runtime"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(root / "home"),
        "HERMES_HOME": str(root / "hermes" / "profiles" / "fred"),
        "HERMES_KANBAN_DB": str(root / "board.db"),
        "HERMES_ISOLATION_ROOT": str(_isolation_root()),
        "HERMES_PROTECTED_HERMES": str(Path.home() / ".hermes"),
        "HERMES_PROTECTED_KANBAN_DB": str(Path.home() / ".hermes" / "kanban.db"),
        "HERMES_PROTECTED_X": str(Path.home() / "X"),
        "PYTHONPATH": str(project),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(root / "tmp"),
        "XDG_CONFIG_HOME": str(root / "xdg" / "config"),
        "XDG_CACHE_HOME": str(root / "xdg" / "cache"),
        "XDG_DATA_HOME": str(root / "xdg" / "data"),
        "XDG_STATE_HOME": str(root / "xdg" / "state"),
        "XDG_RUNTIME_DIR": str(root / "xdg" / "runtime"),
        "LANG": "C.UTF-8",
        "TZ": "UTC",
    }


def _script() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "mission_supervisor_poc.py"


def _run(root: Path, *args: str, timeout: float = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_script()), "--root", str(root / "worker"),
         "--db", str(root / "board.db"), *args],
        env=_env(root), capture_output=True, text=True, timeout=timeout, check=False,
    )


def _json(result: subprocess.CompletedProcess[str]) -> dict:
    assert result.stdout.strip(), result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.integration
@pytest.mark.platforms("linux")
def test_supervisor_refuses_live_board_and_outside_workspace():
    root = _new_root()
    env = _env(root)
    live_board = Path.home() / ".hermes" / "kanban.db"
    refused_board = subprocess.run(
        [sys.executable, str(_script()), "--root", str(root / "worker"), "--db", str(live_board)],
        env=env, capture_output=True, text=True, timeout=10, check=False,
    )
    assert refused_board.returncode == 2
    assert "must be inside HERMES_ISOLATION_ROOT" in refused_board.stderr

    outside_root = Path.home() / "X" / ".phase1-supervisor-refusal-probe"
    refused_root = subprocess.run(
        [sys.executable, str(_script()), "--root", str(outside_root), "--db", str(root / "board.db")],
        env=env, capture_output=True, text=True, timeout=10, check=False,
    )
    assert refused_root.returncode == 2
    assert "must be inside the existing HERMES_ISOLATION_ROOT" in refused_root.stderr
    assert not outside_root.exists()


@pytest.mark.integration
@pytest.mark.platforms("linux")
def test_real_scope_bubblewrap_boundary_and_quiescent_handoff():
    root = _new_root()
    result = _run(root, "--mode", "demo", timeout=30)
    assert result.returncode == 0, result.stderr
    evidence = _json(result)

    scope = evidence["scope"]
    assert scope["active"] == "active"
    assert scope["populated"] is True
    assert scope["child_session"] == scope["child_pid"]  # setsid child
    assert scope["child_pid"] in scope["cgroup_pids"]
    assert scope["parent_pid"] not in scope["cgroup_pids"]  # parent exited first
    assert scope["cgroup"].endswith(".scope")

    assert evidence["blocked_handoff"]["transferred"] is False
    assert "not quiescent" in evidence["blocked_handoff"]["reason"]
    assert evidence["transfer"]["transferred"] is True
    assert evidence["transfer"]["generation"] == 2
    assert evidence["transfer"]["populated"] is False

    worker_root = root / "worker"
    assert (worker_root / "output" / "input-read.txt").read_text() == "approved read-only input\n"
    boundary = json.loads((worker_root / "output" / "write-boundary.json").read_text())
    assert boundary["input"]["denied"] is True
    assert boundary["protected"]["denied"] is True
    assert boundary["outside"]["denied"] is True
    assert boundary["unapproved_output"]["denied"] is True
    assert boundary["cgroup_control"]["denied"] is True
    assert all(value != "UNEXPECTED_WRITE" for value in boundary.values())
    assert (worker_root / "protected" / "mission.txt").read_text() == "protected mission state\n"

    escape = evidence["escapes"]
    assert escape["systemd_scope"]["exit_code"] != 0
    assert escape["systemd_timer"]["exit_code"] != 0
    assert "ignoring command 'start'" in escape["systemd_service"]["output"]
    assert escape["host_unit_load_states"]["hermes-poc-escape.scope"] == "not-found"
    assert escape["host_unit_load_states"]["hermes-poc-escape.service"] == "not-found"
    assert escape["host_unit_load_states"]["hermes-poc-escape-timer.timer"] == "not-found"
    assert escape["container"]["exit_code"] != 0
    assert escape["at_scheduler"]["unavailable"] == "FileNotFoundError"
    assert escape["crontab_scheduler"]["unavailable"] == "FileNotFoundError"

    metrics = evidence["metrics"]
    assert len(metrics["launch_ms_samples"]) == 4
    assert metrics["launch_ms_median"] > 0
    assert metrics["scope_memory_bytes"]["memory.peak"] is not None


@pytest.mark.integration
@pytest.mark.platforms("linux")
def test_worker_failure_supervisor_restart_cancel_and_stale_owner_refusal():
    root = _new_root()
    started = _run(root, "--mode", "start", "--delay", "30", "--crash", timeout=20)
    assert started.returncode == 71, started.stderr
    initial = _json(started)
    mission_id = initial["mission_id"]
    child_pid = initial["child_pid"]
    protected = root / "worker" / "protected" / "mission.txt"
    before = hashlib.sha256(protected.read_bytes()).hexdigest()

    # This is a new supervisor process reading the durable operation row and OS scope.
    recovered = _run(root, "--mode", "recover", "--mission-id", mission_id)
    assert recovered.returncode == 0, recovered.stderr
    assert _json(recovered)["recovered"] is False
    assert _json(recovered)["populated"] is True

    blocked = _run(root, "--mode", "handoff", "--mission-id", mission_id)
    assert blocked.returncode == 0, blocked.stderr
    assert _json(blocked)["transferred"] is False

    cancelled = _run(root, "--mode", "recover", "--cancel", "--mission-id", mission_id)
    assert cancelled.returncode == 0, cancelled.stderr
    assert _json(cancelled)["recovered"] is True
    assert _json(cancelled)["populated"] is False
    transferred = _run(root, "--mode", "handoff", "--mission-id", mission_id)
    assert _json(transferred)["transferred"] is True

    stale = _run(root, "--mode", "start", "--mission-id", mission_id, "--delay", "0.1")
    assert stale.returncode == 2
    assert "worker start does not match current mission owner" in stale.stderr
    assert hashlib.sha256(protected.read_bytes()).hexdigest() == before
    assert not Path("/proc", str(child_pid)).exists()
    assert hashlib.sha256(protected.read_bytes()).hexdigest() == before


@pytest.mark.integration
@pytest.mark.platforms("linux")
def test_launch_handoff_termination_and_concurrent_launch_races(monkeypatch):
    from scripts.mission_supervisor_poc import MissionSupervisorPoc, SupervisorError

    root = _new_root()
    for key, value in _env(root).items():
        monkeypatch.setenv(key, value)
    # The test process has a real-home I/O guard, so use inert protected-path
    # sentinels here. Separate subprocess tests exercise live-path rejection.
    monkeypatch.setenv("HERMES_PROTECTED_HERMES", str(root / "protected-hermes"))
    monkeypatch.setenv("HERMES_PROTECTED_KANBAN_DB", str(root / "protected-board.db"))
    monkeypatch.setenv("HERMES_PROTECTED_X", str(root / "protected-x"))
    db = root / "board.db"
    supervisor = MissionSupervisorPoc(db, root / "race-a")
    mission_id = supervisor.admit("poc:launch-race")
    launch_gate, launch_waiting = threading.Event(), threading.Event()
    launched = []
    errors = []

    def start_race():
        try:
            launched.append(supervisor.start(mission_id, delay=0.25,
                                             launch_gate=launch_gate, launch_waiting=launch_waiting))
        except BaseException as exc:
            errors.append(exc)

    starter = threading.Thread(target=start_race)
    starter.start()
    assert launch_waiting.wait(timeout=5)
    during_start = MissionSupervisorPoc(db, root / "race-b")
    refused = during_start.handoff(mission_id)
    assert refused["transferred"] is False
    assert refused["state"] == "starting"
    with pytest.raises(SupervisorError, match="already has a supervisor operation"):
        during_start.start(mission_id, delay=0.2)
    launch_gate.set()
    starter.join(timeout=15)
    assert not starter.is_alive() and not errors
    during_start.wait_empty(mission_id)

    first, second = MissionSupervisorPoc(db, root / "race-c"), MissionSupervisorPoc(db, root / "race-d")
    barrier = threading.Barrier(2)
    outcomes = []

    def transfer(actor):
        barrier.wait()
        try:
            outcomes.append(actor.handoff(mission_id))
        except BaseException as exc:
            outcomes.append(exc)

    racers = [threading.Thread(target=transfer, args=(actor,)) for actor in (first, second)]
    for racer in racers:
        racer.start()
    for racer in racers:
        racer.join(timeout=5)
    assert sum(isinstance(item, dict) and item.get("transferred") is True for item in outcomes) == 1

    # Different missions may run in separate scopes at the same time.
    left, right = MissionSupervisorPoc(db, root / "parallel-left"), MissionSupervisorPoc(db, root / "parallel-right")
    left_id = left.admit("poc:parallel:left", owner="fred-left")
    right_id = right.admit("poc:parallel:right", owner="fred-right")
    barrier = threading.Barrier(2)
    parallel, errors = [], []

    def launch(actor, mid, owner):
        try:
            barrier.wait()
            parallel.append(actor.start(mid, owner_id=owner, delay=0.35))
        except BaseException as exc:
            errors.append(exc)

    workers = [threading.Thread(target=launch, args=(left, left_id, "fred-left")),
               threading.Thread(target=launch, args=(right, right_id, "fred-right"))]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=15)
    assert not errors and len(parallel) == 2
    assert parallel[0]["unit"] != parallel[1]["unit"]
    assert parallel[0]["cgroup"] != parallel[1]["cgroup"]
    for actor, mid, owner in ((left, left_id, "fred-left"), (right, right_id, "fred-right")):
        actor.wait_empty(mid)
        actor.handoff(mid, owner_id=owner)

    # A handoff racing the cancellation phase still sees the durable lease.
    cancel_id = left.admit("poc:cancel-race", owner="fred-cancel")
    left.start(cancel_id, owner_id="fred-cancel", delay=30)
    cancel_gate, cancel_waiting = threading.Event(), threading.Event()
    cancel_errors = []

    def stop_race():
        try:
            left.cancel(cancel_id, cancel_gate=cancel_gate, cancel_waiting=cancel_waiting)
        except BaseException as exc:
            cancel_errors.append(exc)

    stopper = threading.Thread(target=stop_race)
    stopper.start()
    assert cancel_waiting.wait(timeout=5)
    cancel_owner = MissionSupervisorPoc(db, root / "cancel-race-observer")
    blocked_cancel = cancel_owner.handoff(cancel_id, owner_id="fred-cancel")
    assert blocked_cancel["transferred"] is False
    assert blocked_cancel["state"] == "cancelling"
    cancel_gate.set()
    stopper.join(timeout=15)
    assert not stopper.is_alive() and not cancel_errors
    assert left.handoff(cancel_id, owner_id="fred-cancel")["transferred"] is True
