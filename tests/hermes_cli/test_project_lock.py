from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from hermes_cli import kanban
from hermes_cli import kanban_db as kb
from hermes_cli import project_lock as pl


@pytest.fixture
def claimed_owner(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kb.connect() as conn:
        workspace = Path(__file__).resolve().parents[2]
        task_id = kb.create_task(
            conn,
            title="deploy",
            assignee="developer",
            workspace_kind="dir",
            workspace_path=str(workspace),
        )
        task = kb.claim_task(conn, task_id, claimer="host:100")
        assert task is not None
        owner = pl.LeaseOwner(
            task_id=task_id,
            run_id=task.current_run_id,
            claim_lock="host:100",
            instance_id="instance-a",
            host=socket.gethostname(),
            pid=os.getpid(),
            started_at=__import__("psutil").Process(os.getpid()).create_time(),
        )
    return home, owner


def test_production_delivery_key_unifies_deploy_and_migration():
    deploy = pl.project_lock_key("Pryapus/Drip-Research-Hub", target="production")
    migration = pl.project_lock_key(
        "https://github.com/Pryapus/Drip-Research-Hub.git",
        target="production",
    )
    assert deploy == migration == "delivery:github.com/pryapus/drip-research-hub:production"
    with pytest.raises(ValueError, match="Production-only"):
        pl.project_lock_key("Pryapus/Drip-Research-Hub", target="preview")
    with pytest.raises(ValueError, match="owner/repo"):
        pl.project_lock_key("https://github.com/Pryapus/Drip-Research-Hub/tree/main")


def test_worker_production_commands_require_lock_wrapper(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_deploy")
    monkeypatch.setenv(
        "HERMES_KANBAN_WORKSPACE", str(Path(__file__).resolve().parents[2])
    )
    assert pl.production_delivery_guard("vercel --prod")
    assert pl.production_delivery_guard("supabase db push")
    assert pl.production_delivery_guard("supabase db push && echo --local")
    assert pl.production_delivery_guard("bash -c 'supabase db push && echo --local'")
    assert pl.production_delivery_guard("gh pr merge 123 --squash")
    assert pl.production_delivery_guard("gh --repo owner/repo pr merge 123")
    assert pl.production_delivery_guard("git push origin main")
    assert pl.production_delivery_guard("git -C /tmp/repo push origin main")
    assert pl.production_delivery_guard("vercel") is None
    assert pl.production_delivery_guard("supabase db push --local") is None
    assert pl.production_delivery_guard(
        "hermes kanban lock run --project owner/repo --operation deploy -- vercel --prod"
    )
    assert pl.production_delivery_guard(
        f"{Path(sys.executable).with_name('hermes')} kanban lock run "
        "--project NousResearch/hermes-agent --operation deploy -- vercel --prod"
    ) is None
    assert pl.production_delivery_guard(
        "/tmp/hermes kanban lock run --project NousResearch/hermes-agent "
        "--operation deploy -- vercel --prod"
    )
    assert pl.production_delivery_guard("python3 release.py")
    assert pl.production_delivery_guard("python3 -u release.py")
    assert pl.production_delivery_guard("python3 -O -W ignore release.py")
    assert pl.production_delivery_guard(
        "hermes() { true; }\n"
        "hermes kanban lock run --project NousResearch/hermes-agent "
        "--operation deploy -- python3 -u release.py"
    )
    assert pl.production_delivery_guard(
        f"{Path(sys.executable).with_name('hermes')} kanban lock run "
        "--project NousResearch/hermes-agent --operation deploy -- "
        "echo $(python3 -u release.py)"
    )
    assert pl.production_delivery_guard(
        f"{Path(sys.executable).with_name('hermes')} kanban lock run "
        "--project NousResearch/hermes-agent --operation deploy -- "
        "cat <(python3 -u release.py)"
    )
    assert pl.production_delivery_guard(
        "hermes kanban lock run --project owner/repo --operation deploy -- true && vercel --prod"
    )
    assert pl.production_delivery_guard("bash -c 'vercel --prod'")
    from tools.terminal_tool import terminal_tool

    blocked = json.loads(terminal_tool("vercel --prod", force=True))
    assert blocked["status"] == "blocked"
    assert "kanban lock run" in blocked["error"]


def test_same_project_lock_is_shared_across_boards(claimed_owner, tmp_path):
    home, owner_a = claimed_owner
    board_b_path = tmp_path / "board-b.db"
    kb.init_db(board_b_path)
    with kb.connect_closing(board_b_path) as board_b:
        task_id = kb.create_task(board_b, title="deploy b", assignee="developer")
        task = kb.claim_task(board_b, task_id, claimer="host:200")
        assert task is not None
        assert task.current_run_id is not None
        owner_b = pl.LeaseOwner(
            task_id=task_id,
            run_id=task.current_run_id,
            claim_lock="host:200",
            instance_id="instance-b",
            host=socket.gethostname(),
            pid=os.getpid(),
        )
        with kb.connect_closing(home / "kanban.db") as board_a, pl.connect_project_locks() as locks:
            first = pl.acquire_project_lock(
                locks, owner_conn=board_a, project="Pryapus/Drip-Research-Hub",
                operation="deploy", owner=owner_a, lease_seconds=10, now=100.0,
            )
            assert first is not None
            assert pl.acquire_project_lock(
                locks, owner_conn=board_b, project="Pryapus/Drip-Research-Hub",
                operation="migration", owner=owner_b, lease_seconds=10, now=100.0,
            ) is None
            assert pl.release_project_lock(locks, first, owner_conn=board_a) is True


def test_wait_deadline_is_bounded_during_sqlite_contention(claimed_owner):
    _, owner = claimed_owner
    with (
        kb.connect_closing() as owner_conn,
        pl.connect_project_locks() as blocker,
        pl.connect_project_locks() as contender,
    ):
        blocker.execute("BEGIN IMMEDIATE")
        waits = []
        started = time.monotonic()
        try:
            lease = pl.acquire_with_wait(
                contender, owner_conn=owner_conn,
                project="Pryapus/Drip-Research-Hub", operation="deploy",
                owner=owner, lease_seconds=10, wait_seconds=0.05,
                poll_seconds=0.01, on_wait=lambda holder: waits.append(holder),
            )
        finally:
            blocker.execute("ROLLBACK")
        elapsed = time.monotonic() - started
    assert lease is None
    assert waits
    assert elapsed < 0.2


def test_wait_deadline_includes_lock_store_initialization(claimed_owner):
    home, _ = claimed_owner
    lock_path = home / "kanban" / "project-delivery-locks.db"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    blocker = __import__("sqlite3").connect(lock_path, isolation_level=None)
    blocker.execute(
        pl._LOCK_SCHEMA
        .replace(", owner_started_at REAL", "")
        .replace(", command_started_at REAL", "")
    )
    blocker.execute("BEGIN IMMEDIATE")
    try:
        waits = []
        started = time.monotonic()
        with pytest.raises(pl.ProjectLockTimeout):
            with pl.connect_project_locks(
                deadline=started + 0.05,
                on_wait=lambda holder: waits.append(holder),
            ):
                pass
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    assert lock_path.exists()
    assert waits
    assert time.monotonic() - started < 0.5


def test_atomic_handoff_and_stale_token_rejection(claimed_owner):
    _, owner_a = claimed_owner
    owner_b = pl.LeaseOwner(
        task_id=owner_a.task_id,
        run_id=owner_a.run_id,
        claim_lock=owner_a.claim_lock,
        instance_id="instance-b",
        host=socket.gethostname(),
        pid=os.getpid(),
    )
    barrier = __import__("threading").Barrier(2)

    def contender(owner):
        with kb.connect() as conn:
            barrier.wait()
            return pl.acquire_project_lock(
                conn,
                project="Pryapus/Drip-Research-Hub",
                operation="deploy",
                owner=owner,
                lease_seconds=10,
                now=100.0,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        leases = list(pool.map(contender, (owner_a, owner_b)))

    winner = next(lease for lease in leases if lease is not None)
    loser_owner = owner_b if winner.owner.instance_id == owner_a.instance_id else owner_a
    assert sum(lease is not None for lease in leases) == 1

    with kb.connect() as conn:
        assert pl.release_project_lock(conn, winner) is True
        successor = pl.acquire_project_lock(
            conn,
            project="Pryapus/Drip-Research-Hub",
            operation="migration",
            owner=loser_owner,
            lease_seconds=10,
            now=101.0,
        )
        assert successor is not None
        assert successor.fence == winner.fence + 1
        assert pl.release_project_lock(conn, winner) is False
        assert pl.renew_project_lock(conn, winner, lease_seconds=10, now=102.0) is False


def test_expired_holder_takeover_and_owner_claim_fencing(claimed_owner):
    _, owner = claimed_owner
    dead_owner = replace(owner, pid=2 ** 30)
    with kb.connect() as conn:
        first = pl.acquire_project_lock(
            conn,
            project="Pryapus/Drip-Research-Hub",
            operation="deploy",
            owner=dead_owner,
            lease_seconds=4,
            now=100.0,
        )
        assert first is not None
        assert pl.acquire_project_lock(
            conn,
            project="Pryapus/Drip-Research-Hub",
            operation="migration",
            owner=dead_owner,
            lease_seconds=4,
            now=103.9,
        ) is None

        successor_owner = pl.LeaseOwner(
            task_id=owner.task_id,
            run_id=owner.run_id,
            claim_lock=owner.claim_lock,
            instance_id="instance-successor",
            host=socket.gethostname(),
            pid=os.getpid(),
        )
        successor = pl.acquire_project_lock(
            conn,
            project="Pryapus/Drip-Research-Hub",
            operation="migration",
            owner=successor_owner,
            lease_seconds=4,
            now=104.0,
        )
        assert successor is not None
        assert successor.fence == first.fence + 1
        assert pl.release_project_lock(conn, first) is False

        conn.execute(
            "UPDATE tasks SET claim_lock='host:successor' WHERE id=?",
            (owner.task_id,),
        )
        conn.commit()
        assert pl.renew_project_lock(conn, successor, lease_seconds=4, now=105.0) is False


def test_expired_lease_does_not_overlap_a_live_local_holder(claimed_owner):
    _, owner = claimed_owner
    successor_owner = replace(owner, instance_id="instance-successor")
    with kb.connect() as conn:
        first = pl.acquire_project_lock(
            conn, project="Pryapus/Drip-Research-Hub", operation="deploy",
            owner=owner, lease_seconds=2, now=100.0,
        )
        assert first is not None
        assert pl.acquire_project_lock(
            conn, project="Pryapus/Drip-Research-Hub", operation="migration",
            owner=successor_owner, lease_seconds=2, now=102.0,
        ) is None
        conn.execute(
            "UPDATE project_delivery_locks SET owner_started_at=? "
            "WHERE resource_key=?",
            (owner.started_at - 1000, first.key),
        )
        successor = pl.acquire_project_lock(
            conn, project="Pryapus/Drip-Research-Hub", operation="migration",
            owner=successor_owner, lease_seconds=2, now=102.0,
        )
        assert successor is not None
        assert successor.fence == first.fence + 1


def test_cli_parser_exposes_bounded_production_lock_run():
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command")
    kanban.build_parser(sub)
    args = root.parse_args(
        [
            "kanban", "lock", "run",
            "--project", "Pryapus/Drip-Research-Hub",
            "--operation", "migration",
            "--wait", "12",
            "--lease", "30",
            "--", "python3", "migrate.py",
        ]
    )
    assert args.kanban_action == "lock"
    assert args.lock_action == "run"
    assert args.wait == 12
    assert args.lock_command == ["--", "python3", "migrate.py"]


def _worker_env(home: Path, owner: pl.LeaseOwner) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "HERMES_HOME": str(home),
            "HERMES_KANBAN_HOME": str(home),
            "HERMES_KANBAN_DB": str(home / "kanban.db"),
            "HERMES_KANBAN_TASK": owner.task_id,
            "HERMES_KANBAN_RUN_ID": str(owner.run_id),
            "HERMES_KANBAN_CLAIM_LOCK": owner.claim_lock,
            "HERMES_KANBAN_WORKSPACE": str(Path(__file__).resolve().parents[2]),
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
        }
    )
    return env


def _cli_lock_command(command: list[str], *, wait: int = 2) -> list[str]:
    return [
        str(Path(sys.executable).with_name("hermes")), "kanban", "lock", "run",
        "--project", "NousResearch/hermes-agent", "--operation", "deploy",
        "--wait", str(wait), "--lease", "2", "--", *command,
    ]


def test_real_lock_cli_propagates_failure_status(claimed_owner):
    home, owner = claimed_owner
    result = subprocess.run(
        _cli_lock_command([sys.executable, "-c", "raise SystemExit(73)"]),
        env=_worker_env(home, owner),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 73, result.stderr


@pytest.mark.skipif(os.name != "posix", reason="Parent-death supervisor is POSIX-only")
def test_locked_command_uses_parent_death_supervisor():
    command = pl._supervised_command(["deploy", "--production"])
    assert command[0] == sys.executable
    assert command[1].endswith("tools/mcp_stdio_watchdog.py")
    assert command[command.index("--") + 1:] == ["deploy", "--production"]


@pytest.mark.skipif(os.name != "posix", reason="Start-gated supervisor is POSIX-only")
def test_supervisor_does_not_spawn_before_durable_attachment(tmp_path):
    marker = tmp_path / "spawned"
    gate_read, gate_write = os.pipe()
    proc = subprocess.Popen(
        pl._supervised_command(
            [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"],
            start_gate_fd=gate_read,
        ),
        pass_fds=(gate_read,),
    )
    os.close(gate_read)
    os.close(gate_write)
    assert proc.wait(timeout=5) == 75
    assert not marker.exists()


@pytest.mark.skipif(os.name != "posix", reason="Process-group cleanup is POSIX-only")
@pytest.mark.live_system_guard_bypass
def test_supervisor_reaps_descendant_after_direct_child_exits(tmp_path):
    descendant_pid_path = tmp_path / "descendant-pid"
    descendant_code = (
        "import os,pathlib,signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(descendant_pid_path)!r}).write_text(str(os.getpid())); "
        "time.sleep(30)"
    )
    leader_code = (
        "import pathlib,subprocess,sys,time; "
        "subprocess.Popen([sys.executable, '-c', "
        f"{descendant_code!r}], stdin=subprocess.DEVNULL, "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
        f"marker=pathlib.Path({str(descendant_pid_path)!r}); "
        "exec('while not marker.exists(): time.sleep(0.01)')"
    )
    supervisor = subprocess.Popen(
        pl._supervised_command([sys.executable, "-c", leader_code]),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    descendant_pid = None
    try:
        assert supervisor.wait(timeout=10) == 0
        deadline = time.time() + 2
        while time.time() < deadline and not descendant_pid_path.exists():
            time.sleep(0.05)
        assert descendant_pid_path.exists()
        descendant_pid = int(descendant_pid_path.read_text())
        assert not kb._pid_alive(descendant_pid)
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait()
        if descendant_pid and kb._pid_alive(descendant_pid):
            os.kill(descendant_pid, signal.SIGKILL)


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL crash probe is POSIX-only")
@pytest.mark.live_system_guard_bypass
def test_real_cli_killed_holder_hands_off_after_lease(claimed_owner, tmp_path):
    home, owner = claimed_owner
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="successor", assignee="developer")
        task = kb.claim_task(conn, task_id, claimer="host:200")
        assert task is not None
        assert task.current_run_id is not None
    successor_owner = pl.LeaseOwner(
        task_id=task_id,
        run_id=task.current_run_id,
        claim_lock="host:200",
        instance_id="successor",
    )
    entered = tmp_path / "entered"
    handed_off = tmp_path / "handed-off"
    child_pid_path = tmp_path / "child-pid"
    descendant_pid_path = tmp_path / "descendant-pid"
    completed = tmp_path / "completed"
    descendant_code = (
        "import os,pathlib,signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(descendant_pid_path)!r}).write_text(str(os.getpid())); "
        "time.sleep(30); "
        f"pathlib.Path({str(completed)!r}).write_text(str(time.time()))"
    )
    holder = subprocess.Popen(
        _cli_lock_command(
            [
                sys.executable,
                "-c",
                "import os,pathlib,subprocess,sys,time; "
                f"subprocess.Popen([sys.executable, '-c', {descendant_code!r}]); "
                f"pathlib.Path({str(child_pid_path)!r}).write_text(str(os.getpid())); "
                f"pathlib.Path({str(entered)!r}).write_text(str(time.time())); "
                "time.sleep(30)",
            ],
            wait=5,
        ),
        env=_worker_env(home, owner),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    command_pid = None
    child_pid = None
    descendant_pid = None
    try:
        deadline = time.time() + 5
        while time.time() < deadline:
            with pl.connect_project_locks() as conn:
                status = pl.project_lock_status(conn, "NousResearch/hermes-agent")
            if (
                entered.exists()
                and child_pid_path.exists()
                and descendant_pid_path.exists()
                and status
                and status["command_pid"]
            ):
                command_pid = status["command_pid"]
                child_pid = int(child_pid_path.read_text())
                descendant_pid = int(descendant_pid_path.read_text())
                break
            time.sleep(0.05)
        if command_pid is None:
            if holder.poll() is None:
                holder.terminate()
            stdout, stderr = holder.communicate(timeout=5)
            pytest.fail(
                f"holder never entered the critical section: "
                f"rc={holder.returncode} stdout={stdout!r} stderr={stderr!r}"
            )

        os.kill(holder.pid, signal.SIGKILL)
        holder.wait(timeout=5)
        successor = subprocess.run(
            _cli_lock_command(
                [
                    sys.executable,
                    "-c",
                    f"import pathlib; pathlib.Path({str(handed_off)!r}).write_text('successor')",
                ],
                wait=10,
            ),
            env=_worker_env(home, successor_owner),
            capture_output=True,
            text=True,
            timeout=16,
            check=False,
        )
        assert successor.returncode == 0, successor.stderr
        assert handed_off.read_text() == "successor"
        events = [json.loads(line) for line in successor.stderr.splitlines()]
        states = [event["state"] for event in events]
        assert "waiting" in states
        assert states[-2:] == ["acquired", "released"]
        assert events[-2]["timestamp"] >= float(entered.read_text())
        assert child_pid is not None and not kb._pid_alive(child_pid)
        assert descendant_pid is not None and not kb._pid_alive(descendant_pid)
        assert not completed.exists()
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait()
        if command_pid and kb._pid_alive(command_pid):
            os.kill(command_pid, signal.SIGKILL)
        if child_pid and kb._pid_alive(child_pid):
            os.kill(child_pid, signal.SIGKILL)
        if descendant_pid and kb._pid_alive(descendant_pid):
            os.kill(descendant_pid, signal.SIGKILL)


def test_non_finite_wait_is_rejected(claimed_owner):
    _, owner = claimed_owner
    with kb.connect() as conn, pytest.raises(ValueError, match="finite"):
        pl.acquire_with_wait(
            conn, project="Pryapus/Drip-Research-Hub", operation="deploy",
            owner=owner, lease_seconds=30, wait_seconds=float("nan"),
        )
