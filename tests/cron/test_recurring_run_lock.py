"""Cross-process full-lifetime ownership for cron job executions."""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timedelta, timezone

import pytest

from cron import jobs


_REPO_ROOT = str(Path(jobs.__file__).resolve().parent.parent)


def _child_env(hermes_home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (_REPO_ROOT, env.get("PYTHONPATH", "")) if part
    )
    return env


def _wait_for(predicate, *, timeout: float = 10.0, message: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(message)


def _side_effect_count(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except FileNotFoundError:
        return 0


def test_two_scheduler_processes_cannot_overlap_one_recurring_run(tmp_path):
    """A later due slot must not run or advance while its prior run is alive."""
    hermes_home = tmp_path / "hermes-home"
    cron_dir = hermes_home / "cron"
    scripts_dir = hermes_home / "scripts"
    cron_dir.mkdir(parents=True)
    scripts_dir.mkdir(parents=True)

    side_effects = tmp_path / "side-effects.txt"
    script_ready = tmp_path / "script-ready"
    release_script = tmp_path / "release-script"
    script = scripts_dir / "blocking.py"
    script.write_text(
        textwrap.dedent(
            f"""
            from pathlib import Path
            import os
            import time

            effects = Path({str(side_effects)!r})
            with effects.open("a", encoding="utf-8") as stream:
                stream.write(str(os.getpid()) + "\\n")
                stream.flush()
            Path({str(script_ready)!r}).write_text("ready", encoding="utf-8")
            deadline = time.monotonic() + 20
            while not Path({str(release_script)!r}).exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("test did not release blocking cron script")
                time.sleep(0.02)
            print("completed")
            """
        ),
        encoding="utf-8",
    )

    now = datetime.now(timezone.utc)
    job = {
        "id": "recurring-lock-regression",
        "name": "recurring lock regression",
        "prompt": "",
        "script": script.name,
        "no_agent": True,
        "schedule": {"kind": "interval", "minutes": 1},
        "schedule_display": "every 1m",
        "repeat": {"times": None, "completed": 0},
        "enabled": True,
        "state": "scheduled",
        "next_run_at": (now - timedelta(seconds=1)).isoformat(),
        "last_run_at": None,
        "last_status": None,
        "last_error": None,
        "deliver": "local",
    }
    with jobs.use_cron_store(hermes_home):
        jobs.save_jobs([job])

    owner_started = tmp_path / "owner-started"
    owner_done = tmp_path / "owner-done"
    owner_program = tmp_path / "owner.py"
    owner_program.write_text(
        textwrap.dedent(
            f"""
            from pathlib import Path
            import time
            from cron.scheduler import get_running_job_ids, tick

            result = tick(verbose=False, sync=False)
            Path({str(owner_started)!r}).write_text(str(result), encoding="utf-8")
            deadline = time.monotonic() + 25
            while get_running_job_ids():
                if time.monotonic() >= deadline:
                    raise TimeoutError("cron worker did not finish")
                time.sleep(0.02)
            Path({str(owner_done)!r}).write_text("done", encoding="utf-8")
            """
        ),
        encoding="utf-8",
    )
    contender_program = tmp_path / "contender.py"
    contender_program.write_text(
        "from cron.scheduler import tick\nraise SystemExit(0 if tick(verbose=False, sync=True) == 0 else 3)\n",
        encoding="utf-8",
    )

    owner = subprocess.Popen(
        [sys.executable, str(owner_program)], env=_child_env(hermes_home)
    )
    contender = None
    try:
        _wait_for(
            script_ready.exists,
            message="first scheduler never began the recurring side effect",
        )
        assert _side_effect_count(side_effects) == 1

        # Simulate the next recurrence becoming due before the first invocation
        # has finished. A second scheduler process can acquire .tick.lock now
        # because the async first tick released that dispatch-only lock.
        forced_due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        with jobs.use_cron_store(hermes_home):
            stored = jobs.load_jobs()
            assert stored[0]["next_run_at"] > now.isoformat()
            stored[0]["next_run_at"] = forced_due
            jobs.save_jobs(stored)

        contender = subprocess.Popen(
            [sys.executable, str(contender_program)], env=_child_env(hermes_home)
        )
        _wait_for(
            lambda: contender.poll() is not None or _side_effect_count(side_effects) > 1,
            timeout=5,
            message="contending scheduler neither exited nor attempted a duplicate run",
        )

        assert contender.poll() == 0, "contender dispatched the already-running job"
        assert _side_effect_count(side_effects) == 1
        with jobs.use_cron_store(hermes_home):
            # Losing ownership must happen before advance_next_run().
            contended_job = jobs.get_job(job["id"])
            assert contended_job is not None
            assert contended_job["next_run_at"] == forced_due

        release_script.write_text("release", encoding="utf-8")
        owner.wait(timeout=15)
        assert owner.returncode == 0
        assert owner_done.exists()
        assert _side_effect_count(side_effects) == 1

        with jobs.use_cron_store(hermes_home):
            completed = jobs.get_job(job["id"])
        assert completed is not None
        assert completed["repeat"]["completed"] == 1
        assert completed["last_status"] == "ok"
        assert datetime.fromisoformat(completed["next_run_at"]) > datetime.now(timezone.utc)
    finally:
        release_script.touch()
        for child in (contender, owner):
            if child is not None and child.poll() is None:
                child.kill()
            if child is not None:
                child.wait(timeout=10)


@pytest.mark.skipif(
    jobs.fcntl is None and jobs.msvcrt is None,
    reason="platform has no supported advisory file-lock backend",
)
def test_losing_immediate_runner_does_not_advance_or_execute(tmp_path):
    """cronjob(action='run') must own a recurrence before its store claim."""
    hermes_home = tmp_path / "hermes-home"
    cron_dir = hermes_home / "cron"
    scripts_dir = hermes_home / "scripts"
    cron_dir.mkdir(parents=True)
    scripts_dir.mkdir(parents=True)

    side_effects = tmp_path / "immediate-side-effects.txt"
    owner_ready = tmp_path / "immediate-owner-ready"
    release_owner = tmp_path / "immediate-owner-release"
    owner_result = tmp_path / "immediate-owner-result.json"
    script = scripts_dir / "blocking-immediate.py"
    script.write_text(
        textwrap.dedent(
            f"""
            from pathlib import Path
            import os
            import time

            effects = Path({str(side_effects)!r})
            with effects.open("a", encoding="utf-8") as stream:
                stream.write(str(os.getpid()) + "\\n")
                stream.flush()
            Path({str(owner_ready)!r}).write_text("ready", encoding="utf-8")
            deadline = time.monotonic() + 20
            while not Path({str(release_owner)!r}).exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("test did not release immediate owner")
                time.sleep(0.02)
            print("completed")
            """
        ),
        encoding="utf-8",
    )

    job_id = "immediate-run-lock-regression"
    job = {
        "id": job_id,
        "name": "immediate run lock regression",
        "prompt": "",
        "script": script.name,
        "no_agent": True,
        "schedule": {"kind": "interval", "minutes": 1},
        "schedule_display": "every 1m",
        "repeat": {"times": None, "completed": 0},
        "enabled": True,
        "state": "scheduled",
        "next_run_at": datetime.now(timezone.utc).isoformat(),
        "last_run_at": None,
        "last_status": None,
        "last_error": None,
        "deliver": "local",
    }
    with jobs.use_cron_store(hermes_home):
        jobs.save_jobs([job])

    owner_program = tmp_path / "immediate-owner.py"
    owner_program.write_text(
        textwrap.dedent(
            f"""
            import json
            from pathlib import Path
            from tools.cronjob_tools import cronjob

            result = json.loads(cronjob(action="run", job_id={job_id!r}))
            Path({str(owner_result)!r}).write_text(json.dumps(result), encoding="utf-8")
            job_result = result["job"]
            raise SystemExit(
                0 if job_result["executed"] and job_result["execution_success"] else 3
            )
            """
        ),
        encoding="utf-8",
    )
    contender_program = tmp_path / "immediate-contender.py"
    contender_program.write_text(
        textwrap.dedent(
            f"""
            from tools.cronjob_tools import cronjob

            print(cronjob(action="run", job_id={job_id!r}))
            """
        ),
        encoding="utf-8",
    )

    owner = subprocess.Popen(
        [sys.executable, str(owner_program)], env=_child_env(hermes_home)
    )
    try:
        _wait_for(owner_ready.exists, message="immediate owner never entered its script")
        assert _side_effect_count(side_effects) == 1

        # Let a second immediate invocation reclaim the bounded fire claim while
        # the first invocation still owns the full-run lock. The loser must be
        # rejected before claim_job_for_fire can advance this forced-due slot.
        forced_due = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        with jobs.use_cron_store(hermes_home):
            stored = jobs.load_jobs()
            stored[0]["fire_claim"]["at"] = (
                datetime.now(timezone.utc) - timedelta(seconds=301)
            ).isoformat()
            stored[0]["next_run_at"] = forced_due
            jobs.save_jobs(stored)
            before = jobs.get_job(job_id)
        assert before is not None

        contender = subprocess.run(
            [sys.executable, str(contender_program)],
            env=_child_env(hermes_home),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert contender.returncode == 0, contender.stderr
        contender_result = json.loads(contender.stdout.strip())
        assert contender_result["success"] is True
        assert contender_result["job"]["executed"] is False
        assert contender_result["job"]["execution_success"] is False
        assert _side_effect_count(side_effects) == 1

        with jobs.use_cron_store(hermes_home):
            after = jobs.get_job(job_id)
        assert after is not None
        assert after["next_run_at"] == forced_due
        assert after["fire_claim"] == before["fire_claim"]

        release_owner.touch()
        owner.wait(timeout=15)
        assert owner.returncode == 0
        completed_result = json.loads(owner_result.read_text(encoding="utf-8"))
        assert completed_result["job"]["execution_success"] is True
        assert _side_effect_count(side_effects) == 1
    finally:
        release_owner.touch()
        if owner.poll() is None:
            owner.kill()
        owner.wait(timeout=10)


@pytest.mark.skipif(
    jobs.fcntl is None and jobs.msvcrt is None,
    reason="platform has no supported advisory file-lock backend",
)
def test_job_run_lock_is_released_when_owner_process_dies(tmp_path, monkeypatch):
    """The OS, not a stale timestamp, must recover ownership after a crash."""
    from cron import scheduler

    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setattr(scheduler, "_hermes_home", hermes_home)
    ready = tmp_path / "lock-ready"
    holder = tmp_path / "lock-holder.py"
    holder.write_text(
        textwrap.dedent(
            f"""
            from pathlib import Path
            import time
            from cron.scheduler import _try_acquire_job_run_lock

            lock = _try_acquire_job_run_lock("crash-release-job")
            if lock is None:
                raise SystemExit(2)
            Path({str(ready)!r}).write_text("ready", encoding="utf-8")
            time.sleep(60)
            """
        ),
        encoding="utf-8",
    )

    child = subprocess.Popen(
        [sys.executable, str(holder)], env=_child_env(hermes_home)
    )
    lock = None
    try:
        _wait_for(ready.exists, message="child never acquired the job run lock")
        with jobs.use_cron_store(hermes_home):
            assert scheduler._try_acquire_job_run_lock("crash-release-job") is None

        child.kill()
        child.wait(timeout=10)

        deadline = time.monotonic() + 5
        while lock is None and time.monotonic() < deadline:
            with jobs.use_cron_store(hermes_home):
                lock = scheduler._try_acquire_job_run_lock("crash-release-job")
            if lock is None:
                time.sleep(0.02)
        assert lock is not None, "OS did not release the job lock after owner death"
    finally:
        if lock is not None:
            lock.release()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)


def test_direct_recurring_run_owns_lock_until_body_returns(monkeypatch):
    """Manual/direct callers get the same full-lifetime ownership as tick()."""
    from cron import scheduler

    events: list[str] = []

    class RecordingLock:
        def release(self):
            events.append("release")

    monkeypatch.setattr(
        scheduler,
        "_try_acquire_job_run_lock",
        lambda job_id: events.append(f"acquire:{job_id}") or RecordingLock(),
    )
    monkeypatch.setattr(
        scheduler,
        "_run_one_job_owned",
        lambda job, **kwargs: events.append("body") or True,
    )

    assert scheduler.run_one_job(
        {"id": "manual-recurring", "schedule": {"kind": "interval"}}
    ) is True
    assert events == ["acquire:manual-recurring", "body", "release"]


def test_run_lock_releases_on_base_exception(monkeypatch):
    """KeyboardInterrupt/SystemExit paths cannot leak ownership until process exit."""
    from cron import scheduler

    released: list[bool] = []

    class RecordingLock(scheduler._JobRunLock):
        def __init__(self):
            pass

        def release(self):
            released.append(True)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(scheduler, "_run_one_job_owned", interrupt)
    with pytest.raises(KeyboardInterrupt):
        scheduler.run_one_job(
            {"id": "interrupted", "schedule": {"kind": "cron"}},
            _run_lock=RecordingLock(),
        )
    assert released == [True]


def test_run_lock_is_held_through_delivery_mark_and_teardown(monkeypatch):
    """Ownership covers every observable and cleanup phase of the run body."""
    from agent import secret_scope
    from cron import scheduler

    events: list[str] = []

    class RecordingLock(scheduler._JobRunLock):
        released = False

        def __init__(self):
            pass

        def release(self):
            self.released = True
            events.append("release")

    run_lock = RecordingLock()

    def still_owned(event):
        assert run_lock.released is False
        events.append(event)

    def fake_run_job(_job, *, defer_agent_teardown):
        still_owned("execute")
        defer_agent_teardown.append(object())
        return True, "output", "final response", None

    monkeypatch.setattr(scheduler, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(secret_scope, "build_profile_secret_scope", lambda _home: None)
    monkeypatch.setattr(secret_scope, "set_secret_scope", lambda _scope: object())
    monkeypatch.setattr(secret_scope, "reset_secret_scope", lambda _token: None)
    monkeypatch.setattr(scheduler, "run_job", fake_run_job)
    monkeypatch.setattr(
        scheduler, "save_job_output", lambda *_args: still_owned("save") or "output.md"
    )
    monkeypatch.setattr(
        scheduler, "_deliver_result", lambda *_args, **_kwargs: still_owned("deliver")
    )
    monkeypatch.setattr(
        scheduler, "_teardown_cron_agent", lambda *_args: still_owned("teardown")
    )
    monkeypatch.setattr(
        scheduler, "mark_job_run", lambda *_args, **_kwargs: still_owned("mark")
    )

    assert scheduler.run_one_job(
        {"id": "lifecycle", "schedule": {"kind": "interval"}},
        _run_lock=run_lock,
    ) is True
    assert events == ["execute", "save", "deliver", "teardown", "mark", "release"]


def test_one_shot_keeps_existing_run_claim_path(monkeypatch):
    """One-shots must not be routed away from their durable run_claim semantics."""
    from cron import scheduler

    monkeypatch.setattr(
        scheduler,
        "_try_acquire_job_run_lock",
        lambda job_id: pytest.fail("one-shot unexpectedly acquired recurring run lock"),
    )
    monkeypatch.setattr(scheduler, "_run_one_job_owned", lambda job, **kwargs: True)

    assert scheduler.run_one_job(
        {"id": "one-shot", "schedule": {"kind": "once"}}
    ) is True


def test_run_lock_isolated_by_active_cron_store(tmp_path):
    """Equal job IDs in separate profile stores must not block one another."""
    from cron import scheduler

    profile_a = tmp_path / "profiles" / "a"
    profile_b = tmp_path / "profiles" / "b"
    with jobs.use_cron_store(profile_a):
        path_a = scheduler._job_run_lock_path("shared-id")
        lock_a = scheduler._try_acquire_job_run_lock("shared-id")
    assert lock_a is not None
    lock_b = None
    try:
        with jobs.use_cron_store(profile_b):
            path_b = scheduler._job_run_lock_path("shared-id")
            lock_b = scheduler._try_acquire_job_run_lock("shared-id")
        assert lock_b is not None
        assert path_a.parent.parent == profile_a.resolve() / "cron"
        assert path_b.parent.parent == profile_b.resolve() / "cron"
        assert path_a != path_b
    finally:
        lock_a.release()
        if lock_b is not None:
            lock_b.release()


def test_run_lock_follows_store_not_process_home(tmp_path, monkeypatch):
    """Different runtime homes still contend when they address one cron store."""
    from cron import scheduler

    shared_store_home = tmp_path / "shared-store"
    with jobs.use_cron_store(shared_store_home):
        monkeypatch.setattr(scheduler, "_hermes_home", tmp_path / "process-a")
        path_a = scheduler._job_run_lock_path("same-job")
        lock_a = scheduler._try_acquire_job_run_lock("same-job")
        assert lock_a is not None
        try:
            monkeypatch.setattr(scheduler, "_hermes_home", tmp_path / "process-b")
            path_b = scheduler._job_run_lock_path("same-job")
            assert path_b == path_a
            assert scheduler._try_acquire_job_run_lock("same-job") is None
        finally:
            lock_a.release()


def test_different_process_homes_contend_on_same_active_store(tmp_path):
    """Real processes derive ownership from store context, not HERMES_HOME."""
    shared_store = tmp_path / "shared-store"
    ready = tmp_path / "holder-ready"
    holder = tmp_path / "store-lock-holder.py"
    holder.write_text(
        textwrap.dedent(
            f"""
            from pathlib import Path
            import time
            from cron import jobs
            from cron.scheduler import _try_acquire_job_run_lock

            with jobs.use_cron_store({str(shared_store)!r}):
                lock = _try_acquire_job_run_lock("shared-process-job")
                if lock is None:
                    raise SystemExit(2)
                Path({str(ready)!r}).write_text("ready", encoding="utf-8")
                time.sleep(30)
            """
        ),
        encoding="utf-8",
    )
    contender = tmp_path / "store-lock-contender.py"
    contender.write_text(
        textwrap.dedent(
            f"""
            from cron import jobs
            from cron.scheduler import _try_acquire_job_run_lock

            with jobs.use_cron_store({str(shared_store)!r}):
                lock = _try_acquire_job_run_lock("shared-process-job")
            raise SystemExit(0 if lock is None else 3)
            """
        ),
        encoding="utf-8",
    )

    owner = subprocess.Popen(
        [sys.executable, str(holder)], env=_child_env(tmp_path / "process-home-a")
    )
    try:
        _wait_for(ready.exists, message="first process never acquired shared-store lock")
        result = subprocess.run(
            [sys.executable, str(contender)],
            env=_child_env(tmp_path / "process-home-b"),
            timeout=10,
            check=False,
        )
        assert result.returncode == 0
    finally:
        if owner.poll() is None:
            owner.kill()
        owner.wait(timeout=10)


def test_lock_backend_error_is_not_reported_as_contention(tmp_path, monkeypatch):
    """Open/permission failures carry a truthful fail-closed reason."""
    from cron import scheduler

    lock_path = tmp_path / "profile" / "cron" / ".run-locks" / "fake.lock"
    monkeypatch.setattr(scheduler, "_job_run_lock_path", lambda _job_id: lock_path)

    def deny_open(*_args, **_kwargs):
        raise PermissionError(errno.EACCES, "permission denied", str(lock_path))

    monkeypatch.setattr(scheduler, "open", deny_open, raising=False)
    with pytest.raises(scheduler._JobRunLockError) as exc_info:
        scheduler._try_acquire_job_run_lock("permission-job")
    message = str(exc_info.value)
    assert "permission denied" in message.lower()
    assert str(lock_path) in message


def test_direct_run_logs_backend_failure_and_does_not_dispatch(monkeypatch, caplog):
    """Fail closed on lock errors, but never call the failure 'already running'."""
    from cron import scheduler

    error = scheduler._JobRunLockError(
        "backend-job", Path("/blocked/run.lock"), PermissionError("denied")
    )

    def raise_lock_error(_job_id):
        raise error

    monkeypatch.setattr(scheduler, "_try_acquire_job_run_lock", raise_lock_error)
    monkeypatch.setattr(
        scheduler,
        "_run_one_job_owned",
        lambda *_args, **_kwargs: pytest.fail("lock failure dispatched the job"),
    )

    with caplog.at_level("ERROR"):
        assert scheduler.run_one_job(
            {"id": "backend-job", "schedule": {"kind": "interval"}}
        ) is False
    assert "/blocked/run.lock" in caplog.text
    assert "refusing" in caplog.text.lower()
    assert "already running" not in caplog.text.lower()


def test_failed_schedule_advance_revalidates_and_skips_stale_snapshot(
    tmp_path, monkeypatch, caplog
):
    """Ownership without a successful state transition is not a dispatch claim."""
    from cron import scheduler

    released: list[bool] = []

    class RecordingLock:
        def release(self):
            released.append(True)

    job = {
        "id": "removed-during-advance",
        "name": "removed during advance",
        "schedule": {"kind": "interval", "minutes": 1},
    }
    monkeypatch.setattr(scheduler, "_hermes_home", tmp_path)
    monkeypatch.setattr(scheduler, "get_due_jobs", lambda **_kwargs: [job])
    monkeypatch.setattr(
        scheduler, "_try_acquire_job_run_lock", lambda _job_id: RecordingLock()
    )
    monkeypatch.setattr(
        scheduler, "advance_next_run", lambda _job_id, **_kwargs: False
    )
    monkeypatch.setattr(scheduler, "get_job", lambda _job_id: None, raising=False)
    monkeypatch.setattr(
        scheduler,
        "run_one_job",
        lambda *_args, **_kwargs: pytest.fail("stale snapshot was dispatched"),
    )

    with caplog.at_level("WARNING"):
        assert scheduler.tick(verbose=False, sync=True) == 0
    assert released == [True]
    assert "no longer exists" in caplog.text.lower()


def test_successful_advance_dispatches_revalidated_active_snapshot(
    tmp_path, monkeypatch
):
    """Edits made after the due scan replace, rather than trail, its snapshot."""
    from cron import scheduler

    released: list[bool] = []
    dispatched: list[dict] = []

    class RecordingLock:
        def release(self):
            released.append(True)

    stale = {
        "id": "edited-during-advance",
        "name": "old name",
        "prompt": "old prompt",
        "schedule": {"kind": "interval", "minutes": 1},
        "next_run_at": "2026-07-11T00:00:00+00:00",
    }
    current = {**stale, "name": "new name", "prompt": "new prompt"}
    monkeypatch.setattr(scheduler, "_hermes_home", tmp_path)
    monkeypatch.setattr(scheduler, "get_due_jobs", lambda **_kwargs: [stale])
    monkeypatch.setattr(
        scheduler, "_try_acquire_job_run_lock", lambda _job_id: RecordingLock()
    )
    monkeypatch.setattr(
        scheduler, "advance_next_run", lambda _job_id, **_kwargs: True
    )
    monkeypatch.setattr(scheduler, "get_job", lambda _job_id: current)

    def fake_run(job, **kwargs):
        dispatched.append(job)
        kwargs["_run_lock"].release()
        return True

    monkeypatch.setattr(scheduler, "run_one_job", fake_run)

    assert scheduler.tick(verbose=False, sync=True) == 1
    assert dispatched == [current]
    assert released == [True]


@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_forked_child_does_not_inherit_run_ownership(tmp_path):
    """A bare-fork child must not prolong a dead scheduler's lock lifetime."""
    from cron import scheduler

    hermes_home = tmp_path / "hermes-home"
    child_ready = tmp_path / "fork-child-ready"
    child_pid_file = tmp_path / "fork-child-pid"
    holder = tmp_path / "fork-holder.py"
    holder.write_text(
        textwrap.dedent(
            f"""
            from pathlib import Path
            import os
            import time
            from cron.scheduler import _try_acquire_job_run_lock

            lock = _try_acquire_job_run_lock("fork-release-job")
            if lock is None:
                raise SystemExit(2)
            pid = os.fork()
            if pid == 0:
                Path({str(child_pid_file)!r}).write_text(str(os.getpid()), encoding="utf-8")
                Path({str(child_ready)!r}).write_text("ready", encoding="utf-8")
                time.sleep(30)
                os._exit(0)
            os._exit(0)
            """
        ),
        encoding="utf-8",
    )

    holder_process = subprocess.Popen(
        [sys.executable, str(holder)], env=_child_env(hermes_home)
    )
    lock = None
    fork_child_pid = None
    try:
        _wait_for(child_ready.exists, message="fork child never became ready")
        holder_process.wait(timeout=10)
        assert holder_process.returncode == 0
        fork_child_pid = int(child_pid_file.read_text(encoding="utf-8"))
        with jobs.use_cron_store(hermes_home):
            lock = scheduler._try_acquire_job_run_lock("fork-release-job")
        assert lock is not None
    finally:
        if lock is not None:
            lock.release()
        if fork_child_pid is not None:
            try:
                os.kill(fork_child_pid, 9)
            except ProcessLookupError:
                pass
        if holder_process.poll() is None:
            holder_process.kill()
        holder_process.wait(timeout=10)


def test_external_fire_loses_ownership_before_store_claim(monkeypatch):
    """A local contender must not let external fire advance schedule via CAS."""
    from cron import scheduler
    from cron import scheduler_provider

    claims: list[str] = []
    monkeypatch.setattr(
        jobs,
        "get_job",
        lambda job_id: {"id": job_id, "schedule": {"kind": "interval"}},
    )
    monkeypatch.setattr(scheduler, "_try_acquire_job_run_lock", lambda job_id: None)
    monkeypatch.setattr(
        jobs,
        "claim_job_for_fire",
        lambda job_id: claims.append(job_id) or True,
    )

    assert scheduler_provider.InProcessCronScheduler().fire_due("contended") is False
    assert claims == []


def test_external_recurring_lock_outlives_expired_store_claim(tmp_path, monkeypatch):
    """Same-host ownership still excludes after the 300s fire claim is stale."""
    from cron import scheduler
    from cron import scheduler_provider

    shared_store = tmp_path / "shared-store"
    owner_ready = tmp_path / "external-owner-ready"
    release_owner = tmp_path / "external-owner-release"
    side_effects = tmp_path / "external-side-effects"
    job_id = "external-long-run"
    job = {
        "id": job_id,
        "enabled": True,
        "state": "scheduled",
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": datetime.now(timezone.utc).isoformat(),
    }
    with jobs.use_cron_store(shared_store):
        jobs.save_jobs([job])

    owner = tmp_path / "external-owner.py"
    owner.write_text(
        textwrap.dedent(
            f"""
            from pathlib import Path
            import time
            from cron import jobs, scheduler
            from cron.scheduler_provider import InProcessCronScheduler

            def fake_run(job, **_kwargs):
                with Path({str(side_effects)!r}).open("a", encoding="utf-8") as stream:
                    stream.write("owner\\n")
                    stream.flush()
                Path({str(owner_ready)!r}).write_text("ready", encoding="utf-8")
                deadline = time.monotonic() + 20
                while not Path({str(release_owner)!r}).exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("test did not release external owner")
                    time.sleep(0.02)
                return True

            scheduler.run_one_job = fake_run
            with jobs.use_cron_store({str(shared_store)!r}):
                ran = InProcessCronScheduler().fire_due({job_id!r})
            raise SystemExit(0 if ran else 3)
            """
        ),
        encoding="utf-8",
    )

    owner_process = subprocess.Popen(
        [sys.executable, str(owner)], env=_child_env(tmp_path / "external-home-a")
    )
    try:
        _wait_for(owner_ready.exists, message="external owner never entered run body")
        assert _side_effect_count(side_effects) == 1

        # Simulate a cross-host retry arriving after the bounded store claim TTL.
        # A same-host process must still lose the full-run advisory lock before
        # it can reclaim or advance the recurring record.
        with jobs.use_cron_store(shared_store):
            stored = jobs.load_jobs()
            stored[0]["fire_claim"]["at"] = (
                datetime.now(timezone.utc) - timedelta(seconds=301)
            ).isoformat()
            jobs.save_jobs(stored)
            before = jobs.get_job(job_id)
            assert before is not None

            monkeypatch.setattr(
                scheduler,
                "run_one_job",
                lambda *_args, **_kwargs: pytest.fail("expired retry dispatched"),
            )
            assert scheduler_provider.InProcessCronScheduler().fire_due(job_id) is False
            after = jobs.get_job(job_id)

        assert after is not None
        assert after["next_run_at"] == before["next_run_at"]
        assert after["fire_claim"] == before["fire_claim"]
        assert _side_effect_count(side_effects) == 1
    finally:
        release_owner.touch()
        try:
            owner_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            owner_process.kill()
            owner_process.wait(timeout=10)
            raise


def test_external_one_shot_uses_durable_claim_without_run_lock(monkeypatch):
    """External one-shots must not depend on the recurring lock backend."""
    from cron import scheduler
    from cron import scheduler_provider

    one_shot = {"id": "external-once", "schedule": {"kind": "once"}}
    claims: list[str] = []
    runs: list[str] = []
    monkeypatch.setattr(jobs, "get_job", lambda _job_id: one_shot)
    monkeypatch.setattr(
        jobs,
        "claim_job_for_fire",
        lambda job_id, **_kwargs: claims.append(job_id) or True,
    )
    monkeypatch.setattr(
        scheduler,
        "_try_acquire_job_run_lock",
        lambda _job_id: pytest.fail("external one-shot acquired recurring run lock"),
    )
    monkeypatch.setattr(
        scheduler,
        "run_one_job",
        lambda job, **_kwargs: runs.append(job["id"]) or True,
    )

    assert scheduler_provider.InProcessCronScheduler().fire_due(one_shot["id"]) is True
    assert claims == [one_shot["id"]]
    assert runs == [one_shot["id"]]


def test_external_fire_rejects_schedule_kind_race_before_claim_mutation(
    tmp_path, monkeypatch
):
    """A one-shot snapshot cannot claim/advance a newly recurring record."""
    from cron import scheduler
    from cron import scheduler_provider

    job_id = "kind-race"
    initial = {
        "id": job_id,
        "enabled": True,
        "state": "scheduled",
        "schedule": {"kind": "once", "run_at": "2099-01-01T00:00:00+00:00"},
        "next_run_at": "2099-01-01T00:00:00+00:00",
    }
    changed = {
        **initial,
        "schedule": {"kind": "interval", "minutes": 5},
    }

    with jobs.use_cron_store(tmp_path):
        jobs.save_jobs([changed])
        monkeypatch.setattr(jobs, "get_job", lambda _job_id: initial)
        monkeypatch.setattr(
            scheduler,
            "run_one_job",
            lambda *_args, **_kwargs: pytest.fail("stale schedule snapshot dispatched"),
        )
        assert scheduler_provider.InProcessCronScheduler().fire_due(job_id) is False

        stored = jobs.load_jobs()[0]
    assert stored.get("fire_claim") is None
    assert stored["next_run_at"] == changed["next_run_at"]


def test_stale_due_scan_defers_recurring_advance_until_run_ownership(tmp_path):
    """The scheduler's due-scan mode leaves a stale slot unchanged for a loser."""
    stale = datetime.now(timezone.utc) - timedelta(hours=2)
    job = {
        "id": "stale-owned-advance",
        "enabled": True,
        "state": "scheduled",
        "schedule": {"kind": "interval", "minutes": 60},
        "next_run_at": stale.isoformat(),
    }
    with jobs.use_cron_store(tmp_path):
        jobs.save_jobs([job])
        due = jobs.get_due_jobs(defer_recurring_advance=True)
        assert [item["id"] for item in due] == [job["id"]]
        current = jobs.get_job(job["id"])
        assert current is not None
        assert current["next_run_at"] == stale.isoformat()

        # Once ownership is established, the compare-and-set advance succeeds.
        assert jobs.advance_next_run(
            job["id"],
            expected_next_run_at=stale.isoformat(),
            expected_schedule_kind="interval",
        ) is True


def test_advance_compare_and_set_rejects_stale_snapshot(tmp_path):
    """A changed active record is not advanced from an old due snapshot."""
    job = {
        "id": "advance-cas",
        "enabled": True,
        "state": "scheduled",
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": "2026-07-11T00:00:00+00:00",
    }
    with jobs.use_cron_store(tmp_path):
        jobs.save_jobs([job])
        assert jobs.advance_next_run(
            job["id"],
            expected_next_run_at="2026-07-10T00:00:00+00:00",
            expected_schedule_kind="interval",
        ) is False
        current = jobs.get_job(job["id"])
        assert current is not None
        assert current["next_run_at"] == job["next_run_at"]


def test_windows_lock_backend_uses_one_stable_byte(tmp_path, monkeypatch):
    """The Windows path uses non-blocking byte 0 and explicitly unlocks it."""
    from cron import scheduler

    calls: list[tuple[str, int, int]] = []

    class FakeMsvcrt:
        LK_NBLCK = "nonblocking"
        LK_UNLCK = "unlock"

        @staticmethod
        def locking(fd, mode, count):
            calls.append((mode, count, os.lseek(fd, 0, os.SEEK_CUR)))

    monkeypatch.setattr(scheduler, "_hermes_home", tmp_path / "hermes-home")
    monkeypatch.setattr(scheduler, "fcntl", None)
    monkeypatch.setattr(scheduler, "msvcrt", FakeMsvcrt)

    with jobs.use_cron_store(tmp_path / "hermes-home"):
        lock = scheduler._try_acquire_job_run_lock("windows-job")
        assert lock is not None
        assert calls == [("nonblocking", 1, 0)]
        lock.release()
    assert calls == [("nonblocking", 1, 0), ("unlock", 1, 0)]
