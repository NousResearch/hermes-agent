from __future__ import annotations

import copy
import errno
import importlib
import json
import multiprocessing
import os
from types import SimpleNamespace

import pytest


@pytest.fixture
def jobs_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_constants
    import cron.jobs

    importlib.reload(hermes_constants)
    importlib.reload(cron.jobs)
    return cron.jobs


def _job(_jobs, job_id: str):
    return {
        "id": job_id,
        "name": job_id,
        "enabled": True,
        "prompt": "test",
    }


def _write_jobs(jobs, rows) -> None:
    jobs._current_cron_store().jobs_file.write_text(
        json.dumps({"jobs": rows}), encoding="utf-8"
    )


def test_three_way_merge_preserves_disjoint_changes(jobs_env):
    jobs = jobs_env
    base = [_job(jobs, job_id) for job_id in ("a", "b", "e")]
    desired = [copy.deepcopy(base[1]), copy.deepcopy(base[2]), _job(jobs, "c")]
    current = copy.deepcopy(base) + [_job(jobs, "d")]
    desired[0]["enabled"] = False
    current[2]["prompt"] = "current"
    merged = jobs._merge_jobs_three_way(base, desired, current)

    by_id = {row["id"]: row for row in merged}
    assert list(by_id) == ["b", "e", "c", "d"]
    assert by_id["b"]["enabled"] is False
    assert by_id["e"]["prompt"] == "current"


@pytest.mark.parametrize(
    ("desired", "current"),
    [
        ({"enabled": False}, {"prompt": "other"}),
        (None, {"enabled": False}),
        ({"enabled": False}, None),
    ],
)
def test_three_way_merge_rejects_same_id_conflicts(jobs_env, desired, current):
    jobs = jobs_env
    base = [_job(jobs, "a")]

    def changed(delta):
        if delta is None:
            return []
        row = copy.deepcopy(base[0])
        row.update(delta)
        return [row]

    with pytest.raises(RuntimeError, match="conflicting concurrent cron job"):
        jobs._merge_jobs_three_way(base, changed(desired), changed(current))


@pytest.mark.parametrize("ambiguous", ["missing", "duplicate"])
def test_ambiguous_rows_fail_closed_during_concurrent_merge(jobs_env, ambiguous):
    jobs = jobs_env
    first = _job(jobs, "a") if ambiguous == "duplicate" else "legacy"
    base = [first, _job(jobs, "a"), _job(jobs, "b")]
    desired = copy.deepcopy(base)
    current = copy.deepcopy(base)
    desired[1]["enabled"] = False
    current[2]["prompt"] = "current"

    with pytest.raises(RuntimeError, match="missing or duplicate ids"):
        jobs._merge_jobs_three_way(base, desired, current)


@pytest.mark.parametrize("sibling_change", ["pause", "delete"])
def test_loaded_save_preserves_sibling_pause_or_delete(jobs_env, sibling_change):
    jobs = jobs_env
    seed = _job(jobs, "a")
    jobs.save_jobs([seed], replace=True)

    with jobs._jobs_lock():
        stale = jobs.load_jobs()
        current = copy.deepcopy(stale)
        if sibling_change == "pause":
            current[0]["enabled"] = False
        else:
            current = []
        _write_jobs(jobs, current)
        jobs._save_jobs_unlocked(stale)

    assert jobs.load_jobs() == current


def test_unknown_stamps_never_certify_a_stale_loaded_snapshot(jobs_env, monkeypatch):
    jobs = jobs_env
    seed = _job(jobs, "a")
    jobs.save_jobs([seed], replace=True)

    with jobs._jobs_lock():
        monkeypatch.setattr(jobs, "_jobs_file_stamp", lambda _path: None)
        stale = jobs.load_jobs()
        current = copy.deepcopy(stale)
        current[0]["enabled"] = False
        _write_jobs(jobs, current)
        jobs._save_jobs_unlocked(stale)

    assert jobs.load_jobs() == current


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership semantics")
def test_commit_lock_preserves_cron_directory_owner(jobs_env, monkeypatch):
    jobs = jobs_env
    calls = []
    monkeypatch.setattr(jobs.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        jobs.os,
        "fchown",
        lambda fd, uid, gid: calls.append((fd, uid, gid)),
    )

    with jobs._jobs_commit_lock():
        pass

    directory = os.stat(jobs._current_cron_store().cron_dir)
    assert len(calls) == 1
    assert calls[0][1:] == (directory.st_uid, directory.st_gid)
    assert jobs._jobs_commit_lock_file().stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "O_NOFOLLOW"),
    reason="POSIX no-follow semantics",
)
def test_commit_lock_refuses_symlink_before_chmod(jobs_env):
    jobs = jobs_env
    victim = jobs._current_cron_store().cron_dir / "victim"
    victim.write_text("unchanged", encoding="utf-8")
    victim.chmod(0o644)
    jobs._jobs_commit_lock_file().symlink_to(victim)

    with pytest.raises(RuntimeError, match="Unable to open"):
        with jobs._jobs_commit_lock():
            pass

    assert victim.stat().st_mode & 0o777 == 0o644


def test_windows_commit_lock_seeds_and_locks_byte_zero(jobs_env, monkeypatch):
    jobs = jobs_env
    calls = []
    fake = SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2)

    def locking(fd, mode, size):
        calls.append((os.lseek(fd, 0, os.SEEK_CUR), mode, size))

    fake.locking = locking
    monkeypatch.setattr(jobs, "fcntl", None)
    monkeypatch.setattr(jobs, "msvcrt", fake)

    with jobs._jobs_commit_lock():
        assert jobs._jobs_commit_lock_file().read_bytes() == b" "

    assert calls == [(0, fake.LK_NBLCK, 1), (0, fake.LK_UNLCK, 1)]


def test_commit_lock_without_backend_uses_historical_fallback(jobs_env, monkeypatch):
    jobs = jobs_env
    monkeypatch.setattr(jobs, "fcntl", None)
    monkeypatch.setattr(jobs, "msvcrt", None)

    with jobs._jobs_commit_lock():
        pass


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock semantics")
def test_unsupported_flock_uses_generation_fallback(jobs_env, monkeypatch):
    jobs = jobs_env

    def unsupported(_fd, _operation):
        raise OSError(errno.ENOTSUP, "unsupported")

    monkeypatch.setattr(jobs.fcntl, "flock", unsupported)
    with jobs._jobs_commit_lock():
        pass


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock semantics")
def test_unlock_error_does_not_fail_published_save(jobs_env, monkeypatch):
    jobs = jobs_env
    real_flock = jobs.fcntl.flock

    def fail_unlock(fd, operation):
        if operation == jobs.fcntl.LOCK_UN:
            raise OSError("synthetic unlock failure")
        return real_flock(fd, operation)

    with jobs._jobs_lock():
        monkeypatch.setattr(jobs.fcntl, "flock", fail_unlock)
        jobs._save_jobs_unlocked([_job(jobs, "saved")], replace=True)

    payload = json.loads(jobs._current_cron_store().jobs_file.read_text())
    assert [row["id"] for row in payload["jobs"]] == ["saved"]


def test_loaded_save_honors_removed_ids_for_current_only_job(jobs_env):
    jobs = jobs_env
    jobs.save_jobs([_job(jobs, "a")], replace=True)

    with jobs._jobs_lock():
        desired = jobs.load_jobs()
        _write_jobs(jobs, desired + [_job(jobs, "external")])
        jobs._save_jobs_unlocked(desired, removed_ids={"external"})

    assert [row["id"] for row in jobs.load_jobs()] == ["a"]


def test_generation_churn_fails_closed_without_final_unchecked_write(
    jobs_env, monkeypatch
):
    jobs = jobs_env
    seed = _job(jobs, "a")
    jobs.save_jobs([seed], replace=True)

    with jobs._jobs_lock():
        desired = jobs.load_jobs()
        desired[0]["prompt"] = "desired"
        real_stamp = jobs._jobs_file_stamp
        calls = 0

        def churning_stamp(path):
            nonlocal calls
            calls += 1
            if calls in {2, 5, 8}:
                _write_jobs(jobs, [seed, _job(jobs, f"external-{calls}")])
            return real_stamp(path)

        monkeypatch.setattr(jobs, "_jobs_file_stamp", churning_stamp)
        with pytest.raises(RuntimeError, match="kept changing"):
            jobs._save_jobs_unlocked(desired)

    stored = jobs.load_jobs()
    assert stored[0]["prompt"] == "test"
    assert stored[1]["id"].startswith("external-")
    assert not list(jobs._current_cron_store().cron_dir.glob(".jobs_*.tmp"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock semantics")
@pytest.mark.parametrize(
    "lock_error",
    [BlockingIOError(), OSError(errno.ENOLCK, "lock records unavailable")],
    ids=["contended", "enolck"],
)
def test_commit_lock_contention_fails_closed(jobs_env, monkeypatch, lock_error):
    jobs = jobs_env
    seed = _job(jobs, "a")
    jobs.save_jobs([seed], replace=True)
    before = jobs._current_cron_store().jobs_file.read_bytes()
    real_flock = jobs.fcntl.flock

    def contend(fd, operation):
        if operation & jobs.fcntl.LOCK_NB:
            raise lock_error
        return real_flock(fd, operation)

    with jobs._jobs_lock():
        monkeypatch.setattr(jobs.fcntl, "flock", contend)
        monkeypatch.setattr(jobs, "_JOBS_COMMIT_LOCK_TIMEOUT_SECONDS", 0.0)
        with pytest.raises(RuntimeError, match="Timed out"):
            jobs._save_jobs_unlocked([_job(jobs, "b")])

    assert jobs._current_cron_store().jobs_file.read_bytes() == before


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock subprocess test")
def test_real_degraded_writers_serialize_publication(jobs_env):
    jobs = jobs_env
    jobs.save_jobs([_job(jobs, "a"), _job(jobs, "b")], replace=True)
    context = multiprocessing.get_context("fork")
    a_loaded = context.Event()
    b_loaded = context.Event()
    start_saves = context.Event()
    a_committing = context.Event()
    b_committing = context.Event()
    a_done = context.Event()
    b_done = context.Event()

    def writer(loaded_event, committing_event, done_event, first):
        jobs._JOBS_LOCK_TIMEOUT_SECONDS = 0.05
        with jobs._jobs_lock():
            loaded = jobs.load_jobs()
            if first:
                loaded[0]["prompt"] = "writer-a"
                loaded.append(_job(jobs, "c"))
            else:
                loaded[1]["enabled"] = False
                loaded.append(_job(jobs, "d"))
            loaded_event.set()
            if not start_saves.wait(10):
                raise RuntimeError("writer start timed out")
            committing_event.set()
            jobs.save_jobs(loaded)
        done_event.set()

    processes = [
        context.Process(
            target=writer, args=(a_loaded, a_committing, a_done, True)
        ),
        context.Process(
            target=writer, args=(b_loaded, b_committing, b_done, False)
        ),
    ]
    processes[0].start()
    assert a_loaded.wait(5)
    processes[1].start()
    assert b_loaded.wait(5)
    with jobs._jobs_commit_lock():
        start_saves.set()
        assert a_committing.wait(5)
        assert b_committing.wait(5)
        assert not a_done.is_set()
        assert not b_done.is_set()

    for process in processes:
        process.join(15)
        if process.is_alive():
            process.terminate()
            process.join(5)
        assert process.exitcode == 0

    stored = {row["id"]: row for row in jobs.load_jobs()}
    assert stored["a"]["prompt"] == "writer-a"
    assert stored["b"]["enabled"] is False
    assert {"c", "d"} <= stored.keys()
