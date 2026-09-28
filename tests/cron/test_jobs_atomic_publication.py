"""The cron store must never publish through a partial in-place write."""

import errno
import json
import multiprocessing
import os
import stat

import pytest

from cron import jobs


def _concurrent_job_change(home, start, result, index, remove_id):
    start.wait()
    with jobs.use_cron_store(home):
        if remove_id:
            jobs.remove_job(remove_id)
        job = jobs.create_job(
            prompt=f"watch-end {index}", schedule="every 1h", name=f"watcher-{index}",
        )
        result.put(job["id"])


@pytest.fixture
def store(tmp_path, monkeypatch):
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")
    jobs.save_jobs([{"id": "old", "prompt": "old"}])
    return cron_dir / "jobs.json"


def test_rename_failure_preserves_complete_old_store(store, monkeypatch):
    old = store.read_bytes()
    real_replace = os.replace

    def fail_store_rename(source, target):
        if str(target) == str(store):
            raise OSError(errno.EXDEV, "simulated cross-device rename")
        return real_replace(source, target)

    monkeypatch.setattr(jobs.os, "replace", fail_store_rename)
    with pytest.raises(OSError, match="cross-device rename"):
        jobs.save_jobs([{"id": "new", "prompt": "new"}])
    assert store.read_bytes() == old
    assert [j["id"] for j in json.loads(store.read_text())["jobs"]] == ["old"]
    assert not list(store.parent.glob(".jobs_*.tmp"))


def test_save_refuses_degraded_cross_process_lock(store, monkeypatch):
    old = store.read_bytes()
    monkeypatch.setattr(jobs, "_acquire_flock", lambda *_: False)
    with pytest.raises(RuntimeError, match="cron jobs lock"):
        jobs.save_jobs([{"id": "new", "prompt": "new"}])
    assert store.read_bytes() == old


@pytest.mark.parametrize("failure", [errno.EXDEV, errno.EBUSY])
def test_rename_errors_never_fall_back_to_in_place_write(store, monkeypatch, failure):
    old = store.read_bytes()
    real_replace = os.replace

    def fail_store_rename(source, target):
        if str(target) == str(store):
            raise OSError(failure, "simulated rename failure")
        return real_replace(source, target)

    monkeypatch.setattr(jobs.os, "replace", fail_store_rename)
    with pytest.raises(OSError, match="simulated rename failure"):
        jobs.save_jobs([{"id": "new"}])
    assert store.read_bytes() == old
    assert not list(store.parent.glob(".jobs_*.tmp"))


def test_symlinked_store_stages_and_renames_beside_resolved_target(tmp_path, monkeypatch):
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    target = target_dir / "jobs.json"
    link_dir = tmp_path / "link"
    link_dir.mkdir()
    link = link_dir / "jobs.json"
    link.symlink_to(target)
    monkeypatch.setattr(jobs, "CRON_DIR", link_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", link)
    monkeypatch.setattr(jobs, "OUTPUT_DIR", link_dir / "output")

    assert jobs._jobs_lock_file() == target_dir / ".jobs.lock"
    jobs.save_jobs([{"id": "kept"}])
    assert link.is_symlink()
    assert [item["id"] for item in jobs.load_jobs()] == ["kept"]
    assert json.loads((target_dir / "jobs.json.good").read_text())["jobs"][0]["id"] == "kept"
    assert not list(link_dir.glob(".jobs_*.tmp"))
    target.write_bytes(b'{"jobs": [')
    with pytest.raises(RuntimeError, match="Forensic copy"):
        jobs.validate_jobs_store()
    assert len(list(target_dir.glob("jobs.json.corrupt-*"))) == 1
    assert not list(link_dir.glob("jobs.json.corrupt-*"))


def test_corrupt_primary_is_preserved_and_recovery_is_explicit(store):
    jobs.save_jobs([{"id": "old"}, {"id": "new"}])
    latest = store.read_bytes()
    assert (store.parent / "jobs.json.good").read_bytes() == latest
    broken = b'{"jobs": [{"id": "old"}, {"id": '
    store.write_bytes(broken)

    with pytest.raises(RuntimeError, match="Forensic copy"):
        jobs.validate_jobs_store()
    with pytest.raises(RuntimeError, match="corrupted"):
        jobs.save_jobs([{"id": "replacement"}])
    assert store.read_bytes() == broken
    forensic = list(store.parent.glob("jobs.json.corrupt-*"))
    assert len(forensic) == 1 and forensic[0].read_bytes() == broken
    assert stat.S_IMODE(forensic[0].stat().st_mode) == 0o600

    jobs.recover_jobs_from_good_backup()
    assert [item["id"] for item in jobs.load_jobs()] == ["old", "new"]
    assert forensic[0].read_bytes() == broken


def test_directory_sync_failure_reports_complete_new_store(store, monkeypatch):
    real_sync = jobs._sync_jobs_directory
    calls = []

    def fail_first_sync(path):
        calls.append(path)
        if len(calls) == 2:  # prior .good snapshot synced; primary rename has happened
            raise OSError(errno.EIO, "simulated directory fsync failure")
        return real_sync(path)

    monkeypatch.setattr(jobs, "_sync_jobs_directory", fail_first_sync)
    with pytest.raises(OSError, match="directory fsync failure"):
        jobs.save_jobs([{"id": "new"}])
    assert [item["id"] for item in json.loads(store.read_text())["jobs"]] == ["new", "old"]
    assert json.loads((store.parent / "jobs.json.good").read_text())["jobs"][0]["id"] == "old"


@pytest.mark.parametrize("boundary", ["serialize", "file_fsync", "backup", "rename", "dir_fsync"])
def test_interruption_boundaries_keep_complete_store_and_prior_jobs(store, monkeypatch, boundary):
    old = store.read_bytes()
    real_dump = jobs.json.dump
    real_fsync = jobs.os.fsync
    real_backup = jobs._backup_previous_store
    real_replace = jobs.os.replace
    real_dir_sync = jobs._sync_jobs_directory
    sync_calls = []

    def interrupted_dump(data, stream, **kwargs):
        stream.write('{"jobs": [')
        raise RuntimeError("interrupted serialize")

    def interrupted_fsync(fd):
        raise RuntimeError("interrupted file_fsync")

    def interrupted_backup(*args, **kwargs):
        raise RuntimeError("interrupted backup")

    def interrupted_replace(source, target):
        if str(target) == str(store):
            raise RuntimeError("interrupted rename")
        return real_replace(source, target)

    def interrupted_dir_sync(path):
        sync_calls.append(path)
        if len(sync_calls) == 2:
            raise RuntimeError("interrupted dir_fsync")
        return real_dir_sync(path)

    if boundary == "serialize":
        monkeypatch.setattr(jobs.json, "dump", interrupted_dump)
    elif boundary == "file_fsync":
        monkeypatch.setattr(jobs.os, "fsync", interrupted_fsync)
    elif boundary == "backup":
        monkeypatch.setattr(jobs, "_backup_previous_store", interrupted_backup)
    elif boundary == "rename":
        monkeypatch.setattr(jobs.os, "replace", interrupted_replace)
    else:
        monkeypatch.setattr(jobs, "_sync_jobs_directory", interrupted_dir_sync)

    with pytest.raises(RuntimeError, match="interrupted"):
        jobs.save_jobs([{"id": "new"}])
    # A failure after publication can return an error with the new complete store.
    data = json.loads(store.read_text())["jobs"]
    assert "old" in {item["id"] for item in data}
    if boundary == "dir_fsync":
        assert len(data) == 2
    else:
        assert store.read_bytes() == old
    assert not list(store.parent.glob(".jobs_*.tmp"))


@pytest.mark.skipif(os.name != "posix", reason="fork-based cross-process lock test")
def test_concurrent_create_remove_and_watcher_readback(tmp_path):
    home = tmp_path / "home"
    with jobs.use_cron_store(home):
        seed = jobs.create_job(prompt="seed", schedule="every 1h")
    ctx = multiprocessing.get_context("fork")
    start = ctx.Event()
    result = ctx.Queue()
    workers = [ctx.Process(target=_concurrent_job_change,
                           args=(home, start, result, index, seed["id"] if index == 0 else None))
               for index in range(4)]
    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        worker.join(15)
        assert worker.exitcode == 0
    ids = [result.get(timeout=1) for _ in workers]
    assert len(set(ids)) == len(workers)
    with jobs.use_cron_store(home):
        durable_ids = {job["id"] for job in jobs.load_jobs()}
    assert durable_ids == set(ids)
