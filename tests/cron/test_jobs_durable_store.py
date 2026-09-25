"""Crash and corruption boundaries for the cron job store."""

import json
import errno
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from cron import jobs


@pytest.fixture
def store(tmp_path):
    with jobs.use_cron_store(tmp_path):
        yield tmp_path / "cron" / "jobs.json"


def test_malformed_store_refuses_save_and_preserves_forensic_bytes(store):
    jobs.save_jobs([{"id": "prior", "prompt": "private payload"}])
    malformed = b'{"jobs": [{"id": "prior"}, {"id": "partial",'
    store.write_bytes(malformed)

    with pytest.raises(RuntimeError, match="corrupt"):
        jobs.save_jobs([{"id": "replacement"}])

    assert store.read_bytes() == malformed
    assert [j["id"] for j in json.loads(store.with_name("jobs.json.last-good").read_text())["jobs"]] == ["prior"]
    assert store.with_name("jobs.json.corrupt").read_bytes() == malformed


def test_interrupted_stage_and_publish_keep_complete_previous_generation(store, monkeypatch):
    jobs.save_jobs([{"id": "prior", "prompt": "full body"}])
    before = store.read_bytes()
    original_dump = jobs.json.dump

    def partial_dump(value, stream, **kwargs):
        stream.write('{"jobs": [{"id": "cut",')
        raise KeyboardInterrupt("interrupted during serialization")

    monkeypatch.setattr(jobs.json, "dump", partial_dump)
    with pytest.raises(KeyboardInterrupt):
        jobs.save_jobs([{"id": "new"}])
    assert store.read_bytes() == before

    monkeypatch.setattr(jobs.json, "dump", original_dump)
    original_replace = jobs.os.replace

    def interrupted_publish(src, dst):
        if os.fspath(dst) == os.fspath(store):
            raise KeyboardInterrupt("interrupted before rename")
        return original_replace(src, dst)

    monkeypatch.setattr(jobs.os, "replace", interrupted_publish)
    with pytest.raises(KeyboardInterrupt):
        jobs.save_jobs([{"id": "new"}])
    assert store.read_bytes() == before


def test_rename_failure_never_falls_back_to_in_place_copy(store, monkeypatch):
    jobs.save_jobs([{"id": "prior", "prompt": "whole"}])
    before = store.read_bytes()
    real_replace = jobs.os.replace

    def cross_device(src, dst):
        if os.fspath(dst) == os.fspath(store):
            raise OSError(errno.EXDEV, "simulated rename failure")
        return real_replace(src, dst)

    monkeypatch.setattr(jobs.os, "replace", cross_device)
    with pytest.raises(OSError) as error:
        jobs.save_jobs([{"id": "new"}])
    assert error.value.errno == errno.EXDEV
    assert store.read_bytes() == before


def test_interrupted_backup_keeps_new_store_and_prior_good_backup(store, monkeypatch):
    jobs.save_jobs([{"id": "prior", "prompt": "whole"}])
    previous_backup = store.with_name("jobs.json.last-good").read_bytes()
    real_replace = jobs.os.replace

    def interrupt_backup(src, dst):
        if os.fspath(dst) == os.fspath(store.with_name("jobs.json.last-good")):
            raise KeyboardInterrupt("interrupted before backup rename")
        return real_replace(src, dst)

    monkeypatch.setattr(jobs.os, "replace", interrupt_backup)
    with pytest.raises(KeyboardInterrupt):
        jobs.save_jobs([{"id": "prior", "prompt": "whole"}, {"id": "new"}])
    assert [j["id"] for j in json.loads(store.read_text())["jobs"]] == ["prior", "new"]
    assert store.with_name("jobs.json.last-good").read_bytes() == previous_backup


def test_startup_read_refuses_corrupt_store_and_keeps_forensic_copy(store):
    jobs.save_jobs([{"id": "prior"}])
    store.write_bytes(b'{"jobs": [{"id": "cut",')
    with pytest.raises(RuntimeError, match="corrupted"):
        jobs.load_jobs()
    assert store.with_name("jobs.json.corrupt").read_bytes() == store.read_bytes()


def test_newest_good_backup_contains_complete_committed_generation(store):
    prior = {"id": "prior", "prompt": "full body", "opaque": {"a": [1, 2]}}
    jobs.save_jobs([prior])
    jobs.save_jobs([prior, {"id": "new", "prompt": "watcher"}])

    assert json.loads(store.read_text())["jobs"] == [prior, {"id": "new", "prompt": "watcher"}]
    backup = store.with_name("jobs.json.last-good")
    assert json.loads(backup.read_text())["jobs"] == [prior, {"id": "new", "prompt": "watcher"}]
    assert backup.stat().st_mode & 0o077 == 0


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
def test_existing_world_readable_store_is_tightened_before_publish(store):
    jobs.save_jobs([{"id": "prior", "prompt": "sensitive"}])
    os.chmod(store, 0o644)

    jobs.save_jobs([{"id": "prior", "prompt": "sensitive"}, {"id": "new"}])

    assert stat.S_IMODE(store.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.with_name("jobs.json.last-good").stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
@pytest.mark.parametrize("policy,mode", [("HERMES_MANAGED", 0o640), ("HERMES_CONTAINER", 0o644)])
def test_managed_and_container_store_mode_stays_operator_owned(store, monkeypatch, policy, mode):
    jobs.save_jobs([{"id": "prior"}])
    os.chmod(store, mode)
    monkeypatch.setenv(policy, "nixos" if policy == "HERMES_MANAGED" else "1")

    jobs.save_jobs([{"id": "prior"}, {"id": "new"}])

    assert stat.S_IMODE(store.stat().st_mode) == mode
    assert stat.S_IMODE(store.with_name("jobs.json.last-good").stat().st_mode) == mode


def test_lock_timeout_refuses_mutation(store, monkeypatch):
    jobs.save_jobs([{"id": "prior"}])
    before = store.read_bytes()
    monkeypatch.setattr(jobs, "_acquire_flock", lambda *_: False)
    with pytest.raises(TimeoutError):
        jobs.save_jobs([{"id": "new"}])
    assert store.read_bytes() == before


def test_publish_uses_rename_and_syncs_directory(store, monkeypatch):
    seen = []
    real_replace = jobs.os.replace
    real_fsync = jobs.os.fsync

    def track_replace(src, dst):
        seen.append(("rename", os.fspath(dst)))
        return real_replace(src, dst)

    def track_fsync(fd):
        seen.append(("fsync", os.fstat(fd).st_mode))
        return real_fsync(fd)

    monkeypatch.setattr(jobs.os, "replace", track_replace)
    monkeypatch.setattr(jobs.os, "fsync", track_fsync)
    jobs.save_jobs([{"id": "complete"}])

    assert ("rename", os.fspath(store)) in seen
    rename_index = seen.index(("rename", os.fspath(store)))
    assert any(kind == "fsync" and stat.S_ISDIR(mode) for kind, mode in seen[rename_index + 1:])


@pytest.mark.skipif(os.name != "posix", reason="flock concurrency contract is POSIX")
def test_concurrent_watcher_create_and_remove_preserve_job_payloads(store):
    jobs.save_jobs([{"id": "remove-me", "prompt": "old"},
                    {"id": "keep-me", "prompt": "full prior payload"}])
    home = store.parent.parent
    (home / "scripts").mkdir()
    (home / "scripts" / "watch.sh").write_text("#!/bin/sh\necho alert\n")
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    create_code = ("from cron.jobs import create_job; "
                   "j=create_job(prompt=None,schedule='every 2m',script='watch.sh',"
                   "no_agent=True,deliver='local',name='watcher',repeat=0); print(j['id'])")
    remove_code = "from cron.jobs import remove_job; assert remove_job('remove-me')"
    create = subprocess.Popen([sys.executable, "-c", create_code], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    remove = subprocess.Popen([sys.executable, "-c", remove_code], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    created_id, create_error = create.communicate(timeout=20)
    _, remove_error = remove.communicate(timeout=20)
    assert create.returncode == 0, create_error
    assert remove.returncode == 0, remove_error
    assert {j["id"] for j in jobs.load_jobs()} == {"keep-me", created_id.strip()}
    assert next(j for j in jobs.load_jobs() if j["id"] == "keep-me")["prompt"] == "full prior payload"
