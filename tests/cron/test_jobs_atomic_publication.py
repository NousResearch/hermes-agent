"""The cron store must never publish through a partial in-place write."""

import errno
import json
import os

import pytest

from cron import jobs


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
