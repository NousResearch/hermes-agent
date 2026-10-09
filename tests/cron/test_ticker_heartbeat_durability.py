"""Periodic liveness is atomic, but only persistent records require fsync."""
import json
import os

import cron.jobs as jobs
import utils


def test_heartbeat_profile_isolation_without_forced_disk_commits(tmp_path, monkeypatch):
    synced = []
    monkeypatch.setattr(utils.os, "fsync", lambda fd: synced.append(fd))
    homes = [tmp_path / "a", tmp_path / "b"]
    for home in homes:
        home.mkdir()
    stamps = {}
    for home in [homes[0], homes[1], homes[0]]:
        with jobs.use_cron_store(home):
            jobs.record_ticker_heartbeat(success=True)
            assert jobs.get_ticker_heartbeat_age() is not None
            assert jobs.get_ticker_success_age() is not None
        marker = home / "cron" / "ticker_heartbeat"
        stamp = marker.read_text()
        epoch, pid = stamp.split()
        assert float(epoch) > 0
        assert int(pid) == os.getpid()
        assert not list(marker.parent.glob(".hb_*"))
        other = homes[1] if home == homes[0] else homes[0]
        if other in stamps:
            assert (other / "cron" / "ticker_heartbeat").read_text() == stamps[other]
        stamps[home] = stamp
    assert not synced, "best-effort heartbeat files must not force filesystem commits"


def test_persistent_records_keep_file_sync_enabled_by_default(tmp_path, monkeypatch):
    synced = []
    monkeypatch.setattr(utils.os, "fsync", lambda fd: synced.append(fd))
    target = tmp_path / "durable-state"
    utils.atomic_write_text(target, "saved", mode=0o600)
    assert target.read_text() == "saved"
    assert synced
    home = tmp_path / "jobs-home"
    home.mkdir()
    with jobs.use_cron_store(home):
        for write in (lambda: jobs.save_jobs([]), jobs.record_catch_up_occurrence,
                      lambda: jobs.record_ticker_error("synthetic error")):
            synced.clear()
            write()
            assert synced, "persistent records must retain their durability barriers"
    assert json.loads((home / "cron" / "jobs.json").read_text())["jobs"] == []
