# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def hermes_home(tmp_path: Path) -> Path:
    home = tmp_path / ".hermes"
    jobs_path = home / "cron" / "jobs.json"
    jobs_path.parent.mkdir(parents=True)
    jobs_path.write_text('{"jobs": []}\n')
    return home


def test_restore_uses_destination_profile_lock(
    hermes_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cron.jobs as cron_jobs
    from hermes_cli.backup import create_quick_snapshot, restore_quick_snapshot

    jobs_path = hermes_home / "cron" / "jobs.json"
    jobs_path.write_text('{"jobs": [{"id": "snapshot"}]}\n')
    snap_id = create_quick_snapshot(hermes_home=hermes_home)
    assert snap_id is not None
    jobs_path.write_text('{"jobs": []}\n')

    locked_paths: list[Path] = []
    real_lock = cron_jobs._jobs_lock

    @contextlib.contextmanager
    def _spy_lock():
        locked_paths.append(cron_jobs._jobs_lock_file())
        with real_lock():
            yield

    monkeypatch.setattr(cron_jobs, "_jobs_lock", _spy_lock)

    process_default = hermes_home.parent / "process-default"
    with cron_jobs.use_cron_store(process_default):
        default_lock = process_default.resolve() / "cron" / ".jobs.lock"
        assert cron_jobs._jobs_lock_file() == default_lock
        assert restore_quick_snapshot(snap_id, hermes_home=hermes_home) is True
        assert cron_jobs._jobs_lock_file() == default_lock

    assert locked_paths == [hermes_home.resolve() / "cron" / ".jobs.lock"]
    data = json.loads(jobs_path.read_text())
    assert [job["id"] for job in data["jobs"]] == ["snapshot"]


def test_conditional_restore_preserves_scheduler_update(
    hermes_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cron.jobs as cron_jobs
    from hermes_cli.backup import (
        create_quick_snapshot,
        restore_cron_jobs_if_emptied,
    )

    jobs_path = hermes_home / "cron" / "jobs.json"
    jobs_path.write_text('{"jobs": [{"id": "snapshot-a"}, {"id": "snapshot-b"}]}\n')
    snap_id = create_quick_snapshot(hermes_home=hermes_home)
    assert snap_id is not None
    jobs_path.write_text('{"jobs": []}\n')

    @contextlib.contextmanager
    def _scheduler_wins_before_lock():
        jobs_path.write_text('{"jobs": [{"id": "concurrent"}]}\n')
        yield

    monkeypatch.setattr(cron_jobs, "_jobs_lock", _scheduler_wins_before_lock)

    assert restore_cron_jobs_if_emptied(snap_id, hermes_home=hermes_home) is None
    data = json.loads(jobs_path.read_text())
    assert [job["id"] for job in data["jobs"]] == ["concurrent"]


def test_conditional_restore_uses_validated_snapshot_bytes(
    hermes_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cron.jobs as cron_jobs
    from hermes_cli.backup import (
        create_quick_snapshot,
        restore_cron_jobs_if_emptied,
    )

    jobs_path = hermes_home / "cron" / "jobs.json"
    jobs_path.write_text('{"jobs": [{"id": "snapshot-a"}, {"id": "snapshot-b"}]}\n')
    snap_id = create_quick_snapshot(hermes_home=hermes_home)
    assert snap_id is not None
    snapshot_jobs = hermes_home / "state-snapshots" / snap_id / "cron" / "jobs.json"
    jobs_path.write_text('{"jobs": []}\n')

    @contextlib.contextmanager
    def _snapshot_changes_before_lock():
        snapshot_jobs.write_text('{"jobs": [{"id": "changed"}]}\n')
        yield

    monkeypatch.setattr(cron_jobs, "_jobs_lock", _snapshot_changes_before_lock)

    result = restore_cron_jobs_if_emptied(snap_id, hermes_home=hermes_home)
    assert result and result["job_count"] == 2
    data = json.loads(jobs_path.read_text())
    assert [job["id"] for job in data["jobs"]] == ["snapshot-a", "snapshot-b"]
