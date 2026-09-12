import threading

import pytest

from cron import jobs


def test_bulk_resnapshot_does_not_clobber_concurrent_pause(tmp_path, monkeypatch):
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")
    monkeypatch.setattr(
        jobs,
        "_compute_provider_model_snapshots",
        lambda **_kwargs: ("new-provider", "new-model"),
    )
    jobs.save_jobs(
        [
            {
                "id": "job-1",
                "name": "race witness",
                "schedule": {"kind": "interval", "minutes": 60},
                "enabled": True,
                "state": "scheduled",
                "provider_snapshot": "old-provider",
                "model_snapshot": "old-model",
            }
        ]
    )

    original_load = jobs.load_jobs
    resnapshot_thread = threading.local()
    loaded = threading.Event()
    release = threading.Event()
    pause_started = threading.Event()

    def blocking_load():
        rows = original_load()
        if getattr(resnapshot_thread, "active", False) and not loaded.is_set():
            loaded.set()
            assert release.wait(5)
        return rows

    monkeypatch.setattr(jobs, "load_jobs", blocking_load)

    def resnapshot():
        resnapshot_thread.active = True
        jobs.resnapshot_all_unpinned()

    def pause():
        pause_started.set()
        jobs.pause_job("job-1", reason="user requested")

    resnap = threading.Thread(target=resnapshot)
    pauser = threading.Thread(target=pause)
    resnap.start()
    assert loaded.wait(5)
    pauser.start()
    assert pause_started.wait(5)
    pauser.join(timeout=0.1)
    assert pauser.is_alive(), "pause must wait for the resnapshot transaction"

    release.set()
    resnap.join(timeout=5)
    pauser.join(timeout=5)
    assert not resnap.is_alive()
    assert not pauser.is_alive()

    [saved] = original_load()
    assert saved["state"] == "paused"
    assert saved["enabled"] is False
    assert saved["paused_reason"] == "user requested"
    assert saved["provider_snapshot"] == "new-provider"
    assert saved["model_snapshot"] == "new-model"


@pytest.mark.parametrize(
    ("resolved", "expected"),
    [
        ((None, "new-model"), ("old-provider", "new-model")),
        ((None, None), ("old-provider", "old-model")),
    ],
)
def test_single_resnapshot_retains_snapshots_when_resolution_fails(
    tmp_path, monkeypatch, resolved, expected
):
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")
    jobs.save_jobs(
        [
            {
                "id": "job-1",
                "name": "resolution witness",
                "schedule": {"kind": "interval", "minutes": 60},
                "enabled": True,
                "state": "scheduled",
                "provider_snapshot": "old-provider",
                "model_snapshot": "old-model",
            }
        ]
    )
    monkeypatch.setattr(jobs, "_compute_provider_model_snapshots", lambda **_kwargs: resolved)

    updated = jobs.resnapshot_job("job-1")

    assert updated is not None
    assert (updated["provider_snapshot"], updated["model_snapshot"]) == expected


def test_drifted_resnapshot_persists_the_confirmed_assignment(tmp_path, monkeypatch):
    cron_dir = tmp_path / "cron"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")
    (tmp_path / "config.yaml").write_text(
        "model:\n  provider: new-provider\n  default: new-model\n",
        encoding="utf-8",
    )
    jobs.save_jobs(
        [
            {
                "id": "job-1",
                "name": "assignment witness",
                "schedule": {"kind": "interval", "minutes": 60},
                "enabled": True,
                "state": "scheduled",
                "provider_snapshot": "old-provider",
                "model_snapshot": "old-model",
            }
        ]
    )
    monkeypatch.setattr(
        jobs,
        "_compute_provider_model_snapshots",
        lambda **_kwargs: ("raced-provider", "raced-model"),
    )

    [updated] = jobs.resnapshot_all_unpinned(
        expected_provider="new-provider",
        expected_model="new-model",
        drifted_only=True,
    )

    assert (updated["provider_snapshot"], updated["model_snapshot"]) == (
        "new-provider",
        "new-model",
    )
