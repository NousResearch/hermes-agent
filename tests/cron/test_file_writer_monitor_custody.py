"""Private TEST-FIRST proposal for the paired monitor snapshot/jobs-state writer.

The test calls the actual provider-free mutation seam ``monitor._persist_monitor_state``.  It
never runs a monitor script, launches an OS-dependent command, or treats ``check_monitor`` as a
writer fence.  Jobs are routed to B while the child imports with Hermes home A, and the parent
holds B's real EX lease through the whole paired mutation.
"""
from __future__ import annotations

from pathlib import Path

from tests.cron.writer_test_support import WriterChild, profile_env


def test_monitor_snapshot_and_jobs_state_wait_for_b_ex_together(tmp_path: Path) -> None:
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()
    store_b = {"store_home": str(home_b)}

    with WriterChild(home_a) as child:
        job = child.run_unfenced(
            "seed_job", {**store_b, "name": "opaque monitor job"}
        )
        child.run_under_exclusive(
            "monitor_state", {**store_b, "job_id": job["id"]}, lease_home=home_b
        )

        snapshot = home_b / "cron" / "output" / job["id"] / "monitor_last_output.txt"
        assert snapshot.read_text(encoding="utf-8") == "opaque monitor output\n"
        with profile_env(home_a):
            from cron import jobs

            with jobs.use_cron_store(home_b):
                stored = jobs.get_job(job["id"])
        assert stored is not None
        assert stored["monitor_state"]["last_output_hash"] == "opaque-monitor-hash"

    with profile_env(home_a):
        from cron import jobs

        with jobs.use_cron_store(home_b):
            stored = jobs.get_job(job["id"])
    assert stored is not None
    assert stored["monitor_state"]["last_output_hash"] == "opaque-monitor-hash"
    assert isinstance(stored["monitor_state"]["last_changed_at"], str)

    assert snapshot.read_text(encoding="utf-8") == "opaque monitor output\n"
    assert not (
        home_a / "cron" / "output" / job["id"] / "monitor_last_output.txt"
    ).exists()
    assert (home_b / "cron" / "jobs.json").is_file()
    assert not (home_a / "cron" / "jobs.json").exists()
