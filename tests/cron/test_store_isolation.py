"""Pins for the cron store isolation contract (tests/conftest.py step 3b).

cron/jobs.py computes HERMES_DIR/CRON_DIR/JOBS_FILE/TICKER_*_FILE/OUTPUT_DIR
at module import from the LAUNCH environment, and _current_cron_store() falls
back to those baked constants. Before the conftest repin, every test that
created jobs outside use_cron_store() wrote the launch environment's real
store — a live ~/.hermes/cron/jobs.json accumulated 65 pytest fixture jobs
("claim job"/"oneshot"/"paused job") from repeated single-file runs of
test_ticker_stall_60703.py exactly this way. These tests pin the isolation
contract itself so a refactor that reintroduces the leak fails loudly.
"""
import json
import os
from pathlib import Path

import cron.jobs as cron_jobs


def _hermetic_home() -> Path:
    return Path(os.environ["HERMES_HOME"]).resolve()


def test_baked_constants_point_at_the_hermetic_home():
    """Every import-time path constant must resolve under this test's home."""
    home = _hermetic_home()
    assert cron_jobs.HERMES_DIR == home
    assert cron_jobs.CRON_DIR == home / "cron"
    assert cron_jobs.JOBS_FILE == home / "cron" / "jobs.json"
    assert cron_jobs.TICKER_HEARTBEAT_FILE == home / "cron" / "ticker_heartbeat"
    assert cron_jobs.TICKER_SUCCESS_FILE == home / "cron" / "ticker_last_success"
    assert cron_jobs.OUTPUT_DIR == home / "cron" / "output"


def test_unwrapped_create_job_writes_the_hermetic_store():
    """create_job() with no use_cron_store() wrapper lands in the test home.

    This is the exact call shape that leaked into the real store: the
    fallback path of _current_cron_store() must now resolve to the
    per-test home, not the process launch environment.
    """
    job = cron_jobs.create_job(name="isolation probe", schedule="0 7 * * *", prompt="x")
    jobs_file = _hermetic_home() / "cron" / "jobs.json"
    assert jobs_file.exists(), (
        "create_job() outside use_cron_store() must write the hermetic home"
    )
    data = json.loads(jobs_file.read_text())
    jobs = data["jobs"] if isinstance(data, dict) else data
    assert any(j["id"] == job["id"] for j in jobs)
