"""A failing cron attempt must not store an unbounded failure payload.

Regression: a script-only job that printed a ~70 MB report to stdout and exited
non-zero every 5 minutes stored that whole payload on every attempt, which grew
cron/executions.db to 20 GB and cron/jobs.json to 77 MB. Retention bounds rows,
not bytes, so one noisy job is enough to fill a disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated cron environment with temp HERMES_HOME."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    (hermes_home / "scripts").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")

    return hermes_home


def _point_ledger(monkeypatch, tmp_path):
    import cron.executions as executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    return executions


def test_clip_error_text_leaves_small_text_untouched():
    from cron.executions import clip_error_text

    assert clip_error_text("boom") == "boom"
    assert clip_error_text(None) is None
    assert clip_error_text("") == ""


def test_clip_error_text_keeps_head_and_tail_with_a_marker():
    from cron.executions import MAX_STORED_ERROR_CHARS, clip_error_text

    huge = "HEAD" * 5000 + "MIDDLE" * 20000 + "TAIL" * 5000
    clipped = clip_error_text(huge)

    assert clipped is not None
    assert len(clipped) == MAX_STORED_ERROR_CHARS
    assert clipped.startswith("HEAD")
    assert clipped.endswith("TAIL")
    assert "chars omitted" in clipped


def test_finish_execution_stores_a_bounded_error(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("noisy-job", source="builtin")
    executions.mark_execution_running(record["id"])

    done = executions.finish_execution(record["id"], success=False, error="x" * 5_000_000)

    assert done["status"] == "failed"
    assert done["error"]
    assert len(done["error"]) <= executions.MAX_STORED_ERROR_CHARS


def test_ledger_keeps_small_errors_verbatim(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("small-job", source="builtin")
    executions.mark_execution_running(record["id"])

    done = executions.finish_execution(record["id"], success=False, error="Script exited with code 3")

    assert done["error"] == "Script exited with code 3"


def test_job_record_clips_an_oversized_last_error():
    import cron.jobs as jobs

    job = {"id": "j1", "name": "noisy"}
    jobs._record_run_outcome(
        job, False, "y" * 5_000_000, None, None, "2026-09-13T00:00:00+00:00")

    assert job["last_error"]
    from cron.executions import MAX_STORED_ERROR_CHARS

    assert len(job["last_error"]) <= MAX_STORED_ERROR_CHARS
    assert job["failure_streak"] == 1


def test_script_failure_text_is_bounded(cron_env):
    from cron.scheduler_script import _run_job_script

    script = cron_env / "scripts" / "noisy.py"
    script.write_text(
        "import sys\n"
        "sys.stdout.write('n' * 5_000_000)\n"
        "sys.stdout.flush()\n"
        "sys.stderr.write('e' * 1_000_000)\n"
        "sys.stderr.flush()\n"
        "raise SystemExit(3)\n",
        encoding="utf-8",
    )

    success, output = _run_job_script("noisy.py")

    assert success is False
    assert output
    from cron.executions import MAX_STORED_ERROR_CHARS

    assert len(output) <= MAX_STORED_ERROR_CHARS


def test_script_validation_failure_text_is_bounded():
    from cron.executions import MAX_STORED_ERROR_CHARS
    from cron.scheduler_script import _run_job_script

    success, output = _run_job_script("x" * 20_000 + "\x00")

    assert success is False
    assert len(output) <= MAX_STORED_ERROR_CHARS


def test_script_exception_failure_text_is_bounded(cron_env, monkeypatch):
    from cron.executions import MAX_STORED_ERROR_CHARS
    from cron import scheduler_script

    script = cron_env / "scripts" / "raises-before-start.py"
    script.write_text("print('not reached')\n", encoding="utf-8")

    def fail_to_start(*_args, **_kwargs):
        raise RuntimeError("z" * 20_000)

    monkeypatch.setattr(scheduler_script.subprocess, "Popen", fail_to_start)
    success, output = scheduler_script._run_job_script(script.name)

    assert success is False
    assert len(output) <= MAX_STORED_ERROR_CHARS


def test_script_success_output_is_not_truncated(cron_env):
    """Only FAILURE text is capped; a successful script's output is untouched."""
    from cron.scheduler_script import _run_job_script

    script = cron_env / "scripts" / "chatty.py"
    script.write_text("import sys; sys.stdout.write('ok' * 100_000)\n", encoding="utf-8")

    success, output = _run_job_script("chatty.py")

    assert success is True
    assert len(output) == 200_000
