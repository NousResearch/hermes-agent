"""A failed model connection does not undo scripts already run by the cron fire."""

from datetime import datetime, timezone, timedelta
from pathlib import Path

import httpx
import pytest


@pytest.mark.parametrize("script_field", ["script", "monitor_script", None])
def test_only_unexecuted_jobs_get_connection_retry(tmp_path, monkeypatch, script_field):
    from cron import scheduler
    from cron.jobs import create_job, get_job, mark_job_run

    home = tmp_path / ".hermes"
    scripts = home / "scripts"
    scripts.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (home / "config.yaml").write_text(
        "model: {default: test}\ncron: {preflight: false}\n", encoding="utf-8"
    )
    counter = scripts / "counter.txt"
    (scripts / "collect.py").write_text(
        "from pathlib import Path\np=Path(__file__).with_name('counter.txt')\n"
        "p.write_text(str(int(p.read_text())+1) if p.exists() else '1')\nprint('data ready')\n",
        encoding="utf-8",
    )

    def unavailable(*args):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(scheduler, "_resolve_job_runtime", unavailable)
    job = create_job(
        "Summarize collected data",
        "every 24h",
        **({script_field: "collect.py"} if script_field else {}),
    )
    success, _, _, error = scheduler.run_job(job)
    assert not success and "ConnectError" in error
    assert mark_job_run(
        job["id"], success, error, model_unreachable=bool(job.get("_model_unreachable"))
    )
    after = get_job(job["id"])
    next_in = datetime.fromisoformat(after["next_run_at"]) - datetime.now(timezone.utc)
    if script_field:
        assert counter.read_text(encoding="utf-8") == "1"
        assert not job.get("_model_unreachable")
        assert "unreachable_retry" not in after
        assert next_in > timedelta(hours=23)
    else:
        assert after["unreachable_retry"]["attempt"] == 1
        assert timedelta(0) < next_in < timedelta(minutes=7)
