"""Saved execution receipts cross the real script, delivery and ledger boundaries."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from cron import executions, jobs, scheduler
from gateway.config import GatewayConfig, Platform, PlatformConfig
from tools.cronjob_tools import _latest_job_output_excerpt


@pytest.mark.parametrize("preclaimed", [False, True])
@pytest.mark.parametrize("script_fails", [False, True])
def test_real_script_run_saves_delivers_and_links_attempt(
    tmp_path, monkeypatch, preclaimed, script_fails
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    script = scripts_dir / "receipt.py"
    script.write_text(
        'import sys\nprint("script receipt evidence")\n'
        f"sys.exit({int(script_fails)})\n",
        encoding="utf-8",
    )
    config = GatewayConfig()
    config.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    sent = []

    async def send(platform, pconfig, chat_id, text, **kwargs):
        sent.append(text)
        return {"success": True, "message_id": "receipt"}

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", send)
    job = jobs.create_job(
        prompt="fixture only", schedule="every 1h", script=str(script),
        no_agent=True, deliver="telegram:fixture",
    )
    attempt = None
    if preclaimed:
        attempt = executions.create_execution(job["id"], source="fixture")
        job["execution_id"] = attempt["id"]
    assert scheduler.run_one_job(job) is True
    rows = executions.list_executions(job_id=job["id"])
    assert len(rows) == 1
    row = rows[0]
    if preclaimed:
        assert attempt is not None
        assert row["id"] == attempt["id"]
    assert row["status"] == ("failed" if script_fails else "completed")
    assert row["finished_at"]
    assert row["delivery_outcome"] == "delivered"
    assert len(sent) == 1
    saved = jobs.get_job(job["id"])
    assert saved is not None
    assert saved["last_status"] == ("error" if script_fails else "ok")
    assert saved["last_delivery_error"] is None
    output_file = Path(row["output_path"])
    assert output_file.parent == jobs.get_cron_output_dir() / job["id"]
    assert "script receipt evidence" in output_file.read_text(encoding="utf-8")
    assert "<object" not in output_file.name


@pytest.mark.parametrize("step", [timedelta(days=1), timedelta(microseconds=1)])
def test_timestamp_recency_preserves_retention_and_latest_excerpt(tmp_path, monkeypatch, step):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(jobs, "_cron_output_keep", lambda: 2)
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    paths = []
    for index, execution_id in enumerate(["f" * 32, "b" * 32, "0" * 32]):
        monkeypatch.setattr(executions.uuid, "uuid4", lambda: SimpleNamespace(hex=execution_id))
        attempt = executions.create_execution("recency", source="fixture")
        monkeypatch.setattr(jobs, "_hermes_now", lambda: start + step * index)
        paths.append(jobs.save_job_output("recency", f"result {index}", attempt["id"]))
        assert executions.get_execution(attempt["id"])["output_path"] == str(paths[-1])
        assert _latest_job_output_excerpt("recency") == f"result {index}"
    assert not paths[0].exists()
    assert paths[1].exists() and paths[2].exists()
    assert paths[2].name.startswith((start + step * 2).strftime("%Y-%m-%d_%H-%M-%S"))


@pytest.mark.parametrize("with_execution", [False, True])
def test_same_timestamp_outputs_do_not_overwrite(tmp_path, monkeypatch, with_execution):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    fixed = datetime(2026, 10, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: fixed)
    attempt = executions.create_execution("collision", source="fixture") if with_execution else None
    execution_id = attempt["id"] if attempt else None
    first = jobs.save_job_output("collision", "first", execution_id)
    second = jobs.save_job_output("collision", "second", execution_id)
    assert first != second
    assert first.read_text() == "first"
    assert second.read_text() == "second"
    if attempt:
        assert executions.get_execution(attempt["id"])["output_path"] == str(second)


def test_missing_execution_output_link_warns(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    existing = executions.create_execution("existing", source="fixture")
    with caplog.at_level("WARNING", logger="cron.executions"):
        executions.set_execution_output_path("missing-attempt", "saved.md")
    assert "missing-attempt" in caplog.text
    assert "output" in caplog.text.lower()
    assert executions.get_execution("missing-attempt") is None
    assert executions.get_execution(existing["id"])["output_path"] is None
