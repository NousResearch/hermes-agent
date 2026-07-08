import json
from datetime import UTC, datetime

from hermes_telemetry import emit_event, error_fingerprint, stable_hash


def test_emit_event_writes_privacy_preserving_jsonl(tmp_path):
    ts = datetime(2026, 7, 8, 12, 0, tzinfo=UTC)

    path = emit_event(
        "cron_run",
        {
            "job_id_hash": stable_hash("raw-job-id"),
            "prompt": "do not store this",
            "api_key": "sk-secret",
            "safe_count": 3,
        },
        status="ok",
        source="test",
        ts=ts,
        hermes_home=tmp_path,
    )

    assert path == tmp_path / "ops" / "events" / "2026-07-08.jsonl"
    line = path.read_text(encoding="utf-8").strip()
    event = json.loads(line)

    assert event["schema_version"] == 1
    assert event["event_type"] == "cron_run"
    assert event["source"] == "test"
    assert event["status"] == "ok"
    assert event["payload"]["safe_count"] == 3
    assert event["payload"]["prompt"] == "[REDACTED]"
    assert event["payload"]["api_key"] == "[REDACTED]"
    assert "raw-job-id" not in line
    assert "sk-secret" not in line
    assert "do not store this" not in line


def test_error_fingerprint_groups_without_storing_error_text():
    fp1 = error_fingerprint("RuntimeError: provider key failed")
    fp2 = error_fingerprint("RuntimeError: provider key failed")

    assert fp1 == fp2
    assert fp1
    assert "provider" not in fp1
