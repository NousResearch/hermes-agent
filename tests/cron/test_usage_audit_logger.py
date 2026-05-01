"""Tests for the Phase 0.5 cron usage_audit.jsonl logger.

Covers:
- successful write produces a single valid JSONL line with full schema
- missing token info still writes a line with null fields
- writer exception is swallowed (json.dumps raises) — call must return cleanly
- file path is created if parent dir is missing
- timestamp format is RFC3339 UTC with millisecond precision and 'Z' suffix
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from cron import scheduler


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    """Redirect Path.home() so the audit logger writes under tmp_path."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    return fake_home


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class TestUsageAuditPath:
    def test_resolves_under_user_home(self, tmp_home):
        p = scheduler._usage_audit_path()
        assert p == tmp_home / ".hermes" / "cron" / "usage_audit.jsonl"


class TestUtcnowIsoMs:
    def test_format_has_millisecond_precision_and_z(self):
        ts = scheduler._utcnow_iso_ms()
        # YYYY-MM-DDTHH:MM:SS.mmmZ
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$", ts), ts


class TestWriteUsageAudit:
    def test_successful_write_produces_valid_jsonl(self, tmp_home):
        record = {
            "ts": "2026-05-01T04:23:11.123Z",
            "job_id": "bluenode-dispatch-recommend-sweep",
            "fire_id": "deadbeefcafe",
            "prompt_tokens": 11894,
            "completion_tokens": 287,
            "total_tokens": 12181,
            "response_silent": False,
            "deliver_target": None,
            "model": "google/gemma-4-31b-it",
            "duration_ms": 4231,
            "error": None,
        }
        scheduler._write_usage_audit(record)

        path = scheduler._usage_audit_path()
        assert path.exists()
        lines = _read_jsonl(path)
        assert len(lines) == 1
        assert lines[0] == record

    def test_missing_token_info_writes_line_with_null_fields(self, tmp_home):
        record = {
            "ts": "2026-05-01T04:23:11.123Z",
            "job_id": "j",
            "fire_id": "f",
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "response_silent": True,
            "deliver_target": "telegram",
            "model": None,
            "duration_ms": 12,
            "error": "boom",
        }
        scheduler._write_usage_audit(record)
        lines = _read_jsonl(scheduler._usage_audit_path())
        assert len(lines) == 1
        assert lines[0]["prompt_tokens"] is None
        assert lines[0]["completion_tokens"] is None
        assert lines[0]["total_tokens"] is None
        assert lines[0]["error"] == "boom"

    def test_writer_exception_swallowed(self, tmp_home, caplog):
        # Force json.dumps to raise — writer must NOT propagate.
        with patch("cron.scheduler.json.dumps", side_effect=RuntimeError("kaboom")):
            scheduler._write_usage_audit({"job_id": "x"})

        # File never created.
        assert not scheduler._usage_audit_path().exists()
        # Warning logged with our marker.
        assert any("usage_audit write failed" in rec.message for rec in caplog.records)

    def test_parent_dir_created_if_missing(self, tmp_home):
        # Ensure the .hermes/cron path does not exist yet.
        target = tmp_home / ".hermes" / "cron"
        assert not target.exists()

        scheduler._write_usage_audit({"k": "v"})

        assert target.exists() and target.is_dir()
        assert (target / "usage_audit.jsonl").exists()

    def test_appends_multiple_records(self, tmp_home):
        scheduler._write_usage_audit({"i": 1})
        scheduler._write_usage_audit({"i": 2})
        scheduler._write_usage_audit({"i": 3})
        lines = _read_jsonl(scheduler._usage_audit_path())
        assert [r["i"] for r in lines] == [1, 2, 3]

    def test_unicode_preserved_not_escaped(self, tmp_home):
        # ensure_ascii=False so non-ASCII model names / job names round-trip cleanly.
        scheduler._write_usage_audit({"job_id": "한글", "model": "gemma"})
        text = scheduler._usage_audit_path().read_text(encoding="utf-8")
        assert "한글" in text
