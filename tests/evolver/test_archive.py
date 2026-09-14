"""Invariant tests for the failure archive (evolver/archive.py).

Regression for the Harness Evolver Phase 0 plan
(frontier-kb: notes/harness-evolver-hermes-plan.md).
"""
import json

import pytest

from evolver.archive import (FAILURE_CLASSES, TraceArchive, make_record,
                             make_span, validate_record)


def _task():
    return {"task_id": "hermes-fix-1", "kind": "bugfix", "summary": "x"}


def _source():
    return {"type": "factory", "path": "/tmp/fix_results/1.json"}


def test_valid_record_passes():
    record = make_record(_task(), [make_span("t", status="ok")],
                         "regression", _source())
    assert validate_record(record) == []


def test_missing_top_level_key_rejected():
    record = make_record(_task(), [make_span("t")], "unknown", _source())
    del record["trace"]
    errors = validate_record(record)
    assert any("trace" in e for e in errors)


def test_unknown_failure_class_rejected():
    record = make_record(_task(), [make_span("t")], "unknown", _source())
    record["failure_class"] = "mystery"
    errors = validate_record(record)
    assert any("failure_class" in e for e in errors)
    # every documented class is accepted
    for cls in FAILURE_CLASSES:
        record["failure_class"] = cls
        assert validate_record(record) == [], cls


def test_empty_spans_rejected():
    record = make_record(_task(), [make_span("t")], "unknown", _source())
    record["trace"]["spans"] = []
    assert any("spans" in e for e in validate_record(record))


def test_make_record_rejects_invalid_failure_class():
    with pytest.raises(ValueError):
        make_record(_task(), [make_span("t")], "mystery", _source())


def test_make_span_records_duration():
    from datetime import datetime
    span = make_span("slow", status="error", duration_s=90.0)
    assert span["attributes"]["duration_s"] == "90.0"
    started = datetime.fromisoformat(span["started_at"])
    ended = datetime.fromisoformat(span["ended_at"])
    assert (ended - started).total_seconds() >= 90.0


def test_archive_round_trip_jsonl(tmp_path):
    archive = TraceArchive(tmp_path / "a.jsonl")
    record = make_record(_task(), [make_span("t", status="error")],
                         "test_failure", _source(),
                         outcome={"status": "failed"})
    rid = archive.append(record)
    assert rid == record["record_id"]
    # one JSON object per line, reloadable
    lines = (tmp_path / "a.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["record_id"] == rid
    assert archive.count() == 1


def test_archive_filters_and_skips_bad_lines(tmp_path):
    archive = TraceArchive(tmp_path / "a.jsonl")
    archive.append(make_record(_task(), [make_span("t")], "regression",
                               _source()))
    archive.append(make_record({**_task(), "task_id": "other",
                                "kind": "trap"}, [make_span("t")],
                               "timeout", _source()))
    with open(tmp_path / "a.jsonl", "a", encoding="utf-8") as f:
        f.write("not json at all\n")
    assert archive.count() == 2  # bad line skipped, not fatal
    assert sum(1 for _ in archive.iter_records(failure_class="timeout")) == 1
    assert sum(1 for _ in archive.iter_records(task_kind="trap")) == 1


def test_append_rejects_invalid_record(tmp_path):
    archive = TraceArchive(tmp_path / "a.jsonl")
    with pytest.raises(ValueError):
        archive.append({"not": "a record"})
    assert archive.count() == 0
