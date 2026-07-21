from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _append(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))
        fh.write("\n")


def test_rebuild_current_view_handles_assert_supersede_retract(tmp_path, projection_mod):
    ledger_file = tmp_path / "ledger" / "2026-07.jsonl"
    _append(ledger_file, {
        "event_id": "e1", "event": "assert", "fact_id": "f1",
        "scope": "user", "subject": "u1", "key": "timezone", "value": "UTC",
        "occurred_at": "2026-07-17T20:00:00Z",
    })
    _append(ledger_file, {
        "event_id": "e2", "event": "supersede", "fact_id": "f2", "supersedes": "f1",
        "scope": "user", "subject": "u1", "key": "timezone", "value": "PST",
        "occurred_at": "2026-07-17T20:01:00Z",
    })
    _append(ledger_file, {
        "event_id": "e3", "event": "retract", "fact_id": "f3", "retracts": "f2",
        "scope": "user", "subject": "u1", "key": "timezone",
        "occurred_at": "2026-07-17T20:02:00Z",
    })

    out = projection_mod.rebuild_current_view(tmp_path)
    assert out["applied"] == 3
    assert out["active"] == 0

    current = tmp_path / "views" / "current.jsonl"
    assert current.exists()
    assert current.read_text(encoding="utf-8") == ""


def test_rebuild_current_view_atomic_write(tmp_path, projection_mod):
    ledger_file = tmp_path / "ledger" / "2026-07.jsonl"
    _append(ledger_file, {
        "event_id": "e1", "event": "assert", "fact_id": "fact_1",
        "scope": "user", "subject": "u1", "key": "lang", "value": "en",
        "occurred_at": "2026-07-17T20:00:00Z",
    })

    out = projection_mod.rebuild_current_view(tmp_path)
    assert out["active"] == 1

    current = tmp_path / "views" / "current.jsonl"
    lines = [json.loads(x) for x in current.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) == 1
    assert lines[0]["fact_id"] == "fact_1"
    assert lines[0]["value"] == "en"
    assert lines[0]["schema_name"] == "truth-ledger.current-projection.v1"
    assert lines[0]["logical_key"] == {"scope": "user", "subject": "u1", "key": "lang"}
    assert lines[0]["state"] == "active"
    assert lines[0]["updated_at"] == "2026-07-17T20:00:00Z"


def test_rebuild_current_view_writes_records_that_validate_against_projection_schema(
    tmp_path, projection_mod
):
    ledger_file = tmp_path / "ledger" / "2026-07.jsonl"
    _append(ledger_file, {
        "event_id": "evt_1", "operation": "assert", "fact_id": "fact_1",
        "fact": {"scope": "user", "subject": "u1", "key": "lang", "value": "en"},
        "occurred_at": "2026-07-17T20:00:00Z",
    })

    out = projection_mod.rebuild_current_view(tmp_path)

    rows = [json.loads(line) for line in Path(out["path"]).read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    projection_mod.validate_document("current-projection.v1", rows[0])


def test_rebuild_current_view_accepts_operation_field_and_retract_supersedes(tmp_path, projection_mod):
    ledger_file = tmp_path / "ledger" / "2026-07.jsonl"
    _append(
        ledger_file,
        {
            "event_id": "e1",
            "operation": "assert",
            "fact_id": "fact_1",
            "scope": "user",
            "subject": "u1",
            "key": "timezone",
            "value": "UTC",
            "occurred_at": "2026-07-17T20:00:00Z",
        },
    )
    _append(
        ledger_file,
        {
            "event_id": "e2",
            "operation": "retract",
            "fact_id": "fact_2",
            "supersedes": "fact_1",
            "scope": "user",
            "subject": "u1",
            "key": "timezone",
            "occurred_at": "2026-07-17T20:01:00Z",
        },
    )

    out = projection_mod.rebuild_current_view(tmp_path)
    assert out["applied"] == 2
    assert out["active"] == 0


def test_rebuild_current_view_accepts_operation_field_for_assert(tmp_path, projection_mod):
    ledger_file = tmp_path / "ledger" / "2026-07.jsonl"
    _append(
        ledger_file,
        {
            "event_id": "e1",
            "operation": "assert",
            "fact_id": "fact_1",
            "scope": "user",
            "subject": "u1",
            "key": "timezone",
            "value": "UTC",
            "occurred_at": "2026-07-17T20:00:00Z",
        },
    )

    out = projection_mod.rebuild_current_view(tmp_path)
    assert out["applied"] == 1
    assert out["active"] == 1


def test_rebuild_current_view_supports_reconciliation_nested_fact_shape(tmp_path, projection_mod):
    ledger_file = tmp_path / "ledger" / "2026-07.jsonl"
    _append(
        ledger_file,
        {
            "event_id": "e1",
            "operation": "assert",
            "fact_id": "fact_1",
            "fact": {
                "scope": "user",
                "subject": "u1",
                "key": "timezone",
                "value": "UTC",
            },
            "occurred_at": "2026-07-17T20:00:00Z",
        },
    )
    _append(
        ledger_file,
        {
            "event_id": "e2",
            "operation": "assert",
            "fact_id": "fact_2",
            "fact": {
                "scope": "user",
                "subject": "u1",
                "key": "language",
                "value": "en",
            },
            "occurred_at": "2026-07-17T20:01:00Z",
        },
    )

    out = projection_mod.rebuild_current_view(tmp_path)
    assert out["applied"] == 2
    assert out["active"] == 2


def test_rebuild_current_view_reports_and_quarantines_invalid_source_lines(tmp_path, projection_mod):
    ledger_file = tmp_path / "ledger" / "2026-07.jsonl"
    ledger_file.parent.mkdir(parents=True, exist_ok=True)
    ledger_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_id": "e1",
                        "operation": "assert",
                        "fact_id": "fact_1",
                        "fact": {
                            "scope": "user",
                            "subject": "u1",
                            "key": "timezone",
                            "value": "UTC",
                        },
                        "occurred_at": "2026-07-17T20:00:00Z",
                    },
                    separators=(",", ":"),
                ),
                '{"event_id":"bad"',
            ]
        ),
        encoding="utf-8",
    )

    out = projection_mod.rebuild_current_view(tmp_path)
    assert out["applied"] == 1
    assert out["active"] == 1
    assert out["invalid_source_records"] == 1
    assert out["quarantined_files"] == 1


def test_rebuild_skips_generated_projection_records_that_fail_schema_validation(tmp_path, projection_mod):
    ledger_file = tmp_path / "ledger" / "2026-07.jsonl"
    _append(ledger_file, {
        "event_id": "evt_bad", "operation": "assert", "fact_id": "fact_bad",
        "fact": {"scope": "user", "subject": "u1", "key": "", "value": "en"},
        "occurred_at": "2026-07-17T20:00:00Z",
    })

    out = projection_mod.rebuild_current_view(tmp_path)

    assert out["active"] == 0
    assert out["invalid_source_records"] == 1
    assert Path(out["path"]).read_text(encoding="utf-8") == ""


def test_rebuild_current_view_is_deterministic_after_derived_state_deletion(tmp_path, projection_mod):
    ledger_file = tmp_path / "ledger" / "2026-07.jsonl"
    _append(
        ledger_file,
        {
            "event_id": "e1",
            "operation": "assert",
            "fact_id": "fact_1",
            "fact": {
                "scope": "user",
                "subject": "u1",
                "key": "timezone",
                "value": "UTC",
            },
            "occurred_at": "2026-07-17T20:00:00Z",
        },
    )
    _append(
        ledger_file,
        {
            "event_id": "e2",
            "operation": "confirm",
            "fact_id": "fact_1",
            "fact": {
                "scope": "user",
                "subject": "u1",
                "key": "timezone",
                "value": "UTC",
            },
            "occurred_at": "2026-07-17T20:01:00Z",
        },
    )

    first = projection_mod.rebuild_current_view(tmp_path)
    hash_one = hashlib.sha256(Path(first["path"]).read_bytes()).hexdigest()

    Path(first["path"]).unlink()
    second = projection_mod.rebuild_current_view(tmp_path)
    hash_two = hashlib.sha256(Path(second["path"]).read_bytes()).hexdigest()

    assert hash_one == hash_two
