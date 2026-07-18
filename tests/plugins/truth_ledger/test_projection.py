from __future__ import annotations

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
        "event_id": "e1", "event": "assert", "fact_id": "f1",
        "scope": "user", "subject": "u1", "key": "lang", "value": "en",
        "occurred_at": "2026-07-17T20:00:00Z",
    })

    out = projection_mod.rebuild_current_view(tmp_path)
    assert out["active"] == 1

    current = tmp_path / "views" / "current.jsonl"
    lines = [json.loads(x) for x in current.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) == 1
    assert lines[0]["fact_id"] == "f1"
    assert lines[0]["value"] == "en"


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
