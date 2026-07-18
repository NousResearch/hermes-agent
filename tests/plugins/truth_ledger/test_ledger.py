from __future__ import annotations

import json
from pathlib import Path


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8").strip().splitlines()) if path.read_text(encoding="utf-8").strip() else 0


def test_append_event_is_idempotent_by_event_key(tmp_path, ledger_mod):
    store = ledger_mod.LedgerStore(tmp_path)
    event = {
        "event_id": "evt-1",
        "event": "assert",
        "scope": "user",
        "subject": "u1",
        "key": "pref",
        "value": "dark",
        "occurred_at": "2026-07-17T20:00:00Z",
    }

    first = store.append_event(event=event, event_key="k1")
    second = store.append_event(event=event, event_key="k1")

    assert first["status"] == "indexed"
    assert second["status"] == "duplicate"

    ledger_file = tmp_path / "ledger" / "2026-07.jsonl"
    assert _line_count(ledger_file) == 1


def test_append_event_rejects_oversized_record(tmp_path, ledger_mod):
    store = ledger_mod.LedgerStore(tmp_path, record_hard_bytes=200)
    event = {
        "event_id": "evt-big",
        "event": "assert",
        "scope": "user",
        "subject": "u1",
        "key": "blob",
        "value": "x" * 500,
        "occurred_at": "2026-07-17T20:00:00Z",
    }

    out = store.append_event(event=event, event_key="big-key")
    assert out["status"] == "rejected"
    assert out["reason"] == "record_hard_cap"


def test_scan_quarantines_partial_tail(tmp_path, ledger_mod):
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir(parents=True)
    f = ledger_dir / "2026-07.jsonl"
    f.write_text('{"event_id":"1","event":"assert"}\n{"event_id":"2","event":"assert"}\n{"event_id":"3"', encoding="utf-8")

    parsed = ledger_mod.scan_jsonl_with_tail_quarantine(
        f,
        tmp_path / "errors",
    )

    assert len(parsed) == 2
    quarantined = list((tmp_path / "errors").glob("corrupt-tail-*.jsonl"))
    assert len(quarantined) == 1
    qtxt = quarantined[0].read_text(encoding="utf-8")
    assert "event_id" in qtxt
