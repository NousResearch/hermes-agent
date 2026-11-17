"""Round-trip export/import must restore archived message state (#122679).

The JSONL transfer projection feeds ``import_sessions``: rows in-place compaction
archived must come back archived — visible in the display history, never as live
model context — or a restored compacted session silently loses every turn the
summary replaced. ``active``/``compacted`` are the row state; absent flags (older
exports) keep meaning live rows.
"""

from __future__ import annotations

import json

from hermes_state import SessionDB

STRANDED_ID = "20260823_043331_c93770"


def _seed_and_compact(db, turns=4):
    db.create_session(STRANDED_ID, source="tui")
    db.set_session_title(STRANDED_ID, "Bot Chat")
    for i in range(1, turns + 1):
        db.append_message(STRANDED_ID, "user", f"question {i}")
        db.append_message(STRANDED_ID, "assistant", f"answer {i}")
    tail = db.get_messages(STRANDED_ID)[-2:]
    db.archive_and_compact(
        STRANDED_ID, [{"role": "user", "content": "[summary of turns 1-3]"}, *tail], tail_count=2)


def _shape(db, **flags):
    return [(m["role"], m["content"], m.get("active", 1), m.get("compacted", 0))
            for m in db.get_messages(STRANDED_ID, **flags)]


def test_export_all_round_trips_compacted_history(tmp_path):
    src = SessionDB(db_path=tmp_path / "src.db")
    dst = SessionDB(db_path=tmp_path / "dst.db")
    try:
        _seed_and_compact(src)
        shown_before = _shape(src, include_compacted=True)
        live_before = _shape(src)
        all_before = _shape(src, include_inactive=True)
        assert len(shown_before) > len(live_before), "compacted turns must be archived, not deleted"

        # The transfer projection through real JSONL (what `sessions export` writes).
        payload = json.loads(json.dumps(src.export_all(include_inactive=True)))

        result = dst.import_sessions(payload)

        assert result["ok"] and result["imported"] == 1, result
        assert _shape(dst, include_inactive=True) == all_before, "every row must survive with its state"
        assert _shape(dst, include_compacted=True) == shown_before, "display history must be unchanged"
        assert _shape(dst) == live_before, "archived turns must not re-enter live model context"
        # Session counters count live rows only.
        assert dst.get_session(STRANDED_ID)["message_count"] == len(live_before)

        # Hand-edited/foreign JSONL: string flags ("0" is truthy) and null for live rows must not
        # flip archived rows live or live rows archived.
        for msg in payload[0]["messages"]:
            msg["active"], msg["compacted"] = (str(msg["active"]) if not msg["active"] else None,
                                               str(msg["compacted"]))
        edited = SessionDB(db_path=tmp_path / "edited.db")
        try:
            assert edited.import_sessions(payload)["ok"]
            assert _shape(edited, include_inactive=True) == all_before
        finally:
            edited.close()
    finally:
        src.close()
        dst.close()
