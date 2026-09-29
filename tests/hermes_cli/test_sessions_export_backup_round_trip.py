"""`hermes sessions export` JSONL — the documented backup format — must round-trip a compacted
session: import restores the turns in-place compaction archived, still archived (#122679 fixed
only the console export; the primary CLI kept the live-only projection)."""

import json
import sys

import hermes_cli.main as main_mod
from hermes_state import SessionDB

SID = "20260929_120000_abcdef"


def _shape(db, **flags):
    return [(m["role"], m["content"], m.get("active", 1), m.get("compacted", 0))
            for m in db.get_messages(SID, **flags)]


def test_jsonl_backup_round_trips_compaction_archived_turns(tmp_path, monkeypatch):
    src = SessionDB()
    src.create_session(SID, source="cli")
    for i in range(1, 5):
        src.append_message(SID, "user", f"question {i}")
        src.append_message(SID, "assistant", f"answer {i}")
    tail = src.get_messages(SID)[-2:]
    src.archive_and_compact(SID, [{"role": "user", "content": "[summary]"}, *tail], tail_count=2)
    shown, live = _shape(src, include_compacted=True), _shape(src)
    src.close()
    assert len(shown) > len(live)

    backup = tmp_path / "backup.jsonl"
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "export", str(backup)])
    main_mod.main()

    dst = SessionDB(db_path=tmp_path / "restored.db")
    try:
        payload = [json.loads(line) for line in backup.read_text(encoding="utf-8").splitlines() if line]
        assert dst.import_sessions(payload)["ok"]
        assert _shape(dst, include_compacted=True) == shown, "display history must survive the backup"
        assert _shape(dst) == live, "archived turns must not come back as live context"
    finally:
        dst.close()
