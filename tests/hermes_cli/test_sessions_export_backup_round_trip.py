"""`hermes sessions export` JSONL — the documented backup format — must round-trip a compacted
session: import restores the turns in-place compaction archived, still archived (#122679 fixed
only the console export; the primary CLI kept the live-only projection). That backup materializes
every stored row in memory, so it is bounded per session by ``sessions.max_export_messages``, as the
console export is."""

import json
import sys

import pytest

import hermes_cli.main as main_mod
from hermes_state import SessionDB

SID = "20260929_120000_abcdef"


def _shape(db, **flags):
    return [(m["role"], m["content"], m.get("active", 1), m.get("compacted", 0))
            for m in db.get_messages(SID, **flags)]


def _seed_compacted_session() -> SessionDB:
    """8 turns compacted to a summary + 2-row live tail: few live rows, many stored rows."""
    src = SessionDB()
    src.create_session(SID, source="cli")
    for i in range(1, 5):
        src.append_message(SID, "user", f"question {i}")
        src.append_message(SID, "assistant", f"answer {i}")
    tail = src.get_messages(SID)[-2:]
    src.archive_and_compact(SID, [{"role": "user", "content": "[summary]"}, *tail], tail_count=2)
    return src


def _export(monkeypatch, backup, *selection):
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "export", str(backup), *selection])
    main_mod.main()


def test_jsonl_backup_round_trips_compaction_archived_turns(tmp_path, monkeypatch):
    src = _seed_compacted_session()
    shown, live = _shape(src, include_compacted=True), _shape(src)
    src.close()
    assert len(shown) > len(live)

    backup = tmp_path / "backup.jsonl"
    _export(monkeypatch, backup)

    dst = SessionDB(db_path=tmp_path / "restored.db")
    try:
        payload = [json.loads(line) for line in backup.read_text(encoding="utf-8").splitlines() if line]
        assert dst.import_sessions(payload)["ok"]
        assert _shape(dst, include_compacted=True) == shown, "display history must survive the backup"
        assert _shape(dst) == live, "archived turns must not come back as live context"
    finally:
        dst.close()


@pytest.mark.parametrize("selection", [("--session-id", SID), ("--source", "cli"), ()],
                         ids=["session-id", "filter", "bare"])
def test_jsonl_backup_refuses_a_session_over_max_export_messages(tmp_path, monkeypatch, capsys, selection):
    """Every selection path applies the per-session guard to the STORED row count the backup
    materializes, so a small live tail over a large archive cannot slip past it."""
    from hermes_cli.config import load_config, save_config

    src = _seed_compacted_session()
    live = len(_shape(src))
    stored = len(_shape(src, include_inactive=True))
    src.end_session(SID, "user_exit")  # bulk filters match ended sessions
    src.close()
    cfg = load_config()
    cfg.setdefault("sessions", {})["max_export_messages"] = live + 1
    save_config(cfg)
    assert live < live + 1 < stored

    backup = tmp_path / "backup.jsonl"
    _export(monkeypatch, backup, *selection)

    assert not backup.exists()
    assert SID in capsys.readouterr().out
