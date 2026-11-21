"""`hermes sessions export` JSONL — the documented backup format — must round-trip a compacted
session: import restores the turns in-place compaction archived, still archived (#122679 fixed
only the console export; the primary CLI kept the live-only projection). That backup materializes
every stored row in memory, so it is bounded per session by ``sessions.max_export_messages``, as the
console export is."""

import json
import sys

import pytest

import hermes_cli.main as main_mod
from persistence.conversation.messages import NewMessage, TranscriptRead, TranscriptView
from persistence_assembly import Persistence, open_persistence, profile_scope
from tests.fixtures.storage_read import transcript_rows
from tests.fixtures.storage_seed import seed_messages, seed_session

SID = "20260929_120000_abcdef"


def _shape(db, view=TranscriptView.ACTIVE):
    return [(m["role"], m["content"], m.get("active", 1), m.get("compacted", 0))
            for m in transcript_rows(db, SID, TranscriptRead(view=view))]


def _seed_compacted_session() -> Persistence:
    """8 turns compacted to a summary + 2-row live tail: few live rows, many stored rows."""
    src = open_persistence(profile_scope())
    seed_session(src, SID, "cli")
    for i in range(1, 5):
        seed_messages(src, SID, {"role": "user", "content": f"question {i}"},
                      {"role": "assistant", "content": f"answer {i}"})
    tail = transcript_rows(src, SID)[-2:]
    src.messages.compact(SID, [NewMessage.from_live(m) for m in ({"role": "user", "content": "[summary]"}, *tail)],
                         carried_tail_count=2)
    return src


def _export(monkeypatch, backup, *selection):
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "export", str(backup), *selection])
    main_mod.main()


def test_jsonl_backup_round_trips_compaction_archived_turns(tmp_path, monkeypatch):
    src = _seed_compacted_session()
    shown, live = _shape(src, TranscriptView.DISPLAY), _shape(src)
    src.close()
    assert len(shown) > len(live)

    backup = tmp_path / "backup.jsonl"
    _export(monkeypatch, backup)

    dst = open_persistence(profile_scope(home=tmp_path / "restored"))
    try:
        payload = [json.loads(line) for line in backup.read_text(encoding="utf-8-sig").splitlines() if line]
        assert dst.maintenance.import_sessions(payload).ok
        assert _shape(dst, TranscriptView.DISPLAY) == shown, "display history must survive the backup"
        assert _shape(dst) == live, "archived turns must not come back as live context"
    finally:
        dst.close()


@pytest.mark.parametrize("selection, disabled",
                         [(("--session-id", SID), False), (("--source", "cli"), False), ((), False), ((), True)],
                         ids=["session-id", "filter", "bare", "limit-0"])
def test_jsonl_backup_refuses_a_session_over_max_export_messages(tmp_path, monkeypatch, capsys, selection,
                                                                 disabled):
    """Every selection path applies the per-session guard to the STORED row count the backup
    materializes, so a small live tail over a large archive cannot slip past it. ``limit-0``: a
    disabled guard writes the backup without the guard's own full-session scan (export_all's is
    the only one)."""
    from hermes_cli.config import load_config, save_config

    src = _seed_compacted_session()
    live = len(_shape(src))
    stored = len(_shape(src, TranscriptView.ALL))
    src.sessions.end(SID, "user_exit")  # bulk filters match ended sessions
    src.close()
    cfg = load_config()
    cfg.setdefault("sessions", {})["max_export_messages"] = 0 if disabled else live + 1
    save_config(cfg)
    assert stored > live + 1
    from persistence.sqlite.conversation import sessions as session_repository
    from persistence.sqlite.housekeeping import _portability
    scans = []
    real_search = session_repository.search_sessions
    for module in (session_repository, _portability):
        monkeypatch.setattr(module, "search_sessions",
                            lambda *a, **kw: scans.append(1) or real_search(*a, **kw))

    backup = tmp_path / "backup.jsonl"
    if disabled:
        _export(monkeypatch, backup, *selection)
        assert backup.exists() and len(scans) == 1
        return
    with pytest.raises(SystemExit) as refused:
        _export(monkeypatch, backup, *selection)
    assert refused.value.code == 1
    assert not backup.exists()
    out = capsys.readouterr().out
    assert SID in out and "max_export_messages" in out
