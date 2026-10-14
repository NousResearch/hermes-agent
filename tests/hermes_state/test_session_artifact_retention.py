"""Session-owned recovery artifacts follow the SessionDB lifecycle."""

from __future__ import annotations

import json
import time

from hermes_state import SessionDB


def _write_artifacts(home, session_id: str, token: str):
    sessions_dir = home / "sessions"
    pending_dir = home / "pending_messages"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    pending_dir.mkdir(parents=True, exist_ok=True)
    request_dump = sessions_dir / f"request_dump_{session_id}_{token}.json"
    emergency_archive = pending_dir / f"pending-{token}.json"
    request_dump.write_text("{}", encoding="utf-8")
    emergency_archive.write_text(json.dumps({
        "reason": "shutdown-with-unpersisted-agent-history",
        "session_id": session_id,
        "messages": [{"role": "user", "content": token}],
    }), encoding="utf-8")
    return request_dump, emergency_archive


def test_retention_and_explicit_delete_remove_only_owned_session_artifacts(tmp_path):
    home = tmp_path / ".hermes"
    sessions_dir = home / "sessions"
    db = SessionDB(db_path=home / "state.db")
    stale_id, live_id, neighbour_id = "stale", "live", "neighbour"
    for session_id in (stale_id, live_id, neighbour_id):
        db.create_session(session_id, source="cli")
        db.end_session(session_id, end_reason="done")

    stale_artifacts = _write_artifacts(home, stale_id, "stale")
    live_artifacts = _write_artifacts(home, live_id, "live")
    neighbour_artifacts = _write_artifacts(home, neighbour_id, "neighbour")
    old = time.time() - 10 * 86400
    db._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, last_activity_at = ? WHERE id = ?",
        (old, old, old, stale_id),
    ))

    assert db.prune_sessions(older_than_days=1, sessions_dir=sessions_dir) == 1
    assert not any(path.exists() for path in stale_artifacts)
    assert all(path.exists() for path in live_artifacts + neighbour_artifacts)

    assert db.delete_session(live_id, sessions_dir=sessions_dir)
    assert not any(path.exists() for path in live_artifacts)
    assert all(path.exists() for path in neighbour_artifacts)
    assert db.get_session(neighbour_id) is not None
    db.close()


def _archive_reads(monkeypatch, pending_dir):
    """Count archive parses: each ``read_text`` of a file under *pending_dir*."""
    import pathlib

    reads = []
    original = pathlib.Path.read_text

    def counting(self, *args, **kwargs):
        if self.parent == pending_dir:
            reads.append(self.name)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", counting)
    return reads


def test_bulk_cleanup_parses_each_archive_once(tmp_path, monkeypatch):
    """Bulk deletes index the spool once per call: N sessions x M archives was N*M parses."""
    home = tmp_path / ".hermes"
    sessions_dir = home / "sessions"
    db = SessionDB(db_path=home / "state.db")
    try:
        doomed = [f"s{n}" for n in range(20)]
        for sid in doomed + ["keep"]:
            db.create_session(sid, source="cli")
            db.end_session(sid, end_reason="done")
        owned = [_write_artifacts(home, sid, f"tok{sid}")[1] for sid in doomed[:5]]
        kept_archive = _write_artifacts(home, "keep", "tokkeep")[1]
        reads = _archive_reads(monkeypatch, home / "pending_messages")

        assert db.delete_sessions(doomed[:10], sessions_dir=sessions_dir) == 10
        assert len(reads) == 6  # 6 archives on disk, each parsed once for 10 sessions
        assert not any(path.exists() for path in owned) and kept_archive.exists()

        reads.clear()
        assert db.delete_empty_sessions(sessions_dir=sessions_dir) == 11  # s10..s19 + keep are empty
        assert len(reads) == 1  # only kept_archive is left, parsed once
        assert not kept_archive.exists()
    finally:
        db.close()


def test_bom_prefixed_archive_is_matched_like_the_spool_reader_does(tmp_path):
    home = tmp_path / ".hermes"
    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session("sess-bom", source="cli")
        _, archive = _write_artifacts(home, "sess-bom", "bom")
        archive.write_bytes(b"\xef\xbb\xbf" + archive.read_bytes())
        assert json.loads(archive.read_text(encoding="utf-8-sig"))["session_id"] == "sess-bom"

        assert db.delete_session("sess-bom", sessions_dir=home / "sessions")
        assert not archive.exists()
    finally:
        db.close()


def test_malformed_or_unowned_archives_are_preserved(tmp_path):
    home = tmp_path / ".hermes"
    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session("target", source="cli")
        _, owned = _write_artifacts(home, "target", "owned")
        pending = home / "pending_messages"
        malformed = pending / "pending-malformed.json"
        malformed.write_text('{"session_id": "target", ', encoding="utf-8")
        undecodable = pending / "pending-undecodable.json"
        undecodable.write_bytes(b'{"session_id": "target", "x": "\xff"}')
        list_payload = pending / "pending-list.json"
        list_payload.write_text(json.dumps([{"session_id": "target"}]), encoding="utf-8")
        prefix_only = pending / "pending-prefix.json"
        prefix_only.write_text(json.dumps({"session_id": "target-2"}), encoding="utf-8")

        assert db.delete_session("target", sessions_dir=home / "sessions")
        assert not owned.exists()
        assert all(p.exists() for p in (malformed, undecodable, list_payload, prefix_only))
    finally:
        db.close()
