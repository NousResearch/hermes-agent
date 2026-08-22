"""``sessions import --from hermes`` — restoring Hermes's own export.

``sessions export --format jsonl`` is documented as a backup ("Suitable for
writing to a JSONL file for backup/analysis") and ``SessionDB.import_sessions``
already restores exactly that payload — but the only caller was the dashboard's
HTTP endpoint, so from a terminal the export was a one-way door.

These tests drive the real CLI and two real ``SessionDB`` stores; the restore
path is never mocked.
"""

import json
import sys
from pathlib import Path

import pytest

from hermes_cli.foreign_sessions import (
    import_hermes_export,
    looks_like_hermes_export,
    read_hermes_export,
)
from hermes_state import SessionDB

SID = "20260817_140000_native"


def _store(tmp_path, name):
    return SessionDB(db_path=tmp_path / name / "state.db")


def _seed(db, sid=SID, title="Merkle notes"):
    db.create_session(sid, source="cli", cwd="/w/proj")
    db.append_message(sid, "user", "what is a Merkle tree")
    db.append_message(sid, "assistant", "A hash tree where each leaf is a data hash.")
    db.set_session_title(sid, title)  # titles are unique per store
    db.update_session_cwd(sid, "/w/proj", git_branch="main", git_repo_root="/w/proj")
    return sid


def _export_file(db, tmp_path, sid=SID, name="backup.jsonl"):
    path = tmp_path / name
    path.write_text(json.dumps(db.export_session(sid)) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def exported(tmp_path):
    """A populated source store and the backup file taken from it."""
    src = _store(tmp_path, "src")
    try:
        _seed(src)
        yield _export_file(src, tmp_path)
    finally:
        src.close()


class TestRestore:
    def test_a_backup_restores_into_an_empty_store(self, tmp_path, exported):
        """The disaster-recovery case: nothing here yet, bring it back."""
        dst = _store(tmp_path, "dst")
        try:
            result = import_hermes_export(exported, db=dst)

            assert result["ok"] and result["imported"] == 1
            row = dst.get_session(SID) or {}
            assert row.get("id") == SID, "the session id was not preserved"
            assert row.get("title") == "Merkle notes"
            assert row.get("cwd") == "/w/proj"
            assert row.get("git_branch") == "main"
            assert [
                (m["role"], m["content"])
                for m in dst.get_messages_as_conversation(SID)
            ] == [
                ("user", "what is a Merkle tree"),
                ("assistant", "A hash tree where each leaf is a data hash."),
            ]
        finally:
            dst.close()

    def test_reimport_skips_and_never_overwrites(self, tmp_path, exported):
        """A second run must not clobber a session that moved on since export."""
        dst = _store(tmp_path, "dst")
        try:
            import_hermes_export(exported, db=dst)
            dst.append_message(SID, "user", "a turn added after the backup")

            result = import_hermes_export(exported, db=dst)

            assert result["ok"]
            assert result["imported"] == 0 and result["skipped"] == 1
            assert len(dst.get_messages_as_conversation(SID)) == 3, (
                "re-importing overwrote history recorded after the backup"
            )
        finally:
            dst.close()

    def test_several_sessions_in_one_file(self, tmp_path):
        src = _store(tmp_path, "src")
        dst = _store(tmp_path, "dst")
        try:
            ids = [
                _seed(src, f"20260817_1500{i:02d}_multi", title=f"note {i}")
                for i in range(3)
            ]
            path = tmp_path / "all.jsonl"
            path.write_text(
                "".join(json.dumps(src.export_session(i)) + "\n" for i in ids),
                encoding="utf-8",
            )

            result = import_hermes_export(path, db=dst)

            assert result["imported"] == 3
            assert sorted(result["imported_ids"]) == sorted(ids)
        finally:
            src.close()
            dst.close()


class TestFileShapes:
    def test_jsonl_one_object_per_line(self, tmp_path, exported):
        assert len(read_hermes_export(exported)) == 1

    def test_a_whole_file_array_is_accepted(self, tmp_path, exported):
        """The shape the dashboard's import endpoint takes."""
        arr = tmp_path / "arr.json"
        arr.write_text(json.dumps(read_hermes_export(exported)), encoding="utf-8")
        assert len(read_hermes_export(arr)) == 1

    def test_a_single_object_is_accepted(self, tmp_path, exported):
        one = tmp_path / "one.json"
        one.write_text(json.dumps(read_hermes_export(exported)[0]), encoding="utf-8")
        assert len(read_hermes_export(one)) == 1

    def test_an_empty_file_is_empty_not_an_error(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("   \n\n", encoding="utf-8")
        assert read_hermes_export(p) == []

    def test_a_malformed_line_names_itself(self, tmp_path, exported):
        broken = tmp_path / "broken.jsonl"
        broken.write_text(
            exported.read_text(encoding="utf-8") + "{not json\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match=r":2:"):
            read_hermes_export(broken)


class TestSourceDetection:
    def test_a_hermes_export_is_recognised(self, exported):
        assert looks_like_hermes_export(exported)

    def test_a_claude_transcript_is_not(self, tmp_path):
        """Claude Code lines have `type`/`message`, never a top-level `messages`."""
        p = tmp_path / "session.jsonl"
        p.write_text(
            json.dumps({
                "type": "user", "sessionId": "abc",
                "message": {"role": "user", "content": "hi"},
            }) + "\n",
            encoding="utf-8",
        )
        assert not looks_like_hermes_export(p)

    def test_a_missing_file_is_not(self, tmp_path):
        assert not looks_like_hermes_export(tmp_path / "nope.jsonl")

    def test_binary_garbage_is_not(self, tmp_path):
        p = tmp_path / "x.jsonl"
        p.write_bytes(b"\x00\x01\x02not json at all\n")
        assert not looks_like_hermes_export(p)


class TestRejection:
    def test_a_foreign_transcript_names_the_right_flag(self, tmp_path):
        """Wrong --from must not half-import; it must say what to use."""
        p = tmp_path / "claude.jsonl"
        p.write_text(
            json.dumps({"type": "user", "message": {"role": "user", "content": "hi"}})
            + "\n",
            encoding="utf-8",
        )
        dst = _store(tmp_path, "dst")
        try:
            with pytest.raises(ValueError, match=r"--from claude\|codex"):
                import_hermes_export(p, db=dst)
        finally:
            dst.close()

    def test_a_missing_file_is_reported(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            import_hermes_export(tmp_path / "nope.jsonl")

    def test_an_empty_export_is_reported(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("\n", encoding="utf-8")
        with pytest.raises(ValueError, match="No session records"):
            import_hermes_export(p)


class TestCli:
    """The surface a user actually types."""

    def _run(self, monkeypatch, argv):
        import hermes_cli.main as main_mod

        monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "import", *argv])
        main_mod.main()

    def test_end_to_end_restore(self, monkeypatch, tmp_path, capsys):
        db = SessionDB()
        try:
            _seed(db)
            backup = _export_file(db, tmp_path)
            db.delete_session(SID)
            assert db.get_session(SID) is None, "precondition: session is gone"
        finally:
            db.close()

        self._run(monkeypatch, ["--from", "hermes", str(backup)])

        out = capsys.readouterr().out
        assert "Imported 1 session" in out, out
        assert f"hermes --resume {SID}" in out
        db = SessionDB()
        try:
            assert (db.get_session(SID) or {}).get("title") == "Merkle notes"
        finally:
            db.close()

    def test_content_beats_a_misleading_path(self, monkeypatch, tmp_path, capsys):
        """A user-named export can easily land on a claude-looking path."""
        db = SessionDB()
        try:
            _seed(db)
            backup = _export_file(db, tmp_path, name="claude-notes.jsonl")
            db.delete_session(SID)
        finally:
            db.close()

        self._run(monkeypatch, [str(backup)])

        assert "Imported 1 session" in capsys.readouterr().out

    def test_from_hermes_without_a_path_says_so(self, monkeypatch, capsys):
        """There is no store to scan, so the picker must not be offered."""
        self._run(monkeypatch, ["--from", "hermes"])
        out = capsys.readouterr().out
        assert "needs the export file" in out, out

    def test_a_bad_file_writes_nothing(self, monkeypatch, tmp_path, capsys):
        p = tmp_path / "junk.jsonl"
        p.write_text(json.dumps({"type": "user"}) + "\n", encoding="utf-8")
        before = len(SessionDB().search_sessions(limit=1000))

        self._run(monkeypatch, ["--from", "hermes", str(p)])

        assert "Error:" in capsys.readouterr().out
        assert len(SessionDB().search_sessions(limit=1000)) == before
