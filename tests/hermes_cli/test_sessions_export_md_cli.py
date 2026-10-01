import sys

import pytest


def test_sessions_export_md_writes_single_session(monkeypatch, tmp_path, capsys):
    import hermes_cli.main as main_mod
    import hermes_state

    captured = {}

    class FakeDB:
        def resolve_session_id(self, session_id):
            captured["resolved_from"] = session_id
            return "20260706_123456_abcd1234"

        def export_session(self, session_id, include_compacted=False):
            captured["exported"] = session_id
            return {
                "id": session_id,
                "title": "Export CLI Test",
                "source": "cli",
                "message_count": 1,
                "messages": [{"role": "user", "content": "hello"}],
            }

        def delete_session(self, *args, **kwargs):
            raise AssertionError("markdown export must not delete sessions")

        def prune_sessions(self, *args, **kwargs):
            raise AssertionError("markdown export must not prune sessions")

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(hermes_state, "SessionDB", lambda *args, **kwargs: FakeDB())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes",
            "sessions",
            "export",
            "--format",
            "md",
            "--session-id",
            "20260706_123456",
            str(tmp_path),
        ],
    )

    main_mod.main()

    output = capsys.readouterr().out
    files = list(tmp_path.glob("*.md"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert "# Export CLI Test" in text
    assert "hello" in text
    assert captured == {
        "resolved_from": "20260706_123456",
        "exported": "20260706_123456_abcd1234",
        "closed": True,
    }
    assert str(files[0]) in output


def test_sessions_export_redact_scrubs_secrets(monkeypatch, tmp_path):
    """--redact runs exported content through force-mode secret redaction."""
    import hermes_cli.main as main_mod
    import hermes_state

    secret = "sk-proj-Zz12345678901234567890123456789012345678"

    class FakeDB:
        def resolve_session_id(self, session_id):
            return "s1"

        def export_session(self, session_id, include_compacted=False):
            return {
                "id": "s1",
                "title": "Redact",
                "messages": [
                    {"role": "tool", "name": "terminal", "content": f"api key: {secret}"}
                ],
            }

        def close(self):
            pass

    monkeypatch.setattr(hermes_state, "SessionDB", lambda *args, **kwargs: FakeDB())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes", "sessions", "export", "--format", "md",
            "--session-id", "s1", "--redact", str(tmp_path),
        ],
    )

    main_mod.main()

    text = next(tmp_path.glob("*.md")).read_text(encoding="utf-8")
    assert secret not in text
    assert "api key:" in text


def _real_store(monkeypatch, tmp_path):
    import hermes_state

    real_session_db = hermes_state.SessionDB
    db_path = tmp_path / "state.db"

    class StoreAtTmp(real_session_db):
        def __init__(self, *args, **kwargs):
            super().__init__(db_path=db_path)

    monkeypatch.setattr(hermes_state, "SessionDB", StoreAtTmp)
    return StoreAtTmp


def _seed_six_turns(open_db, session_id, *, compact):
    db = open_db()
    try:
        db.create_session(session_id, "cli")
        for i in range(1, 7):
            db.append_message(session_id, "user", f"question {i}")
            db.append_message(session_id, "assistant", f"answer {i}")
        if compact:
            watermark = db.get_active_message_watermark(session_id)
            tail = [{"role": "user", "content": "question 6"},
                    {"role": "assistant", "content": "answer 6"}]
            db.archive_and_compact(
                session_id, [{"role": "user", "content": "[CONTEXT COMPACTION] summary"}, *tail],
                watermark=watermark, tail_count=len(tail),
            )
    finally:
        db.close()


def _export_delete(monkeypatch, out_dir, session_id, *extra):
    import hermes_cli.main as main_mod

    monkeypatch.setattr(sys, "argv", [
        "hermes", "sessions", "export", "--format", "md", "--session-id", session_id,
        "--delete-after-verified", "--yes", *extra, str(out_dir),
    ])
    main_mod.main()


@pytest.mark.parametrize("lineage", ["single", "logical"])
def test_delete_after_verified_exports_compacted_display_history(monkeypatch, tmp_path, capsys, lineage):
    open_db = _real_store(monkeypatch, tmp_path)
    _seed_six_turns(open_db, "s1", compact=True)

    _export_delete(monkeypatch, tmp_path / "out", "s1", "--lineage", lineage)

    text = next((tmp_path / "out").glob("*.md")).read_text(encoding="utf-8")
    assert [f"answer {i}" in text for i in range(1, 7)] == [True] * 6
    assert "Deleted exported session 's1'." in capsys.readouterr().out
    db = open_db()
    try:
        assert db.get_session("s1") is None
    finally:
        db.close()


def test_delete_after_verified_rejects_same_count_content_change(monkeypatch, tmp_path, capsys):
    """A content rewrite is a real concurrent write that a count-only guard cannot see."""
    import hermes_cli.session_export_md as session_export_md

    open_db = _real_store(monkeypatch, tmp_path)
    _seed_six_turns(open_db, "s1", compact=False)
    write_session_markdown = session_export_md.write_session_markdown

    def write_then_rewrite(*args, **kwargs):
        path = write_session_markdown(*args, **kwargs)
        writer = open_db()
        try:
            row = next(message for message in writer.get_messages("s1") if message.get("role") == "user")
            assert writer.set_user_message_content("s1", row["id"], "changed after export") == 1
        finally:
            writer.close()
        return path

    monkeypatch.setattr(session_export_md, "write_session_markdown", write_then_rewrite)
    _export_delete(monkeypatch, tmp_path / "out", "s1")

    output = capsys.readouterr().out
    assert "was not deleted because its history or delegate set changed after export" in output
    db = open_db()
    try:
        assert db.get_session("s1") is not None
        assert db.get_messages("s1")[0]["content"] == "changed after export"
    finally:
        db.close()
