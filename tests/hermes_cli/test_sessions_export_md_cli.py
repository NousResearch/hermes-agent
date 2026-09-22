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
    assert "Exported 1 session" in output
    assert "1 message" in output
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


def _trace_fake_db(captured):
    class FakeDB:
        def resolve_session_id(self, session_id):
            return "s1"

        def get_session(self, session_id):
            return {"id": session_id, "model": "test-model"}

        def get_messages_as_conversation(self, session_id):
            captured["conv"] = session_id
            return [
                {"role": "user", "content": "hello trace"},
                {"role": "assistant", "content": "hi"},
            ]

        def close(self):
            captured["closed"] = True

    return FakeDB()


def _real_store(monkeypatch, tmp_path):
    """Point the CLI's SessionDB at one real file; returns an opener for the test's own handles."""
    import hermes_state

    real_session_db = hermes_state.SessionDB
    db_path = tmp_path / "state.db"

    class _StoreAtTmp(real_session_db):
        def __init__(self, *args, **kwargs):
            super().__init__(db_path=db_path)

    monkeypatch.setattr(hermes_state, "SessionDB", _StoreAtTmp)
    return _StoreAtTmp


def _seed_six_turns(open_db, session_id, *, compact):
    db = open_db()
    try:
        db.create_session(session_id, "cli")
        for i in range(1, 7):
            db.append_message(session_id, "user", f"question {i}")
            db.append_message(session_id, "assistant", f"answer {i}")
        if compact:
            # Default in-place compaction, production shape: watermark from compression start, last turn carried.
            watermark = db.get_active_message_watermark(session_id)
            tail = [{"role": "user", "content": "question 6"}, {"role": "assistant", "content": "answer 6"}]
            db.archive_and_compact(session_id, [{"role": "user", "content": "[CONTEXT COMPACTION] summary"}, *tail],
                                   watermark=watermark, tail_count=len(tail))
    finally:
        db.close()


def _export_and_delete(monkeypatch, out_dir, session_id, *extra):
    import hermes_cli.main as main_mod

    monkeypatch.setattr(sys, "argv", [
        "hermes", "sessions", "export", "--format", "md", "--session-id", session_id,
        "--delete-after-verified", "--yes", *extra, str(out_dir),
    ])
    main_mod.main()


@pytest.mark.parametrize("lineage", ["single", "logical"])
def test_delete_after_verified_exports_the_turns_in_place_compaction_archived(monkeypatch, tmp_path, capsys, lineage):
    open_db = _real_store(monkeypatch, tmp_path)
    _seed_six_turns(open_db, "s1", compact=True)

    _export_and_delete(monkeypatch, tmp_path / "out", "s1", "--lineage", lineage)

    text = next((tmp_path / "out").glob("*.md")).read_text(encoding="utf-8")
    assert [f"answer {i}" in text for i in range(1, 7)] == [True] * 6
    assert "Deleted exported session 's1'." in capsys.readouterr().out
    db = open_db()
    try:
        assert db.get_session("s1") is None
    finally:
        db.close()


def test_delete_after_verified_keeps_a_session_that_gained_a_message_after_the_export(monkeypatch, tmp_path, capsys):
    import hermes_cli.session_export_md as session_export_md

    open_db = _real_store(monkeypatch, tmp_path)
    _seed_six_turns(open_db, "s1", compact=False)
    write_session_markdown = session_export_md.write_session_markdown

    def write_then_a_turn_lands(*args, **kwargs):
        path = write_session_markdown(*args, **kwargs)
        writer = open_db()
        try:
            writer.append_message("s1", "user", "sent after the export was read")
        finally:
            writer.close()
        return path

    monkeypatch.setattr(session_export_md, "write_session_markdown", write_then_a_turn_lands)
    _export_and_delete(monkeypatch, tmp_path / "out", "s1")

    assert "Export verification failed; not deleting session 's1'" in capsys.readouterr().out
    db = open_db()
    try:
        assert db.get_messages("s1")[-1]["content"] == "sent after the export was read"
    finally:
        db.close()


@pytest.mark.parametrize("argv, marker, expected", [
    pytest.param(["--format", "html", "--session-id", "s1"], "answer", 6, id="html"),
    pytest.param(["--format", "html"], "answer", 6, id="html-every-session"),
    pytest.param(["--format", "md", "--only", "user-prompts", "--session-id", "s1"], "question", 6, id="only-prompts"),
    # The importable payload keeps the live rows: import_sessions would replay archived turns as live context.
    pytest.param(["--format", "jsonl", "--session-id", "s1"], "answer", 1, id="jsonl-live-only"),
])
def test_human_readable_exports_carry_the_turns_in_place_compaction_archived(
        monkeypatch, tmp_path, argv, marker, expected):
    import hermes_cli.main as main_mod

    open_db = _real_store(monkeypatch, tmp_path)
    _seed_six_turns(open_db, "s1", compact=True)
    out = tmp_path / "export.out"
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "export", *argv, str(out)])
    main_mod.main()

    text = out.read_text(encoding="utf-8")
    assert sum(f"{marker} {i}" in text for i in range(1, 7)) == expected
