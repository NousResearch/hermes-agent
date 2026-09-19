from contextlib import closing

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("existing", [False, True])
def test_import_preserves_history_with_colliding_titles(tmp_path, existing):
    with closing(SessionDB(db_path=tmp_path / "state.db")) as db:
        if existing:
            db.create_session("existing", "cli")
            db.set_session_title("existing", "Shared title")
        payload = [
            {"id": sid, "source": "cli", "title": "Shared title",
             "messages": [{"role": "user", "content": sid}]}
            for sid in ("incoming-a", "incoming-b")
        ]
        result = db.import_sessions(payload)
        assert result["ok"] and result["imported"] == len(payload)
        titles = [db.get_session_title(item["id"]) for item in payload]
        if existing:
            assert db.get_session_title("existing") == "Shared title"
            titles.append(db.get_session_title("existing"))
        assert len(set(titles)) == len(titles)
        for item in payload:
            assert db.get_messages(item["id"])[0]["content"] == item["id"]
        assert db.import_sessions(payload)["skipped"] == len(payload)


@pytest.mark.parametrize("title", ["Report", "R" * SessionDB.MAX_TITLE_LENGTH])
def test_import_keeps_unique_titles_and_handles_suffix_collisions(tmp_path, title):
    with closing(SessionDB(db_path=tmp_path / "state.db")) as db:
        suffix = " (imported 2)"
        for sid, name in [("a", title), ("b", title[:SessionDB.MAX_TITLE_LENGTH - len(suffix)] + suffix)]:
            db.create_session(sid, "cli")
            db.set_session_title(sid, name)
        payload = [
            {"id": "c", "title": title, "messages": []},
            {"id": "d", "title": "Unique report", "messages": []},
            {"id": "e", "title": None, "messages": []},
        ]
        assert db.import_sessions(payload)["imported"] == len(payload)
        titles = [db.get_session_title(sid) for sid in ("a", "b", "c")]
        assert len(set(titles)) == len(titles)
        assert all(len(name) <= SessionDB.MAX_TITLE_LENGTH for name in titles)
        assert db.get_session_title("d") == "Unique report"
        assert db.get_session_title("e") is None
        origin = {"tool": "codex", "foreign_session_id": "foreign", "path": "fixture.jsonl"}
        result = db.import_foreign_history(
            origin, [{"role": "user", "content": "foreign history"}],
            title=title, cwd=None, profile="default",
        )
        imported_title = db.get_session_title(result["session_id"])
        assert imported_title not in titles
        assert len(imported_title) <= SessionDB.MAX_TITLE_LENGTH
        assert db.import_foreign_history(
            origin, [], title=title, cwd=None, profile="default"
        )["already_imported"]
