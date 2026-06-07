import json
from types import SimpleNamespace

from hermes_cli.webui_session_import import import_webui_sessions, import_webui_sessions_by_profile
from hermes_state import SessionDB


def _write_session(path, session_id, *, title="Legacy WebUI", messages=None, **extra):
    payload = {
        "session_id": session_id,
        "title": title,
        "workspace": "/tmp/legacy-workspace",
        "profile": "default",
        "model": "openai/gpt-4.1",
        "created_at": 1000.0,
        "updated_at": 2000.0,
        "input_tokens": 12,
        "output_tokens": 34,
        "messages": messages if messages is not None else [
            {"role": "user", "content": "hello from webui", "timestamp": 1001.0},
            {"role": "assistant", "content": "hello back", "timestamp": 1002.0},
        ],
    }
    payload.update(extra)
    (path / f"{session_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_import_webui_sessions_dry_run_does_not_write(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    _write_session(sessions_dir, "webui_missing")

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        report = import_webui_sessions(db, sessions_dir=sessions_dir, dry_run=True)

        assert report.scanned == 1
        assert report.imported == 1
        assert report.changed == 1
        assert db.get_session("webui_missing") is None
    finally:
        db.close()


def test_import_webui_sessions_apply_imports_missing_session(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    _write_session(sessions_dir, "webui_missing")

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        report = import_webui_sessions(db, sessions_dir=sessions_dir, dry_run=False)

        assert report.imported == 1
        session = db.get_session("webui_missing")
        assert session["source"] == "webui"
        assert session["title"] == "Legacy WebUI"
        assert session["model"] == "openai/gpt-4.1"
        assert session["cwd"] == "/tmp/legacy-workspace"
        assert session["message_count"] == 2
        assert session["input_tokens"] == 12
        assert session["output_tokens"] == 34
        messages = db.get_messages("webui_missing")
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[0]["content"] == "hello from webui"
    finally:
        db.close()


def test_import_webui_sessions_filters_to_profile(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    _write_session(sessions_dir, "default_webui", profile="default")
    _write_session(sessions_dir, "uni_webui", profile="uni_work")

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        report = import_webui_sessions(db, sessions_dir=sessions_dir, dry_run=False, profile="default")

        assert report.imported == 1
        assert report.skipped_other_profile == 1
        assert db.get_session("default_webui") is not None
        assert db.get_session("uni_webui") is None
    finally:
        db.close()


def test_import_webui_sessions_by_profile_routes_to_profile_dbs(tmp_path, monkeypatch):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    default_home = tmp_path / "default"
    uni_home = tmp_path / "profiles" / "uni_work"
    default_home.mkdir()
    uni_home.mkdir(parents=True)
    _write_session(sessions_dir, "default_webui", profile="default")
    _write_session(sessions_dir, "uni_webui", profile="uni_work")

    from hermes_cli import profiles

    monkeypatch.setattr(
        profiles,
        "list_profiles",
        lambda: [
            SimpleNamespace(name="default", path=default_home),
            SimpleNamespace(name="uni_work", path=uni_home),
        ],
    )

    report = import_webui_sessions_by_profile(sessions_dir=sessions_dir, dry_run=False)

    assert report.changed == 2
    default_db = SessionDB(db_path=default_home / "state.db")
    uni_db = SessionDB(db_path=uni_home / "state.db")
    try:
        assert default_db.get_session("default_webui") is not None
        assert default_db.get_session("uni_webui") is None
        assert uni_db.get_session("default_webui") is None
        assert uni_db.get_session("uni_webui") is not None
    finally:
        default_db.close()
        uni_db.close()


def test_import_webui_sessions_skips_empty_by_default(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    _write_session(sessions_dir, "empty_webui", messages=[])

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        report = import_webui_sessions(db, sessions_dir=sessions_dir, dry_run=False)

        assert report.skipped_empty == 1
        assert db.get_session("empty_webui") is None
    finally:
        db.close()


def test_import_webui_sessions_skips_non_object_json(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    (sessions_dir / "not_a_session.json").write_text(json.dumps([]), encoding="utf-8")

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        report = import_webui_sessions(db, sessions_dir=sessions_dir, dry_run=False)

        assert report.skipped_invalid == 1
        assert report.errors[0].reason == "session JSON must be an object"
    finally:
        db.close()


def test_import_webui_sessions_refreshes_short_existing_webui_row(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    _write_session(
        sessions_dir,
        "short_webui",
        title="JSON title should not clobber rename",
        messages=[
            {"role": "user", "content": "first", "timestamp": 1001.0},
            {"role": "assistant", "content": "second", "timestamp": 1002.0},
            {"role": "user", "content": "third", "timestamp": 1003.0},
        ],
    )

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("short_webui", "webui", model="old-model")
        db.set_session_title("short_webui", "User renamed title")
        db.append_message("short_webui", role="user", content="first")

        report = import_webui_sessions(db, sessions_dir=sessions_dir, dry_run=False)

        assert report.refreshed == 1
        session = db.get_session("short_webui")
        assert session["title"] == "User renamed title"
        assert session["message_count"] == 3
        assert [m["content"] for m in db.get_messages("short_webui")] == [
            "first",
            "second",
            "third",
        ]
    finally:
        db.close()


def test_import_webui_sessions_does_not_overwrite_non_webui_row(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    _write_session(sessions_dir, "cli_session")

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("cli_session", "cli")
        db.append_message("cli_session", role="user", content="cli-owned")

        report = import_webui_sessions(db, sessions_dir=sessions_dir, dry_run=False)

        assert report.skipped_foreign_source == 1
        assert db.get_session("cli_session")["source"] == "cli"
        assert [m["content"] for m in db.get_messages("cli_session")] == ["cli-owned"]
    finally:
        db.close()


def test_import_webui_sessions_suffixes_duplicate_titles(tmp_path):
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    _write_session(sessions_dir, "webui_a", title="Repeated")
    _write_session(sessions_dir, "webui_b", title="Repeated")

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        report = import_webui_sessions(db, sessions_dir=sessions_dir, dry_run=False)

        assert report.imported == 2
        assert db.get_session("webui_a")["title"] == "Repeated"
        assert db.get_session("webui_b")["title"] == "Repeated (webui_b)"
    finally:
        db.close()
