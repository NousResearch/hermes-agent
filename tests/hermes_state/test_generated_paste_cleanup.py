import json
import sqlite3

import pytest

from agent.context_references import parse_context_references
from hermes_state import SessionDB


@pytest.fixture
def paste_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    paste_dir = tmp_path / "composer-pastes"
    paste_dir.mkdir()
    try:
        yield db, tmp_path / "sessions", paste_dir
    finally:
        db.close()


def test_delete_session_reclaims_only_unreferenced_generated_pastes(paste_db):
    db, sessions_dir, paste_dir = paste_db
    shared = paste_dir / "pasted_content_shared.txt"
    doomed = paste_dir / "pasted_content_doomed.txt"
    ordinary = paste_dir / "manual-notes.txt"
    for path in (shared, doomed, ordinary):
        path.write_text(path.name, encoding="utf-8")

    db.create_session("deleted", source="desktop")
    db.create_session("survivor", source="desktop")
    db.append_message("deleted", "user", f"@file:{shared} @file:{doomed}")
    db.append_message("survivor", "user", f"@file:{shared}")

    assert db.delete_session("deleted", sessions_dir=sessions_dir) is True
    assert shared.exists()
    assert not doomed.exists()
    assert ordinary.exists()


@pytest.mark.parametrize("target", [
    "{path}", "C:/h/composer-pastes/{name}", r"C:\h\composer-pastes\{name}",
    '`{path}`', '"{path}"', "'{path}'",
])
@pytest.mark.parametrize("suffix", [".", ")", ",", ":10", ":10-20"])
def test_surviving_parser_references_preserve_paste(paste_db, target, suffix):
    db, sessions_dir, paste_dir = paste_db
    shared = paste_dir / "pasted_content_2026-10-03_abcdef-2.txt"
    shared.write_text("still in use", encoding="utf-8")
    target = target.format(path=shared, name=shared.name)
    content = f"see @file:{target}{suffix}"
    assert parse_context_references(content)[0].target == target.strip("`\"'")
    db.create_session("deleted", source="desktop")
    db.create_session("survivor", source="desktop")
    db.append_message("deleted", "user", f"@file:{shared}")
    db.append_message("survivor", "user", content)

    assert db.delete_session("deleted", sessions_dir=sessions_dir)
    assert db.get_session("survivor") is not None
    assert shared.read_text(encoding="utf-8") == "still in use"


@pytest.mark.parametrize("quote", ["", "`", '"', "'"])
def test_deleted_ranged_reference_reclaims_generated_paste(paste_db, quote):
    db, sessions_dir, paste_dir = paste_db
    generated = paste_dir / "pasted_content_2026-10-03_abcdef-2.txt"
    generated.write_text("generated", encoding="utf-8")
    db.create_session("deleted", source="desktop")
    db.append_message("deleted", "user", f"@file:{quote}{generated}{quote}:10-20")

    assert db.delete_session("deleted", sessions_dir=sessions_dir)
    assert not generated.exists()


def test_delete_preserves_user_created_dotted_lookalikes(paste_db):
    db, sessions_dir, paste_dir = paste_db
    ordinary = paste_dir / "pasted_content_notes.v2.txt"
    generated = paste_dir / "pasted_content_2026-10-03_abcdef-2.txt"
    for path in (ordinary, generated):
        path.write_text(path.name, encoding="utf-8")
    db.create_session("deleted", source="desktop")
    db.append_message("deleted", "user", f"@file:{ordinary} @file:{generated}")

    assert db.delete_session("deleted", sessions_dir=sessions_dir)
    assert ordinary.read_text(encoding="utf-8") == ordinary.name
    assert not generated.exists()


@pytest.mark.parametrize("with_sessions_dir", [False, True])
def test_delete_wide_delegate_tree_under_legacy_sqlite_limit(paste_db, with_sessions_dir):
    db, sessions_dir, paste_dir = paste_db
    generated = paste_dir / "pasted_content_2026-10-03_abcdef.txt"
    generated.write_text("child paste", encoding="utf-8")
    db.create_session("root", source="desktop")
    db.create_session("survivor", source="desktop")
    child_ids = [f"child_{i}" for i in range(1200)]

    def seed(conn):
        conn.executemany(
            "INSERT INTO sessions (id, source, started_at, parent_session_id, model_config) "
            "VALUES (?, 'desktop', 1, 'root', ?)",
            [(sid, json.dumps({"_delegate_from": "root"})) for sid in child_ids],
        )
        conn.executemany(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, 'user', ?, 1)",
            [(sid, f"@file:{generated}") for sid in child_ids],
        )

    db._execute_write(seed)
    db._conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
    assert db.delete_session("root", sessions_dir=sessions_dir if with_sessions_dir else None)
    assert db._read_one("SELECT COUNT(*) FROM sessions")[0] == 1
    assert db.get_session("survivor") is not None
    assert db._read_one("SELECT COUNT(*) FROM messages")[0] == 0
    assert generated.exists() is not with_sessions_dir


def test_delete_without_candidates_does_not_scan_surviving_messages(paste_db, monkeypatch):
    db, sessions_dir, paste_dir = paste_db
    db.create_session("deleted", source="desktop")
    db.create_session("survivor", source="desktop")
    db.append_message("deleted", "user", "no generated paste")
    db.append_message("survivor", "user", "keep this message")

    def unexpected_read():
        pytest.fail("no-candidate deletion must not scan surviving messages")

    monkeypatch.setattr(db, "_read_ctx", unexpected_read)
    assert db.delete_session("deleted", sessions_dir=sessions_dir)
