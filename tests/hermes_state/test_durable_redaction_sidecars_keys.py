"""Durable-redaction coverage for sidecars and structured dictionary keys."""

import hashlib
import hmac
import json
import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_common import SCHEMA_VERSION
from hermes_state_messages import _redact_durable_projection


SECRET = "sk-sidecarLeak012345"


def _assert_absent_from_fts(conn):
    for table in ("messages_fts", "messages_fts_trigram", "messages_fts_cjk"):
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        ).fetchone() is not None:
            assert conn.execute(
                f"SELECT 1 FROM {table} WHERE {table} MATCH ?", (f'"{SECRET}"',)
            ).fetchone() is None


def test_api_content_backfills_apply_durable_redaction(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("sidecar", source="cli")
    try:
        first_id = db.append_message("sidecar", "user", content="first")
        second_id = db.append_message("sidecar", "user", content="second")
        assert db.set_latest_user_api_content("sidecar", "second", SECRET) == 1
        assert db.set_message_api_content("sidecar", first_id, "first", SECRET) == 1

        rows = db._conn.execute(
            "SELECT id, api_content FROM messages WHERE session_id = ? ORDER BY id", ("sidecar",)
        ).fetchall()
        assert {row["id"] for row in rows} == {first_id, second_id}
        assert all(SECRET not in row["api_content"] for row in rows)
    finally:
        db.close()


def test_fresh_durable_projections_redact_every_message_string_projection_and_fts(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("fresh", source="cli")
    raw = {"outer": {SECRET: {"nested": SECRET}}}
    try:
        db.append_message(
            "fresh", SECRET, "safe", tool_calls=raw, tool_call_id=SECRET,
            tool_name=SECRET, effect_disposition=SECRET, finish_reason=SECRET,
            platform_message_id=SECRET, display_kind=SECRET,
        )
        db.append_message("fresh", "assistant", "setter-target")
        row = db._conn.execute(
            "SELECT role, tool_call_id, tool_calls, tool_name, effect_disposition, finish_reason, "
            "platform_message_id, display_kind FROM messages WHERE session_id = ?", ("fresh",)
        ).fetchone()
        stored = row["tool_calls"]
        assert raw["outer"][SECRET]["nested"] == SECRET  # live operands remain unmodified
        assert SECRET not in " ".join(str(value) for value in row)
        assert json.loads(stored)["outer"]
        assert db.set_latest_matching_message_display_kind(
            "fresh", role="assistant", content="setter-target", display_kind=SECRET,
        )
        assert SECRET not in db._conn.execute(
            "SELECT display_kind FROM messages WHERE session_id = ?", ("fresh",)
        ).fetchone()[0]
        _assert_absent_from_fts(db._conn)
    finally:
        db.close()


def test_fresh_session_title_is_a_redacted_durable_projection(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    db.create_session("fresh-title", source="cli")
    try:
        assert db.set_session_title("fresh-title", f"investigate {SECRET}")
        stored_title = db.get_session_title("fresh-title")
        assert stored_title is not None
        assert SECRET not in stored_title
    finally:
        db.close()

    secret_bytes = SECRET.encode("utf-8")
    for path in (db_path, db_path.with_name(f"{db_path.name}-wal")):
        assert not path.exists() or secret_bytes not in path.read_bytes(), path


def test_fresh_session_diagnostics_are_redacted_durable_projections(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    db.create_session("fresh-diagnostics", source="cli")
    try:
        db.touch_session_activity("fresh-diagnostics", description=f"working {SECRET}")
        db.fail_handoff("fresh-diagnostics", f"handoff failed: {SECRET}")
        db.record_compression_failure_cooldown("fresh-diagnostics", 9999999999, f"compression failed: {SECRET}")

        row = db._conn.execute(
            "SELECT last_activity_description, handoff_error, compression_failure_error "
            "FROM sessions WHERE id = ?", ("fresh-diagnostics",),
        ).fetchone()
        assert SECRET not in " ".join(str(value) for value in row)
    finally:
        db.close()

    secret_bytes = SECRET.encode("utf-8")
    for path in (db_path, db_path.with_name(f"{db_path.name}-wal")):
        assert not path.exists() or secret_bytes not in path.read_bytes(), path


def test_direct_imported_title_is_redacted_and_sanitizes_bytes(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    title = f"imported {SECRET} title"
    try:
        result = db.import_sessions([{
            "id": "direct-import", "source": "import", "title": title, "messages": [],
        }])
        assert result["ok"] and result["imported"] == 1
        stored_title = db.get_session_title("direct-import")
        assert stored_title is not None and SECRET not in stored_title
    finally:
        db.close()

    secret_bytes = SECRET.encode("utf-8")
    for path in (db_path, db_path.with_name(f"{db_path.name}-wal")):
        assert not path.exists() or secret_bytes not in path.read_bytes(), path


def test_foreign_imported_title_is_redacted_and_sanitizes_bytes(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    title = f"foreign {SECRET} title"
    try:
        foreign = db.import_foreign_history(
            {"tool": "foreign-tool", "path": "/foreign/history", "foreign_session_id": "foreign-1"},
            [{"role": "user", "content": "safe"}], title=title, cwd="/foreign", profile="default",
        )
        assert not foreign["already_imported"]
        stored_title = db.get_session_title(foreign["session_id"])
        assert stored_title is not None and SECRET not in stored_title
    finally:
        db.close()

    secret_bytes = SECRET.encode("utf-8")
    for path in (db_path, db_path.with_name(f"{db_path.name}-wal")):
        assert not path.exists() or secret_bytes not in path.read_bytes(), path


def test_foreign_imported_origin_json_is_redacted_and_sanitizes_bytes(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    origin = {
        "tool": "foreign-tool",
        "path": f"/foreign/{SECRET}/history",
        "foreign_session_id": "foreign-1",
        "metadata": {SECRET: {"nested": SECRET}},
    }
    try:
        foreign = db.import_foreign_history(
            origin, [{"role": "user", "content": "safe"}], title="safe", cwd="/foreign", profile="default",
        )
        assert not foreign["already_imported"]
        stored = db._conn.execute(
            "SELECT origin_json FROM sessions WHERE id = ?", (foreign["session_id"],),
        ).fetchone()[0]
        assert SECRET not in stored
        assert origin["metadata"][SECRET]["nested"] == SECRET  # live provenance remains raw
        duplicate = db.import_foreign_history(
            origin, [{"role": "user", "content": "safe"}], title="safe", cwd="/foreign", profile="default",
        )
        assert duplicate == {"session_id": foreign["session_id"], "already_imported": True}
    finally:
        db.close()

    secret_bytes = SECRET.encode("utf-8")
    for path in (db_path, db_path.with_name(f"{db_path.name}-wal")):
        assert not path.exists() or secret_bytes not in path.read_bytes(), path


def test_fresh_create_session_origin_json_is_redacted_and_sanitizes_bytes(tmp_path):
    db_path = tmp_path / "state.db"
    raw_origin = {"platform": "test", "credential": SECRET, SECRET: {"nested": SECRET}}
    db = SessionDB(db_path)
    try:
        db.create_session("fresh-origin", source="test", origin_json=json.dumps(raw_origin))
        stored = db._conn.execute(
            "SELECT origin_json FROM sessions WHERE id = ?", ("fresh-origin",),
        ).fetchone()[0]
        assert SECRET not in stored
        assert raw_origin[SECRET]["nested"] == SECRET
    finally:
        db.close()

    for path in (db_path, db_path.with_name(f"{db_path.name}-wal")):
        assert not path.exists() or SECRET.encode("utf-8") not in path.read_bytes(), path


def test_fresh_gateway_peer_origin_json_is_redacted_and_sanitizes_bytes(tmp_path):
    db_path = tmp_path / "state.db"
    raw_origin = {"platform": "test", "credential": SECRET, SECRET: {"nested": SECRET}}
    db = SessionDB(db_path)
    try:
        db.record_gateway_session_peer(
            "fresh-gateway-origin", source="test", session_key="agent:main:test",
            origin_json=json.dumps(raw_origin),
        )
        stored = db._conn.execute(
            "SELECT origin_json FROM sessions WHERE id = ?", ("fresh-gateway-origin",),
        ).fetchone()[0]
        assert SECRET not in stored
        assert raw_origin[SECRET]["nested"] == SECRET
    finally:
        db.close()

    for path in (db_path, db_path.with_name(f"{db_path.name}-wal")):
        assert not path.exists() or SECRET.encode("utf-8") not in path.read_bytes(), path


def test_gateway_routing_entry_redacts_fresh_data_and_restores_operational_key(tmp_path):
    """The table key is routing state; its JSON is a display-safe projection."""
    db_path = tmp_path / "state.db"
    session_key = "agent:main:telegram:dm:operational-key"
    entry = {"session_key": session_key, "metadata": {SECRET: {"credential": SECRET}}}
    db = SessionDB(db_path)
    try:
        db.save_gateway_routing_entry(session_key, json.dumps(entry), scope="gateway")
        stored = db._conn.execute(
            "SELECT entry_json FROM gateway_routing WHERE scope = ? AND session_key = ?", ("gateway", session_key),
        ).fetchone()[0]
        assert SECRET not in stored
        # The live loader rehydrates only the operational key from its dedicated column.
        loaded = json.loads(db.load_gateway_routing_entries(scope="gateway")[session_key])
        assert loaded["session_key"] == session_key
        assert entry["metadata"][SECRET]["credential"] == SECRET
    finally:
        db.close()
    assert SECRET.encode("utf-8") not in db_path.read_bytes()


def test_gateway_routing_migration_redacts_legacy_entry_bytes(tmp_path):
    """A pre-v33 routing row is rewritten before the migration is accepted."""
    db_path = tmp_path / "legacy-routing.db"
    session_key = "agent:main:telegram:dm:operational-key"
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        conn.executescript("""
            CREATE TABLE gateway_routing (
                scope TEXT NOT NULL DEFAULT '', session_key TEXT NOT NULL,
                entry_json TEXT NOT NULL, updated_at REAL NOT NULL,
                PRIMARY KEY (scope, session_key)
            );
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version VALUES (32);
        """)
        conn.execute(
            "INSERT INTO gateway_routing VALUES (?, ?, ?, ?)",
            ("gateway", session_key, json.dumps({"session_key": session_key, "metadata": {SECRET: SECRET}}), 1.0),
        )
    finally:
        conn.close()
    migrated = SessionDB(db_path)
    try:
        stored = migrated._conn.execute("SELECT entry_json FROM gateway_routing").fetchone()[0]
        assert SECRET not in stored
        assert json.loads(migrated.load_gateway_routing_entries(scope="gateway")[session_key])["session_key"] == session_key
        assert migrated._conn.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        migrated.close()
    assert SECRET.encode("utf-8") not in db_path.read_bytes()


def test_foreign_import_identity_uses_raw_digest_not_redacted_origin(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    first_origin = {"tool": "foreign-tool", "path": "/foreign/history", "foreign_session_id": SECRET}
    second_origin = {"tool": "foreign-tool", "path": "/foreign/history", "foreign_session_id": "«redacted:sk-other…»"}
    try:
        first = db.import_foreign_history(
            first_origin, [{"role": "user", "content": "safe"}], title="safe", cwd="/foreign", profile="default",
        )
        second = db.import_foreign_history(
            second_origin, [{"role": "user", "content": "safe"}], title="safe", cwd="/foreign", profile="default",
        )
        repeat = db.import_foreign_history(
            first_origin, [{"role": "user", "content": "safe"}], title="safe", cwd="/foreign", profile="default",
        )
        assert not first["already_imported"]
        assert not second["already_imported"]
        assert first["session_id"] != second["session_id"]
        assert repeat == {"session_id": first["session_id"], "already_imported": True}
        rows = db._conn.execute(
            "SELECT origin_json, import_identity_digest FROM sessions WHERE id IN (?, ?)",
            (first["session_id"], second["session_id"]),
        ).fetchall()
        assert len(rows) == 2
        assert len({row["import_identity_digest"] for row in rows}) == 2
        assert all(SECRET not in row["origin_json"] for row in rows)
    finally:
        db.close()


def test_foreign_import_identity_digest_is_profile_keyed_and_not_recoverable_from_db(tmp_path, monkeypatch):
    """The durable digest is HMAC, so copying state.db alone is not a raw-ID oracle."""
    home = tmp_path / "profile-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    origin = {"tool": "foreign-tool", "path": "/foreign/history", "foreign_session_id": SECRET}
    db = SessionDB(tmp_path / "state.db")
    try:
        first = db.import_foreign_history(
            origin, [{"role": "user", "content": "safe"}], title="safe", cwd="/foreign", profile="default",
        )
        repeat = db.import_foreign_history(
            origin, [{"role": "user", "content": "safe"}], title="safe", cwd="/foreign", profile="default",
        )
        digest = db._conn.execute(
            "SELECT import_identity_digest FROM sessions WHERE id = ?", (first["session_id"],),
        ).fetchone()[0]
        identity = {"tool": origin["tool"], "foreign_session_id": origin["foreign_session_id"]}
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        unkeyed = hashlib.sha256(f"hermes:foreign-import:v1:{canonical}".encode("utf-8")).hexdigest()
        key_path = home / ".foreign-import-identity.key"

        assert repeat == {"session_id": first["session_id"], "already_imported": True}
        assert digest != unkeyed
        assert digest == hmac.new(
            key_path.read_bytes(), f"hermes:foreign-import:v1:{canonical}".encode("utf-8"), hashlib.sha256,
        ).hexdigest()
        assert SECRET.encode("utf-8") not in (tmp_path / "state.db").read_bytes()
        assert key_path.exists()
    finally:
        db.close()


def test_v30_migration_backfills_raw_foreign_identity_before_redacting_origin(tmp_path):
    db_path = tmp_path / "state.db"
    origin = {"tool": "foreign-tool", "path": "/foreign/history", "foreign_session_id": SECRET}
    db = SessionDB(db_path)
    db.create_session("legacy-foreign", source="foreign-tool")
    db.close()

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE sessions SET origin_json = ? WHERE id = ?",
            (json.dumps({"imported_from": origin}), "legacy-foreign"),
        )
        conn.execute("UPDATE schema_version SET version = 30")

    migrated = SessionDB(db_path)
    try:
        row = migrated._conn.execute(
            "SELECT origin_json, import_identity_digest FROM sessions WHERE id = ?", ("legacy-foreign",),
        ).fetchone()
        assert SECRET not in row["origin_json"]
        assert row["import_identity_digest"]
        assert row["import_identity_digest"] == migrated._foreign_import_identity_digest(origin)
    finally:
        migrated.close()


def test_v31_migration_drops_legacy_unkeyed_import_digest(tmp_path):
    db_path = tmp_path / "state.db"
    origin = {"tool": "foreign-tool", "path": "/foreign/history", "foreign_session_id": SECRET}
    identity = {"tool": origin["tool"], "foreign_session_id": origin["foreign_session_id"]}
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    legacy_digest = hashlib.sha256(f"hermes:foreign-import:v1:{canonical}".encode("utf-8")).hexdigest()
    db = SessionDB(db_path)
    db.create_session("legacy-v31-foreign", source="foreign-tool")
    db.close()
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE sessions SET import_identity_digest = ? WHERE id = ?", (legacy_digest, "legacy-v31-foreign"))
        conn.execute("UPDATE schema_version SET version = 31")

    migrated = SessionDB(db_path)
    try:
        assert migrated._conn.execute(
            "SELECT import_identity_digest FROM sessions WHERE id = ?", ("legacy-v31-foreign",),
        ).fetchone()[0] is None
    finally:
        migrated.close()
    for path in (db_path, db_path.with_name(f"{db_path.name}-wal")):
        assert not path.exists() or legacy_digest.encode("ascii") not in path.read_bytes()


def test_v31_migration_redacts_origin_json_and_sanitizes_its_bytes(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    db.create_session("legacy-origin", source="cli")
    db.close()

    legacy_origin = {"imported_from": {"path": SECRET, SECRET: {"nested": SECRET}}}
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE sessions SET origin_json = ? WHERE id = ?", (json.dumps(legacy_origin), "legacy-origin"))
        conn.execute("UPDATE schema_version SET version = 30")

    migrated = SessionDB(db_path)
    try:
        stored = migrated._conn.execute(
            "SELECT origin_json FROM sessions WHERE id = ?", ("legacy-origin",),
        ).fetchone()[0]
        assert SECRET not in stored
        _assert_absent_from_fts(migrated._conn)
    finally:
        migrated.close()

    secret_bytes = SECRET.encode("utf-8")
    for path in (db_path, db_path.with_name(f"{db_path.name}-wal")):
        assert not path.exists() or secret_bytes not in path.read_bytes(), path


def test_direct_import_redacted_title_collision_keeps_both_sessions(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    session_ids = (
        "direct-sk-abc" + "1" * 14 + "ZZZZ",
        "direct-sk-abc" + "2" * 14 + "ZZZZ",
    )
    try:
        first = db.import_sessions([{
            "id": session_ids[0], "source": "import", "title": SECRET, "messages": [],
        }])
        second = db.import_sessions([{
            "id": session_ids[1], "source": "import", "title": SECRET, "messages": [],
        }])
        assert first["ok"] and first["imported"] == 1 and first["imported_ids"] == [session_ids[0]]
        assert second["ok"] and second["imported"] == 1 and second["imported_ids"] == [session_ids[1]]
        titles = [row["title"] for row in db._conn.execute(
            "SELECT title FROM sessions WHERE id IN (?, ?) ORDER BY id",
            session_ids,
        )]
        assert len(set(titles)) == 2
        assert all(SECRET not in title for title in titles)
        for session_id in session_ids:
            assert all(session_id not in title for title in titles)
            assert all(hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12] not in title for title in titles)
        repeat = db.import_sessions([{
            "id": session_ids[1], "source": "import", "title": SECRET, "messages": [],
        }])
        assert repeat["ok"] and repeat["imported"] == 0 and repeat["skipped_ids"] == [session_ids[1]]
        assert titles == [row["title"] for row in db._conn.execute(
            "SELECT title FROM sessions WHERE id IN (?, ?) ORDER BY id", session_ids,
        )]
    finally:
        db.close()


def test_v31_migration_redacts_session_title_and_sanitizes_its_bytes(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    db.create_session("legacy-title", source="cli")
    db.close()

    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (f"legacy {SECRET}", "legacy-title"))
        conn.execute("UPDATE schema_version SET version = 30")

    migrated = SessionDB(db_path)
    try:
        stored_title = migrated.get_session_title("legacy-title")
        assert stored_title is not None
        assert SECRET not in stored_title
    finally:
        migrated.close()

    secret_bytes = SECRET.encode("utf-8")
    for path in (db_path, db_path.with_name(f"{db_path.name}-wal")):
        assert not path.exists() or secret_bytes not in path.read_bytes(), path


def test_v30_migration_retries_colliding_redacted_credential_titles(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    session_ids = (
        "legacy-sk-abc" + "1" * 14 + "ZZZZ",
        "legacy-sk-abc" + "2" * 14 + "ZZZZ",
    )
    db.create_session(session_ids[0], source="cli")
    db.create_session(session_ids[1], source="cli")
    db.close()

    raw_titles = (
        "credential " + "sk-abc" + "1" * 14 + "ZZZZ",
        "credential " + "sk-abc" + "2" * 14 + "ZZZZ",
    )
    assert _redact_durable_projection(raw_titles[0]) == _redact_durable_projection(raw_titles[1])
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "UPDATE sessions SET title = ? WHERE id = ?",
            zip(raw_titles, session_ids),
        )
        conn.execute("UPDATE schema_version SET version = 30")

    with monkeypatch.context() as patch:
        patch.setattr(
            SessionDB,
            "_sanitize_v31_legacy_redaction_storage",
            lambda self: (_ for _ in ()).throw(sqlite3.OperationalError("simulated sanitation failure")),
        )
        with pytest.raises(sqlite3.OperationalError, match="simulated sanitation failure"):
            SessionDB(db_path)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 30

    migrated = SessionDB(db_path)
    try:
        rows = migrated._conn.execute(
            "SELECT id, title FROM sessions WHERE id IN (?, ?) ORDER BY id",
            session_ids,
        ).fetchall()
        assert [row["id"] for row in rows] == list(session_ids)
        assert len({row["title"] for row in rows}) == 2
        assert all(raw not in row["title"] for raw in raw_titles for row in rows)
        for session_id in session_ids:
            assert all(session_id not in row["title"] for row in rows)
            assert all(hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12] not in row["title"] for row in rows)
        assert migrated._conn.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION
        titles = [row["title"] for row in rows]
    finally:
        migrated.close()

    reopened = SessionDB(db_path)
    try:
        assert titles == [row["title"] for row in reopened._conn.execute(
            "SELECT title FROM sessions WHERE id IN (?, ?) ORDER BY id", session_ids,
        )]
    finally:
        reopened.close()

    for raw in raw_titles:
        for path in (db_path, db_path.with_name(f"{db_path.name}-wal")):
            assert not path.exists() or raw.encode("utf-8") not in path.read_bytes(), path


def test_v31_migration_redacts_session_diagnostics_and_sanitizes_bytes(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    db.create_session("legacy-diagnostics", source="cli")
    db.close()

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE sessions SET last_activity_description = ?, handoff_error = ?, "
            "compression_failure_error = ? WHERE id = ?",
            (SECRET, SECRET, SECRET, "legacy-diagnostics"),
        )
        conn.execute("UPDATE schema_version SET version = 30")

    migrated = SessionDB(db_path)
    try:
        row = migrated._conn.execute(
            "SELECT last_activity_description, handoff_error, compression_failure_error "
            "FROM sessions WHERE id = ?", ("legacy-diagnostics",),
        ).fetchone()
        assert SECRET not in " ".join(str(value) for value in row)
        _assert_absent_from_fts(migrated._conn)
    finally:
        migrated.close()

    secret_bytes = SECRET.encode("utf-8")
    for path in (db_path, db_path.with_name(f"{db_path.name}-wal")):
        assert not path.exists() or secret_bytes not in path.read_bytes(), path


def test_redacted_dict_key_collisions_are_deterministic_and_json_safe(monkeypatch):
    monkeypatch.setattr("agent.redact.redact_sensitive_text", lambda text, **_: "[REDACTED]")
    raw = {"first-key": "first", "second-key": "second"}
    projected = _redact_durable_projection(raw)

    assert projected == {"[REDACTED]": "[REDACTED]", "[REDACTED] [redacted key 2]": "[REDACTED]"}
    assert projected == _redact_durable_projection(raw)
    assert json.loads(json.dumps(projected)) == projected


def test_v31_migration_redacts_every_message_string_projection_and_fts(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    db.create_session("legacy", source="cli")
    db.append_message("legacy", "assistant", content="safe")
    db.close()

    legacy = {"outer": {SECRET: {"nested": SECRET}}}
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE messages SET role = ?, api_content = ?, tool_call_id = ?, tool_calls = ?, tool_name = ?, "
            "effect_disposition = ?, finish_reason = ?, platform_message_id = ?, display_kind = ? "
            "WHERE session_id = ?",
            (SECRET, SECRET, SECRET, json.dumps(legacy), SECRET, SECRET, SECRET, SECRET, SECRET, "legacy"),
        )
        conn.execute("UPDATE schema_version SET version = 30")

    migrated = SessionDB(db_path)
    try:
        row = migrated._conn.execute(
            "SELECT role, api_content, tool_call_id, tool_calls, tool_name, effect_disposition, finish_reason, "
            "platform_message_id, display_kind FROM messages WHERE session_id = ?", ("legacy",)
        ).fetchone()
        assert SECRET not in " ".join(str(value) for value in row)
        parsed = json.loads(row["tool_calls"])
        assert SECRET not in json.dumps(parsed)
        assert isinstance(parsed["outer"], dict)
        _assert_absent_from_fts(migrated._conn)
    finally:
        migrated.close()


def test_v31_migration_sanitizes_legacy_secret_bytes_from_db_and_wal(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    db.create_session("legacy-physical", source="cli")
    db.append_message("legacy-physical", "assistant", content="safe")
    db.close()

    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE messages SET role = ? WHERE session_id = ?", (SECRET, "legacy-physical"))
        conn.execute("UPDATE schema_version SET version = 30")

    migrated = SessionDB(db_path)
    migrated.close()  # a clean close must leave no raw legacy bytes in either active file

    secret_bytes = SECRET.encode("utf-8")
    for path in (db_path, db_path.with_name(f"{db_path.name}-wal")):
        assert not path.exists() or secret_bytes not in path.read_bytes(), path


def test_v31_does_not_advance_when_storage_sanitization_fails(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    db.create_session("legacy-sanitation-failure", source="cli")
    db.close()
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE schema_version SET version = 30")

    monkeypatch.setattr(
        SessionDB,
        "_sanitize_v31_legacy_redaction_storage",
        lambda self: (_ for _ in ()).throw(sqlite3.OperationalError("simulated sanitation failure")),
    )
    with pytest.raises(sqlite3.OperationalError, match="simulated sanitation failure"):
        SessionDB(db_path)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 30


def test_system_prompt_write_and_v31_migration_redact_rows_and_references(tmp_path):
    db_path = tmp_path / "state.db"
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    db = SessionDB(db_path)
    db.create_session("fresh-prompt", source="cli", system_prompt=f"fresh {secret}")
    try:
        assert secret not in db._conn.execute("SELECT prompt FROM system_prompts").fetchone()[0]
    finally:
        db.close()

    with sqlite3.connect(db_path) as conn:
        raw_hash = "raw-prompt-hash"
        conn.execute("INSERT INTO system_prompts (hash, prompt) VALUES (?, ?)", (raw_hash, f"legacy {secret}"))
        conn.execute("UPDATE sessions SET system_prompt_hash = ? WHERE id = ?", (raw_hash, "fresh-prompt"))
        conn.execute("UPDATE schema_version SET version = 30")

    migrated = SessionDB(db_path)
    try:
        prompt_rows = migrated._conn.execute("SELECT hash, prompt FROM system_prompts").fetchall()
        assert all(secret not in row["prompt"] for row in prompt_rows)
        referenced = migrated._conn.execute(
            "SELECT sp.prompt FROM sessions s JOIN system_prompts sp ON sp.hash = s.system_prompt_hash "
            "WHERE s.id = ?", ("fresh-prompt",)
        ).fetchone()[0]
        assert secret not in referenced
        assert migrated._conn.execute(
            "SELECT 1 FROM messages_fts WHERE messages_fts MATCH ?", (f'\"{secret}\"',)
        ).fetchone() is None
    finally:
        migrated.close()


def test_v31_migration_redacts_and_advances_without_fts5_or_rebuilding_existing_fts(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    db.create_session("legacy-without-fts5", source="cli")
    db.append_message("legacy-without-fts5", "assistant", content="safe")
    db.close()

    with sqlite3.connect(db_path) as conn:
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'messages_fts'"
        ).fetchone() is not None
        conn.execute(
            "UPDATE messages SET api_content = ? WHERE session_id = ?",
            (SECRET, "legacy-without-fts5"),
        )
        conn.execute("UPDATE schema_version SET version = 30")

    rebuild_attempts = []
    monkeypatch.setattr(SessionDB, "_sqlite_supports_fts5", lambda self, cursor: False)
    monkeypatch.setattr(
        SessionDB,
        "_rebuild_fts_indexes",
        lambda self, *args, **kwargs: rebuild_attempts.append((args, kwargs)),
    )

    migrated = SessionDB(db_path)
    try:
        row = migrated._conn.execute(
            "SELECT api_content FROM messages WHERE session_id = ?", ("legacy-without-fts5",)
        ).fetchone()
        assert SECRET not in row["api_content"]
        assert rebuild_attempts == []
        assert migrated._conn.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        migrated.close()


def test_v31_without_fts5_drops_existing_cjk_triggers_before_message_updates(tmp_path, monkeypatch):
    """A tokenizer-less opener must detach CJK triggers before v31 rewrites rows."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    db.create_session("legacy-cjk", source="cli")
    db.append_message("legacy-cjk", "assistant", content="safe")
    db.close()

    with sqlite3.connect(db_path) as conn:
        for name in ("messages_fts_cjk_insert", "messages_fts_cjk_delete", "messages_fts_cjk_update"):
            conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        conn.execute("UPDATE messages SET api_content = ? WHERE session_id = ?", (
            "sk-abcdefghijklmnopqrstuvwxyz123456", "legacy-cjk",
        ))
        # Any surviving CJK trigger would abort the migration UPDATE below.
        conn.execute("""
            CREATE TRIGGER messages_fts_cjk_update AFTER UPDATE ON messages
            BEGIN SELECT RAISE(ABORT, 'CJK trigger was not detached'); END
        """)
        conn.execute("UPDATE schema_version SET version = 30")

    monkeypatch.setattr(SessionDB, "_sqlite_supports_fts5", lambda self, cursor: False)
    migrated = SessionDB(db_path)
    try:
        assert migrated._conn.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION
        remaining = migrated._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'messages_fts_cjk_%'"
        ).fetchall()
        assert remaining == []
    finally:
        migrated.close()