"""Causal contract tests for the fenced complete-export core component.

The capture cases seed a real SessionDB before importing/calling the new module.  They
use a real temporary stream and do not manufacture an owner-stopped receipt or use a
raw SQLite backup.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from hermes_state import SessionDB
from hermes_state_complete_export_policy import (
    BUSINESS_METADATA,
    BUSINESS_PAYLOAD,
    CONTROL_RECOVERY,
    DELIVERY_METADATA,
    DERIVED_INTERNAL,
    MIXED_REQUIRES_PROJECTION,
    OPTIONAL_TELEGRAM_TABLE_POLICIES,
    SCHEMA32_FIELD_POLICY_REVIEW_SHA256,
    SCHEMA32_SCHEMA_CENSUS_SHA256,
    SCHEMA32_SOURCE_ID,
    SCHEMA32_WRITER_SOURCE_ID,
    SECURITY_IDENTITY,
    TABLE_POLICIES,
    ModelConfigProjectionRefused,
    project_model_config,
    schema_policy_for_version,
    validate_schema,
)


class _DelegatingSink:
    """Inject one sink-boundary result while retaining a real temporary file."""

    def __init__(self, raw, *, write_result=None, on_flush=None, seekable=True):
        self._raw = raw
        self._write_result = write_result
        self._on_flush = on_flush
        self._seekable = seekable

    def write(self, data):
        self._raw.write(data)
        if callable(self._write_result):
            return self._write_result(len(data))
        return self._write_result if self._write_result is not None else None

    def flush(self):
        if self._on_flush is not None:
            self._on_flush()
        return self._raw.flush()

    def tell(self):
        if not self._seekable:
            raise io.UnsupportedOperation("tell")
        return self._raw.tell()

    def seek(self, *args):
        if not self._seekable:
            raise io.UnsupportedOperation("seek")
        return self._raw.seek(*args)

    def truncate(self, *args):
        if not self._seekable:
            raise io.UnsupportedOperation("truncate")
        return self._raw.truncate(*args)

    def seekable(self):
        return self._seekable


def _drop_sqlite_relation(conn: sqlite3.Connection, name: str) -> None:
    row = conn.execute(
        "SELECT type FROM sqlite_master WHERE name = ? LIMIT 1", (name,)
    ).fetchone()
    if row is not None:
        kind = str(row[0]).upper()
        assert kind in {"TABLE", "VIEW"}
        conn.execute(f'DROP {kind} "{name}"')


_SCHEMA30_FIXTURE = Path(__file__).parent / "fixtures" / "complete_export_schema30.sql"
_SCHEMA30_FIXTURE_SHA256 = "f89dfb66cb33497cf354bd68f80b32287f4fda059f563f60a7792462587a1674"


def _materialize_schema30_profile(tmp_path: Path, version: object = 30) -> tuple[Path, Path]:
    """Materialize the independent pinned schema-30 DDL into an existing profile."""
    profile_home = tmp_path / "schema30-profile"
    profile_home.mkdir()
    state_path = profile_home / "state.db"
    fixture = _SCHEMA30_FIXTURE.read_bytes()
    assert hashlib.sha256(fixture).hexdigest() == _SCHEMA30_FIXTURE_SHA256
    with sqlite3.connect(state_path) as conn:
        conn.executescript(fixture.decode("utf-8"))
        conn.execute(
            "INSERT INTO sessions(id, source, user_id, session_key, chat_id, chat_type, "
            "display_name, origin_json, model, model_config, system_prompt, started_at, "
            "profile_name, transport_profile) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "legacy30-session",
                "telegram",
                "legacy30-user",
                "telegram:legacy30-chat",
                "legacy30-chat",
                "private",
                "Legacy synthetic chat",
                json.dumps({"platform": "telegram", "chat_id": "legacy30-chat"}),
                "legacy30-model",
                json.dumps({"provider": "synthetic-provider", "opaque": "business"}),
                "legacy30 system prompt",
                1.0,
                "default",
                "telegram-default",
            ),
        )
        conn.execute(
            "INSERT INTO messages(session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (
                "legacy30-session",
                "user",
                "legacy30 synthetic business message",
                2.0,
            ),
        )
        conn.execute(
            "INSERT INTO gateway_routing(scope, session_key, entry_json, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (
                "default",
                "telegram:legacy30-chat",
                json.dumps({"session_id": "legacy30-session", "opaque": "business"}),
                3.0,
            ),
        )
        conn.execute(
            "INSERT INTO state_meta(key, value) VALUES (?, ?)",
            ("legacy30-control", json.dumps({"state": "synthetic", "opaque": "business"})),
        )
        if version is not None:
            conn.execute("INSERT INTO schema_version(version) VALUES (?)", (version,))
        conn.commit()
    return profile_home, state_path


def _materialize_schema32_profile(tmp_path: Path) -> tuple[Path, Path]:
    """Create a current real v32 SessionDB profile, then close setup ownership."""
    profile_home = tmp_path / "schema32-profile"
    profile_home.mkdir()
    state_path = profile_home / "state.db"
    db = SessionDB(state_path)
    _seed_real_session_db(db)
    db.close()
    return profile_home, state_path


_SCHEMA31_FIXTURE = Path(__file__).parent / "fixtures" / "complete_export_schema31.sql"
_SCHEMA31_FIXTURE_SHA256 = "7aea9ccb4293e51d747cf05350268aa34921d60af954c5442acf3f156f4e033d"


class _HistoricalSchema31SessionDB(SessionDB):
    """Use the real SessionDB writer while preserving a source-backed v31 fixture."""

    def _run_data_migrations(self, cursor, current_version, fts5_available):
        assert current_version == 31


def _materialize_historical_schema31_profile(tmp_path: Path) -> tuple[Path, Path]:
    """Materialize DDL extracted from the historical v31 source, without relabeling v32."""
    profile_home = tmp_path / "historical-schema31-profile"
    profile_home.mkdir()
    state_path = profile_home / "state.db"
    fixture = _SCHEMA31_FIXTURE.read_bytes()
    assert hashlib.sha256(fixture).hexdigest() == _SCHEMA31_FIXTURE_SHA256
    with sqlite3.connect(state_path) as conn:
        conn.executescript(fixture.decode("utf-8"))
        conn.execute("INSERT INTO schema_version(version) VALUES (31)")
        conn.execute(
            "INSERT INTO sessions(id, source, started_at, profile_name, transport_profile) "
            "VALUES (?, ?, ?, ?, ?)",
            ("historical31-session", "cli", 1.0, "default", "cli-default"),
        )
        conn.execute(
            "INSERT INTO messages(session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            ("historical31-session", "user", "historical31 business", 2.0),
        )
        conn.commit()
    return profile_home, state_path


def _profile_file_snapshot(state_path: Path) -> tuple[tuple[str, bytes | None], ...]:
    """Capture bytes for the DB and journal companions without opening SQLite."""
    paths = (
        state_path,
        Path(f"{state_path}-wal"),
        Path(f"{state_path}-shm"),
        Path(f"{state_path}-journal"),
    )
    return tuple((str(path), path.read_bytes() if path.exists() else None) for path in paths)



def _assert_persistent_profile_files_unchanged(before, state_path: Path) -> None:
    """Keep database, committed WAL and rollback-journal bytes exact.

    SQLite's wal-index is transient coordination/cache state, not database
    content: https://sqlite.org/walformat.html#the_wal_index_or_shm_file .
    SQLite may rebuild it; production code must never restore or unlink it.
    """
    after = dict(_profile_file_snapshot(state_path))
    for name, expected in before:
        if name.endswith("-shm"):
            continue
        observed = after[name]
        if name.endswith("-wal") and expected is None:
            assert observed in (None, b""), "capture minted committed WAL content"
        else:
            assert observed == expected, f"persistent SQLite state changed: {name}"



def _bind_factory_profile(monkeypatch, complete_export, profile_home: Path, state_path: Path) -> None:
    """Bind the official resolver and test environment to one synthetic profile."""
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", state_path)
    assert complete_export._capture_profile_path() == state_path.resolve()


def _independent_sql_schema(state_path: Path) -> dict[str, tuple[str, ...]]:
    """Read the named user-table fieldset directly from the materialized SQLite schema."""
    with sqlite3.connect(state_path) as conn:
        table_names = tuple(
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        )
        return {
            table: tuple(
                row[1]
                for row in conn.execute(
                    f'PRAGMA table_info("{table.replace(chr(34), chr(34) * 2)}")'
                )
            )
            for table in table_names
        }


def _seed_real_session_db(db: SessionDB) -> None:
    """Create representative business, relation, control, and Telegram rows."""
    db.create_session(
        "session-1",
        source="telegram",
        model="synthetic-model",
        model_config={
            "provider": "synthetic-provider",
            "api_mode": "responses",
            "api_key": "<SECRET>",
            "opaque_business": {"text": "api_key is user-authored text"},
        },
        system_prompt="synthetic system prompt",
        user_id="user-1",
        session_key="telegram:chat-1",
        chat_id="chat-1",
        chat_type="private",
        thread_id="thread-1",
        display_name="Synthetic chat",
        origin_json=json.dumps(
            {"platform": "telegram", "chat_id": "chat-1", "chat_name": "Opaque display text"}
        ),
        profile_name="default",
        transport_profile="telegram-default",
    )
    db.append_message(
        "session-1",
        role="user",
        content='{"api_key":"user-authored opaque content"}',
        reasoning='{"opaque":"api_key appears in business reasoning"}',
        display_metadata={"opaque": "api_key appears in display metadata"},
        timestamp=1.0,
    )
    db.append_message(
        "session-1",
        role="assistant",
        content="synthetic answer",
        tool_name="synthetic_tool",
        tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "synthetic"}}],
        timestamp=2.0,
    )

    # These methods exercise the actual optional Telegram DDL and relation writers.
    db.apply_telegram_topic_migration()
    db.enable_telegram_topic_mode(
        chat_id="chat-1",
        user_id="user-1",
        profile_name="default",
        has_topics_enabled=True,
        allows_users_to_create_topics=True,
    )
    db.bind_telegram_topic(
        chat_id="chat-1",
        thread_id="thread-1",
        user_id="user-1",
        session_key="telegram:chat-1",
        session_id="session-1",
        profile_name="default",
    )

    def seed_control_rows(conn: Any) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO runtime_epoch(singleton, epoch, instance_id) VALUES (1, 1, 'fixture')"
        )
        conn.execute(
            "INSERT OR REPLACE INTO gateway_routing(scope, session_key, entry_json, updated_at) "
            "VALUES ('default', 'telegram:chat-1', ?, 3.0)",
            (json.dumps({"session_key": "telegram:chat-1", "display_name": "Opaque"}),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO gateway_hygiene_state(session_key, failure_streak) VALUES (?, 0)",
            ("telegram:chat-1",),
        )
        conn.execute(
            "INSERT OR REPLACE INTO conversation_generations(source, session_key, generation) "
            "VALUES ('telegram', 'telegram:chat-1', 1)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO gateway_heartbeats(backend_id, pid, started_at, last_heartbeat, profile, host) "
            "VALUES ('backend-1', 1, 1.0, 3.0, 'default', 'synthetic-host')"
        )
        conn.execute(
            "INSERT OR REPLACE INTO compression_locks(session_id, holder, acquired_at, expires_at) "
            "VALUES ('session-1', 'fixture', 1.0, 30.0)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO session_turn_leases(conversation_id, holder, acquired_at, expires_at) "
            "VALUES ('session-1', 'fixture', 1.0, 30.0)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO session_model_usage(session_id, model, task, input_tokens, output_tokens) "
            "VALUES ('session-1', 'synthetic-model', 'synthetic-task', 2, 3)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO session_admissions("
            "admission_id, request_id, principal_id, target_session_id, lineage_json, payload_json, "
            "payload_digest, intent, status, outcome, owner_epoch, generation) "
            "VALUES ('admission-1', 'request-1', 'principal-1', 'session-1', ?, ?, "
            "'digest-1', 'queue', 'queued', NULL, NULL, NULL)",
            (json.dumps(["session-1"]), json.dumps({"input": {"opaque": "business"}, "intent": "queue"})),
        )
        conn.execute(
            "INSERT OR REPLACE INTO worker_executions("
            "execution_id, session_id, kind, owner_epoch, generation, status, adoption_digest, last_sequence) "
            "VALUES ('execution-1', 'session-1', 'compute', 1, 0, 'registered', 'digest-2', 1)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO worker_receipts(execution_id, sequence, payload_digest, result_json) "
            "VALUES ('execution-1', 1, 'digest-3', ?)",
            (json.dumps({"opaque_result": "business"}),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO async_delegations("
            "delegation_id, origin_session, origin_ui_session_id, state, dispatched_at, updated_at, "
            "event_json, result_json, task_json, origin_session_id) "
            "VALUES ('delegation-1', 'session-1', 'ui-1', 'completed', 1.0, 3.0, ?, ?, ?, 'session-1')",
            (
                json.dumps({"event": "synthetic", "opaque": "business"}),
                json.dumps({"result": "synthetic", "opaque": "business"}),
                json.dumps({"task": "synthetic", "opaque": "business"}),
            ),
        )
        conn.execute(
            "INSERT OR REPLACE INTO state_meta(key, value) VALUES ('synthetic-business-key', ?)",
            (json.dumps({"opaque": "business"}),),
        )

    db._execute_write(seed_control_rows)


@pytest.fixture
def seeded_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    _seed_real_session_db(db)
    try:
        yield db
    finally:
        db.close()


def test_live_schema_is_closed_and_optional_telegram_is_explicit(seeded_db):
    with seeded_db._read_ctx() as conn:
        validation = validate_schema(conn, schema_policy=schema_policy_for_version(32))
        present = set(validation.present_tables)
        actual_columns = {
            table: tuple(row[1] for row in conn.execute(f'PRAGMA table_info("{table}")'))
            for table in present
            if table in TABLE_POLICIES
        }
    assert validation.valid, validation.refusal_reason()
    assert validation.optional_tables == tuple(
        sorted(name for name, policy in OPTIONAL_TELEGRAM_TABLE_POLICIES.items() if name in present)
    )
    assert set(TABLE_POLICIES) <= present
    dispositions = {
        column.disposition for policy in TABLE_POLICIES.values() for column in policy.columns
    }
    assert {
        BUSINESS_PAYLOAD,
        BUSINESS_METADATA,
        CONTROL_RECOVERY,
        DELIVERY_METADATA,
        SECURITY_IDENTITY,
        MIXED_REQUIRES_PROJECTION,
        DERIVED_INTERNAL,
    } <= dispositions
    for table, policy in TABLE_POLICIES.items():
        if table not in present:
            assert policy.optional
            continue
        assert actual_columns[table] == policy.column_names
        assert all(column.raw_value_retention == "exact" for column in policy.columns)
        assert all(column.disposition for column in policy.columns)


def test_model_config_boundary_preserves_opaque_business_members():
    # Keep these bytes deliberately awkward: a projection must not round-trip
    # through binary64 or re-serialize duplicate/Unicode/escaped business values.
    raw = (
        b'{"decimal":900719925474099312345678901234567890,'
        b'"api_key":"<SECRET>","opaque":"\\u0061pi_key is business text",'
        b'"exponent":1e+400,"opaque":2}'
    )
    projection = project_model_config(raw)
    assert projection is not None
    assert projection.raw_value_allowed is False
    assert projection.excluded_json_paths == ("$.api_key",)
    assert b"<SECRET>" not in projection.value
    assert projection.value == (
        b'{"decimal":900719925474099312345678901234567890,'
        b'"opaque":"\\u0061pi_key is business text","exponent":1e+400,"opaque":2}'
    )

    with pytest.raises(ModelConfigProjectionRefused):
        project_model_config(b'{"api_key":"one","api_key":"two","opaque":true}')

    opaque_raw = b'{"opaque":"api_key remains opaque business text"}'
    opaque_projection = project_model_config(opaque_raw)
    assert opaque_projection is not None
    assert opaque_projection.raw_value_allowed is True
    assert opaque_projection.value == opaque_raw


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b'{"api_key":"<SECRET>","opaque":1}', b'{"opaque":1}'),
        (b'{"opaque":1,"api_key":"<SECRET>"}', b'{"opaque":1}'),
        (b'{"api_key":"<SECRET>"}', b'{}'),
        (
            b'{"\\u0061pi_key":"<SECRET>","opaque":"\\u6d4b"}',
            b'{"opaque":"\\u6d4b"}',
        ),
        (
            '{"api_key":"<SECRET>","hebrew":"שלום"}'.encode(),
            '{"hebrew":"שלום"}'.encode(),
        ),
    ],
)
def test_model_config_projection_positions_and_unicode_are_byte_exact(raw, expected):
    projection = project_model_config(raw)
    assert projection is not None
    assert projection.raw_value_allowed is False
    assert projection.excluded_json_paths == ("$.api_key",)
    assert projection.value == expected
    assert b"<SECRET>" not in projection.value


def test_model_config_projection_keeps_nested_and_numeric_opaque_bytes():
    raw = (
        b'{"nested":{"api_key":"opaque"},"big":90071992547409931234567890,'
        b'"negative_zero":-0,"exponent":1e+400,"duplicate":1,"duplicate":2}'
    )
    projection = project_model_config(raw)
    assert projection is not None
    assert projection.raw_value_allowed is True
    assert projection.value == raw


@pytest.mark.parametrize(
    "raw",
    [
        b'{"api_key":',
        b'{"api_key":"<SECRET>"} trailing',
        b'{"api_key":NaN}',
        b'{"api_key":"\xff"}',
    ],
)
def test_model_config_projection_refuses_malformed_or_nonstandard_json(raw):
    with pytest.raises(ModelConfigProjectionRefused):
        project_model_config(raw)


@pytest.mark.parametrize("raw", [None, b"null", b"42", b'["api_key","<SECRET>"]'])
def test_model_config_projection_keeps_nonobjects_opaque(raw):
    projection = project_model_config(raw)
    if raw is None:
        assert projection is None
    else:
        assert projection is not None
        assert projection.raw_value_allowed is True
        assert projection.value == raw


def test_unknown_core_relation_is_an_explicit_schema_refusal(seeded_db):
    seeded_db._execute_write(
        lambda conn: conn.execute("CREATE TABLE unregistered_relation (value TEXT NOT NULL)")
    )
    with seeded_db._read_ctx() as conn:
        validation = validate_schema(conn, schema_policy=schema_policy_for_version(32))
    assert "unregistered_relation" in validation.unknown_tables
    assert not validation.valid


def test_schema_refuses_same_columns_view_for_a_policy_table(seeded_db):
    def replace_with_view(conn: sqlite3.Connection) -> None:
        conn.execute("DROP TABLE runtime_epoch")
        conn.execute(
            "CREATE VIEW runtime_epoch AS "
            "SELECT 1 AS singleton, 1 AS epoch, 'fixture' AS instance_id"
        )

    seeded_db._execute_write(replace_with_view)
    with seeded_db._read_ctx() as conn:
        validation = validate_schema(conn, schema_policy=schema_policy_for_version(32))
    assert not validation.valid
    assert "runtime_epoch" in validation.relation_type_mismatches


def test_schema_refuses_missing_required_trusted_extension_table(seeded_db):
    with seeded_db._read_ctx() as conn:
        validation = validate_schema(
            conn,
            schema_policy=schema_policy_for_version(32),
            registered_extension_tables={"addon_table": ("value",)},
        )
    assert not validation.valid
    assert "addon_table" in validation.missing_extension_tables


def test_schema_refuses_wrong_derived_relation_type_and_columns(seeded_db):
    def replace_derived_relation(conn: sqlite3.Connection) -> None:
        _drop_sqlite_relation(conn, "messages_fts_src")
        conn.execute(
            "CREATE TABLE messages_fts_src "
            "(id INTEGER, content TEXT, tool_name TEXT, tool_calls TEXT)"
        )

    seeded_db._execute_write(replace_derived_relation)
    with seeded_db._read_ctx() as conn:
        validation = validate_schema(conn, schema_policy=schema_policy_for_version(32))
    assert not validation.valid
    assert "messages_fts_src" in validation.derived_relation_mismatches


def test_schema_reports_absent_derived_relation_explicitly(seeded_db):
    seeded_db._execute_write(
        lambda conn: _drop_sqlite_relation(conn, "messages_fts_src")
    )
    with seeded_db._read_ctx() as conn:
        validation = validate_schema(conn, schema_policy=schema_policy_for_version(32))
    reported = set(validation.absent_optional_derived_relations) | set(
        validation.missing_required_derived_relations
    )
    assert "messages_fts_src" in reported


@pytest.mark.parametrize("existing_profile", [False, True])
def test_normal_profile_without_optional_telegram_tables_exports_business(
    tmp_path, monkeypatch, existing_profile
):
    """Normal CLI profiles never need Telegram migration to be exportable."""
    complete_export = importlib.import_module("hermes_state_complete_export")
    profile_home = tmp_path / "ordinary-profile"
    profile_home.mkdir()
    state_path = profile_home / "state.db"
    db = SessionDB(state_path)
    try:
        db.create_session("ordinary-session", source="cli")
        db.append_message(
            "ordinary-session", role="user", content="ordinary non-Telegram business"
        )
        with db._read_ctx() as connection:
            present = {
                row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        assert not set(OPTIONAL_TELEGRAM_TABLE_POLICIES) & present
        if existing_profile:
            db.close()
            _bind_factory_profile(monkeypatch, complete_export, profile_home, state_path)
            context = complete_export.open_existing_complete_state_capture()
        else:
            context = complete_export.open_complete_state_capture(db)
        stream = io.BytesIO()
        with context as capture:
            assert capture.manifest.optional_tables == ()
            receipt = capture.write_to(stream)
        assert receipt.record_count > 0
        assert b"ordinary non-Telegram business" in stream.getvalue()
    finally:
        db.close()


def test_positive_core_capture_from_real_db(seeded_db, tmp_path):
    """A valid source and sink reach the complete fenced core implementation."""
    complete_export = importlib.import_module("hermes_state_complete_export")
    with TemporaryDirectory(dir=tmp_path) as directory:
        output_path = Path(directory) / "core.capture"
        with output_path.open("w+b") as stream:
            with complete_export.open_complete_state_capture(seeded_db) as capture:
                assert capture.manifest.schema_version == 32
                assert capture.manifest.official_schema_source == SCHEMA32_SOURCE_ID
                assert capture.manifest.policy_writer_provenance == SCHEMA32_WRITER_SOURCE_ID
                assert capture.manifest.schema_census_sha256 == SCHEMA32_SCHEMA_CENSUS_SHA256
                assert capture.manifest.field_policy_review_sha256 == SCHEMA32_FIELD_POLICY_REVIEW_SHA256
                receipt = capture.write_to(stream)
            stream.flush()
            stream.seek(0)
            payload = stream.read()

        assert receipt.record_count >= 1
        assert receipt.byte_count == len(payload)
        assert b"synthetic answer" in payload
        assert b"<SECRET>" not in payload
        records = [json.loads(line) for line in payload.splitlines()]
        assert records[0]["kind"] == "manifest"
        assert records[0]["schema_version"] == 32
        assert records[0]["schema_census_sha256"] == SCHEMA32_SCHEMA_CENSUS_SHA256
        assert records[0]["field_policy_review_sha256"] == SCHEMA32_FIELD_POLICY_REVIEW_SHA256
        assert "telegram_dm_topic_mode" in records[0]["optional_tables"]
        assert any(record["relations"] for record in records[1:] if record["evidence_class"] == "core")
        assert any(record["evidence_class"] == "derived_evidence" for record in records[1:])


def test_historical_schema31_fixture_captures_without_relabeling_current32(tmp_path):
    """A real v31 DDL fixture remains supported independently of the current v32 writer."""
    complete_export = importlib.import_module("hermes_state_complete_export")
    _profile_home, state_path = _materialize_historical_schema31_profile(tmp_path)
    db = _HistoricalSchema31SessionDB(state_path)
    try:
        with db._read_ctx() as conn:
            assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 31
        with io.BytesIO() as stream:
            with complete_export.open_complete_state_capture(db) as capture:
                assert capture.manifest.schema_version == 31
                assert capture.manifest.official_schema_source == "official-521-schema31"
                assert capture.manifest.schema_census_sha256
                assert capture.manifest.field_policy_review_sha256
                receipt = capture.write_to(stream)
            assert receipt.record_count > 0
            manifest = json.loads(stream.getvalue().splitlines()[0])
            assert "schema_census_sha256" not in manifest
            assert "field_policy_review_sha256" not in manifest
            assert b"historical31 business" in stream.getvalue()
    finally:
        db.close()


@pytest.mark.parametrize("schema_version", [30, 32])
def test_existing_profile_factory_captures_supported_version_without_pre_capture_mutation(
    tmp_path, monkeypatch, schema_version
):
    """The official resolver/factory must attach an existing 30/32 profile safely."""
    complete_export = importlib.import_module("hermes_state_complete_export")
    if schema_version == 30:
        profile_home, state_path = _materialize_schema30_profile(tmp_path)
    else:
        profile_home, state_path = _materialize_schema32_profile(tmp_path)
    _bind_factory_profile(monkeypatch, complete_export, profile_home, state_path)
    independent_schema = _independent_sql_schema(state_path) if schema_version == 30 else None
    before = _profile_file_snapshot(state_path)
    with TemporaryDirectory(dir=tmp_path) as directory:
        with (Path(directory) / "core.capture").open("w+b") as stream:
            with complete_export.open_existing_complete_state_capture(
                trusted_extensions=(),
                budget=complete_export.CaptureBudget(timeout_seconds=10),
            ) as capture:
                assert capture.manifest.schema_version == schema_version
                receipt = capture.write_to(stream)
            stream.flush()
            stream.seek(0)
            payload = stream.read()
    assert receipt.byte_count == len(payload)
    assert payload
    records = [json.loads(line) for line in payload.splitlines()]
    manifest = records[0]
    assert manifest["kind"] == "manifest"
    if schema_version == 30:
        assert "schema_census_sha256" not in manifest
        assert "field_policy_review_sha256" not in manifest
    core_records = [record for record in records[1:] if record["evidence_class"] == "core"]
    assert core_records
    if independent_schema is not None:
        assert tuple(sorted(manifest["core_tables"])) == tuple(independent_schema)
        for record in core_records:
            assert record["table"] in independent_schema
            assert (
                tuple(field["column"] for field in record["fields"])
                == independent_schema[record["table"]]
            )
        assert any(
            record["table"] == "sessions"
            and any(
                field["column"] == "id"
                and field["value"].get("text") == "legacy30-session"
                for field in record["fields"]
            )
            for record in core_records
        )
        assert any(
            record["table"] == "messages"
            and any(
                field["column"] == "content"
                and field["value"].get("text") == "legacy30 synthetic business message"
                for field in record["fields"]
            )
            for record in core_records
        )
        assert any(
            record["table"] == "state_meta"
            and any(
                field["column"] == "key"
                and field["value"].get("text") == "legacy30-control"
                for field in record["fields"]
            )
            and any(
                field["column"] == "value"
                and field["value"].get("text") == '{"state": "synthetic", "opaque": "business"}'
                for field in record["fields"]
            )
            for record in core_records
        )
    else:
        assert any(
            record["table"] == "messages"
            and any(
                relation["target_key"] == "str:session-1"
                for relation in record["relations"]
            )
            for record in core_records
        )
    _assert_persistent_profile_files_unchanged(before, state_path)


def test_existing_profile_factory_refuses_missing_profile_without_creating_state(
    tmp_path, monkeypatch
):
    complete_export = importlib.import_module("hermes_state_complete_export")
    profile_home = tmp_path / "missing-profile"
    state_path = profile_home / "state.db"
    _bind_factory_profile(monkeypatch, complete_export, profile_home, state_path)
    with pytest.raises(complete_export.CoreCaptureRefused):
        with complete_export.open_existing_complete_state_capture():
            pass
    assert not profile_home.exists()


def test_existing_profile_factory_refuses_invalid_header_without_repairing_file(
    tmp_path, monkeypatch
):
    complete_export = importlib.import_module("hermes_state_complete_export")
    profile_home = tmp_path / "invalid-profile"
    profile_home.mkdir()
    state_path = profile_home / "state.db"
    state_path.write_bytes(b"not a sqlite database")
    before = _profile_file_snapshot(state_path)
    _bind_factory_profile(monkeypatch, complete_export, profile_home, state_path)
    with pytest.raises(complete_export.CoreCaptureRefused):
        with complete_export.open_existing_complete_state_capture():
            pass
    _assert_persistent_profile_files_unchanged(before, state_path)


@pytest.mark.parametrize("unsupported_version", [29, 33, "thirty"])
def test_existing_profile_factory_refuses_unsupported_or_noninteger_version(
    tmp_path, monkeypatch, unsupported_version
):
    complete_export = importlib.import_module("hermes_state_complete_export")
    if unsupported_version == 33:
        profile_home, state_path = _materialize_schema32_profile(tmp_path)
        with sqlite3.connect(state_path) as conn:
            conn.execute("UPDATE schema_version SET version = 33")
            conn.commit()
    else:
        profile_home, state_path = _materialize_schema30_profile(tmp_path, unsupported_version)
    before = _profile_file_snapshot(state_path)
    _bind_factory_profile(monkeypatch, complete_export, profile_home, state_path)
    with pytest.raises(complete_export.CoreCaptureRefused):
        with complete_export.open_existing_complete_state_capture():
            pass
    _assert_persistent_profile_files_unchanged(before, state_path)


def test_existing_profile_factory_refuses_missing_schema_version(tmp_path, monkeypatch):
    complete_export = importlib.import_module("hermes_state_complete_export")
    profile_home, state_path = _materialize_schema30_profile(tmp_path, version=None)
    before = _profile_file_snapshot(state_path)
    _bind_factory_profile(monkeypatch, complete_export, profile_home, state_path)
    with pytest.raises(complete_export.CoreCaptureRefused):
        with complete_export.open_existing_complete_state_capture():
            pass
    _assert_persistent_profile_files_unchanged(before, state_path)


def test_existing_profile_factory_refuses_duplicate_schema_versions(tmp_path, monkeypatch):
    complete_export = importlib.import_module("hermes_state_complete_export")
    profile_home, state_path = _materialize_schema30_profile(tmp_path, version=30)
    with sqlite3.connect(state_path) as conn:
        conn.execute("INSERT INTO schema_version(version) VALUES (30)")
        conn.commit()
    before = _profile_file_snapshot(state_path)
    _bind_factory_profile(monkeypatch, complete_export, profile_home, state_path)
    with pytest.raises(complete_export.CoreCaptureRefused):
        with complete_export.open_existing_complete_state_capture():
            pass
    _assert_persistent_profile_files_unchanged(before, state_path)


def test_existing_profile_factory_refuses_live_same_path_owner(tmp_path, monkeypatch):
    complete_export = importlib.import_module("hermes_state_complete_export")
    profile_home = tmp_path / "live-profile"
    profile_home.mkdir()
    state_path = profile_home / "state.db"
    owner = SessionDB(state_path)
    try:
        _bind_factory_profile(monkeypatch, complete_export, profile_home, state_path)
        with pytest.raises(complete_export.CoreCaptureRefused):
            with complete_export.open_existing_complete_state_capture():
                pass
        owner._conn.execute("SELECT 1").fetchone()
    finally:
        owner.close()


def test_existing_profile_factory_does_not_follow_mutable_session_db_binding(
    tmp_path, monkeypatch
):
    """An addon-like SessionDB substitution must not run constructor DDL in the factory."""
    complete_export = importlib.import_module("hermes_state_complete_export")
    import hermes_state

    profile_home, state_path = _materialize_schema32_profile(tmp_path)
    before = _profile_file_snapshot(state_path)
    constructor_calls: list[str] = []
    synthetic_table = "synthetic_constructor_migration"

    class ConstructorWritingSessionDB(SessionDB):
        def __init__(self, *args, **kwargs):
            constructor_calls.append("called")
            super().__init__(*args, **kwargs)
            self._execute_write(
                lambda conn: conn.execute(
                    f'CREATE TABLE "{synthetic_table}" (id INTEGER PRIMARY KEY)'
                )
            )

    # A correct factory either binds the immutable official class or refuses the
    # substituted binding before constructing a handle.  Both outcomes preserve
    # the existing profile and never execute the synthetic migration.
    monkeypatch.setattr(hermes_state, "SessionDB", ConstructorWritingSessionDB)
    _bind_factory_profile(monkeypatch, complete_export, profile_home, state_path)
    try:
        with TemporaryDirectory(dir=tmp_path) as directory:
            with (Path(directory) / "core.capture").open("w+b") as stream:
                with complete_export.open_existing_complete_state_capture() as capture:
                    capture.write_to(stream)
    except complete_export.CoreCaptureRefused:
        pass

    assert constructor_calls == []
    _assert_persistent_profile_files_unchanged(before, state_path)
    with sqlite3.connect(state_path) as conn:
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ?", (synthetic_table,)
        ).fetchone() is None


@pytest.mark.parametrize("race", ["unlink", "rename"])
def test_existing_profile_factory_uses_existing_only_rw_attach_across_path_race(
    tmp_path, monkeypatch, race
):
    """A path disappearing after preflight cannot make SQLite create a new profile."""
    complete_export = importlib.import_module("hermes_state_complete_export")
    import hermes_state_dbfile

    profile_home, state_path = _materialize_schema32_profile(tmp_path)
    original_bytes = state_path.read_bytes()
    replacement = profile_home / "state.db.replaced"
    original_connect = hermes_state_dbfile._connect_tracked_db
    raced = False

    def race_before_connect(db_path, *args, **kwargs):
        nonlocal raced
        if not raced:
            raced = True
            path = Path(kwargs.get("tracking_path", db_path))
            if race == "unlink":
                path.unlink()
            else:
                path.rename(replacement)
        return original_connect(db_path, *args, **kwargs)

    monkeypatch.setattr(hermes_state_dbfile, "_connect_tracked_db", race_before_connect)
    _bind_factory_profile(monkeypatch, complete_export, profile_home, state_path)
    with pytest.raises(complete_export.CoreCaptureRefused):
        with complete_export.open_existing_complete_state_capture():
            pass

    assert raced
    assert not state_path.exists(), "mode=rw attach must not recreate the raced path"
    if race == "rename":
        assert replacement.read_bytes() == original_bytes


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_rows", True),
        ("max_rows", 1.5),
        ("max_rows", float("nan")),
        ("max_rows", float("inf")),
        ("max_bytes", False),
        ("max_bytes", 1.5),
        ("max_bytes", float("nan")),
        ("max_bytes", float("inf")),
        ("timeout_seconds", True),
        ("timeout_seconds", float("nan")),
        ("timeout_seconds", float("inf")),
    ],
)
def test_capture_budget_rejects_nonfinite_boolean_or_nonintegral_limits(field, value):
    complete_export = importlib.import_module("hermes_state_complete_export")
    with pytest.raises(ValueError):
        complete_export.CaptureBudget(**{field: value})


def test_capture_budget_accepts_finite_positive_exact_integral_limits_without_a_hidden_cap():
    complete_export = importlib.import_module("hermes_state_complete_export")
    budget = complete_export.CaptureBudget(
        max_rows=10**18,
        max_bytes=2**63 - 1,
        timeout_seconds=0.000001,
    )
    assert budget.max_rows == 10**18
    assert budget.max_bytes == 2**63 - 1
    assert budget.timeout_seconds == 0.000001


def _leave_preexisting_wal_frames(state_path: Path) -> None:
    """Leave a linked WAL generation behind without running SQLite close cleanup."""
    script = """
import os
import sqlite3
import sys

path = sys.argv[1]
conn = sqlite3.connect(path, isolation_level=None)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA wal_autocheckpoint=0")
conn.execute("BEGIN IMMEDIATE")
conn.execute(
    "INSERT INTO messages(session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
    ("session-1", "assistant", "pre-existing WAL frame", 99.0),
)
conn.execute("COMMIT")
assert os.path.exists(path + "-wal")
os._exit(0)
"""
    subprocess.run(
        [sys.executable, "-c", script, os.fspath(state_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert Path(os.fspath(state_path) + "-wal").exists()


@pytest.mark.parametrize("disable_close_checkpoint", [False, True], ids=["normal", "no-ckpt-unavailable"])
def test_existing_profile_factory_preserves_preexisting_wal_bytes(
    tmp_path, monkeypatch, disable_close_checkpoint
):
    """Existing-only capture must not checkpoint or discard a pre-existing WAL generation."""
    complete_export = importlib.import_module("hermes_state_complete_export")
    import hermes_state

    profile_home, state_path = _materialize_schema32_profile(tmp_path)
    _leave_preexisting_wal_frames(state_path)
    before = _profile_file_snapshot(state_path)
    original_open = Path.open
    original_unlink = Path.unlink

    def reject_sidecar_write(path, mode="r", *args, **kwargs):
        if str(path).endswith(("-shm", "-wal", "-journal")) and any(
            flag in mode for flag in ("w", "a", "x", "+")
        ):
            raise AssertionError("capture manually rewrites a SQLite sidecar")
        return original_open(path, mode, *args, **kwargs)

    def reject_sidecar_unlink(path, *args, **kwargs):
        if str(path).endswith(("-shm", "-wal", "-journal")):
            raise AssertionError("capture manually unlinks a SQLite sidecar")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_sidecar_write)
    monkeypatch.setattr(Path, "unlink", reject_sidecar_unlink)
    if disable_close_checkpoint:
        monkeypatch.setattr(hermes_state, "_close_time_checkpoint_configurable", lambda: False)
    _bind_factory_profile(monkeypatch, complete_export, profile_home, state_path)

    try:
        with TemporaryDirectory(dir=tmp_path) as directory:
            with (Path(directory) / "core.capture").open("w+b") as stream:
                with complete_export.open_existing_complete_state_capture() as capture:
                    capture.write_to(stream)
                stream.seek(0)
                assert b"pre-existing WAL frame" in stream.read()
    except complete_export.CoreCaptureRefused:
        # A runtime without a safe NO_CKPT_ON_CLOSE mechanism may fail closed
        # before publication; it must still leave the existing generation exact.
        if not disable_close_checkpoint:
            raise

    _assert_persistent_profile_files_unchanged(before, state_path)


@pytest.mark.platforms("linux")
def test_existing_profile_constructor_failure_releases_wal_guard_before_close(
    tmp_path, monkeypatch
):
    """A post-open constructor error must release the OFD guard before dropping SQLite."""
    complete_export = importlib.import_module("hermes_state_complete_export")
    import hermes_state
    import hermes_state_lockguard

    if not hermes_state_lockguard.supported():
        pytest.skip("OFD WAL lock guard is unavailable on this platform")

    profile_home, state_path = _materialize_schema32_profile(tmp_path)
    raw = sqlite3.connect(str(state_path), isolation_level=None)
    try:
        raw.execute("PRAGMA journal_mode=WAL")
        raw.execute("PRAGMA wal_autocheckpoint=0")
        raw.execute(
            "INSERT INTO messages(session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            ("session-1", "assistant", "constructor failure WAL", 100.0),
        )
        events: list[str] = []
        original_release = hermes_state_lockguard.release
        original_close = hermes_state.SessionDB._close_connection_quietly

        def record_release(held):
            events.append("release")
            return original_release(held)

        def record_close(conn):
            events.append("close")
            return original_close(conn)

        def fail_after_open(self):
            raise RuntimeError("synthetic post-open constructor failure")

        monkeypatch.setattr(hermes_state_lockguard, "release", record_release)
        monkeypatch.setattr(
            hermes_state.SessionDB,
            "_close_connection_quietly",
            staticmethod(record_close),
        )
        monkeypatch.setattr(hermes_state.SessionDB, "_record_db_file_identity", fail_after_open)
        _bind_factory_profile(monkeypatch, complete_export, profile_home, state_path)

        with pytest.raises(complete_export.CoreCaptureRefused):
            with complete_export.open_existing_complete_state_capture():
                pass

        assert "release" in events
        assert "close" in events
        assert events.index("release") < events.index("close")
        identity = (state_path.stat().st_dev, state_path.stat().st_ino)
        assert hermes_state_lockguard._HANDLES.get(identity, 0) == 0
    finally:
        raw.close()


def test_capture_deadline_interrupts_a_real_derived_relation_count(
    seeded_db, tmp_path
):
    """A finite deadline must interrupt an expensive derived SQL count itself."""
    complete_export = importlib.import_module("hermes_state_complete_export")

    def replace_source_view(conn: Any) -> None:
        conn.execute("DROP VIEW messages_fts_src")
        conn.execute(
            """
            CREATE VIEW messages_fts_src AS
            WITH RECURSIVE
              left_value(value) AS (
                SELECT 1
                UNION ALL
                SELECT value + 1 FROM left_value WHERE value < 100000
              ),
              right_value(value) AS (
                SELECT 1
                UNION ALL
                SELECT value + 1 FROM right_value WHERE value < 100000
              )
            SELECT left_value.value * 100000 + right_value.value AS id,
                   NULL AS content,
                   NULL AS tool_name,
                   NULL AS tool_calls
            FROM left_value CROSS JOIN right_value
            """
        )

    seeded_db._execute_write(replace_source_view)
    started: list[float] = []
    capture_ref: list[Any] = []
    watchdog: list[threading.Timer] = []

    def trace(sql: str) -> None:
        if not started and "COUNT(*)" in sql.upper() and "MESSAGES_FTS" in sql.upper():
            started.append(time.monotonic())
            capture_ref[0]._deadline = started[0] - 1.0

    seeded_db._conn.set_trace_callback(trace)
    try:
        with TemporaryDirectory(dir=tmp_path) as directory:
            output_path = Path(directory) / "core.capture"
            with output_path.open("w+b") as stream:
                with pytest.raises(complete_export.CoreCaptureRefused):
                    with complete_export.open_complete_state_capture(
                        seeded_db,
                        budget=complete_export.CaptureBudget(timeout_seconds=10),
                    ) as capture:
                        capture_ref.append(capture)
                        timer = threading.Timer(3.0, seeded_db._conn.interrupt)
                        watchdog.append(timer)
                        timer.start()
                        capture.write_to(stream)
                stream.seek(0)
                assert stream.read() == b""
    finally:
        seeded_db._conn.set_trace_callback(None)
        for timer in watchdog:
            timer.cancel()
            timer.join(timeout=2)

    assert started, "the real derived relation count was not reached"
    assert time.monotonic() - started[0] < 2.0


def test_capture_holds_the_canonical_writer_lock_until_stream_completion(seeded_db, tmp_path):
    complete_export = importlib.import_module("hermes_state_complete_export")
    writer_started = threading.Event()
    writer_finished = threading.Event()

    def attempt_write() -> None:
        writer_started.set()
        seeded_db._execute_write(
            lambda conn: conn.execute(
                "INSERT OR REPLACE INTO state_meta(key, value) VALUES ('after-capture', 'yes')"
            )
        )
        writer_finished.set()

    writer = threading.Thread(target=attempt_write)
    with complete_export.open_complete_state_capture(seeded_db) as capture:
        with TemporaryDirectory(dir=tmp_path) as directory:
            with (Path(directory) / "core.capture").open("w+b") as stream:
                capture.write_to(stream)
        assert writer_finished.is_set() is False
        writer.start()
        assert writer_started.wait(2)
        assert writer_finished.wait(2) is False
    assert writer_finished.wait(2)
    writer.join(timeout=2)


def test_capture_budget_refuses_before_publishing_partial_stream(seeded_db, tmp_path):
    complete_export = importlib.import_module("hermes_state_complete_export")
    budget = complete_export.CaptureBudget(max_rows=1, max_bytes=1024 * 1024, timeout_seconds=2)
    with TemporaryDirectory(dir=tmp_path) as directory:
        output_path = Path(directory) / "core.capture"
        with output_path.open("w+b") as stream:
            with pytest.raises(complete_export.CoreCaptureRefused):
                with complete_export.open_complete_state_capture(seeded_db, budget=budget) as capture:
                    capture.write_to(stream)
            stream.seek(0)
            assert stream.read() == b""


def test_capture_rejects_nested_or_stale_sessions_and_releases_the_lock(seeded_db, tmp_path, monkeypatch):
    complete_export = importlib.import_module("hermes_state_complete_export")
    with pytest.raises(complete_export.CoreCaptureRefused):
        with complete_export.open_complete_state_capture(seeded_db) as outer:
            with pytest.raises(complete_export.CoreCaptureRefused):
                with complete_export.open_complete_state_capture(
                    seeded_db,
                    budget=complete_export.CaptureBudget(timeout_seconds=0.01),
                ):
                    pass
            original_check = seeded_db._raise_if_db_replaced
            checks = 0

            def stale_after_open() -> None:
                nonlocal checks
                checks += 1
                if checks > 2:
                    raise RuntimeError("synthetic generation drift")
                original_check()

            monkeypatch.setattr(seeded_db, "_raise_if_db_replaced", stale_after_open)
            with TemporaryDirectory(dir=tmp_path) as directory:
                with (Path(directory) / "core.capture").open("w+b") as stream:
                    outer.write_to(stream)
    assert seeded_db._lock.acquire(blocking=False)
    seeded_db._lock.release()


def test_capture_retains_blob_and_opaque_text_storage_classes(seeded_db, tmp_path):
    complete_export = importlib.import_module("hermes_state_complete_export")
    seeded_db._execute_write(
        lambda conn: conn.execute(
            "UPDATE messages SET content = ? WHERE id = 2",
            ("opaque café",),
        )
    )
    # The real messages_display_identity_update trigger clears the derived identity
    # whenever content changes.  Set the BLOB in a second owner write so this fixture
    # proves the encoder against an actual stored BLOB rather than the trigger's NULL.
    seeded_db._execute_write(
        lambda conn: conn.execute(
            "UPDATE messages SET display_identity = ? WHERE id = 2",
            (b"\x00\xff",),
        )
    )
    with seeded_db._read_ctx() as conn:
        storage = conn.execute(
            "SELECT typeof(display_identity), hex(display_identity) FROM messages WHERE id = 2"
        ).fetchone()
    assert tuple(storage) == ("blob", "00FF")
    with TemporaryDirectory(dir=tmp_path) as directory:
        with (Path(directory) / "core.capture").open("w+b") as stream:
            with complete_export.open_complete_state_capture(seeded_db) as capture:
                capture.write_to(stream)
            stream.seek(0)
            payload = stream.read()
    assert b"opaque caf" in payload
    records = [json.loads(line) for line in payload.splitlines()]
    message = next(
        record
        for record in records
        if record.get("table") == "messages"
        and record.get("primary_key") == [{"type": "integer", "decimal": "2"}]
    )
    display_identity = next(
        field for field in message["fields"] if field["column"] == "display_identity"
    )
    assert display_identity["value"] == {
        "type": "blob",
        "base64": "AP8=",
        "raw_b64": "AP8=",
    }


def test_capture_begin_immediate_respects_remaining_deadline_and_restores_busy_timeout(
    seeded_db,
):
    complete_export = importlib.import_module("hermes_state_complete_export")
    foreign = sqlite3.connect(str(seeded_db.db_path), timeout=0, isolation_level=None)
    try:
        foreign.execute("PRAGMA busy_timeout=0")
        foreign.execute("BEGIN IMMEDIATE")
        seeded_db._conn.execute("PRAGMA busy_timeout=1000")
        started = time.monotonic()
        with pytest.raises(complete_export.CoreCaptureRefused):
            with complete_export.open_complete_state_capture(
                seeded_db,
                budget=complete_export.CaptureBudget(timeout_seconds=0.05),
            ):
                pass
        assert time.monotonic() - started < 0.5
        assert seeded_db._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 1000
    finally:
        foreign.rollback()
        foreign.close()


@pytest.mark.parametrize(
    "write_result",
    [
        pytest.param(None, id="none"),
        pytest.param(lambda size: float(size), id="noninteger"),
        pytest.param(lambda size: size - 1, id="short"),
        pytest.param(lambda size: size + 1, id="overlong"),
    ],
)
def test_capture_refuses_unknown_or_inexact_sink_write_counts(seeded_db, tmp_path, write_result):
    complete_export = importlib.import_module("hermes_state_complete_export")
    with TemporaryDirectory(dir=tmp_path) as directory:
        output_path = Path(directory) / "core.capture"
        with output_path.open("w+b") as raw:
            sink = _DelegatingSink(raw, write_result=write_result)
            with pytest.raises(complete_export.CoreCaptureFailed):
                with complete_export.open_complete_state_capture(seeded_db) as capture:
                    capture.write_to(sink)
            raw.seek(0)
            assert raw.read() == b""


def test_capture_refuses_nonseekable_sink_before_publishing_bytes(seeded_db, tmp_path):
    complete_export = importlib.import_module("hermes_state_complete_export")
    with TemporaryDirectory(dir=tmp_path) as directory:
        output_path = Path(directory) / "core.capture"
        with output_path.open("w+b") as raw:
            sink = _DelegatingSink(raw, write_result=lambda size: size, seekable=False)
            with pytest.raises(complete_export.CoreCaptureFailed):
                with complete_export.open_complete_state_capture(seeded_db) as capture:
                    capture.write_to(sink)
            raw.seek(0)
            assert raw.read() == b""


def test_capture_final_currentness_failure_discards_already_written_bytes(
    seeded_db, tmp_path, monkeypatch
):
    complete_export = importlib.import_module("hermes_state_complete_export")
    flushed = False

    def mark_flushed() -> None:
        nonlocal flushed
        flushed = True

    original_check = seeded_db._raise_if_db_replaced

    def fail_after_sink_flush() -> None:
        original_check()
        if flushed:
            raise RuntimeError("synthetic final currentness drift")

    monkeypatch.setattr(seeded_db, "_raise_if_db_replaced", fail_after_sink_flush)
    with TemporaryDirectory(dir=tmp_path) as directory:
        output_path = Path(directory) / "core.capture"
        with output_path.open("w+b") as raw:
            sink = _DelegatingSink(raw, write_result=lambda size: size, on_flush=mark_flushed)
            with pytest.raises(complete_export.CoreCaptureRefused):
                with complete_export.open_complete_state_capture(seeded_db) as capture:
                    capture.write_to(sink)
            raw.seek(0)
            assert raw.read() == b""


def test_capture_discards_bytes_when_source_drifts_after_write_before_context_exit(
    seeded_db, tmp_path, monkeypatch
):
    """A post-write context-exit currentness failure must still roll back the sink."""
    complete_export = importlib.import_module("hermes_state_complete_export")
    original_check = seeded_db._raise_if_db_replaced

    def fail_after_write() -> None:
        original_check()
        raise RuntimeError("synthetic post-write currentness drift")

    with TemporaryDirectory(dir=tmp_path) as directory:
        output_path = Path(directory) / "core.capture"
        with output_path.open("w+b") as raw:
            with pytest.raises(complete_export.CoreCaptureRefused):
                with complete_export.open_complete_state_capture(seeded_db) as capture:
                    capture.write_to(raw)
                    monkeypatch.setattr(seeded_db, "_raise_if_db_replaced", fail_after_write)
            raw.seek(0)
            assert raw.read() == b""


def test_capture_deadline_covers_extension_preflight(seeded_db, tmp_path, monkeypatch):
    complete_export = importlib.import_module("hermes_state_complete_export")
    original_registry = complete_export._extension_registry

    def slow_registry(extensions):
        time.sleep(0.2)  # health: allow HX013 -- delayed extension preflight is the elapsed-budget boundary under test
        return original_registry(extensions)

    monkeypatch.setattr(complete_export, "_extension_registry", slow_registry)
    with TemporaryDirectory(dir=tmp_path) as directory:
        with (Path(directory) / "core.capture").open("w+b") as stream:
            with pytest.raises(complete_export.CoreCaptureRefused):
                with complete_export.open_complete_state_capture(
                    seeded_db,
                    budget=complete_export.CaptureBudget(timeout_seconds=0.15),
                ) as capture:
                    capture.write_to(stream)
            stream.seek(0)
            assert stream.read() == b""


def test_capture_deadline_covers_sink_flush_and_discards_bytes(seeded_db, tmp_path):
    complete_export = importlib.import_module("hermes_state_complete_export")

    def delayed_flush() -> None:
        time.sleep(0.35)  # health: allow HX013 -- delayed sink flush is the elapsed-budget boundary under test

    with TemporaryDirectory(dir=tmp_path) as directory:
        output_path = Path(directory) / "core.capture"
        with output_path.open("w+b") as raw:
            sink = _DelegatingSink(raw, write_result=lambda size: size, on_flush=delayed_flush)
            with pytest.raises(complete_export.CoreCaptureRefused):
                with complete_export.open_complete_state_capture(
                    seeded_db,
                    budget=complete_export.CaptureBudget(timeout_seconds=0.2),
                ) as capture:
                    capture.write_to(sink)
            raw.seek(0)
            assert raw.read() == b""


def test_capture_reads_more_than_the_old_hidden_row_boundary(seeded_db, tmp_path):
    complete_export = importlib.import_module("hermes_state_complete_export")
    row_count = 100_001

    def seed_many(conn: Any) -> None:
        conn.executemany(
            "INSERT INTO messages(session_id, role, content, timestamp) VALUES (?, 'user', ?, ?)",
            (("session-1", f"row-{index}", float(index)) for index in range(row_count)),
        )

    seeded_db._execute_write(seed_many)
    budget = complete_export.CaptureBudget(
        max_rows=row_count + 10_000, max_bytes=512 * 1024 * 1024, timeout_seconds=120
    )
    with TemporaryDirectory(dir=tmp_path) as directory:
        with (Path(directory) / "core.capture").open("w+b") as stream:
            with complete_export.open_complete_state_capture(seeded_db, budget=budget) as capture:
                receipt = capture.write_to(stream)
    assert receipt.record_count >= row_count


def test_manifest_preserves_existing_positional_constructor_contract():
    """Adding v32 provenance must not shift the published v30/v31 arguments."""
    from hermes_state_complete_export import CoreCaptureManifest

    manifest = CoreCaptureManifest(
        "synthetic-api", "synthetic-policy", "synthetic-source", "synthetic-writer", 31,
        ("sessions",), ("optional",), ("extension",), ("validated",),
        ("declared",), ("derived",),
    )
    assert manifest.core_tables == ("sessions",)
    assert manifest.optional_tables == ("optional",)
    assert manifest.extension_sources == ("extension",)
    assert manifest.validated_extension_tables == ("validated",)
    assert manifest.declared_relations == ("declared",)
    assert manifest.derived_relations == ("derived",)
    assert manifest.schema_census_sha256 == ""
    assert manifest.field_policy_review_sha256 == ""
