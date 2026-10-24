"""Schema-30 optional Telegram preservation contracts for AGC-1663.

These cases attach an existing d337 profile through the production factory.  The
positive case seeds every optional scalar from an independent census-backed DDL
fixture; the no-option case uses the unchanged schema-30 fixture and compares the
complete stream to a frozen two-line legacy wire projection.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import sqlite3
import struct
from pathlib import Path
from tempfile import TemporaryDirectory

from hermes_state_complete_export_policy import (
    SCHEMA30_SOURCE_ID,
    SCHEMA30_TABLE_POLICIES,
    SCHEMA30_WRITER_SOURCE_ID,
)


_FIXTURES = Path(__file__).parent / "fixtures"
_SCHEMA30_FIXTURE = _FIXTURES / "complete_export_schema30.sql"
_SCHEMA30_FIXTURE_SHA256 = "f89dfb66cb33497cf354bd68f80b32287f4fda059f563f60a7792462587a1674"
_OPTIONAL_FIXTURE = _FIXTURES / "complete_export_schema30_optional_telegram.sql"
_OPTIONAL_FIXTURE_SHA256 = "5af4f3150d94b84e88c80bea8e3983da34b1125e6287a1b203d7859a7c97ddaa"
_NO_OPTION_EXPECTED = _FIXTURES / "complete_export_schema30_no_option.expected.jsonl"
_NO_OPTION_EXPECTED_SHA256 = "cdbfd84ff0c319c437d74a6ccc81b247ccc3945d3b8f3dacfeb4d44cb8847f64"

_OPTIONAL_COLUMNS = {
    "telegram_dm_topic_bindings": (
        "profile_name",
        "chat_id",
        "thread_id",
        "user_id",
        "session_key",
        "session_id",
        "managed_mode",
        "linked_at",
        "updated_at",
    ),
    "telegram_dm_topic_mode": (
        "profile_name",
        "chat_id",
        "user_id",
        "enabled",
        "activated_at",
        "updated_at",
        "has_topics_enabled",
        "allows_users_to_create_topics",
        "capability_checked_at",
        "intro_message_id",
        "pinned_message_id",
    ),
}

_OPTIONAL_VALUES = {
    "telegram_dm_topic_bindings": (
        "opt-profile",
        "opt-chat",
        "opt-thread",
        "opt-user",
        "telegram:opt-chat",
        "optional30-session",
        "managed",
        13.0,
        14.0,
    ),
    "telegram_dm_topic_mode": (
        "opt-profile",
        "opt-chat",
        "opt-user",
        1,
        10.0,
        11.0,
        1,
        1,
        12.0,
        "opt-intro",
        "opt-pin",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _materialize_schema30_profile(tmp_path: Path, *, with_optional: bool) -> tuple[Path, Path]:
    profile_home = tmp_path / ("schema30-optional-profile" if with_optional else "schema30-no-option-profile")
    profile_home.mkdir()
    state_path = profile_home / "state.db"
    assert _sha256(_SCHEMA30_FIXTURE) == _SCHEMA30_FIXTURE_SHA256
    if with_optional:
        assert _sha256(_OPTIONAL_FIXTURE) == _OPTIONAL_FIXTURE_SHA256
    with sqlite3.connect(state_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(_SCHEMA30_FIXTURE.read_text(encoding="utf-8"))
        if with_optional:
            conn.executescript(_OPTIONAL_FIXTURE.read_text(encoding="utf-8"))
            conn.execute(
                "INSERT INTO sessions(id, source, started_at, profile_name, transport_profile) "
                "VALUES (?, ?, ?, ?, ?)",
                ("optional30-session", "telegram", 1.0, "opt-profile", "telegram-opt"),
            )
            conn.execute(
                "INSERT INTO telegram_dm_topic_mode("
                "profile_name, chat_id, user_id, enabled, activated_at, updated_at, "
                "has_topics_enabled, allows_users_to_create_topics, capability_checked_at, "
                "intro_message_id, pinned_message_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                _OPTIONAL_VALUES["telegram_dm_topic_mode"],
            )
            conn.execute(
                "INSERT INTO telegram_dm_topic_bindings("
                "profile_name, chat_id, thread_id, user_id, session_key, session_id, "
                "managed_mode, linked_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                _OPTIONAL_VALUES["telegram_dm_topic_bindings"],
            )
        conn.execute("INSERT INTO schema_version(version) VALUES (30)")
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        conn.commit()
    return profile_home, state_path


def _bind_factory_profile(monkeypatch, complete_export, profile_home: Path, state_path: Path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", state_path)
    assert complete_export._capture_profile_path() == state_path.resolve()


def _profile_file_snapshot(state_path: Path) -> tuple[tuple[str, bytes | None], ...]:
    paths = (
        state_path,
        Path(f"{state_path}-wal"),
        Path(f"{state_path}-shm"),
        Path(f"{state_path}-journal"),
    )
    return tuple((str(path), path.read_bytes() if path.exists() else None) for path in paths)


def _assert_persistent_profile_files_unchanged(
    before: tuple[tuple[str, bytes | None], ...], state_path: Path
) -> None:
    after = dict(_profile_file_snapshot(state_path))
    for name, expected in before:
        if name.endswith("-shm"):
            continue
        observed = after[name]
        if name.endswith("-wal") and expected is None:
            assert observed in (None, b"")
        else:
            assert observed == expected, f"existing schema-30 profile changed: {name}"


def _text_wire(value: str) -> dict[str, str]:
    encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
    return {"type": "text", "text": value, "utf8_b64": encoded, "raw_b64": encoded}


def _real_wire(value: float) -> dict[str, str]:
    encoded = base64.b64encode(struct.pack(">d", value)).decode("ascii")
    return {"type": "real", "ieee754_be_b64": encoded}


def _integer_wire(value: int) -> dict[str, str]:
    return {"type": "integer", "decimal": str(value)}


def test_schema30_optional_telegram_capture_preserves_all_values_and_sessions_fk(
    tmp_path, monkeypatch
):
    """The existing d337 factory captures both opt-in tables and their FK edge."""
    complete_export = importlib.import_module("hermes_state_complete_export")
    profile_home, state_path = _materialize_schema30_profile(tmp_path, with_optional=True)
    _bind_factory_profile(monkeypatch, complete_export, profile_home, state_path)
    before = _profile_file_snapshot(state_path)

    with TemporaryDirectory(dir=tmp_path) as directory:
        with (Path(directory) / "schema30.optional.capture").open("w+b") as stream:
            with complete_export.open_existing_complete_state_capture(
                trusted_extensions=(),
                budget=complete_export.CaptureBudget(timeout_seconds=10),
            ) as capture:
                assert capture.manifest.schema_version == 30
                assert capture.manifest.official_schema_source == SCHEMA30_SOURCE_ID
                assert capture.manifest.policy_writer_provenance == SCHEMA30_WRITER_SOURCE_ID
                assert capture.manifest.optional_tables == (
                    "telegram_dm_topic_bindings",
                    "telegram_dm_topic_mode",
                )
                assert "official-e003-schema32" not in capture.manifest.official_schema_source
                assert "current-writer-18c525b669b6ea4ba62e6967d228af21d89aa9d3" not in (
                    capture.manifest.policy_writer_provenance
                )
                receipt = capture.write_to(stream)
            stream.flush()
            stream.seek(0)
            payload = stream.read()

    assert receipt.byte_count == len(payload)
    records = [json.loads(line) for line in payload.splitlines()]
    optional_records = {
        record["table"]: record
        for record in records[1:]
        if record["table"] in _OPTIONAL_COLUMNS
    }
    assert set(optional_records) == set(_OPTIONAL_COLUMNS)
    for table, columns in _OPTIONAL_COLUMNS.items():
        record = optional_records[table]
        assert tuple(field["column"] for field in record["fields"]) == columns
        expected_values = _OPTIONAL_VALUES[table]
        expected_wire = [
            _text_wire(value) if isinstance(value, str)
            else _real_wire(value) if isinstance(value, float)
            else _integer_wire(value)
            for value in expected_values
        ]
        assert [field["value"] for field in record["fields"]] == expected_wire

    binding = optional_records["telegram_dm_topic_bindings"]
    assert binding["relations"] == [
        {
            "relation": "telegram_dm_topic_bindings.session_id->sessions.id",
            "source_key": "str:optional30-session",
            "target_key": "str:optional30-session",
        }
    ]
    manifest = records[0]
    assert "telegram_dm_topic_bindings.session_id->sessions.id" in manifest["declared_relations"]
    assert _OPTIONAL_COLUMNS["telegram_dm_topic_bindings"] == tuple(
        column.name for column in SCHEMA30_TABLE_POLICIES["telegram_dm_topic_bindings"].columns
    )
    assert _OPTIONAL_COLUMNS["telegram_dm_topic_mode"] == tuple(
        column.name for column in SCHEMA30_TABLE_POLICIES["telegram_dm_topic_mode"].columns
    )
    assert all(
        SCHEMA30_TABLE_POLICIES[table].optional for table in _OPTIONAL_COLUMNS
    )
    _assert_persistent_profile_files_unchanged(before, state_path)


def test_schema30_no_option_wire_matches_frozen_legacy_projection(tmp_path, monkeypatch):
    """Adding optional policy entries leaves the historical no-option stream exact."""
    complete_export = importlib.import_module("hermes_state_complete_export")
    assert _sha256(_NO_OPTION_EXPECTED) == _NO_OPTION_EXPECTED_SHA256
    profile_home, state_path = _materialize_schema30_profile(tmp_path, with_optional=False)
    _bind_factory_profile(monkeypatch, complete_export, profile_home, state_path)
    before = _profile_file_snapshot(state_path)

    with TemporaryDirectory(dir=tmp_path) as directory:
        with (Path(directory) / "schema30.no-option.capture").open("w+b") as stream:
            with complete_export.open_existing_complete_state_capture(
                trusted_extensions=(),
                budget=complete_export.CaptureBudget(timeout_seconds=10),
            ) as capture:
                assert capture.manifest.schema_version == 30
                assert capture.manifest.optional_tables == ()
                receipt = capture.write_to(stream)
            stream.flush()
            stream.seek(0)
            payload = stream.read()

    assert receipt.byte_count == len(payload)
    assert payload == _NO_OPTION_EXPECTED.read_bytes()
    _assert_persistent_profile_files_unchanged(before, state_path)
