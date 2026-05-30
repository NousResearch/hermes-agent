"""Tests for the report-only Discord thread mapping diagnostic."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.discord_thread_mapping_diagnostic import (
    inspect_discord_thread_mapping,
    main,
)
from gateway.session import SessionSource, SessionStore, build_session_key


THREAD_ID = "222222222222222222"
EXPECTED_THREAD_KEY = f"agent:main:discord:thread:{THREAD_ID}:{THREAD_ID}"
PRIVATE_TRANSCRIPT_TEXT = "do not leak this transcript content"


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    import hermes_state

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")

    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    yield store
    if store._db:
        store._db.close()


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id=THREAD_ID,
        chat_type="thread",
        user_id="333333333333333333",
        thread_id=THREAD_ID,
        parent_chat_id="111111111111111111",
        guild_id="999999999999999999",
    )


def _create_thread_session(store: SessionStore) -> tuple[str, Path, Path]:
    source = _source()
    assert build_session_key(source) == EXPECTED_THREAD_KEY
    entry = store.get_or_create_session(source)
    store.append_to_transcript(
        entry.session_id,
        {"role": "user", "content": PRIVATE_TRANSCRIPT_TEXT},
    )
    store.append_to_transcript(
        entry.session_id,
        {"role": "assistant", "content": "metadata only"},
    )
    if store._db:
        store._db.close()
    return entry.session_id, store.sessions_dir / "sessions.json", store._db.db_path


def test_mapped_discord_thread_reports_active_session_metadata(temp_store):
    session_id, sessions_json, state_db = _create_thread_session(temp_store)

    report = inspect_discord_thread_mapping(
        sessions_json=sessions_json,
        state_db=state_db,
        session_key=EXPECTED_THREAD_KEY,
    )

    assert report["session_key"] == EXPECTED_THREAD_KEY
    assert report["mapping"]["exists"] is True
    assert report["mapping"]["active_session_id"] == session_id
    assert report["active_session"]["session_id"] == session_id
    assert report["active_session"]["message_count"] == 2
    assert PRIVATE_TRANSCRIPT_TEXT not in json.dumps(report)


def test_missing_mapping_reports_candidates_without_transcript_content(temp_store):
    session_id, sessions_json, state_db = _create_thread_session(temp_store)
    sessions_json.unlink()

    report = inspect_discord_thread_mapping(
        sessions_json=sessions_json,
        state_db=state_db,
        session_key=EXPECTED_THREAD_KEY,
    )

    assert report["mapping"]["exists"] is False
    assert "missing" in report["mapping"]["status"]
    assert report["active_session"] is None
    assert report["state_db"]["can_reliably_match_orphans_to_thread_key"] is True
    assert report["exact_orphan_sessions"][0]["session_id"] == session_id
    assert report["exact_orphan_sessions"][0]["message_count"] == 2
    assert PRIVATE_TRANSCRIPT_TEXT not in json.dumps(report)


def test_missing_mapping_exactly_matches_orphan_by_routing_metadata(temp_store):
    session_id, sessions_json, state_db = _create_thread_session(temp_store)
    sessions_json.unlink()

    report = inspect_discord_thread_mapping(
        sessions_json=sessions_json,
        state_db=state_db,
        session_key=EXPECTED_THREAD_KEY,
    )

    assert report["mapping"]["exists"] is False
    assert report["state_db"]["can_reliably_match_orphans_to_thread_key"] is True
    assert report["exact_orphan_sessions"][0]["session_id"] == session_id
    assert report["exact_orphan_sessions"][0]["match_type"] == "exact_metadata_match"
    assert report["exact_orphan_sessions"][0]["routing_metadata"]["thread_id"] == THREAD_ID
    assert report["candidate_orphan_sessions"] == []
    assert PRIVATE_TRANSCRIPT_TEXT not in json.dumps(report)


def test_old_state_db_without_routing_metadata_degrades_to_candidate_only(tmp_path):
    sessions_json = tmp_path / "sessions.json"
    sessions_json.write_text("{}", encoding="utf-8")
    state_db = tmp_path / "old-state.db"
    conn = sqlite3.connect(state_db)
    try:
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                user_id TEXT,
                parent_session_id TEXT,
                started_at REAL NOT NULL,
                ended_at REAL,
                end_reason TEXT,
                message_count INTEGER DEFAULT 0
            );
            INSERT INTO sessions (
                id, source, user_id, parent_session_id, started_at,
                ended_at, end_reason, message_count
            ) VALUES (
                'old-session', 'discord', '333333333333333333', NULL,
                1.0, NULL, NULL, 1
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

    report = inspect_discord_thread_mapping(
        sessions_json=sessions_json,
        state_db=state_db,
        session_key=EXPECTED_THREAD_KEY,
    )

    assert report["state_db"]["can_reliably_match_orphans_to_thread_key"] is False
    assert report["exact_orphan_sessions"] == []
    assert report["candidate_orphan_sessions"][0]["session_id"] == "old-session"
    assert report["candidate_orphan_sessions"][0]["match_type"] == "candidate_only"


def test_missing_sessions_json_is_graceful(temp_store):
    _session_id, sessions_json, state_db = _create_thread_session(temp_store)
    sessions_json.unlink()

    report = inspect_discord_thread_mapping(
        sessions_json=sessions_json,
        state_db=state_db,
        session_key=EXPECTED_THREAD_KEY,
    )

    assert report["sessions_json"]["exists"] is False
    assert report["mapping"]["exists"] is False
    assert report["errors"] == []


def test_missing_state_db_is_graceful(tmp_path):
    sessions_json = tmp_path / "sessions.json"
    sessions_json.write_text("{}", encoding="utf-8")

    report = inspect_discord_thread_mapping(
        sessions_json=sessions_json,
        state_db=tmp_path / "missing-state.db",
        session_key=EXPECTED_THREAD_KEY,
    )

    assert report["state_db"]["exists"] is False
    assert report["state_db"]["available"] is False
    assert report["candidate_orphan_sessions"] == []


def test_diagnostic_is_read_only(temp_store):
    _session_id, sessions_json, state_db = _create_thread_session(temp_store)
    before = {
        sessions_json: sessions_json.stat().st_mtime_ns,
        state_db: state_db.stat().st_mtime_ns,
    }

    inspect_discord_thread_mapping(
        sessions_json=sessions_json,
        state_db=state_db,
        session_key=EXPECTED_THREAD_KEY,
    )

    after = {
        sessions_json: sessions_json.stat().st_mtime_ns,
        state_db: state_db.stat().st_mtime_ns,
    }
    assert after == before


def test_cli_prints_json_report_without_transcript_content(temp_store, capsys):
    _session_id, sessions_json, state_db = _create_thread_session(temp_store)

    exit_code = main(
        [
            "--sessions-json",
            str(sessions_json),
            "--state-db",
            str(state_db),
            "--session-key",
            EXPECTED_THREAD_KEY,
        ]
    )

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert exit_code == 0
    assert payload["mapping"]["exists"] is True
    assert PRIVATE_TRANSCRIPT_TEXT not in output
