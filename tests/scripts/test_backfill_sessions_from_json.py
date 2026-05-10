"""Tests for manually repairing SQLite sessions from JSON session logs."""

import importlib
import json
from pathlib import Path

import pytest

from hermes_state import SessionDB


def _backfill_module():
    try:
        return importlib.import_module("scripts.backfill_sessions_from_json")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing backfill script module: {exc}")


@pytest.fixture()
def state_db(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    yield db, db_path
    db.close()


@pytest.fixture()
def sessions_dir(tmp_path):
    path = tmp_path / "sessions"
    path.mkdir()
    return path


def _messages(unique_term: str = "JSONBACKFILLNEEDLE"):
    return [
        {"role": "user", "content": f"please preserve {unique_term}"},
        {
            "role": "assistant",
            "content": "I will call a tool.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup_context",
                        "arguments": json.dumps({"query": unique_term}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "tool_name": "lookup_context",
            "content": f"tool result for {unique_term}",
        },
    ]


def _write_session_json(
    sessions_dir: Path,
    session_id: str,
    messages: list[dict],
    *,
    model: str = "test-model",
    platform: str = "cli",
    ended_at: float | None = 123.0,
) -> Path:
    payload = {
        "session_id": session_id,
        "model": model,
        "platform": platform,
        "ended_at": ended_at,
        "message_count": len(messages),
        "messages": messages,
    }
    path = sessions_dir / f"session_{session_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _status(summary: dict, session_id: str) -> str:
    for report in summary["reports"]:
        if report["session_id"] == session_id:
            return report["status"]
    raise AssertionError(f"no report for {session_id}: {summary}")


def test_backfills_ended_zero_message_session_and_reindexes_search(
    state_db, sessions_dir
):
    db, db_path = state_db
    session_id = "ended-zero"
    db.create_session(session_id, source="cli", model="old-model")
    db.end_session(session_id, end_reason="user_exit")
    assert db.get_session(session_id)["message_count"] == 0
    _write_session_json(sessions_dir, session_id, _messages("RESTORED_UNIQUE_TERM"))

    summary = _backfill_module().backfill_sessions_from_json(
        db_path=db_path,
        sessions_dir=sessions_dir,
    )

    assert summary["scanned"] == 1
    assert summary["backfilled"] == 1
    assert summary["errors"] == 0
    assert _status(summary, session_id) == "backfilled"
    assert db.get_session(session_id)["message_count"] == 3
    assert [msg["role"] for msg in db.get_messages(session_id)] == [
        "user",
        "assistant",
        "tool",
    ]
    assert len(db.search_messages("RESTORED_UNIQUE_TERM")) >= 1


def test_second_backfill_is_idempotent_by_default(state_db, sessions_dir):
    db, db_path = state_db
    session_id = "idempotent"
    db.create_session(session_id, source="cli")
    db.end_session(session_id, end_reason="user_exit")
    _write_session_json(sessions_dir, session_id, _messages("IDEMPOTENT_TERM"))

    module = _backfill_module()
    first = module.backfill_sessions_from_json(db_path, sessions_dir)
    second = module.backfill_sessions_from_json(db_path, sessions_dir)

    assert first["backfilled"] == 1
    assert second["backfilled"] == 0
    assert second["skipped_existing"] == 1
    assert _status(second, session_id) == "skipped_existing"
    assert db.get_session(session_id)["message_count"] == 3
    assert len(db.get_messages(session_id)) == 3


def test_dry_run_reports_without_mutating_database(state_db, sessions_dir):
    db, db_path = state_db
    session_id = "dry-run"
    db.create_session(session_id, source="cli")
    db.end_session(session_id, end_reason="user_exit")
    _write_session_json(sessions_dir, session_id, _messages("DRY_RUN_TERM"))

    summary = _backfill_module().backfill_sessions_from_json(
        db_path,
        sessions_dir,
        dry_run=True,
    )

    assert summary["backfilled"] == 0
    assert summary["would_backfill"] == 1
    assert _status(summary, session_id) == "would_backfill"
    assert db.get_session(session_id)["message_count"] == 0
    assert db.get_messages(session_id) == []


def test_active_sessions_are_skipped_unless_included(state_db, sessions_dir):
    db, db_path = state_db
    session_id = "active-session"
    db.create_session(session_id, source="cli")
    assert db.get_session(session_id)["ended_at"] is None
    _write_session_json(sessions_dir, session_id, _messages("ACTIVE_TERM"))

    module = _backfill_module()
    skipped = module.backfill_sessions_from_json(db_path, sessions_dir)
    included = module.backfill_sessions_from_json(
        db_path,
        sessions_dir,
        include_active=True,
    )

    assert skipped["backfilled"] == 0
    assert skipped["skipped_active"] == 1
    assert _status(skipped, session_id) == "skipped_active"
    assert included["backfilled"] == 1
    assert db.get_session(session_id)["message_count"] == 3


def test_force_replaces_stale_database_messages(state_db, sessions_dir):
    db, db_path = state_db
    session_id = "force-repair"
    db.create_session(session_id, source="cli")
    db.append_message(session_id, "user", "stale database transcript")
    db.end_session(session_id, end_reason="user_exit")
    canonical = [
        {"role": "user", "content": "canonical prompt FORCE_CANONICAL"},
        {"role": "assistant", "content": "canonical answer"},
    ]
    _write_session_json(sessions_dir, session_id, canonical)

    module = _backfill_module()
    default = module.backfill_sessions_from_json(db_path, sessions_dir)

    assert default["backfilled"] == 0
    assert default["skipped_existing"] == 1
    assert [msg["content"] for msg in db.get_messages(session_id)] == [
        "stale database transcript"
    ]

    forced = module.backfill_sessions_from_json(
        db_path,
        sessions_dir,
        force=True,
    )

    assert forced["backfilled"] == 1
    assert forced["forced"] == 1
    assert _status(forced, session_id) == "forced"
    assert [
        {"role": msg["role"], "content": msg["content"]}
        for msg in db.get_messages(session_id)
    ] == canonical


def test_malformed_json_reports_error_without_aborting_valid_backfill(
    state_db, sessions_dir
):
    db, db_path = state_db
    valid_id = "valid-new-row"
    _write_session_json(
        sessions_dir,
        valid_id,
        _messages("VALID_BATCH_TERM"),
        platform="telegram",
    )
    (sessions_dir / "session_broken.json").write_text("{not-json", encoding="utf-8")

    summary = _backfill_module().backfill_sessions_from_json(
        db_path,
        sessions_dir,
    )

    assert summary["scanned"] == 2
    assert summary["backfilled"] == 1
    assert summary["created_sessions"] == 1
    assert summary["errors"] == 1
    assert _status(summary, valid_id) == "backfilled"
    row = db.get_session(valid_id)
    assert row["source"] == "telegram"
    assert row["model"] == "test-model"
    assert row["message_count"] == 3
    assert len(db.search_messages("VALID_BATCH_TERM")) >= 1


def test_missing_db_row_real_session_json_is_backfilled_and_tool_name_preserved(
    state_db, sessions_dir
):
    db, db_path = state_db
    session_id = "json-only-real-shape"
    payload = {
        "session_id": session_id,
        "model": "test-model",
        "platform": "cli",
        "session_start": "2026-05-10T00:00:00",
        "last_updated": "2026-05-10T00:01:00",
        "message_count": 3,
        "messages": [
            {"role": "user", "content": "recover JSON_ONLY_REAL_TERM"},
            {"role": "assistant", "content": "calling a tool"},
            {
                "role": "tool",
                "tool_call_id": "call_real",
                "name": "web_search",
                "content": "tool result JSON_ONLY_REAL_TERM",
            },
        ],
    }
    (sessions_dir / f"session_{session_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    summary = _backfill_module().backfill_sessions_from_json(db_path, sessions_dir)

    assert summary["backfilled"] == 1
    assert summary["created_sessions"] == 1
    assert _status(summary, session_id) == "backfilled"
    row = db.get_session(session_id)
    assert row["ended_at"] is not None
    messages = db.get_messages(session_id)
    assert len(messages) == 3
    assert messages[2]["tool_name"] == "web_search"
    assert len(db.search_messages("JSON_ONLY_REAL_TERM")) >= 1


def test_db_write_error_reports_error_and_continues_batch(
    state_db, sessions_dir, monkeypatch
):
    db, db_path = state_db
    bad_id = "write-fails"
    good_id = "write-succeeds"
    for session_id in (bad_id, good_id):
        db.create_session(session_id, source="cli")
        db.end_session(session_id, end_reason="user_exit")
        _write_session_json(
            sessions_dir,
            session_id,
            _messages(f"{session_id}_TERM"),
        )

    module = _backfill_module()
    original_replace = module.SessionDB.replace_messages

    def flaky_replace(self, session_id, messages):
        if session_id == bad_id:
            raise RuntimeError("simulated write failure")
        return original_replace(self, session_id, messages)

    monkeypatch.setattr(module.SessionDB, "replace_messages", flaky_replace)

    summary = module.backfill_sessions_from_json(db_path, sessions_dir)

    assert summary["scanned"] == 2
    assert summary["backfilled"] == 1
    assert summary["errors"] == 1
    assert _status(summary, bad_id) == "error"
    assert _status(summary, good_id) == "backfilled"
    assert db.get_session(bad_id)["message_count"] == 0
    assert db.get_session(good_id)["message_count"] == 3
