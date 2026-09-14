"""Replay diverted JSONL via ``hermes sessions import --from diverted``."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from hermes_constants import get_hermes_home
from hermes_state import SessionDB


def _write_diverted(path: Path, records) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def test_import_diverted_appends_jsonl_into_session(capsys):
    home = get_hermes_home()
    session_id = "sess-diverted"
    db = SessionDB()
    try:
        db.create_session(session_id, "cli")
        db.append_message(session_id, "user", "already-in-store")
    finally:
        db.close()

    jsonl = _write_diverted(
        home / "sessions" / f"{session_id}.jsonl",
        [
            {"role": "user", "content": "hello-from-divert"},
            {"role": "assistant", "content": "reply-from-divert"},
        ],
    )

    from hermes_cli.foreign_sessions import run_sessions_import

    inspect_args = Namespace(
        from_source="diverted",
        path=None,
        session_id=session_id,
        inspect_only=True,
    )
    assert run_sessions_import(inspect_args) is not None
    inspect_out = capsys.readouterr().out
    assert str(jsonl) in inspect_out
    assert "2" in inspect_out

    db = SessionDB()
    try:
        contents = [m.get("content") for m in db.get_messages(session_id)]
        assert "hello-from-divert" not in contents
        assert "reply-from-divert" not in contents
    finally:
        db.close()

    apply_args = Namespace(
        from_source="diverted",
        path=None,
        session_id=session_id,
        inspect_only=False,
    )
    assert run_sessions_import(apply_args) == session_id

    db = SessionDB()
    try:
        contents = [m.get("content") for m in db.get_messages(session_id)]
        assert "already-in-store" in contents
        assert "hello-from-divert" in contents
        assert "reply-from-divert" in contents
        before = len(contents)
    finally:
        db.close()

    assert run_sessions_import(apply_args) == session_id
    db = SessionDB()
    try:
        assert len(db.get_messages(session_id)) == before
    finally:
        db.close()


def test_reimport_after_session_continues_does_not_duplicate():
    """JSONL prefix must not re-append after the live session grew past it."""
    from hermes_cli.foreign_sessions import run_sessions_import

    home = get_hermes_home()
    session_id = "sess-diverted-continue"
    db = SessionDB()
    try:
        db.create_session(session_id, "cli")
    finally:
        db.close()

    jsonl = _write_diverted(
        home / "sessions" / f"{session_id}.jsonl",
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
        ],
    )
    apply_args = Namespace(
        from_source="diverted",
        path=str(jsonl),
        session_id=session_id,
        inspect_only=False,
    )
    assert run_sessions_import(apply_args) == session_id

    db = SessionDB()
    try:
        db.append_message(session_id, "user", "C")
        db.append_message(session_id, "assistant", "D")
    finally:
        db.close()

    assert run_sessions_import(apply_args) == session_id
    db = SessionDB()
    try:
        pairs = [(m.get("role"), m.get("content")) for m in db.get_messages(session_id)]
        assert pairs == [
            ("user", "A"),
            ("assistant", "B"),
            ("user", "C"),
            ("assistant", "D"),
        ]
    finally:
        db.close()


def test_import_preserves_null_content_tool_call_rows():
    """Assistant tool_calls with null content and tool results keep native fields."""
    from hermes_cli.foreign_sessions import run_sessions_import
    from hermes_state import divert_session_transcript_jsonl

    home = get_hermes_home()
    session_id = "sess-diverted-tools"
    db = SessionDB()
    try:
        db.create_session(session_id, "cli")
    finally:
        db.close()

    jsonl = divert_session_transcript_jsonl(
        session_id,
        [
            {"role": "user", "content": "calculate"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "add", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "42", "tool_call_id": "call_1", "tool_name": "add"},
        ],
    )
    assert jsonl is not None
    apply_args = Namespace(
        from_source="diverted",
        path=str(jsonl),
        session_id=session_id,
        inspect_only=False,
    )
    assert run_sessions_import(apply_args) == session_id

    db = SessionDB()
    try:
        convo = db.get_messages_as_conversation(session_id)
        roles = [m.get("role") for m in convo]
        assert roles == ["user", "assistant", "tool"]
        tool_call = convo[1]
        assert tool_call.get("tool_calls")
        assert tool_call["tool_calls"][0]["id"] == "call_1"
        result = convo[2]
        assert result.get("tool_call_id") == "call_1"
        assert result.get("content") == "42"
    finally:
        db.close()


def _apply_diverted(jsonl: Path, session_id: str, db=None):
    from hermes_cli.foreign_sessions import run_sessions_import

    return run_sessions_import(Namespace(
        from_source="diverted",
        path=str(jsonl),
        session_id=session_id,
        inspect_only=False,
    ), db=db)


def test_import_into_empty_destination_ignores_source_sidecar():
    """A source-file watermark must not skip restore into another session or rebuilt db."""
    home = get_hermes_home()
    jsonl = _write_diverted(
        home / "sessions" / "shared-divert.jsonl",
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
        ],
    )
    db = SessionDB()
    try:
        db.create_session("sess-first", "cli")
    finally:
        db.close()
    assert _apply_diverted(jsonl, "sess-first") == "sess-first"

    db = SessionDB()
    try:
        db.create_session("sess-second", "cli")
    finally:
        db.close()
    assert _apply_diverted(jsonl, "sess-second") == "sess-second"
    db = SessionDB()
    try:
        pairs = [(m.get("role"), m.get("content")) for m in db.get_messages("sess-second")]
        assert pairs == [("user", "A"), ("assistant", "B")]
    finally:
        db.close()

    recovered = home / "recovered-state.db"
    db2 = SessionDB(db_path=recovered)
    try:
        assert _apply_diverted(jsonl, "sess-first", db=db2) == "sess-first"
        pairs = [(m.get("role"), m.get("content")) for m in db2.get_messages("sess-first")]
        assert pairs == [("user", "A"), ("assistant", "B")]
    finally:
        db2.close()


def test_retry_after_partial_apply_does_not_duplicate():
    """Committed prefix plus a later source append must apply only the missing tail."""
    home = get_hermes_home()
    session_id = "sess-diverted-partial"
    jsonl = _write_diverted(
        home / "sessions" / f"{session_id}.jsonl",
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
        ],
    )
    db = SessionDB()
    try:
        db.create_session(session_id, "cli")
    finally:
        db.close()
    assert _apply_diverted(jsonl, session_id) == session_id

    _write_diverted(
        jsonl,
        [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "C"},
            {"role": "assistant", "content": "D"},
        ],
    )
    db = SessionDB()
    try:
        db.append_message(session_id, "user", "C")
    finally:
        db.close()

    assert _apply_diverted(jsonl, session_id) == session_id
    db = SessionDB()
    try:
        pairs = [(m.get("role"), m.get("content")) for m in db.get_messages(session_id)]
        assert pairs == [
            ("user", "A"),
            ("assistant", "B"),
            ("user", "C"),
            ("assistant", "D"),
        ]
    finally:
        db.close()
