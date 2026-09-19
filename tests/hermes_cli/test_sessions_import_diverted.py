"""Replay diverted JSONL via ``hermes sessions import --from diverted``."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

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


def test_retry_after_partial_apply_does_not_duplicate(monkeypatch):
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
    original_append = db.append_message

    def fail_after_commit(*args, **kwargs):
        original_append(*args, **kwargs)
        raise OSError("injected failure after committed append")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(db, "append_message", fail_after_commit)
            assert _apply_diverted(jsonl, session_id, db=db) is None
    finally:
        db.close()

    with jsonl.open("a", encoding="utf-8") as source:
        source.write(json.dumps({"role": "user", "content": "E"}) + "\n")
    assert _apply_diverted(jsonl, session_id) == session_id
    assert _apply_diverted(jsonl, session_id) == session_id
    db = SessionDB()
    try:
        pairs = [(m.get("role"), m.get("content")) for m in db.get_messages(session_id)]
        assert pairs == [
            ("user", "A"),
            ("assistant", "B"),
            ("user", "C"),
            ("assistant", "D"),
            ("user", "E"),
        ]
    finally:
        db.close()


def test_reimport_after_intervening_turns_does_not_duplicate_later_divert():
    """Recovered A/B, ordinary C/D, then diverted E/F must not re-append E/F on retry."""
    from hermes_state import divert_session_transcript_jsonl

    home = get_hermes_home()
    session_id = "sess-diverted-gap"
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

    db = SessionDB()
    try:
        db.append_message(session_id, "user", "C")
        db.append_message(session_id, "assistant", "D")
    finally:
        db.close()

    appended = divert_session_transcript_jsonl(
        session_id,
        [
            {"role": "user", "content": "E"},
            {"role": "assistant", "content": "F"},
        ],
    )
    assert appended == jsonl
    assert _apply_diverted(jsonl, session_id) == session_id
    assert _apply_diverted(jsonl, session_id) == session_id

    db = SessionDB()
    try:
        pairs = [(m.get("role"), m.get("content")) for m in db.get_messages(session_id)]
        assert pairs == [
            ("user", "A"),
            ("assistant", "B"),
            ("user", "C"),
            ("assistant", "D"),
            ("user", "E"),
            ("assistant", "F"),
        ]
    finally:
        db.close()


def test_import_keeps_new_turn_that_reuses_historical_content():
    """A later divert with the same role/content but a new timestamp must still persist."""
    home = get_hermes_home()
    session_id = "sess-diverted-new-status"
    db = SessionDB()
    try:
        db.create_session(session_id, "cli")
        db.append_message(session_id, "user", "status", timestamp=100)
        db.append_message(session_id, "assistant", "working", timestamp=101)
        db.append_message(session_id, "user", "finish", timestamp=102)
        db.append_message(session_id, "assistant", "done", timestamp=103)
    finally:
        db.close()

    jsonl = _write_diverted(
        home / "sessions" / f"{session_id}.jsonl",
        [
            {"role": "user", "content": "status", "timestamp": 200},
            {"role": "assistant", "content": "done", "timestamp": 201},
        ],
    )
    assert _apply_diverted(jsonl, session_id) == session_id
    db = SessionDB()
    try:
        rows = db.get_messages(session_id)
        pairs = [(m.get("role"), m.get("content"), m.get("timestamp")) for m in rows]
        assert pairs == [
            ("user", "status", 100),
            ("assistant", "working", 101),
            ("user", "finish", 102),
            ("assistant", "done", 103),
            ("user", "status", 200),
            ("assistant", "done", 201),
        ]
    finally:
        db.close()


@pytest.mark.parametrize("timestamp", [None, "invalid", float("nan"), float("inf"), -float("inf")])
def test_identityless_collision_retries_and_growth(timestamp):
    """An older identical event is not proof of recovery, even across failed retries."""
    home = get_hermes_home()
    sid = "reload-collision"
    record = {"role": "system", "content": "MCP tools reloaded", "timestamp": timestamp}
    jsonl = _write_diverted(home / "sessions" / f"{sid}.jsonl", [record])
    db = SessionDB()
    try:
        db.create_session(sid, "cli")
        db.append_message(sid, "system", record["content"], timestamp=100)
        assert _apply_diverted(jsonl, sid, db=db) == sid
        assert len(db.get_messages(sid)) == 2
        assert _apply_diverted(jsonl, sid, db=db) == sid
        assert len(db.get_messages(sid)) == 2
        _write_diverted(jsonl, [record, record])
        assert _apply_diverted(jsonl, sid, db=db) == sid
        assert _apply_diverted(jsonl, sid, db=db) == sid
        assert len(db.get_messages(sid)) == 3
    finally:
        db.close()


def test_recovery_match_moves_forward_in_destination():
    home = get_hermes_home()
    sid = "reversed-recovery"
    records = [
        {"role": "user", "content": "status", "timestamp": 200},
        {"role": "assistant", "content": "done", "timestamp": 201},
    ]
    jsonl = _write_diverted(home / "sessions" / f"{sid}.jsonl", records)
    db = SessionDB()
    try:
        db.create_session(sid, "cli")
        for record in reversed(records):
            db.append_message(sid, **record)
        assert _apply_diverted(jsonl, sid, db=db) == sid
        rows = db.get_messages(sid)
        assert [row["content"] for row in rows] == ["done", "status", "done"]
        assert _apply_diverted(jsonl, sid, db=db) == sid
        assert len(db.get_messages(sid)) == 3
    finally:
        db.close()


@pytest.mark.parametrize("encoded_metadata", [False, True])
def test_producer_durable_fields_survive_diversion_and_retry(encoded_metadata):
    from types import SimpleNamespace
    from agent.session_persistence import _db_flush_row
    from agent.turn_context import substitute_api_content
    from hermes_cli.cli_agent_setup_mixin import _collect_resume_entries
    from hermes_state import divert_session_transcript_jsonl

    sid = "durable-divert"
    metadata = {"source": "delegation", "nested": {"task": 1}}
    message = {
        "role": "assistant", "content": "internal handoff", "_compressed_summary": True,
        "display_kind": "hidden",
        "display_metadata": json.dumps(metadata) if encoded_metadata else metadata,
        "platform_message_id": "platform-42", "api_content": " exact API bytes ",
        "finish_reason": "stop", "reasoning": "reason",
        "reasoning_content": "thinking", "reasoning_details": [{"text": "detail"}],
        "codex_reasoning_items": [{"type": "reasoning", "id": "r1"}],
        "codex_message_items": [{"type": "message", "id": "m1"}],
    }
    row = _db_flush_row(SimpleNamespace(), message, False)
    jsonl = divert_session_transcript_jsonl(sid, [row])
    assert jsonl is not None
    db = SessionDB()
    try:
        assert _apply_diverted(jsonl, sid, db=db) == sid
        assert _apply_diverted(jsonl, sid, db=db) == sid
        stored = db.get_messages(sid)
        assert len(stored) == 1
        assert stored[0]["_compressed_summary"]
        restored = db.get_messages_as_conversation(sid)[0]
        assert restored["display_kind"] == "hidden"
        assert _collect_resume_entries([restored], {}, lambda text: text)[0] == []
        assert db.has_platform_message_id(sid, message["platform_message_id"])
        assert restored["message_id"] == message["platform_message_id"]
        wire = dict(restored)
        substitute_api_content(wire)
        assert wire["content"] == message["api_content"]
        for key in ("finish_reason", "reasoning", "reasoning_content", "reasoning_details",
                    "codex_reasoning_items", "codex_message_items"):
            assert restored[key] == message[key]
        assert {key: restored["display_metadata"][key] for key in metadata} == metadata
        assert restored["display_metadata"]["diverted_recovery_id"]
    finally:
        db.close()


def test_legacy_recovery_identity_survives_durable_field_upgrade():
    import hashlib

    sid = "legacy-durable-divert"
    legacy = {"role": "assistant", "content": "already recovered"}
    record = {**legacy, "display_kind": "hidden", "api_content": "wire bytes",
              "display_metadata": {"source": "original"}, "platform_message_id": "p1"}
    jsonl = _write_diverted(get_hermes_home() / "sessions" / f"{sid}.jsonl", [record])
    digest = hashlib.sha256(str(jsonl.resolve()).encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(legacy, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    db = SessionDB()
    try:
        db.create_session(sid, "cli")
        db.append_message(sid, **legacy, display_metadata={"diverted_recovery_id": digest.hexdigest()})
        assert _apply_diverted(jsonl, sid, db=db) == sid
        assert len(db.get_messages(sid)) == 1
        _write_diverted(jsonl, [record, record])
        assert _apply_diverted(jsonl, sid, db=db) == sid
        assert _apply_diverted(jsonl, sid, db=db) == sid
        assert len(db.get_messages(sid)) == 2
    finally:
        db.close()
