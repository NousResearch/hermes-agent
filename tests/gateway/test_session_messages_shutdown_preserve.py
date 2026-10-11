"""Regression tests for #72680 (retargeted).

The earlier attempt (#73171) snapshotted GatewayRunner._pending_messages, which
on current main has no writers — the live container is the per-agent
``agent._session_messages`` flushed via ``_flush_messages_to_session_db``.
When that flush raises (FTS/SQLite corruption) the in-memory transcript must
be dumped to a recovery snapshot instead of lost.

These tests exercise the real preservation path:
``_flush_agent_transcript_at_shutdown`` -> flush raises ->
``flush_agent_history_to_file`` -> ``recover_pending_to_db``.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.shutdown_flush import (
    TRANSCRIPT_CAP_DROP_REASON,
    flush_agent_history_to_file,
    recover_pending_to_db,
    spool_dropped_transcript_message,
)
from hermes_state import SessionDB


def _make_flush_dir(tmp_path: Path) -> Path:
    """Create a temp flush dir and monkeypatch _get_flush_dir to use it."""
    flush_dir = tmp_path / "pending_messages"
    flush_dir.mkdir(parents=True, exist_ok=True)
    return flush_dir


def test_preserves_agent_history_when_flush_raises(tmp_path, monkeypatch):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    history = [{"role": "user", "content": "lost msg"}]
    flush_agent_history_to_file("sess:abc123", history)

    files = list(flush_dir.glob("*.json"))
    assert files, "expected recovery snapshot"
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["issue"] == "#72680"
    assert data["session_id"] == "sess:abc123"
    assert data["count"] == 1
    assert data["messages"][0]["content"] == "lost msg"


def test_no_recovery_file_on_empty_history(tmp_path, monkeypatch):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    flush_agent_history_to_file("sess:abc123", [])
    assert not list(flush_dir.glob("*.json"))


def test_shutdown_history_recovery_round_trips_canonical_transcript(tmp_path, monkeypatch):
    """A failed shutdown flush must replay the whole transcript exactly once.

    The recovery boundary carries more than display text: structured attachment content,
    assistant tool calls and reasoning, tool-result identity, usage fields, and durable
    message/tool-call identity all have to survive together and in role order.
    """
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr("gateway.shutdown_flush._get_flush_dir", lambda: flush_dir)
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "shutdown-codec"
    db.create_session(session_id, "gateway")
    attachment_content = [
        {"type": "text", "text": "inspect the attachment"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVybWVz"}},
    ]
    db.append_message(
        session_id, "user", content=attachment_content,  # type: ignore[arg-type]
        timestamp=100.0)
    persisted_prefix = db.get_messages_as_conversation(session_id, include_row_ids=True)
    tool_calls = [{
        "id": "call-1", "type": "function",
        "function": {"name": "read_file", "arguments": '{"path":"artifact.png"}'},
    }]
    history = persisted_prefix + [
        {
            "role": "assistant", "content": None, "tool_calls": tool_calls,
            "reasoning": "inspect first", "reasoning_content": "inspect first",
            "reasoning_details": [{"type": "reasoning.summary", "text": "inspect"}],
            "token_count": 17, "finish_reason": "tool_calls", "timestamp": 101.0,
            "api_content": "exact assistant wire bytes",
        },
        {
            "role": "tool", "content": "image metadata", "tool_call_id": "call-1",
            "tool_name": "read_file", "effect_disposition": "read_only", "timestamp": 102.0,
        },
        {
            "role": "assistant", "content": "The attachment is valid.",
            "finish_reason": "stop", "token_count": 9, "timestamp": 103.0,
        },
    ]

    def fail_flush(_messages):
        raise RuntimeError("controlled state.db outage")

    from gateway.run_shutdown import GatewayShutdownMixin
    GatewayShutdownMixin._flush_agent_transcript_at_shutdown(SimpleNamespace(
        session_id=session_id,
        _session_messages=history,
        _flush_messages_to_session_db=fail_flush,
    ))

    (spool_file,) = list(flush_dir.glob("*.json"))
    spooled_bytes = spool_file.read_bytes()
    payload = json.loads(spooled_bytes)
    assert payload["data"]["transcript"]["schema"] == "hermes.session.transcript"
    assert payload["data"]["transcript"]["version"] == 1

    assert recover_pending_to_db(db) == 3
    restored = db.get_messages_as_conversation(session_id, include_row_ids=True)
    assert [message["role"] for message in restored] == ["user", "assistant", "tool", "assistant"]
    assert restored[0]["content"] == attachment_content
    assert restored[1]["tool_calls"] == tool_calls
    assert restored[1]["reasoning"] == "inspect first"
    assert restored[1]["reasoning_details"] == [{"type": "reasoning.summary", "text": "inspect"}]
    assert restored[1]["api_content"] == "exact assistant wire bytes"
    assert restored[2]["tool_call_id"] == "call-1"
    assert restored[2]["tool_name"] == "read_file"
    assert restored[2]["effect_disposition"] == "read_only"
    assert restored[1]["_tool_call_uids"]["call-1"] == restored[2]["_tool_call_uid"]
    raw = db.get_messages(session_id)
    assert (raw[1]["token_count"], raw[1]["finish_reason"]) == (17, "tool_calls")
    assert (raw[3]["token_count"], raw[3]["finish_reason"]) == (9, "stop")

    # Simulate a crash after the DB commit but before unlink: the durable message UIDs make replay a no-op.
    (flush_dir / "pending-replayed.json").write_bytes(spooled_bytes)
    assert recover_pending_to_db(db) == 0
    assert len(db.get_messages(session_id)) == 4
    db.close()


@pytest.mark.parametrize("overlapping", [False, True, "conflicting"])
def test_cap_drop_recovery_uses_codec_and_accepts_legacy_payload(tmp_path, monkeypatch, overlapping):
    """The cap-drop producer emits v1, while pre-v1 files retain their full canonical row."""
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr("gateway.shutdown_flush._get_flush_dir", lambda: flush_dir)
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("codec-cap", "gateway")
    db.create_session("legacy-cap", "gateway")
    calls = [{
        "id": "call-cap", "type": "function",
        "function": {"name": "read_file", "arguments": '{"path":"a.txt"}'},
    }]
    spool_dropped_transcript_message("codec-cap", {
        "role": "assistant", "content": None, "tool_calls": calls,
        "reasoning": "need the file", "token_count": 5,
        "finish_reason": "tool_calls", "timestamp": 201.0,
    })
    spool_dropped_transcript_message("codec-cap", {
        "role": "tool", "content": "contents", "tool_call_id": "call-cap",
        "tool_name": "read_file", "effect_disposition": "read_only", "timestamp": 202.0,
    })
    # Give filename order the same order as seq so this test isolates the codec contract.
    for path in list(flush_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["data"]["transcript"]["version"] == 1
        path.rename(flush_dir / f"pending-codec-{payload['seq']}.json")

    (flush_dir / "pending-legacy.json").write_text(json.dumps({
        "session_key": "legacy-cap", "reason": TRANSCRIPT_CAP_DROP_REASON, "ts": 203, "seq": 2,
        "data": {"session_id": "legacy-cap", "message": {
            "role": "assistant", "content": "legacy answer", "reasoning": "legacy thought",
            "token_count": 4, "finish_reason": "stop", "timestamp": 203.0,
        }},
    }), encoding="utf-8")

    if overlapping:
        first = min(flush_dir.glob("pending-codec-*.json"))
        payload = json.loads(first.read_text())
        if overlapping == "conflicting":
            payload["data"]["transcript"]["messages"][0]["content"] = "conflicting content"
        (flush_dir / "pending-overlapping.json").write_text(json.dumps(payload))

    if overlapping == "conflicting":
        preserved_files = sorted(path.name for path in flush_dir.glob("*.json") if path.name != "pending-legacy.json")
        assert (recover_pending_to_db(db), db.get_messages("codec-cap"),
                sorted(path.name for path in flush_dir.glob("*.json"))) == (
            1, [], preserved_files)
        db.close()
        return

    assert recover_pending_to_db(db) == 3
    restored = db.get_messages_as_conversation("codec-cap")
    assert [message["role"] for message in restored] == ["assistant", "tool"]
    assert restored[0]["tool_calls"] == calls
    assert restored[0]["reasoning"] == "need the file"
    assert restored[1]["tool_call_id"] == "call-cap"
    assert restored[0]["_tool_call_uids"]["call-cap"] == restored[1]["_tool_call_uid"]
    raw = db.get_messages("codec-cap")
    assert (raw[0]["token_count"], raw[0]["finish_reason"]) == (5, "tool_calls")
    assert raw[1]["effect_disposition"] == "read_only"
    legacy = db.get_messages("legacy-cap")
    assert (legacy[0]["reasoning"], legacy[0]["token_count"], legacy[0]["finish_reason"]) == (
        "legacy thought", 4, "stop")
    db.close()


