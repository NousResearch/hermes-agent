"""Tests for gateway/shutdown_flush.py — pending message durability (#72680)."""

import json
import os
import stat
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gateway.shutdown_flush import (
    _serialise_value,
    flush_pending_to_file,
    recover_pending_to_db,
)


def _make_flush_dir(tmp_path: Path) -> Path:
    """Create a temp flush dir and monkeypatch _get_flush_dir to use it."""
    flush_dir = tmp_path / "pending_messages"
    flush_dir.mkdir(parents=True, exist_ok=True)
    return flush_dir


def test_flush_writes_string_pending_to_file(tmp_path, monkeypatch):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    pending = {"agent:main:telegram:supergroup:123": "hello world"}
    count = flush_pending_to_file(pending, reason="shutdown")
    assert count == 1
    files = list(flush_dir.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["session_key"] == "agent:main:telegram:supergroup:123"
    assert payload["reason"] == "shutdown"
    assert payload["data"]["text"] == "hello world"
    assert ":" not in files[0].name
    assert "telegram" not in files[0].name


def test_flush_writes_message_event_to_file(tmp_path, monkeypatch):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    event = MagicMock()
    event.text = "user message"
    event.session_id = "20260728_120000_abc"
    event.platform = "telegram"
    event.sender_id = "456"
    event.sender_name = "Alice"
    event.reply_to = None
    event.media = None
    event.raw_event = None

    count = flush_pending_to_file({"session_key_1": event}, reason="adapter_shutdown")
    assert count == 1
    files = list(flush_dir.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["data"]["text"] == "user message"
    assert payload["data"]["session_id"] == "20260728_120000_abc"


def test_recover_inserts_via_append_message_and_deletes_file(tmp_path, monkeypatch):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    ts = int(time.time())
    # Write a flush file with session_id
    payload = {
        "session_key": "agent:main:telegram:supergroup:123",
        "reason": "shutdown",
        "ts": ts,
        "data": {
            "text": "lost message",
            "session_id": "20260728_120000_abc",
        },
    }
    flush_file = flush_dir / "test_session_123.json"
    flush_file.write_text(json.dumps(payload), encoding="utf-8")

    mock_db = MagicMock()
    count = recover_pending_to_db(mock_db)

    assert count == 1
    mock_db.append_message.assert_called_once_with(
        session_id="20260728_120000_abc",
        role="user",
        content="lost message",
        timestamp=ts,
    )
    assert not flush_file.exists()


def test_recover_closes_owned_db_when_unexpected_exception_escapes(
    tmp_path, monkeypatch
):
    """Owned SessionDB must close even when recovery is interrupted."""
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    (flush_dir / "pending.json").write_text(
        json.dumps(
            {
                "session_key": "agent:main:telegram:123",
                "data": {"text": "message", "session_id": "sid"},
            }
        ),
        encoding="utf-8",
    )

    class InterruptingDB:
        closed = False

        def append_message(self, **_kwargs):
            raise KeyboardInterrupt

        def close(self):
            self.closed = True

    db = InterruptingDB()
    monkeypatch.setattr("hermes_state.SessionDB", lambda: db)

    with pytest.raises(KeyboardInterrupt):
        recover_pending_to_db()

    assert db.closed is True


def test_serialise_object_with_text():
    obj = MagicMock()
    obj.text = "msg"
    obj.session_id = "sid"
    obj.platform = None
    obj.sender_id = None
    obj.sender_name = None
    obj.reply_to = None
    obj.media = None
    obj.raw_event = None
    result = _serialise_value(obj)
    assert result is not None
    assert result["text"] == "msg"
    assert result["session_id"] == "sid"


def test_get_flush_dir_uses_get_hermes_home(tmp_path, monkeypatch):
    """Flush dir must use get_hermes_home(), not hardcoded Path.home()."""
    import gateway.shutdown_flush as mod

    captured = {}

    def fake_get_hermes_home():
        from pathlib import Path
        captured["called"] = True
        return tmp_path

    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", fake_get_hermes_home
    )
    result = mod._get_flush_dir()
    assert captured.get("called") is True
    assert result == tmp_path / "pending_messages"




def test_recover_resolves_session_key_via_resolver(tmp_path, monkeypatch):
    """Real flush files carry only `text` (MessageEvent has no session_id
    attribute). Without a resolver they were skipped forever; with one,
    the message must be recovered and the file cleaned up."""
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    ts = int(time.time())
    payload = {
        "session_key": "agent:main:telegram:dm:42",
        "ts": ts,
        "data": {"text": "are you there?"},  # what _serialise_value actually produces
    }
    flush_file = flush_dir / "pending-test.json"
    flush_file.write_text(json.dumps(payload), encoding="utf-8")

    mock_db = MagicMock()
    count = recover_pending_to_db(
        mock_db,
        session_resolver=lambda key: "20260731_abc123" if key == "agent:main:telegram:dm:42" else None,
    )

    assert count == 1
    mock_db.append_message.assert_called_once_with(
        session_id="20260731_abc123",
        role="user",
        content="are you there?",
        timestamp=ts,
    )
    assert not flush_file.exists()


def test_recover_without_resolver_skips_text_only_payload(tmp_path, monkeypatch):
    """Without a resolver, a text-only (real MessageEvent) payload is
    skipped and the file preserved — the pre-fix behavior for every real
    flush. Documents why the call site wires peek_session_id."""
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    payload = {
        "session_key": "agent:main:telegram:dm:42",
        "ts": int(time.time()),
        "data": {"text": "are you there?"},
    }
    flush_file = flush_dir / "pending-test.json"
    flush_file.write_text(json.dumps(payload), encoding="utf-8")

    mock_db = MagicMock()
    count = recover_pending_to_db(mock_db)

    assert count == 0
    mock_db.append_message.assert_not_called()
    assert flush_file.exists()


def test_recover_via_real_session_store_routing_reload(tmp_path, monkeypatch):
    """Integration through the real startup boundary: persist a session-key
    mapping in boot 1, construct a fresh SessionStore in boot 2, and recover
    a text-only payload via peek_session_id — the exact wiring run.py:25449
    installs (not a mocked resolver)."""
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore

    # Boot 1: persist a session_key -> session_id mapping.
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    store1 = SessionStore(sessions_dir=sessions_dir, config=GatewayConfig())
    src = SessionSource(
        platform=Platform.TELEGRAM, chat_id="42", chat_type="dm", user_id="u1",
    )
    entry = store1.get_or_create_session(src)

    # Flush file with a real MessageEvent-shaped (text-only) payload.
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    ts = int(time.time())
    flush_file = flush_dir / "pending-x.json"
    flush_file.write_text(
        json.dumps({
            "session_key": entry.session_key,
            "ts": ts,
            "data": {"text": "are you there?"},
        }),
        encoding="utf-8",
    )

    # Boot 2: fresh store — peek_session_id must reload the mapping from disk.
    store2 = SessionStore(sessions_dir=sessions_dir, config=GatewayConfig())
    assert store2.peek_session_id(entry.session_key) == entry.session_id

    db = hermes_state.SessionDB(db_path=tmp_path / "state.db")
    try:
        count = recover_pending_to_db(
            db, session_resolver=store2.peek_session_id,
        )
        assert count == 1
        convo = db.get_messages_as_conversation(entry.session_id)
        assert any(m["content"] == "are you there?" for m in convo)
        assert not flush_file.exists()
    finally:
        db.close()
