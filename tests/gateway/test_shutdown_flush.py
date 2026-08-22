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


def _write_flush_file(flush_dir: Path, name: str, session_id: str, text: str) -> Path:
    """Write one well-formed pending-message flush file."""
    path = flush_dir / name
    path.write_text(
        json.dumps(
            {
                "session_key": "agent:main:telegram:supergroup:123",
                "reason": "shutdown",
                "ts": 1700000000,
                "data": {"text": text, "session_id": session_id},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_recover_skips_failing_payload_and_continues(tmp_path, monkeypatch):
    """One unrecoverable file must not abort recovery of the others.

    Recovery walks ``sorted(glob("*.json"))``, so a file that raises an
    ordinary exception used to propagate out of the whole pass.  Because
    that file is never unlinked it would then re-poison every subsequent
    boot, stranding a different subset of messages each time.
    """
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    bad = _write_flush_file(flush_dir, "pending-a.json", "sid-bad", "first")
    good = _write_flush_file(flush_dir, "pending-b.json", "sid-good", "second")

    class PartiallyFailingDB:
        def __init__(self):
            self.appended = []

        def append_message(self, **kwargs):
            if kwargs["session_id"] == "sid-bad":
                raise RuntimeError("session is closed")
            self.appended.append(kwargs["session_id"])

    db = PartiallyFailingDB()
    assert recover_pending_to_db(db) == 1
    assert db.appended == ["sid-good"]
    # The good file is consumed; the bad one is preserved for a retry.
    assert not good.exists()
    assert bad.exists()


def test_recover_survives_unparseable_payload(tmp_path, monkeypatch):
    """A corrupt flush file must be preserved, not abort the pass."""
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    corrupt = flush_dir / "pending-a-bad.json"
    corrupt.write_text("not json", encoding="utf-8")
    good = _write_flush_file(flush_dir, "pending-b-good.json", "sid-good", "kept")

    mock_db = MagicMock()
    assert recover_pending_to_db(mock_db) == 1
    mock_db.append_message.assert_called_once_with(
        session_id="sid-good",
        role="user",
        content="kept",
        timestamp=1700000000,
    )
    assert not good.exists()
    assert corrupt.exists()


def test_recover_closes_owned_db_when_interrupt_follows_tolerated_error(
    tmp_path, monkeypatch
):
    """Tolerating ordinary errors must not weaken the interrupt contract.

    #83226's guarantee is that an interrupt closes an owned SessionDB and
    propagates.  That must still hold on a pass where an earlier file
    already failed with an ordinary exception and was skipped.
    """
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    failed = _write_flush_file(flush_dir, "pending-a.json", "sid-bad", "first")
    _write_flush_file(flush_dir, "pending-b.json", "sid-interrupt", "second")

    class InterruptAfterErrorDB:
        closed = False

        def append_message(self, **kwargs):
            if kwargs["session_id"] == "sid-bad":
                raise RuntimeError("session is closed")
            raise KeyboardInterrupt

        def close(self):
            self.closed = True

    db = InterruptAfterErrorDB()
    monkeypatch.setattr("hermes_state.SessionDB", lambda: db)

    with pytest.raises(KeyboardInterrupt):
        recover_pending_to_db()

    assert db.closed is True
    # The tolerated failure is still on disk for the next startup.
    assert failed.exists()


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


