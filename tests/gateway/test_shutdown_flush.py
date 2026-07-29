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


def test_flush_empty_pending_is_noop(tmp_path, monkeypatch):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    assert flush_pending_to_file({}, reason="test") == 0
    assert list(flush_dir.glob("*.json")) == []


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


def test_flush_same_session_twice_does_not_overwrite(tmp_path, monkeypatch):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr("gateway.shutdown_flush._get_flush_dir", lambda: flush_dir)
    monkeypatch.setattr("gateway.shutdown_flush.time.time", lambda: 1234)

    pending = {"agent:main:telegram:supergroup:123": "hello world"}
    assert flush_pending_to_file(pending, reason="shutdown") == 1
    assert flush_pending_to_file(pending, reason="shutdown") == 1

    files = list(flush_dir.glob("*.json"))
    assert len(files) == 2
    assert all(
        json.loads(path.read_text(encoding="utf-8"))["data"]["text"] == "hello world"
        for path in files
    )


def test_flush_write_failure_leaves_no_recovery_file(tmp_path, monkeypatch):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr("gateway.shutdown_flush._get_flush_dir", lambda: flush_dir)

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("utils.os.replace", fail_replace)

    assert flush_pending_to_file({"session": "message"}, reason="test") == 0
    assert list(flush_dir.iterdir()) == []


def test_flush_directory_fsync_failure_keeps_recovery_file(tmp_path, monkeypatch):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr("gateway.shutdown_flush._get_flush_dir", lambda: flush_dir)

    def fail_directory_fsync(path):
        raise OSError("simulated directory fsync failure")

    monkeypatch.setattr(
        "gateway.shutdown_flush._fsync_directory", fail_directory_fsync
    )

    assert flush_pending_to_file({"session": "message"}, reason="test") == 1
    [flush_file] = list(flush_dir.glob("*.json"))
    assert json.loads(flush_file.read_text(encoding="utf-8"))["data"] == {
        "text": "message"
    }


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


def test_recover_no_flush_files_is_noop(tmp_path, monkeypatch):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    mock_db = MagicMock()
    assert recover_pending_to_db(mock_db) == 0
    mock_db.append_message.assert_not_called()


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


def test_recover_skips_file_without_session_id(tmp_path, monkeypatch):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    payload = {
        "session_key": "some_key",
        "reason": "shutdown",
        "ts": int(time.time()),
        "data": {"text": "no session id"},
    }
    flush_file = flush_dir / "no_sid.json"
    flush_file.write_text(json.dumps(payload), encoding="utf-8")

    mock_db = MagicMock()
    count = recover_pending_to_db(mock_db)

    assert count == 0
    mock_db.append_message.assert_not_called()
    # File preserved for manual recovery
    assert flush_file.exists()


def test_recover_preserves_structurally_invalid_file(tmp_path, monkeypatch):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    payload = {
        "session_key": "some_key",
        "reason": "shutdown",
        "ts": int(time.time()),
        "data": {"text": "", "session_id": "sid"},
    }
    flush_file = flush_dir / "empty.json"
    flush_file.write_text(json.dumps(payload), encoding="utf-8")

    mock_db = MagicMock()
    count = recover_pending_to_db(mock_db)

    assert count == 0
    assert flush_file.exists()


def test_serialise_string():
    assert _serialise_value("hello") == {"text": "hello"}


def test_serialise_dict():
    assert _serialise_value({"text": "hi"}) == {"text": "hi"}


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


@pytest.mark.skipif(
    os.name != "posix",
    reason="mode assertions require POSIX permissions",
)
def test_get_flush_dir_and_files_are_private(tmp_path, monkeypatch):
    import gateway.shutdown_flush as mod

    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    flush_dir = mod._get_flush_dir()
    assert stat.S_IMODE(flush_dir.stat().st_mode) == 0o700

    assert flush_pending_to_file({"session": "message"}, reason="test") == 1
    [flush_file] = list(flush_dir.glob("*.json"))
    assert stat.S_IMODE(flush_file.stat().st_mode) == 0o600
