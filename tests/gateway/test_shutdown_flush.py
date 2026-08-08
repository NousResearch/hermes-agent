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
    flush_queued_events_to_file,
    recover_pending_to_db,
)
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.session import SessionSource


class _ShutdownAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True)

    async def get_chat_info(self, chat_id):
        return {}


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


def test_real_fifo_events_flush_and_recover_with_session_identity(tmp_path, monkeypatch):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    session_key = "agent:main:signal:group:shared"
    session_id = "20260805_010203_fifo"
    source = SessionSource(
        platform=Platform.SIGNAL,
        chat_id="group.shared",
        chat_type="group",
        user_id="+15550001111",
        user_id_alt="uuid-alice",
    )
    events = [MessageEvent(text=text, source=source) for text in ("FIRST", "SECOND", "THIRD")]

    flushed = flush_queued_events_to_file(
        {session_key: events},
        session_ids={session_key: session_id},
    )
    mock_db = MagicMock()
    recovered = recover_pending_to_db(mock_db)

    assert flushed == 3
    assert recovered == 3
    assert [call.kwargs["session_id"] for call in mock_db.append_message.call_args_list] == [
        session_id,
        session_id,
        session_id,
    ]
    assert [call.kwargs["content"] for call in mock_db.append_message.call_args_list] == [
        "FIRST",
        "SECOND",
        "THIRD",
    ]
    assert list(flush_dir.glob("*.json")) == []


@pytest.mark.asyncio
async def test_adapter_head_and_runner_overflow_recover_complete_fifo(
    tmp_path, monkeypatch
):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    session_key = "agent:main:signal:group:shared"
    session_id = "20260805_010203_adapter_head"
    source = SessionSource(
        platform=Platform.SIGNAL,
        chat_id="group.shared",
        chat_type="group",
        user_id="+15550001111",
    )
    adapter = _ShutdownAdapter(
        PlatformConfig(enabled=True, token="test"), Platform.SIGNAL
    )
    adapter._pending_messages[session_key] = MessageEvent(text="FIRST", source=source)
    adapter._session_store = MagicMock()
    adapter._session_store.peek_session_id.return_value = session_id

    await adapter.cancel_background_tasks()
    flushed_overflow = flush_queued_events_to_file(
        {
            session_key: [
                MessageEvent(text="SECOND", source=source),
                MessageEvent(text="THIRD", source=source),
            ]
        },
        session_ids={session_key: session_id},
    )
    mock_db = MagicMock()
    recovered = recover_pending_to_db(mock_db)

    assert flushed_overflow == 2
    assert recovered == 3
    assert [call.kwargs["content"] for call in mock_db.append_message.call_args_list] == [
        "FIRST",
        "SECOND",
        "THIRD",
    ]
    assert list(flush_dir.glob("*.json")) == []


def test_recovery_does_not_overtake_failed_same_session_event(tmp_path, monkeypatch):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    session_key = "agent:main:signal:group:retry"
    session_id = "20260805_010203_retry"
    events = [MessageEvent(text=text) for text in ("FIRST", "SECOND", "THIRD")]
    assert flush_queued_events_to_file(
        {session_key: events}, session_ids={session_key: session_id}
    ) == 3

    class _TransientDb:
        def __init__(self):
            self.attempts = []
            self.successes = []
            self.fail_first = True

        def append_message(self, *, content, **kwargs):
            self.attempts.append(content)
            if self.fail_first:
                self.fail_first = False
                raise RuntimeError("transient failure")
            self.successes.append(content)

    db = _TransientDb()
    assert recover_pending_to_db(db) == 0
    assert db.attempts == ["FIRST"]
    assert db.successes == []
    assert len(list(flush_dir.glob("*.json"))) == 3

    assert recover_pending_to_db(db) == 3
    assert db.attempts == ["FIRST", "FIRST", "SECOND", "THIRD"]
    assert db.successes == ["FIRST", "SECOND", "THIRD"]
    assert list(flush_dir.glob("*.json")) == []


def test_recovery_orders_legacy_and_ordered_files_by_payload_time(tmp_path, monkeypatch):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    session_key = "agent:main:signal:group:legacy"
    session_id = "20260805_010203_legacy"
    now = int(time.time())
    legacy_first = {
        "session_key": session_key,
        "reason": "shutdown",
        "ts": now - 1,
        "data": {"text": "FIRST", "session_id": session_id},
    }
    (flush_dir / f"pending-{'f' * 32}.json").write_text(
        json.dumps(legacy_first), encoding="utf-8"
    )
    assert flush_queued_events_to_file(
        {session_key: [MessageEvent(text="SECOND")]},
        session_ids={session_key: session_id},
    ) == 1

    mock_db = MagicMock()
    assert recover_pending_to_db(mock_db) == 2
    assert [call.kwargs["content"] for call in mock_db.append_message.call_args_list] == [
        "FIRST",
        "SECOND",
    ]
    assert list(flush_dir.glob("*.json")) == []


def test_recovery_uses_mtime_for_legacy_same_timestamp_ties(tmp_path, monkeypatch):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    session_key = "agent:main:signal:group:legacy-mtime"
    session_id = "20260805_010203_legacy_mtime"
    timestamp = int(time.time())
    legacy_path = flush_dir / f"pending-{'f' * 32}.json"
    legacy_path.write_text(
        json.dumps(
            {
                "session_key": session_key,
                "reason": "shutdown",
                "ts": timestamp,
                "data": {"text": "FIRST", "session_id": session_id},
            }
        ),
        encoding="utf-8",
    )
    timestamp_ns = timestamp * 1_000_000_000
    os.utime(legacy_path, ns=(timestamp_ns, timestamp_ns))
    monkeypatch.setattr("gateway.shutdown_flush.time.time", lambda: timestamp)
    assert flush_queued_events_to_file(
        {session_key: [MessageEvent(text="SECOND")]},
        session_ids={session_key: session_id},
    ) == 1

    mock_db = MagicMock()
    assert recover_pending_to_db(mock_db) == 2
    assert [call.kwargs["content"] for call in mock_db.append_message.call_args_list] == [
        "FIRST",
        "SECOND",
    ]


def test_recovery_preserves_invalid_json_shape_and_continues(tmp_path, monkeypatch):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    invalid_path = flush_dir / f"pending-{'f' * 32}.json"
    invalid_path.write_text("[]", encoding="utf-8")
    valid_path = flush_dir / "pending-00000000000000000001-valid.json"
    valid_path.write_text(
        json.dumps(
            {
                "type": "queued_event",
                "session_key": "session-1",
                "ts": 2,
                "data": {"text": "VALID", "session_id": "session-1"},
            }
        ),
        encoding="utf-8",
    )
    mock_db = MagicMock()

    assert recover_pending_to_db(mock_db) == 1

    assert [call.kwargs["content"] for call in mock_db.append_message.call_args_list] == [
        "VALID"
    ]
    assert invalid_path.exists()
    assert not valid_path.exists()


def test_ordered_files_ignore_adversarial_mtime_when_payload_times_tie(
    tmp_path, monkeypatch
):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    write_orders = iter((1, 2))
    monkeypatch.setattr(
        "gateway.shutdown_flush._next_write_order", lambda: next(write_orders)
    )
    monkeypatch.setattr("gateway.shutdown_flush.time.time", lambda: 10)
    assert flush_pending_to_file(
        {"session-1": "FIRST"}, session_ids={"session-1": "session-1"}
    ) == 1
    assert flush_pending_to_file(
        {"session-1": "SECOND"}, session_ids={"session-1": "session-1"}
    ) == 1
    first_path, second_path = sorted(flush_dir.glob("*.json"))
    os.utime(first_path, ns=(20_000_000_000, 20_000_000_000))
    os.utime(second_path, ns=(10_000_000_000, 10_000_000_000))
    mock_db = MagicMock()

    assert recover_pending_to_db(mock_db) == 2

    assert [call.kwargs["content"] for call in mock_db.append_message.call_args_list] == [
        "FIRST",
        "SECOND",
    ]


@pytest.mark.parametrize(
    "invalid_timestamp", [float("nan"), 10**400], ids=("nan", "overflow")
)
def test_invalid_payload_time_falls_back_to_mtime(
    tmp_path, monkeypatch, invalid_timestamp
):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )
    first_path = flush_dir / f"pending-{'f' * 32}.json"
    second_path = flush_dir / "pending-00000000000000000001-valid.json"
    first_path.write_text(
        json.dumps(
            {
                "type": "queued_event",
                "session_key": "session-1",
                "ts": invalid_timestamp,
                "data": {"text": "FIRST", "session_id": "session-1"},
            }
        ),
        encoding="utf-8",
    )
    second_path.write_text(
        json.dumps(
            {
                "type": "queued_event",
                "session_key": "session-1",
                "ts": 10,
                "data": {"text": "SECOND", "session_id": "session-1"},
            }
        ),
        encoding="utf-8",
    )
    os.utime(first_path, ns=(1_000_000_000, 1_000_000_000))
    os.utime(second_path, ns=(10_000_000_000, 10_000_000_000))
    mock_db = MagicMock()

    assert recover_pending_to_db(mock_db) == 2

    assert [call.kwargs["content"] for call in mock_db.append_message.call_args_list] == [
        "FIRST",
        "SECOND",
    ]


def test_fifo_recovery_order_does_not_depend_on_random_uuid_names(tmp_path, monkeypatch):
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: flush_dir
    )

    class _ForcedUuid:
        def __init__(self, value):
            self.hex = value

    forced = iter(("f" * 32, "0" * 32, "8" * 32))
    monkeypatch.setattr("gateway.shutdown_flush.time.time_ns", lambda: 123)
    monkeypatch.setattr(
        "gateway.shutdown_flush.uuid.uuid4",
        lambda: _ForcedUuid(next(forced)),
    )
    events = [MessageEvent(text=text) for text in ("FIRST", "SECOND", "THIRD")]

    assert flush_queued_events_to_file({"shared": events}) == 3
    recovered_payloads = [
        json.loads(path.read_text(encoding="utf-8"))["data"]["text"]
        for path in sorted(flush_dir.glob("*.json"))
    ]

    assert recovered_payloads == ["FIRST", "SECOND", "THIRD"]


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


