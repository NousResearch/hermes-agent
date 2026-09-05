"""Tests for gateway/shutdown_flush.py — pending message durability (#72680)."""

import errno
import json
import os
import stat
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.shutdown_flush import (
    _serialise_value,
    flush_overflow_to_file,
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
        released = False

        def append_message(self, **_kwargs):
            raise KeyboardInterrupt

    db = InterruptingDB()
    monkeypatch.setattr("hermes_state_registry.acquire", lambda: db)
    monkeypatch.setattr(
        "hermes_state_registry.release_or_close", lambda _: setattr(db, "released", True)
    )

    with pytest.raises(KeyboardInterrupt):
        recover_pending_to_db()

    assert db.released is True


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




def test_unmanaged_flush_dir_and_payload_are_owner_only(tmp_path, monkeypatch):
    if os.name != "posix":
        pytest.skip("POSIX permission-bit semantics required")

    from gateway.shutdown_flush import _get_flush_dir, _write_payload

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    previous_umask = os.umask(0o022)
    try:
        flush_dir = _get_flush_dir()
        payload_path = _write_payload(flush_dir, {"message": "private"})
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(flush_dir.stat().st_mode) & 0o077 == 0
    assert stat.S_IMODE(payload_path.stat().st_mode) == 0o600


def test_managed_flush_preserves_group_access_without_chmod(tmp_path, monkeypatch):
    if os.name != "posix":
        pytest.skip("POSIX permission-bit semantics required")

    import gateway.shutdown_flush as shutdown_flush

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED", "nixos")
    os.chmod(tmp_path, 0o2770)

    def reject_chmod(*_args, **_kwargs):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    shutdown_os = SimpleNamespace(**vars(shutdown_flush.os))
    monkeypatch.setattr(shutdown_flush, "os", shutdown_os)
    monkeypatch.setattr(shutdown_flush.os, "chmod", reject_chmod)
    previous_umask = os.umask(0o007)
    try:
        flushed = shutdown_flush.flush_pending_to_file(
            {"agent:main:telegram:dm:1": "managed message"}
        )
    finally:
        os.umask(previous_umask)

    flush_dir = tmp_path / "pending_messages"
    mode = stat.S_IMODE(flush_dir.stat().st_mode)
    payload_paths = list(flush_dir.glob("pending-*.json"))

    assert mode & 0o070 == 0o070
    assert mode & 0o007 == 0
    assert flushed == 1
    assert len(payload_paths) == 1
    payload = json.loads(payload_paths[0].read_text(encoding="utf-8"))
    assert payload["data"]["text"] == "managed message"


def test_managed_legacy_directory_is_upgraded_best_effort(tmp_path, monkeypatch):
    if os.name != "posix":
        pytest.skip("POSIX permission-bit semantics required")
    import gateway.shutdown_flush as shutdown_flush

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED", "nixos")
    os.chmod(tmp_path, 0o2770)
    flush_dir = tmp_path / "pending_messages"
    flush_dir.mkdir(mode=0o700)
    os.chmod(flush_dir, 0o700)
    result = shutdown_flush._get_flush_dir()
    assert result == flush_dir
    assert stat.S_IMODE(flush_dir.stat().st_mode) & 0o070 == 0o070


def test_unmanaged_flush_survives_permission_reconciliation_failure(
    tmp_path, monkeypatch
):
    if os.name != "posix":
        pytest.skip("POSIX permission-bit semantics required")

    import gateway.shutdown_flush as shutdown_flush
    import hermes_cli.config as hermes_config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)

    def reject_reconciliation(*_args, **_kwargs):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    shutdown_os = SimpleNamespace(**vars(shutdown_flush.os))
    monkeypatch.setattr(shutdown_flush, "os", shutdown_os)
    monkeypatch.setattr(shutdown_flush.os, "chmod", reject_reconciliation)
    monkeypatch.setattr(hermes_config, "_secure_dir", reject_reconciliation)

    flushed = shutdown_flush.flush_pending_to_file(
        {"agent:main:telegram:dm:1": "unmanaged message"}
    )

    payload_paths = list((tmp_path / "pending_messages").glob("pending-*.json"))
    assert flushed == 1
    assert len(payload_paths) == 1
    payload = json.loads(payload_paths[0].read_text(encoding="utf-8"))
    assert payload["data"]["text"] == "unmanaged message"


# ── FIFO overflow tail durability (#99882) ─────────────────────────────


def _overflow_event(text: str, session_id: str = "20260901_120000_fifo"):
    event = MagicMock()
    event.text = text
    event.session_id = session_id
    event.platform = "telegram"
    event.sender_id = "1572286605"
    event.sender_name = "tester"
    event.reply_to = None
    event.media = None
    event.raw_event = None
    return event


def test_flush_overflow_writes_one_payload_per_event_in_arrival_order(tmp_path, monkeypatch):
    """The FIFO tail (queued_events) must survive shutdown like the slot does.

    Each overflow entry is its own recover_pending_to_db-compatible payload,
    with ``seq`` recording arrival order inside the session.
    """
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr("gateway.shutdown_flush._get_flush_dir", lambda: flush_dir)

    count = flush_overflow_to_file(
        {
            "agent:main:telegram:dm:1": [
                _overflow_event("follow-up B"),
                _overflow_event("follow-up C"),
            ],
            "agent:main:telegram:dm:2": [],
            "": [_overflow_event("keyless — skipped")],
        },
        reason="shutdown",
    )
    assert count == 2
    payloads = sorted(
        (json.loads(f.read_text(encoding="utf-8")) for f in flush_dir.glob("*.json")),
        key=lambda p: p["seq"],
    )
    assert [p["data"]["text"] for p in payloads] == ["follow-up B", "follow-up C"]
    assert {p["session_key"] for p in payloads} == {"agent:main:telegram:dm:1"}
    assert all(p["reason"] == "shutdown" for p in payloads)


def test_flushed_overflow_is_replayed_by_recover_pending_to_db(tmp_path, monkeypatch):
    """Round-trip: overflow payloads use the slot-flush shape, so the existing
    startup recovery inserts them as user rows without any new reader."""
    flush_dir = _make_flush_dir(tmp_path)
    monkeypatch.setattr("gateway.shutdown_flush._get_flush_dir", lambda: flush_dir)
    flush_overflow_to_file({"agent:main:telegram:dm:1": [_overflow_event("orphan-1")]})

    db = MagicMock()
    recovered = recover_pending_to_db(session_db=db)
    assert recovered == 1
    db.append_message.assert_called_once()
    kwargs = db.append_message.call_args.kwargs
    assert kwargs["session_id"] == "20260901_120000_fifo"
    assert kwargs["role"] == "user"
    assert kwargs["content"] == "orphan-1"
    assert list(flush_dir.glob("*.json")) == []


def test_flush_overflow_noop_on_empty():
    assert flush_overflow_to_file({}) == 0
    assert flush_overflow_to_file({"k": []}) == 0
