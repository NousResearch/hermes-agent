"""Tests for the encryption security-audit log."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_crypto import audit


def _audit_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    log_path = tmp_path / "logs" / "security-audit.jsonl"
    monkeypatch.setattr(audit, "audit_log_path", lambda: log_path)
    return log_path


def test_log_event_writes_jsonl():
    audit.log_event(audit.ENCRYPTION_ENABLED, audit.SUCCESS, key_source="keyring", files=3)
    path = audit.audit_log_path()
    assert path.is_file()
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["activity"] == "encryption_enabled"
    assert record["outcome"] == "success"
    assert record["severity"] == "info"
    assert record["detail"] == {"key_source": "keyring", "files": 3}
    assert "timestamp" in record and "pid" in record


def test_failed_unlock_is_critical():
    audit.log_event(audit.KEYSTORE_UNLOCK_FAILED, audit.FAILURE, reason="wrong_key")
    record = audit.read_recent(1)[0]
    assert record["activity"] == "keystore_unlock_failed"
    assert record["severity"] == "critical"


def test_generic_failure_is_warning():
    audit.log_event(audit.DATA_KEY_UNAVAILABLE, audit.FAILURE, reason="no_keystore")
    assert audit.read_recent(1)[0]["severity"] == "warning"


def test_read_recent_is_ordered_and_limited():
    for i in range(8):
        audit.log_event(audit.KEYSTORE_UNLOCKED, audit.SUCCESS, n=i)
    recent = audit.read_recent(3)
    assert len(recent) == 3
    assert [e["detail"]["n"] for e in recent] == [5, 6, 7]  # newest last


def test_read_recent_includes_rotated_segments(tmp_path, monkeypatch):
    log_path = _audit_path(tmp_path, monkeypatch)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    archive = audit._rotated_segment_path(log_path, 1)
    archive.write_text(
        json.dumps({"activity": "old", "detail": {"n": 1}}) + "\n",
        encoding="utf-8",
    )
    log_path.write_text(
        json.dumps({"activity": "new", "detail": {"n": 2}}) + "\n",
        encoding="utf-8",
    )

    recent = audit.read_recent(2)

    assert [event["activity"] for event in recent] == ["old", "new"]


def test_read_recent_empty_when_no_log():
    assert audit.read_recent() == []


def test_log_event_never_raises_on_bad_detail():
    # A non-JSON-serialisable detail value must not propagate an exception.
    audit.log_event(audit.KEYSTORE_CREATED, audit.SUCCESS, bad=object())
    # Either it was skipped or written defensively — the call simply returns.


def test_none_details_are_dropped():
    audit.log_event(audit.KEYSTORE_ROTATED, audit.SUCCESS, key_source="keyring", extra=None)
    assert audit.read_recent(1)[0]["detail"] == {"key_source": "keyring"}


def test_maybe_rotate_no_op_when_missing_or_small(tmp_path, monkeypatch):
    log_path = _audit_path(tmp_path, monkeypatch)
    audit._maybe_rotate(log_path)
    assert not log_path.exists()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(b"x" * 10)
    monkeypatch.setattr(audit, "MAX_SEGMENT_BYTES", 100)
    audit._maybe_rotate(log_path)
    assert log_path.read_bytes() == b"x" * 10


def test_rotation_archives_active_log(tmp_path, monkeypatch):
    log_path = _audit_path(tmp_path, monkeypatch)
    monkeypatch.setattr(audit, "MAX_SEGMENT_BYTES", 64)

    marker = b"archive-me\n"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(marker * 8)

    audit._maybe_rotate(log_path)

    archive = audit._rotated_segment_path(log_path, 1)
    assert archive.is_file()
    assert archive.read_bytes() == marker * 8
    assert log_path.is_file()
    assert log_path.stat().st_size == 0


def test_rotation_trims_to_max_segments(tmp_path, monkeypatch):
    log_path = _audit_path(tmp_path, monkeypatch)
    monkeypatch.setattr(audit, "MAX_SEGMENT_BYTES", 32)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for index in range(audit.MAX_ROTATED_SEGMENTS + 2):
        log_path.write_bytes(f"segment-{index}-{'x' * 24}\n".encode("utf-8"))
        audit._maybe_rotate(log_path)

    for index in range(1, audit.MAX_ROTATED_SEGMENTS + 1):
        assert audit._rotated_segment_path(log_path, index).is_file()
    assert not audit._rotated_segment_path(log_path, audit.MAX_ROTATED_SEGMENTS + 1).exists()


def test_log_event_rotates_before_append(tmp_path, monkeypatch):
    log_path = _audit_path(tmp_path, monkeypatch)
    monkeypatch.setattr(audit, "MAX_SEGMENT_BYTES", 48)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(b"old-event\n" * 6)

    audit.log_event(audit.KEYSTORE_UNLOCKED, audit.SUCCESS, slot="keyring")

    archive = audit._rotated_segment_path(log_path, 1)
    assert archive.is_file()
    assert b"old-event" in archive.read_bytes()
    active = log_path.read_text(encoding="utf-8")
    assert "keystore_unlocked" in active
    assert "old-event" not in active


def test_log_event_never_raises_when_rotation_fails(tmp_path, monkeypatch):
    log_path = _audit_path(tmp_path, monkeypatch)
    monkeypatch.setattr(audit, "MAX_SEGMENT_BYTES", 1)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(b"x")

    def _boom(_path: Path) -> None:
        raise OSError("rotation blocked")

    monkeypatch.setattr(audit, "_maybe_rotate", _boom)
    audit.log_event(audit.ENCRYPTION_ENABLED, audit.SUCCESS)


def test_rotation_recovers_after_stale_lockfile(tmp_path, monkeypatch):
    """A .rotate-lock file left by a crashed rotator must not block future rotation.

    Regression: the prior ``os.open(O_EXCL)`` lockfile
    stuck after SIGKILL/OOM and short-circuited every later
    ``_maybe_rotate`` call, re-introducing ISSUES #12. The current
    kernel-released advisory lock (``fcntl.flock`` / ``msvcrt.locking``)
    is released automatically on process death, so the next caller acquires
    cleanly past a stale lockfile.
    """
    log_path = _audit_path(tmp_path, monkeypatch)
    monkeypatch.setattr(audit, "MAX_SEGMENT_BYTES", 64)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(b"to-be-archived\n" * 8)

    # Simulate a crashed previous rotator: lockfile present, no live holder.
    stale_lock = log_path.with_name(log_path.name + audit._ROTATE_LOCK_SUFFIX)
    stale_lock.write_bytes(b"\0")
    assert stale_lock.is_file()

    audit._maybe_rotate(log_path)

    archive = audit._rotated_segment_path(log_path, 1)
    assert archive.is_file()
    assert b"to-be-archived" in archive.read_bytes()
    assert log_path.is_file()
    assert log_path.stat().st_size == 0
    # The successful rotator removes the lockfile on the way out.
    assert not stale_lock.exists()


def test_rotation_short_circuits_when_lock_is_held(tmp_path, monkeypatch):
    """A live concurrent rotator must short-circuit; the lockfile must survive.

    Pairs with :func:`test_rotation_recovers_after_stale_lockfile`: that test
    proves a *dead* holder doesn't block; this one proves a *live* holder
    does. Together they pin the kernel-released advisory-lock semantics that
    replaced the sticky ``O_EXCL`` lockfile.
    """
    import os as _os

    from hermes_crypto._lockfile import (
        _acquire_exclusive,
        _open_lockfile,
        _release_exclusive,
    )

    log_path = _audit_path(tmp_path, monkeypatch)
    monkeypatch.setattr(audit, "MAX_SEGMENT_BYTES", 64)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(b"x" * 128)

    lock_path = log_path.with_name(log_path.name + audit._ROTATE_LOCK_SUFFIX)
    held_fd = _open_lockfile(lock_path)
    _acquire_exclusive(held_fd)
    try:
        audit._maybe_rotate(log_path)
        # No rotation: archive is absent, active file unchanged.
        assert not audit._rotated_segment_path(log_path, 1).exists()
        assert log_path.stat().st_size == 128
        # The live holder's lockfile must still exist — _maybe_rotate's
        # short-circuit path must not unlink another writer's lock.
        assert lock_path.is_file()
    finally:
        _release_exclusive(held_fd)
        try:
            _os.close(held_fd)
        except OSError:
            pass
        try:
            lock_path.unlink()
        except OSError:
            pass
