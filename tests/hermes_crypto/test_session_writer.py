"""Tests for encrypting session-transcript writer and read-back helpers."""

from __future__ import annotations

import pytest

from hermes_constants import get_hermes_home
from hermes_crypto import audit, detect, envelope, migrate
from hermes_crypto.errors import HermesCryptoError
from hermes_crypto.session_writer import (
    EncryptingSessionWriter,
    _PlainSessionWriter,
    build_session_writer,
    read_session_text,
)

FAST_ARGON2 = {"time_cost": 1, "memory_cost_kib": 8, "parallelism": 1}


def _enable_with_log_encryption() -> None:
    # force=True bypasses the concurrent-Hermes-instance probe — same pattern
    # the existing test_migrate.py uses. The probe matches transient pytest
    # workers when the checkout path contains the literal "hermes-agent".
    migrate.enable(
        "passphrase",
        passphrase="pw",
        argon2_params=FAST_ARGON2,
        force=True,
    )
    migrate._set_config("security.encryption.encrypt_logs", True)


def _session_path(name: str = "test-001.jsonl"):
    path = get_hermes_home() / "sessions" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_build_session_writer_plain_when_flag_off():
    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2, force=True)
    writer = build_session_writer(_session_path())
    try:
        assert isinstance(writer, _PlainSessionWriter)
    finally:
        writer.close()


def test_build_session_writer_encrypting_when_flag_on():
    _enable_with_log_encryption()
    writer = build_session_writer(_session_path())
    try:
        assert isinstance(writer, EncryptingSessionWriter)
    finally:
        writer.close()


def test_encrypting_writer_keeps_live_file_plaintext_seals_on_close():
    _enable_with_log_encryption()
    path = _session_path()
    lock_path = path.with_name(path.name + ".lock")

    with build_session_writer(path) as w:
        w.write('{"role":"user","content":"hello"}\n')
        w.write('{"role":"assistant","content":"hi there"}\n')
        w.flush()
        # Live: file is plaintext, lockfile present.
        live_bytes = path.read_bytes()
        assert not detect.is_encrypted(live_bytes)
        assert b'"role":"user"' in live_bytes
        assert lock_path.is_file()

    # Closed: file is sealed, lockfile is gone.
    assert detect.is_encrypted(path.read_bytes())
    assert not lock_path.exists()

    text = read_session_text(path)
    assert '"role":"user"' in text
    assert '"role":"assistant"' in text


def test_plain_writer_leaves_file_plaintext_and_no_lockfile():
    migrate.enable("passphrase", passphrase="pw", argon2_params=FAST_ARGON2, force=True)
    path = _session_path()
    with build_session_writer(path) as w:
        w.write("plain line\n")
    assert not detect.is_encrypted(path.read_bytes())
    assert path.read_text(encoding="utf-8") == "plain line\n"
    assert not path.with_name(path.name + ".lock").exists()


def test_writer_refuses_to_append_to_already_encrypted_session():
    _enable_with_log_encryption()
    path = _session_path()
    with build_session_writer(path) as w:
        w.write("first session\n")
    assert detect.is_encrypted(path.read_bytes())
    with pytest.raises(HermesCryptoError, match="already sealed"):
        build_session_writer(path)


def test_writer_refuses_when_lock_already_held():
    _enable_with_log_encryption()
    path = _session_path()
    first = build_session_writer(path)
    try:
        with pytest.raises(HermesCryptoError, match="already being written"):
            build_session_writer(path)
    finally:
        first.close()


def test_writer_audits_critical_on_close_failure(monkeypatch):
    _enable_with_log_encryption()
    path = _session_path()

    # The lazy import inside _encrypt_session_if_needed resolves to the
    # package-level ``envelope`` module — patch ``encrypt`` there.
    def boom(_raw, _dek):
        raise RuntimeError("simulated encrypt failure")

    monkeypatch.setattr(envelope, "encrypt", boom)

    log_path = audit.audit_log_path()
    if log_path.is_file():
        log_path.unlink()

    with build_session_writer(path) as w:
        w.write("oops\n")
    # close swallowed the encrypt failure — never raised out of the writer.

    events = audit.read_recent(20)
    assert any(
        e.get("activity") == audit.SESSION_ENCRYPT_FAILED
        and e.get("severity") == "critical"
        for e in events
    ), [e.get("activity") for e in events]
    # File is left plaintext — visible failure mode for the operator.
    assert not detect.is_encrypted(path.read_bytes())
    # Audit detail carries no key material — just path + exception class.
    failed = next(
        e for e in events if e.get("activity") == audit.SESSION_ENCRYPT_FAILED
    )
    detail = failed.get("detail", {})
    assert detail.get("reason") == "RuntimeError"
    assert "key" not in detail
    assert "passphrase" not in detail


def test_read_session_text_passes_through_plaintext():
    path = _session_path("plain.jsonl")
    # write_bytes to avoid the Windows text-mode CRLF translation that would
    # leave \r\n on disk and break the equality check.
    path.write_bytes(b'{"line":1}\n')
    assert read_session_text(path) == '{"line":1}\n'


def test_writer_close_is_idempotent():
    _enable_with_log_encryption()
    path = _session_path()
    w = build_session_writer(path)
    w.write("once\n")
    w.close()
    # Calling close again must not raise, must not re-process the sealed file.
    sealed = path.read_bytes()
    w.close()
    assert path.read_bytes() == sealed


def test_writer_close_seals_even_when_flush_fails(monkeypatch):
    """A failing flush() during close must NOT skip the encrypt-on-close step.

    Regression: the prior close() ran self._fh.flush()
    inside the same try block as _encrypt_session_if_needed(self._path);
    a raising flush() (disk full, I/O error) bypassed the encrypt call,
    leaving plaintext on disk with the lock already released and no
    SESSION_ENCRYPT_FAILED audit event firing. Encryption is now isolated
    so flush failures are non-fatal and the seal always runs against
    whatever made it to disk.
    """
    _enable_with_log_encryption()
    path = _session_path()

    # Clear the audit log so we can assert no SESSION_ENCRYPT_FAILED fires.
    log_path = audit.audit_log_path()
    if log_path.is_file():
        log_path.unlink()

    writer = build_session_writer(path)
    writer.write('{"role":"user","content":"hello"}\n')
    writer.flush()  # push first write to disk *before* sabotaging flush

    def _boom() -> None:
        raise OSError("simulated disk full")

    monkeypatch.setattr(writer._fh, "flush", _boom)

    # close() must swallow the flush failure rather than propagate it.
    writer.close()

    # Encryption ran against the on-disk data despite the flush failure.
    assert detect.is_encrypted(path.read_bytes())
    # Lockfile cleaned up — the outer `finally` ran.
    assert not path.with_name(path.name + ".lock").exists()
    # Data is recoverable through the read helper.
    assert '"role":"user"' in read_session_text(path)
    # Encryption itself succeeded; no spurious critical audit event.
    events = audit.read_recent(20)
    assert not any(
        e.get("activity") == audit.SESSION_ENCRYPT_FAILED for e in events
    ), [e.get("activity") for e in events]


def test_session_writer_hardens_dir_and_file_perms():
    """live session JSONL is 0o600 and parent dir is 0o700.

    Skipped on Windows — POSIX mode bits don't apply and os.chmod is a
    no-op for directories.
    """
    import os
    import sys

    if sys.platform == "win32":
        pytest.skip("POSIX mode bits don't apply on Windows")

    _enable_with_log_encryption()
    path = _session_path("perms-test.jsonl")
    with build_session_writer(path) as w:
        w.write("checking perms\n")
        w.flush()
        # While the live file is open: 0o600 on the file, 0o700 on the dir.
        assert (os.stat(path).st_mode & 0o777) == 0o600, (
            f"live session JSONL has {oct(os.stat(path).st_mode & 0o777)}, "
            "should be 0o600"
        )
        assert (os.stat(path.parent).st_mode & 0o777) == 0o700, (
            f"session dir has {oct(os.stat(path.parent).st_mode & 0o777)}, "
            "should be 0o700"
        )
