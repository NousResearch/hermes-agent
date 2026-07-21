"""Regression tests for issue #68699 — shared Nous token store silently fails.

The shared Nous OAuth store at ``${HERMES_SHARED_AUTH_DIR}/nous_auth.json``
was silently failing to be created when the atomic rename step hit EXDEV
or EBUSY (or any other OSError). The reporter found four orphaned
``nous_auth.json.tmp.*`` files in their install — valid payloads that
never made it to ``nous_auth.json`` because the rename step failed and
the outer ``except Exception`` swallowed the failure at ``logger.debug``
level.

The fix:
1. Uses the existing ``atomic_replace`` helper (handles EXDEV/EBUSY via
   copy/fsync/unlink fallback) instead of bare ``os.replace``.
2. Promotes the failure log to ``logger.warning`` so operators can see it.
3. Sweeps stale ``nous_auth.json.tmp.*`` files before each write so
   prior crashed attempts don't accumulate.
4. Verifies after write that ``nous_auth.json`` exists and re-attempts
   once if it doesn't.

These tests are RED before the fix and GREEN after.
"""

from __future__ import annotations

import errno
import json
import os
import stat
import sys
import threading
from unittest.mock import patch

import pytest


pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="POSIX mode bits and EXDEV/EBUSY not exercised on Windows",
)


# Reuse the toctou-mode test's helpers where possible
def _write_minimal_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_SHARED_AUTH_DIR", str(tmp_path / "shared_override"))
    from hermes_cli import auth as auth_mod

    state = {
        "access_token": "***",
        "refresh_token": "***",
        "token_type": "Bearer",
        "scope": "openid profile",
        "client_id": "test-client",
        "obtained_at": "2026-01-01T00:00:00Z",
        "expires_at": "2026-01-01T01:00:00Z",
    }
    return auth_mod, state


# ---------------------------------------------------------------------------
# Layer 1: EXDEV fallback — the rename fails because tmp and target live on
# different filesystems. The fix must use atomic_replace() which falls back
# to copy + fsync + unlink, instead of letting bare os.replace() raise.
# ---------------------------------------------------------------------------


def test_shared_nous_store_handles_exdev_during_rename(tmp_path, monkeypatch):
    """EXDEV on the atomic rename must NOT leave the shared store missing."""
    auth_mod, state = _write_minimal_state(tmp_path, monkeypatch)
    shared_dir = tmp_path / "shared_override"

    real_replace = os.replace

    def exdev_replace(src, dst):
        if str(dst).endswith("nous_auth.json"):
            raise OSError(errno.EXDEV, "Cross-device link")
        return real_replace(src, dst)

    with patch.object(os, "replace", side_effect=exdev_replace):
        auth_mod._write_shared_nous_state(state)

    store = shared_dir / "nous_auth.json"
    assert store.exists(), (
        "shared Nous store was not written — EXDEV fallback regressed "
        "and bare os.replace raised silently"
    )
    # No orphan temp should remain after a successful fallback
    leftover = list(shared_dir.glob("nous_auth.json.tmp.*"))
    assert not leftover, f"orphan temp files left after EXDEV fallback: {leftover}"


# ---------------------------------------------------------------------------
# Layer 1 (cont): EBUSY fallback — file is busy (e.g., Windows Defender
# scanning, antivirus, or NFS stale handle). Same fallback path.
# ---------------------------------------------------------------------------


def test_shared_nous_store_handles_ebusy_during_rename(tmp_path, monkeypatch):
    """EBUSY on the atomic rename must NOT leave the shared store missing."""
    auth_mod, state = _write_minimal_state(tmp_path, monkeypatch)
    shared_dir = tmp_path / "shared_override"

    real_replace = os.replace

    def ebusy_replace(src, dst):
        if str(dst).endswith("nous_auth.json"):
            raise OSError(errno.EBUSY, "Device or resource busy")
        return real_replace(src, dst)

    with patch.object(os, "replace", side_effect=ebusy_replace):
        auth_mod._write_shared_nous_state(state)

    store = shared_dir / "nous_auth.json"
    assert store.exists(), (
        "shared Nous store was not written — EBUSY fallback regressed"
    )


# ---------------------------------------------------------------------------
# Layer 3: Stale temp file cleanup. The reporter observed multiple orphan
# temp files in their install — each from a previous failed attempt.
# The fix sweeps stale .tmp.* files before each write.
# ---------------------------------------------------------------------------


def test_shared_nous_store_sweeps_stale_temps_before_write(tmp_path, monkeypatch):
    """Pre-existing orphan temp files must be cleaned before the new write."""
    auth_mod, state = _write_minimal_state(tmp_path, monkeypatch)
    shared_dir = tmp_path / "shared_override"
    shared_dir.mkdir(parents=True, exist_ok=True)

    # Simulate two stale temps from prior crashed writes — one with valid
    # JSON content, one 0-byte (the reporter observed a 0-byte temp).
    valid_temp = shared_dir / "nous_auth.json.tmp.99999.deadbeef"
    valid_temp.write_text('{"_schema": 1, "stale": true}')
    zero_temp = shared_dir / "nous_auth.json.tmp.88888.cafebabe"
    zero_temp.write_text("")

    auth_mod._write_shared_nous_state(state)

    leftover = list(shared_dir.glob("nous_auth.json.tmp.*"))
    assert not leftover, f"stale temp files were not swept: {leftover}"
    # New store exists with the correct content
    store = shared_dir / "nous_auth.json"
    assert store.exists()
    data = json.loads(store.read_text())
    assert data["refresh_token"] == "***"


# ---------------------------------------------------------------------------
# Layer 4: Verify-after-write. If the rename reported success but the
# final file is missing or 0 bytes (a race the reporter observed), the
# fix must detect it and re-attempt or surface the failure.
# ---------------------------------------------------------------------------


def test_shared_nous_store_detects_missing_file_after_rename(tmp_path, monkeypatch, caplog):
    """If the rename succeeded but the file is missing/empty, surface it.

    Production guarantee: a WARNING log line must be emitted so operators
    can see the structural condition for #43589 mutual-revocation.
    """
    import logging

    auth_mod, state = _write_minimal_state(tmp_path, monkeypatch)
    shared_dir = tmp_path / "shared_override"

    real_replace = os.replace

    def silent_replace(src, dst):
        # Simulate the race: rename reports success but final file is gone.
        result = real_replace(src, dst)
        if str(dst).endswith("nous_auth.json"):
            try:
                os.unlink(dst)
            except OSError:
                pass
        return result

    with caplog.at_level(logging.DEBUG, logger="hermes_cli.auth"):
        with patch.object(os, "replace", side_effect=silent_replace):
            auth_mod._write_shared_nous_state(state)

    # The verify-after-write path must emit a WARNING log line so the operator
    # sees the silent failure (#68699 primary complaint).
    verify_warnings = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING
        and ("rename reported success" in r.getMessage().lower()
             or "missing or empty" in r.getMessage().lower())
    ]
    assert verify_warnings, (
        "verify-after-write missing file must surface at WARNING — "
        "operators can't see silent failures at default log level"
    )


# ---------------------------------------------------------------------------
# Layer 2: Log level escalation. The reporter's primary complaint:
# "No error is surfaced." The fix must log at WARNING level, not debug.
# ---------------------------------------------------------------------------


def test_shared_nous_store_failure_logs_at_warning_level(tmp_path, monkeypatch, caplog):
    """A failed write must surface at WARNING level so operators see it."""
    import logging

    auth_mod, state = _write_minimal_state(tmp_path, monkeypatch)

    def explode_replace(src, dst):
        raise OSError(errno.ENOSPC, "No space left on device")

    with caplog.at_level(logging.DEBUG, logger="hermes_cli.auth"):
        with patch.object(os, "replace", side_effect=explode_replace):
            auth_mod._write_shared_nous_state(state)

    # Find the failure log
    failure_records = [
        r for r in caplog.records
        if "shared Nous auth store" in r.getMessage().lower()
        or "nous_auth" in r.getMessage().lower()
    ]
    assert failure_records, "no log line recorded for shared Nous store failure"
    # The reporter requires WARNING (not DEBUG)
    severities = {r.levelno for r in failure_records}
    assert logging.WARNING in severities, (
        f"shared Nous store failure logged at {[logging.getLevelName(s) for s in severities]} "
        "instead of WARNING — operators won't see this at default log level"
    )


# ---------------------------------------------------------------------------
# Integration: normal happy path still works after the fix.
# ---------------------------------------------------------------------------


def test_shared_nous_store_happy_path_unchanged(tmp_path, monkeypatch):
    """The basic 0o600 / parent 0o700 contract must still hold after the fix."""
    auth_mod, state = _write_minimal_state(tmp_path, monkeypatch)

    auth_mod._write_shared_nous_state(state)
    path = auth_mod._nous_shared_store_path()

    assert path.exists(), "shared Nous store was not written"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700

    data = json.loads(path.read_text())
    assert data["refresh_token"] == "***"
    assert data["access_token"] == "***"


# ---------------------------------------------------------------------------
# Concurrency: two threads racing the write — the cross-profile lock must
# serialize them and both must end up with a valid store.
# ---------------------------------------------------------------------------


def test_shared_nous_store_concurrent_writes_serialize(tmp_path, monkeypatch):
    """Concurrent writes to the shared store must serialize and both succeed."""
    auth_mod, state = _write_minimal_state(tmp_path, monkeypatch)
    shared_dir = tmp_path / "shared_override"

    errors = []

    def writer(i):
        try:
            local_state = dict(state)
            local_state["refresh_token"] = f"refresh-{i}"
            auth_mod._write_shared_nous_state(local_state)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent writes raised: {errors}"
    store = shared_dir / "nous_auth.json"
    assert store.exists()
    leftover = list(shared_dir.glob("nous_auth.json.tmp.*"))
    assert not leftover, f"orphan temp files after concurrent writes: {leftover}"
