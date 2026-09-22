"""Regression for #119162: rotating an env-backed credential's secret drops its cooldown.

``_merge_disk_cooldown_state`` keeps a persisted ``exhausted``/``dead`` status over a stale
in-memory snapshot, so a writer that predates another process's cooldown cannot resurrect a
rate-limited key. The mirror rule is that a credential whose secret changed *is* a different
credential and must not inherit the old status — but env-backed/borrowed rows are persisted
without their secret, only a ``secret_fingerprint``, so the rotation has to be read from that
fingerprint rather than from ``access_token``.
"""
import time

import pytest

from hermes_cli.auth import _merge_disk_cooldown_state

FINGERPRINT_OLD = "sha256:1111111111111111"
FINGERPRINT_NEW = "sha256:2222222222222222"


def _env_backed_row(fingerprint, location="top_level", **overrides):
    """A sanitized (env-backed) pool row: no ``access_token`` anywhere, only a fingerprint."""
    row = {
        "id": "abc123",
        "label": "env:OPENCODE_GO_API_KEY",
        "source": "env:OPENCODE_GO_API_KEY",
        "last_status": None,
        "last_status_at": None,
        "last_error_code": None,
        "last_error_reset_at": None,
        "failure_reason": None,
    }
    if location == "top_level":
        row["secret_fingerprint"] = fingerprint
    else:
        row["extra"] = {"secret_fingerprint": fingerprint}
    row.update(overrides)
    return row


def _fingerprint(row):
    """Read a fixture row's fingerprint wherever this test put it."""
    return row.get("secret_fingerprint") or (row.get("extra") or {}).get("secret_fingerprint")


def _persisted_cooldown(fingerprint, location="top_level", **overrides):
    return _env_backed_row(
        fingerprint, location,
        last_status="exhausted",
        last_status_at=time.time() - 60.0,
        last_error_code=429,
        last_error_reset_at=time.time() + 15 * 24 * 3600,
        failure_reason="rate_limit",
        **overrides,
    )


@pytest.mark.parametrize("location", ["top_level", "extra"])
def test_rotated_env_backed_secret_does_not_inherit_disk_cooldown(location):
    """Differing fingerprints mean a fresh credential: the old 429 cooldown must be dropped."""
    entry = _env_backed_row(FINGERPRINT_NEW, location)                 # the key now in .env
    disk_entry = _persisted_cooldown(FINGERPRINT_OLD, location)        # what the OLD key persisted

    merged = _merge_disk_cooldown_state(entry, disk_entry, "opencode-go")

    assert merged["last_status"] is None
    assert merged["last_status_at"] is None
    assert merged["last_error_code"] is None
    assert merged["last_error_reset_at"] is None
    assert merged.get("failure_reason") is None
    assert _fingerprint(merged) == FINGERPRINT_NEW


def test_same_env_backed_secret_still_inherits_disk_cooldown():
    """An unchanged fingerprint is the same credential: the other process's cooldown still wins."""
    entry = _env_backed_row(FINGERPRINT_OLD, "top_level", last_status_at=time.time() - 300.0)
    disk_entry = _persisted_cooldown(FINGERPRINT_OLD)

    merged = _merge_disk_cooldown_state(entry, disk_entry, "opencode-go")

    assert merged["last_status"] == "exhausted"
    assert merged["last_error_code"] == 429
    assert merged["last_error_reset_at"] == disk_entry["last_error_reset_at"]
    assert merged["secret_fingerprint"] == FINGERPRINT_OLD
