"""Audit log for dashboard-auth events.

Profile-aware location: ``$HERMES_HOME/logs/dashboard-auth.log``.
Format: one JSON object per line. Token-like kwargs are dropped before
serialisation so we never leak refresh tokens or JWTs to disk.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from hermes_cli.dashboard_auth.audit import audit_log, AuditEvent


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    """Redirect $HERMES_HOME and ~ to a tmp dir for the duration of the test."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Some code paths fall back to Path.home() — patch that too.
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    return home


@pytest.mark.parametrize("non_native", [False, True])
def test_audit_bounds_string_values(profile_home: Path, non_native: bool) -> None:
    oversized = 'value"\n' * 1000
    audit_log(
        AuditEvent.LOGIN_FAILURE, provider=oversized,
        details={"values": [oversized]}, extra=Path(oversized) if non_native else oversized,
        reason="invalid_credentials", attempts=3, allowed=False,
        access_token=oversized,
    )
    lines = (profile_home / "logs" / "dashboard-auth.log").read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    for value in (entry["provider"], entry["details"]["values"][0], entry["extra"]):
        assert len(value) <= 256
        assert value.endswith("...[truncated]")
    assert oversized.startswith(entry["provider"].removesuffix("...[truncated]"))
    assert entry["reason"] == "invalid_credentials"
    assert entry["attempts"] == 3
    assert entry["allowed"] is False
    assert "access_token" not in entry


def test_audit_writes_jsonlines(profile_home):
    audit_log(AuditEvent.LOGIN_START, provider="nous", ip="1.2.3.4")
    audit_log(
        AuditEvent.LOGIN_SUCCESS,
        provider="nous", user_id="u1",
        email="a@b.com", ip="1.2.3.4",
    )

    path = profile_home / "logs" / "dashboard-auth.log"
    assert path.exists(), f"audit log not created at {path}"
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2

    second = json.loads(lines[1])
    assert second["event"] == "login_success"
    assert second["provider"] == "nous"
    assert second["user_id"] == "u1"
    assert second["email"] == "a@b.com"
    assert "ts" in second  # ISO-8601 timestamp


def test_audit_rotates_when_over_max_bytes(profile_home, monkeypatch):
    """Rotation is bounded: at most ``_BACKUP_COUNT`` ``.N`` files, current log under the cap."""
    import hermes_cli.dashboard_auth.audit as audit_module

    monkeypatch.setattr(audit_module, "_MAX_BYTES", 2000)
    monkeypatch.setattr(audit_module, "_BACKUP_COUNT", 3)

    for i in range(500):
        audit_log(AuditEvent.LOGIN_START, provider="x" * 50, i=i)

    logs = profile_home / "logs"
    assert (logs / "dashboard-auth.log").exists()
    for i in (1, 2, 3):
        assert (logs / f"dashboard-auth.log.{i}").exists()
    assert not (logs / "dashboard-auth.log.4").exists()
    assert (logs / "dashboard-auth.log").stat().st_size < 2000


def test_audit_rotates_on_crossing_write(profile_home, monkeypatch):
    """A write that pushes the file to/past the cap must rotate BEFORE appending (the pending
    line's bytes count toward the threshold), not stay over the cap until the next event."""
    import hermes_cli.dashboard_auth.audit as audit_module

    monkeypatch.setattr(audit_module, "_MAX_BYTES", 200)
    monkeypatch.setattr(audit_module, "_BACKUP_COUNT", 3)

    logs = profile_home / "logs"
    logs.mkdir()
    prior = logs / "dashboard-auth.log"
    prior.write_bytes(b"x" * 190)

    audit_log(AuditEvent.LOGIN_START, provider="nous")

    assert (logs / "dashboard-auth.log.1").read_bytes() == b"x" * 190
    assert prior.stat().st_size < 200


def test_audit_redacts_token_like_fields(profile_home):
    audit_log(
        AuditEvent.LOGIN_SUCCESS,
        provider="nous", access_token="should-not-appear",
        refresh_token="also-not", code="not-this", state="nope",
    )
    raw = (profile_home / "logs" / "dashboard-auth.log").read_text()
    for forbidden in ("should-not-appear", "also-not", "not-this", "nope"):
        assert forbidden not in raw, f"token-like value leaked into audit log: {forbidden}"




