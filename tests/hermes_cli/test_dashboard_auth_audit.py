"""Audit log for dashboard-auth events.

Profile-aware location: ``$HERMES_HOME/logs/dashboard-auth.log``.
Format: one JSON object per line. Token-like kwargs are dropped before
serialisation so we never leak refresh tokens or JWTs to disk.
"""
from __future__ import annotations

import json
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


def test_audit_redacts_token_like_fields(profile_home):
    audit_log(
        AuditEvent.LOGIN_SUCCESS,
        provider="nous", access_token="should-not-appear",
        refresh_token="also-not", code="not-this", state="nope",
    )
    raw = (profile_home / "logs" / "dashboard-auth.log").read_text()
    for forbidden in ("should-not-appear", "also-not", "not-this", "nope"):
        assert forbidden not in raw, f"token-like value leaked into audit log: {forbidden}"


def test_audit_all_event_types_have_string_values():
    for ev in AuditEvent:
        assert isinstance(ev.value, str)
        assert ev.value


def test_audit_write_failure_does_not_raise(monkeypatch, tmp_path):
    """A broken audit log must not crash auth."""
    # Point HERMES_HOME at a file (not a dir) so mkdir/open will fail.
    broken = tmp_path / "not-a-dir"
    broken.write_text("blocking file")
    monkeypatch.setenv("HERMES_HOME", str(broken))
    # Should NOT raise.
    audit_log(AuditEvent.LOGIN_FAILURE, provider="nous", reason="x")


def test_audit_creates_logs_dir_if_missing(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # logs/ deliberately does not exist
    audit_log(AuditEvent.LOGIN_START, provider="nous")
    assert (home / "logs").is_dir()
    assert (home / "logs" / "dashboard-auth.log").exists()


def test_audit_rotates_log_when_size_cap_is_reached(profile_home, monkeypatch):
    from hermes_cli.dashboard_auth import audit as audit_mod

    monkeypatch.setattr(audit_mod, "_MAX_LOG_BYTES", 120)
    monkeypatch.setattr(audit_mod, "_BACKUP_COUNT", 2)

    log_path = profile_home / "logs" / "dashboard-auth.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("older-login-line\n" * 8, encoding="utf-8")
    (profile_home / "logs" / "dashboard-auth.log.1").write_text(
        "previous-backup\n", encoding="utf-8",
    )
    (profile_home / "logs" / "dashboard-auth.log.2").write_text(
        "oldest-backup\n", encoding="utf-8",
    )

    audit_log(AuditEvent.LOGIN_START, provider="nous")

    current = log_path.read_text(encoding="utf-8")
    rotated = (profile_home / "logs" / "dashboard-auth.log.1").read_text(encoding="utf-8")
    second_backup = (profile_home / "logs" / "dashboard-auth.log.2").read_text(encoding="utf-8")

    assert "login_start" in current
    assert "older-login-line" not in current
    assert "older-login-line" in rotated
    assert second_backup == "previous-backup\n"
