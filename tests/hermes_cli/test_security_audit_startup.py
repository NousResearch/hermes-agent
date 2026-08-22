"""Tests for the startup security posture audit (hermes_cli.security_audit_startup)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import hermes_cli.security_audit_startup as audit


@pytest.fixture(autouse=True)
def _reset_audit_sentinel():
    audit._AUDIT_RAN = False
    yield
    audit._AUDIT_RAN = False


# ── root check ────────────────────────────────────────────────────────────




# ── SSH password-auth check ─────────────────────────────────────────────────


def test_ssh_check_silent_when_no_daemon_listening(monkeypatch):
    """Config alone is not exposure — macOS ships sshd_config with Remote Login off."""
    monkeypatch.setattr(audit, "_iter_sshd_config_lines", lambda: ["UsePAM yes"])
    monkeypatch.setattr(audit, "_sshd_is_listening", lambda: False)
    assert audit._ssh_password_auth_enabled() is None


def test_ssh_check_warns_when_daemon_listening(monkeypatch):
    monkeypatch.setattr(audit, "_iter_sshd_config_lines", lambda: ["UsePAM yes"])
    monkeypatch.setattr(audit, "_sshd_is_listening", lambda: True)
    msg = audit._ssh_password_auth_enabled()
    assert msg is not None
    assert "default — no explicit directive" in msg


def test_ssh_check_warns_when_detection_inconclusive(monkeypatch):
    """None must never suppress: a detection gap cannot hide a real finding."""
    monkeypatch.setattr(audit, "_iter_sshd_config_lines", lambda: ["UsePAM yes"])
    monkeypatch.setattr(audit, "_sshd_is_listening", lambda: None)
    assert audit._ssh_password_auth_enabled() is not None


def test_ssh_check_explicit_no_still_short_circuits(monkeypatch):
    monkeypatch.setattr(audit, "_iter_sshd_config_lines", lambda: ["PasswordAuthentication no"])
    monkeypatch.setattr(audit, "_sshd_is_listening", lambda: True)
    assert audit._ssh_password_auth_enabled() is None


def test_sshd_is_listening_false_on_closed_port(monkeypatch):
    """A closed loopback port is a positive 'no daemon'."""
    # Port 1 is reserved and never bound in CI sandboxes.
    monkeypatch.setattr(audit, "_iter_sshd_config_lines", lambda: ["Port 1"])
    assert audit._sshd_is_listening() is False


def test_sshd_is_listening_true_on_open_port(monkeypatch):
    """A bound loopback port is a positive 'daemon present'."""
    import socket

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    try:
        monkeypatch.setattr(
            audit, "_iter_sshd_config_lines", lambda: [f"Port {srv.getsockname()[1]}"]
        )
        assert audit._sshd_is_listening() is True
    finally:
        srv.close()


def test_sshd_is_listening_inconclusive_for_external_bind(monkeypatch):
    """An explicit non-loopback ListenAddress cannot be probed from here."""
    monkeypatch.setattr(
        audit, "_iter_sshd_config_lines", lambda: ["ListenAddress 203.0.113.7", "Port 22"]
    )
    assert audit._sshd_is_listening() is None



# ── container / volume-mount check ──────────────────────────────────────────






# ── network listener without auth ──────────────────────────────────────────




# ── orchestration + logging ─────────────────────────────────────────────────


def test_run_security_audit_aggregates(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "_is_root", lambda: True)
    monkeypatch.setattr(audit, "_iter_sshd_config_lines", lambda: ["PasswordAuthentication yes"])
    monkeypatch.setattr(audit, "_sshd_is_listening", lambda: True)
    monkeypatch.setattr(audit, "_in_container", lambda: False)
    findings = audit.run_security_audit(hermes_home=tmp_path, config={})
    assert len(findings) == 2  # root + ssh


def test_run_security_audit_clean_posture(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "_is_root", lambda: False)
    monkeypatch.setattr(audit, "_iter_sshd_config_lines", lambda: ["PasswordAuthentication no"])
    monkeypatch.setattr(audit, "_in_container", lambda: False)
    assert audit.run_security_audit(hermes_home=tmp_path, config={}) == []


def test_log_startup_security_warnings_emits_and_is_idempotent(monkeypatch, tmp_path, caplog):
    import logging

    monkeypatch.setattr(audit, "_is_root", lambda: True)
    monkeypatch.setattr(audit, "_iter_sshd_config_lines", lambda: [])
    monkeypatch.setattr(audit, "_in_container", lambda: False)

    with caplog.at_level(logging.WARNING, logger="hermes.security_audit"):
        first = audit.log_startup_security_warnings(hermes_home=tmp_path, config={})
    assert len(first) == 1
    assert any("ROOT" in r.message for r in caplog.records)

    # Second call is a no-op (idempotent within a process) unless forced.
    second = audit.log_startup_security_warnings(hermes_home=tmp_path, config={})
    assert second == []
    forced = audit.log_startup_security_warnings(hermes_home=tmp_path, config={}, force=True)
    assert len(forced) == 1


