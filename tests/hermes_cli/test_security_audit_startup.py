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


def test_root_check_flags_uid_zero(monkeypatch):
    monkeypatch.setattr(audit, "_is_root", lambda: True)
    msg = audit._running_as_root()
    assert msg and "ROOT" in msg


def test_root_check_silent_for_non_root(monkeypatch):
    monkeypatch.setattr(audit, "_is_root", lambda: False)
    assert audit._running_as_root() is None


# ── SSH password-auth check ─────────────────────────────────────────────────


def test_ssh_password_auth_enabled_explicit_yes(monkeypatch):
    monkeypatch.setattr(
        audit, "_iter_sshd_config_lines",
        lambda: ["PasswordAuthentication yes", "PermitRootLogin no"],
    )
    msg = audit._ssh_password_auth_enabled()
    assert msg and "password authentication is enabled" in msg.lower()


def test_ssh_password_auth_disabled(monkeypatch):
    monkeypatch.setattr(
        audit, "_iter_sshd_config_lines",
        lambda: ["PasswordAuthentication no"],
    )
    assert audit._ssh_password_auth_enabled() is None


def test_ssh_password_auth_default_is_yes(monkeypatch):
    """No explicit directive → sshd default is 'yes' → warn (with qualifier)."""
    monkeypatch.setattr(
        audit, "_iter_sshd_config_lines",
        lambda: ["PermitRootLogin prohibit-password"],
    )
    msg = audit._ssh_password_auth_enabled()
    assert msg and "default" in msg.lower()


def test_ssh_check_silent_when_no_config(monkeypatch):
    """No sshd config readable (e.g. Windows / SSH not installed) → no finding."""
    monkeypatch.setattr(audit, "_iter_sshd_config_lines", lambda: [])
    assert audit._ssh_password_auth_enabled() is None


def test_ssh_last_directive_wins(monkeypatch):
    monkeypatch.setattr(
        audit, "_iter_sshd_config_lines",
        lambda: ["PasswordAuthentication yes", "PasswordAuthentication no"],
    )
    assert audit._ssh_password_auth_enabled() is None


# ── container / volume-mount check ──────────────────────────────────────────


def test_container_no_mount_flags(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "_in_container", lambda: True)
    monkeypatch.setattr(audit, "_path_is_mounted", lambda p: False)
    msg = audit._container_no_volume_mount(tmp_path / ".hermes")
    assert msg and "persistent volume" in msg


def test_container_with_mount_silent(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "_in_container", lambda: True)
    monkeypatch.setattr(audit, "_path_is_mounted", lambda p: True)
    assert audit._container_no_volume_mount(tmp_path / ".hermes") is None


def test_not_in_container_silent(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "_in_container", lambda: False)
    assert audit._container_no_volume_mount(tmp_path / ".hermes") is None


# ── network listener without auth ──────────────────────────────────────────


def test_api_server_network_no_key_flags(monkeypatch):
    monkeypatch.delenv("API_SERVER_KEY", raising=False)
    cfg = {"platforms": {"api_server": {"enabled": True, "extra": {"host": "0.0.0.0", "key": ""}}}}
    findings = audit._network_listener_without_auth(cfg)
    assert any("NO API_SERVER_KEY" in f for f in findings)


def test_api_server_loopback_silent(monkeypatch):
    cfg = {"platforms": {"api_server": {"enabled": True, "extra": {"host": "127.0.0.1", "key": ""}}}}
    assert audit._network_listener_without_auth(cfg) == []


def test_api_server_with_key_silent(monkeypatch):
    cfg = {"platforms": {"api_server": {"enabled": True, "extra": {"host": "0.0.0.0", "key": "a-strong-key-1234567890"}}}}
    assert audit._network_listener_without_auth(cfg) == []


# ── orchestration + logging ─────────────────────────────────────────────────


def test_run_security_audit_aggregates(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "_is_root", lambda: True)
    monkeypatch.setattr(audit, "_iter_sshd_config_lines", lambda: ["PasswordAuthentication yes"])
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


def test_audit_never_raises_on_broken_check(monkeypatch, tmp_path):
    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(audit, "_is_root", _boom)
    # Must not propagate — the broken check is swallowed, others still run.
    findings = audit.run_security_audit(hermes_home=tmp_path, config={})
    assert isinstance(findings, list)
