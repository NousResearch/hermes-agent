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


@pytest.mark.parametrize(
    ("returncode", "expected"),
    [(0, True), (113, False), (1, None)],
)
def test_macos_sshd_service_state_comes_from_launchctl(
    monkeypatch, returncode, expected
):
    """launchd service presence is the macOS Remote Login signal."""
    monkeypatch.setattr(audit.sys, "platform", "darwin")

    class Result:
        def __init__(self, code):
            self.returncode = code

    monkeypatch.setattr(
        audit.subprocess,
        "run",
        lambda *args, **kwargs: Result(returncode),
    )

    assert audit._macos_sshd_service_active() is expected


@pytest.mark.parametrize(
    "error",
    [
        OSError("launchctl unavailable"),
        audit.subprocess.TimeoutExpired(cmd="launchctl", timeout=2),
    ],
)
def test_macos_sshd_inspection_errors_are_unknown(monkeypatch, error):
    """Inspection failures preserve the conservative warning path."""
    monkeypatch.setattr(audit.sys, "platform", "darwin")

    def fail_inspection(*args, **kwargs):
        raise error

    monkeypatch.setattr(audit.subprocess, "run", fail_inspection)

    assert audit._macos_sshd_service_active() is None


def test_non_macos_skips_launchctl_and_preserves_warning(monkeypatch):
    """Other POSIX platforms keep the original config-based behavior."""
    monkeypatch.setattr(audit.sys, "platform", "linux")

    def unexpected_launchctl(*args, **kwargs):
        raise AssertionError("launchctl must not run outside macOS")

    monkeypatch.setattr(audit.subprocess, "run", unexpected_launchctl)
    monkeypatch.setattr(
        audit,
        "_iter_sshd_config_lines",
        lambda: ["PasswordAuthentication yes"],
    )

    assert audit._macos_sshd_service_active() is None
    assert audit._ssh_password_auth_enabled() is not None


def test_macos_inactive_sshd_suppresses_password_auth_warning(monkeypatch):
    """An installed config is not an exposed SSH service on macOS."""
    monkeypatch.setattr(
        audit,
        "_iter_sshd_config_lines",
        lambda: ["PasswordAuthentication yes"],
    )
    monkeypatch.setattr(
        audit,
        "_macos_sshd_service_active",
        lambda: False,
        raising=False,
    )

    assert audit._ssh_password_auth_enabled() is None


def test_active_sshd_with_password_auth_still_warns(monkeypatch):
    """Suppressing inactive macOS noise must not hide a real SSH exposure."""
    monkeypatch.setattr(
        audit,
        "_iter_sshd_config_lines",
        lambda: ["PasswordAuthentication yes"],
    )
    monkeypatch.setattr(
        audit,
        "_macos_sshd_service_active",
        lambda: True,
        raising=False,
    )

    finding = audit._ssh_password_auth_enabled()

    assert finding is not None
    assert "SSH password authentication is ENABLED" in finding


def test_unknown_sshd_state_keeps_conservative_warning(monkeypatch):
    """Inspection failures should not silently clear a possible exposure."""
    monkeypatch.setattr(
        audit,
        "_iter_sshd_config_lines",
        lambda: ["PasswordAuthentication yes"],
    )
    monkeypatch.setattr(
        audit,
        "_macos_sshd_service_active",
        lambda: None,
        raising=False,
    )

    assert audit._ssh_password_auth_enabled() is not None


# ── container / volume-mount check ──────────────────────────────────────────






# ── network listener without auth ──────────────────────────────────────────




# ── orchestration + logging ─────────────────────────────────────────────────


def test_run_security_audit_aggregates(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "_is_root", lambda: True)
    monkeypatch.setattr(audit, "_macos_sshd_service_active", lambda: True)
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


