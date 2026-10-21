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


@pytest.fixture(autouse=True)
def _no_sshd_binary(monkeypatch):
    # The host running the tests may have sshd; the file-reading path is what most cases test.
    monkeypatch.setattr(audit, "_sshd_effective_password_auth", lambda: None)


def _sshd_tree(tmp_path, main, dropins=None):
    (tmp_path / "sshd_config.d").mkdir()
    for name, body in (dropins or {}).items():
        (tmp_path / "sshd_config.d" / name).write_text(body)
    cfg = tmp_path / "sshd_config"
    cfg.write_text(main.replace("@D", str(tmp_path)))
    return cfg


UBUNTU_MAIN = "Include @D/sshd_config.d/*.conf\n#PasswordAuthentication yes\nKbdInteractiveAuthentication no\n"


@pytest.mark.parametrize("main,dropins,warns", [
    # sshd keeps the first value: 20-*.conf's "no" beats a later 50-cloud-init.conf "yes".
    (UBUNTU_MAIN, {"20-keys-only.conf": "PasswordAuthentication no\n", "50-cloud-init.conf": "PasswordAuthentication yes\n"}, False),
    (UBUNTU_MAIN, {"10-a.conf": "PasswordAuthentication yes\n", "20-b.conf": "PasswordAuthentication no\n"}, True),
    (UBUNTU_MAIN, {"50-cloud-init.conf": "PasswordAuthentication yes\n"}, True),
    (UBUNTU_MAIN, {}, True),  # no directive: sshd's default is yes
    ("PasswordAuthentication no\nInclude @D/sshd_config.d/*.conf\n", {"50-x.conf": "PasswordAuthentication yes\n"}, False),
    (UBUNTU_MAIN, {"20-x.conf": "PasswordAuthentication=no\n"}, False),
    # A Match block's value is conditional, not global.
    (UBUNTU_MAIN + "Match User deploy\n  PasswordAuthentication no\n", {}, True),
])
def test_ssh_password_auth_follows_sshd_order(monkeypatch, tmp_path, main, dropins, warns):
    monkeypatch.setattr(audit, "_SSHD_CONFIG", _sshd_tree(tmp_path, main, dropins))
    assert bool(audit._ssh_password_auth_enabled()) is warns


def test_ssh_password_auth_unreadable_dropin_is_not_a_finding(monkeypatch, tmp_path):
    cfg = _sshd_tree(tmp_path, UBUNTU_MAIN, {"50-cloud-init.conf": "PasswordAuthentication yes\n"})
    monkeypatch.setattr(audit, "_SSHD_CONFIG", cfg)
    real_read = Path.read_text

    def read_text(self, *a, **k):
        if self.name == "50-cloud-init.conf":
            raise PermissionError(13, "Permission denied")
        return real_read(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", read_text)
    assert audit._ssh_password_auth_enabled() is None


def test_ssh_password_auth_no_sshd_config(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "_SSHD_CONFIG", tmp_path / "missing")
    assert audit._ssh_password_auth_enabled() is None


@pytest.mark.parametrize("effective,warns", [("no", False), ("yes", True)])
def test_ssh_password_auth_prefers_sshd_T(monkeypatch, tmp_path, effective, warns):
    # sshd -T (the effective config) wins over whatever the files look like.
    monkeypatch.setattr(audit, "_SSHD_CONFIG", _sshd_tree(tmp_path, "PasswordAuthentication yes\n"))
    monkeypatch.setattr(audit, "_sshd_effective_password_auth", lambda: effective)
    assert bool(audit._ssh_password_auth_enabled()) is warns


# ── container / volume-mount check ──────────────────────────────────────────






# ── network listener without auth ──────────────────────────────────────────




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
    monkeypatch.setattr(audit, "_iter_sshd_config_lines", list)
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


