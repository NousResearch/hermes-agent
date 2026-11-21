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


# ── SSH: a stock config on a host that serves no SSH (#155167) ──────────────

# The stock /etc/ssh/sshd_config of an openssh package: many non-comment lines, PasswordAuthentication
# commented out, so no explicit directive is found.
_STOCK_SSHD_LINES = ["Include /etc/ssh/sshd_config.d/*.conf", "UsePAM yes", "Subsystem sftp /usr/lib/ssh/sftp-server"]

_TCP_HEADER = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode"


def _tcp_row(local_port: int, state: str = "0A") -> str:
    return f"   0: 00000000:{local_port:04X} 00000000:0000 {state} 00000000:00000000 00:00000000 00000000     0        0 1"


def _serving_probe(monkeypatch, served: bool | None) -> None:
    """Stub the daemon probe when the tree has it. Without it the finding keeps firing, so the
    caller's own assertion (not a setup error) is what reports the regression.
    """
    if hasattr(audit, "_sshd_serving"):
        monkeypatch.setattr(audit, "_sshd_serving", lambda lines: served)


def _require(monkeypatch, name: str) -> None:
    if not hasattr(audit, name):
        monkeypatch.undo()
        pytest.skip(f"{name} is not implemented on this tree")


def test_stock_sshd_config_is_silent_when_no_daemon_serves(monkeypatch):
    monkeypatch.setattr(audit, "_iter_sshd_config_lines", lambda: list(_STOCK_SSHD_LINES))
    _serving_probe(monkeypatch, False)
    assert audit._ssh_password_auth_enabled() is None


def test_stock_sshd_config_still_warns_when_a_daemon_serves(monkeypatch):
    monkeypatch.setattr(audit, "_iter_sshd_config_lines", lambda: list(_STOCK_SSHD_LINES))
    _serving_probe(monkeypatch, True)
    finding = audit._ssh_password_auth_enabled()
    assert finding is not None and "ENABLED (default — no explicit directive)" in finding


def test_stock_sshd_config_warns_when_daemon_state_is_unknown(monkeypatch):
    monkeypatch.setattr(audit, "_iter_sshd_config_lines", lambda: list(_STOCK_SSHD_LINES))
    _serving_probe(monkeypatch, None)
    assert audit._ssh_password_auth_enabled() is not None  # fail open: keep the warning


def test_explicit_password_authentication_yes_warns_even_when_not_serving(monkeypatch):
    monkeypatch.setattr(audit, "_iter_sshd_config_lines", lambda: ["PasswordAuthentication yes"])
    _serving_probe(monkeypatch, False)
    finding = audit._ssh_password_auth_enabled()
    assert finding is not None and "(default" not in finding


def test_run_security_audit_skips_the_stock_ssh_finding_when_not_serving(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "_is_root", lambda: False)
    monkeypatch.setattr(audit, "_iter_sshd_config_lines", lambda: list(_STOCK_SSHD_LINES))
    _serving_probe(monkeypatch, False)
    monkeypatch.setattr(audit, "_in_container", lambda: False)
    assert audit.run_security_audit(hermes_home=tmp_path, config={}) == []


def test_sshd_serving_is_true_for_a_listener_on_the_configured_port(monkeypatch):
    _require(monkeypatch, "_sshd_serving")
    monkeypatch.setattr(audit, "_listening_tcp_ports", lambda: {22})
    monkeypatch.setattr(audit, "_sshd_process_running", lambda: False)
    assert audit._sshd_serving(list(_STOCK_SSHD_LINES)) is True
    # A Port directive moves the listener: 22 alone then proves nothing about this daemon.
    monkeypatch.setattr(audit, "_listening_tcp_ports", lambda: {2222})
    assert audit._sshd_serving(list(_STOCK_SSHD_LINES)) is False
    assert audit._sshd_serving(["Port 2222"]) is True


def test_sshd_serving_falls_back_to_the_process_for_an_unlisted_port(monkeypatch):
    _require(monkeypatch, "_sshd_serving")
    monkeypatch.setattr(audit, "_listening_tcp_ports", lambda: set())
    monkeypatch.setattr(audit, "_sshd_process_running", lambda: True)  # started with -p 2022
    assert audit._sshd_serving(list(_STOCK_SSHD_LINES)) is True


def test_sshd_serving_is_unknown_when_the_socket_table_is_unreadable(monkeypatch):
    _require(monkeypatch, "_sshd_serving")
    monkeypatch.setattr(audit, "_listening_tcp_ports", lambda: None)  # no /proc (macOS, Windows)
    assert audit._sshd_serving(list(_STOCK_SSHD_LINES)) is None


def test_listening_tcp_ports_reads_listen_state_only(tmp_path, monkeypatch):
    _require(monkeypatch, "_listening_tcp_ports")
    table = tmp_path / "tcp"
    table.write_text("\n".join([_TCP_HEADER, _tcp_row(22), _tcp_row(8080, state="01")]), encoding="utf-8")
    table6 = tmp_path / "tcp6"
    table6.write_text("\n".join([_TCP_HEADER, _tcp_row(2222)]), encoding="utf-8")
    assert audit._listening_tcp_ports((table, table6)) == {22, 2222}
    assert audit._listening_tcp_ports((table, tmp_path / "missing")) == {22}
    assert audit._listening_tcp_ports((tmp_path / "missing",)) is None


def test_sshd_process_running_matches_comm_exactly(tmp_path, monkeypatch):
    _require(monkeypatch, "_sshd_process_running")
    proc = tmp_path / "proc"
    proc.mkdir()
    for pid, comm in (("1", "systemd"), ("42", "sshd"), ("43", "sshd-session")):
        (proc / pid).mkdir()
        (proc / pid / "comm").write_text(comm + "\n", encoding="utf-8")
    assert audit._sshd_process_running(proc) is True
    (proc / "42" / "comm").write_text("entrypoint.sh\n", encoding="utf-8")
    assert audit._sshd_process_running(proc) is False  # a per-connection child is not a listener
    assert audit._sshd_process_running(tmp_path / "missing") is None


