"""Tests for hermes_cli.gateway."""

from types import SimpleNamespace

import hermes_cli.gateway as gateway


class TestSystemdLingerStatus:
    def test_reports_enabled(self, monkeypatch):
        monkeypatch.setattr(gateway, "is_linux", lambda: True)
        monkeypatch.setenv("USER", "alice")
        monkeypatch.setattr(
            gateway.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="yes\n", stderr=""),
        )
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/loginctl")

        assert gateway.get_systemd_linger_status() == (True, "")

    def test_reports_disabled(self, monkeypatch):
        monkeypatch.setattr(gateway, "is_linux", lambda: True)
        monkeypatch.setenv("USER", "alice")
        monkeypatch.setattr(
            gateway.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="no\n", stderr=""),
        )
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/loginctl")

        assert gateway.get_systemd_linger_status() == (False, "")


def test_systemd_status_warns_when_linger_disabled(monkeypatch, tmp_path, capsys):
    unit_path = tmp_path / "hermes-gateway.service"
    unit_path.write_text("[Unit]\n")

    monkeypatch.setattr(gateway, "get_systemd_unit_path", lambda: unit_path)
    monkeypatch.setattr(gateway, "get_systemd_linger_status", lambda: (False, ""))

    def fake_run(cmd, capture_output=False, text=False, check=False):
        if cmd[:4] == ["systemctl", "--user", "status", gateway.SERVICE_NAME]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[:3] == ["systemctl", "--user", "is-active"]:
            return SimpleNamespace(returncode=0, stdout="active\n", stderr="")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(gateway.subprocess, "run", fake_run)

    gateway.systemd_status(deep=False)

    out = capsys.readouterr().out
    assert "Gateway service is running" in out
    assert "Systemd linger is disabled" in out
    assert "loginctl enable-linger" in out


def test_systemd_install_checks_linger_status(monkeypatch, tmp_path, capsys):
    unit_path = tmp_path / "systemd" / "user" / "hermes-gateway.service"

    monkeypatch.setattr(gateway, "get_systemd_unit_path", lambda: unit_path)

    calls = []
    helper_calls = []

    def fake_run(cmd, check=False, **kwargs):
        calls.append((cmd, check))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gateway.subprocess, "run", fake_run)
    monkeypatch.setattr(gateway, "_ensure_linger_enabled", lambda: helper_calls.append(True))

    gateway.systemd_install(force=False)

    out = capsys.readouterr().out
    assert unit_path.exists()
    assert [cmd for cmd, _ in calls] == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", gateway.SERVICE_NAME],
    ]
    assert helper_calls == [True]
    assert "Service installed and enabled" in out
