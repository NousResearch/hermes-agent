"""Tests for gateway linger auto-enable behavior on headless Linux installs."""

from types import SimpleNamespace

import hermes_cli.gateway as gateway


class TestEnsureLingerEnabled:
    def test_linger_already_enabled_via_file(self, monkeypatch, capsys):
        monkeypatch.setattr(gateway, "is_linux", lambda: True)
        monkeypatch.setattr(gateway, "is_termux", lambda: False)
        monkeypatch.setattr("getpass.getuser", lambda: "testuser")
        monkeypatch.setattr(gateway, "Path", lambda _path: SimpleNamespace(exists=lambda: True))

        calls = []
        monkeypatch.setattr(gateway.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

        gateway._ensure_linger_enabled()

        out = capsys.readouterr().out
        assert "Systemd linger is enabled" in out
        assert calls == []

    def test_status_enabled_skips_enable(self, monkeypatch, capsys):
        monkeypatch.setattr(gateway, "is_linux", lambda: True)
        monkeypatch.setattr(gateway, "is_termux", lambda: False)
        monkeypatch.setattr("getpass.getuser", lambda: "testuser")
        monkeypatch.setattr(gateway, "Path", lambda _path: SimpleNamespace(exists=lambda: False))
        monkeypatch.setattr(gateway, "get_systemd_linger_status", lambda: (True, ""))

        calls = []
        monkeypatch.setattr(gateway.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

        gateway._ensure_linger_enabled()

        out = capsys.readouterr().out
        assert "Systemd linger is enabled" in out
        assert calls == []

    def test_loginctl_success_enables_linger(self, monkeypatch, capsys):
        monkeypatch.setattr(gateway, "is_linux", lambda: True)
        monkeypatch.setattr(gateway, "is_termux", lambda: False)
        monkeypatch.setattr("getpass.getuser", lambda: "testuser")
        monkeypatch.setattr(gateway, "Path", lambda _path: SimpleNamespace(exists=lambda: False))
        monkeypatch.setattr(gateway, "get_systemd_linger_status", lambda: (False, ""))
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/loginctl")

        run_calls = []

        def fake_run(cmd, capture_output=False, text=False, check=False, **kwargs):
            run_calls.append((cmd, capture_output, text, check))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway.subprocess, "run", fake_run)

        gateway._ensure_linger_enabled()

        out = capsys.readouterr().out
        assert "Enabling linger" in out
        assert "Linger enabled" in out
        assert run_calls == [(["loginctl", "enable-linger", "testuser"], True, True, False)]

    def test_missing_loginctl_shows_manual_guidance(self, monkeypatch, capsys):
        monkeypatch.setattr(gateway, "is_linux", lambda: True)
        monkeypatch.setattr(gateway, "is_termux", lambda: False)
        monkeypatch.setattr("getpass.getuser", lambda: "testuser")
        monkeypatch.setattr(gateway, "Path", lambda _path: SimpleNamespace(exists=lambda: False))
        monkeypatch.setattr(gateway, "get_systemd_linger_status", lambda: (None, "loginctl not found"))
        monkeypatch.setattr("shutil.which", lambda name: None)

        calls = []
        monkeypatch.setattr(gateway.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

        gateway._ensure_linger_enabled()

        out = capsys.readouterr().out
        assert "sudo loginctl enable-linger testuser" in out
        assert "loginctl not found" in out
        assert calls == []

    def test_loginctl_failure_shows_manual_guidance(self, monkeypatch, capsys):
        monkeypatch.setattr(gateway, "is_linux", lambda: True)
        monkeypatch.setattr(gateway, "is_termux", lambda: False)
        monkeypatch.setattr("getpass.getuser", lambda: "testuser")
        monkeypatch.setattr(gateway, "Path", lambda _path: SimpleNamespace(exists=lambda: False))
        monkeypatch.setattr(gateway, "get_systemd_linger_status", lambda: (False, ""))
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/loginctl")
        monkeypatch.setattr(
            gateway.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="Permission denied"),
        )

        gateway._ensure_linger_enabled()

        out = capsys.readouterr().out
        assert "sudo loginctl enable-linger testuser" in out
        assert "Permission denied" in out


def test_systemd_install_calls_linger_helper(monkeypatch, tmp_path, capsys):
    unit_path = tmp_path / "systemd" / "user" / "hermes-gateway.service"

    monkeypatch.setattr(gateway, "get_systemd_unit_path", lambda system=False: unit_path)
    # Non-temp home so the temp-home write guard (which trips on the
    # hermetic test HERMES_HOME) stays out of the way.
    monkeypatch.setattr(
        gateway,
        "generate_systemd_unit",
        lambda system=False, run_as_user=None: (
            '[Service]\nEnvironment="HERMES_HOME=/home/alice/.hermes"\n'
        ),
    )

    calls = []

    def fake_run(cmd, check=False, **kwargs):
        calls.append((cmd, check))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    helper_calls = []
    monkeypatch.setattr(gateway.subprocess, "run", fake_run)
    monkeypatch.setattr(gateway, "_ensure_linger_enabled", lambda: helper_calls.append(True))

    gateway.systemd_install(force=False)

    out = capsys.readouterr().out
    assert unit_path.exists()
    assert [cmd for cmd, _ in calls] == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", gateway.get_service_name()],
    ]
    assert helper_calls == [True]
    assert "User service installed and enabled" in out


def test_systemd_install_repair_path_calls_linger_helper(monkeypatch, tmp_path):
    # The repair branch used to early-return before the linger helper ran, so
    # upgrades on headless hosts left the service dead after the user logged
    # out. Pin that repair now behaves like fresh-install on user scope.
    unit_path = tmp_path / "hermes-gateway.service"
    unit_path.write_text("old unit\n", encoding="utf-8")

    monkeypatch.setattr(gateway, "get_systemd_unit_path", lambda system=False: unit_path)
    monkeypatch.setattr(gateway, "systemd_unit_is_current", lambda system=False: False)
    monkeypatch.setattr(
        gateway, "generate_systemd_unit", lambda system=False, run_as_user=None: "new unit\n"
    )

    calls = []

    def fake_run(cmd, check=False, **kwargs):
        calls.append((cmd, check))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    helper_calls = []
    monkeypatch.setattr(gateway.subprocess, "run", fake_run)
    monkeypatch.setattr(gateway, "_ensure_linger_enabled", lambda: helper_calls.append(True))

    gateway.systemd_install(force=False)

    assert unit_path.read_text(encoding="utf-8") == "new unit\n"
    assert [cmd for cmd, _ in calls] == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", gateway.get_service_name()],
    ]
    assert helper_calls == [True]


def test_systemd_install_repair_path_system_scope_skips_linger(monkeypatch, tmp_path):
    # Linger is a per-user-session concept; system-scope units run under PID 1
    # and must never touch it. Guards against mirroring the user-scope fix in
    # the wrong branch of systemd_install().
    unit_path = tmp_path / "hermes-gateway.service"
    unit_path.write_text("old unit\n", encoding="utf-8")

    monkeypatch.setattr(gateway, "get_systemd_unit_path", lambda system=False: unit_path)
    monkeypatch.setattr(gateway, "systemd_unit_is_current", lambda system=False: False)
    monkeypatch.setattr(
        gateway, "generate_systemd_unit", lambda system=False, run_as_user=None: "new unit\n"
    )
    monkeypatch.setattr(gateway, "_require_root_for_system_service", lambda _action: None)

    calls = []

    def fake_run(cmd, check=False, **kwargs):
        calls.append((cmd, check))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    helper_calls = []
    monkeypatch.setattr(gateway.subprocess, "run", fake_run)
    monkeypatch.setattr(gateway, "_ensure_linger_enabled", lambda: helper_calls.append(True))

    gateway.systemd_install(force=False, system=True)

    assert helper_calls == []
    assert [cmd for cmd, _ in calls] == [
        ["systemctl", "daemon-reload"],
        ["systemctl", "enable", gateway.get_service_name()],
    ]
