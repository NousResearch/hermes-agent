"""Tests for gateway linger auto-enable behavior on headless Linux installs."""

from types import SimpleNamespace

import pytest

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


    def test_loginctl_success_enables_linger(self, monkeypatch, capsys):
        monkeypatch.setattr(gateway, "is_linux", lambda: True)
        monkeypatch.setattr(gateway, "is_termux", lambda: False)
        monkeypatch.setattr("getpass.getuser", lambda: "testuser")
        monkeypatch.setattr(gateway, "Path", lambda _path: SimpleNamespace(exists=lambda: False))
        monkeypatch.setattr(gateway, "get_systemd_linger_status", lambda username=None: (False, ""))
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


    def test_loginctl_failure_shows_manual_guidance(self, monkeypatch, capsys):
        monkeypatch.setattr(gateway, "is_linux", lambda: True)
        monkeypatch.setattr(gateway, "is_termux", lambda: False)
        monkeypatch.setattr("getpass.getuser", lambda: "testuser")
        monkeypatch.setattr(gateway, "Path", lambda _path: SimpleNamespace(exists=lambda: False))
        monkeypatch.setattr(gateway, "get_systemd_linger_status", lambda username=None: (False, ""))
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

    def test_system_target_user_waits_for_that_users_bus(self, monkeypatch, capsys):
        """Root installing a system unit enables linger for User= and waits for THAT uid's bus, not its own."""
        monkeypatch.setattr(gateway, "is_linux", lambda: True)
        monkeypatch.setattr(gateway, "is_termux", lambda: False)
        monkeypatch.setattr(gateway, "Path", lambda _path: SimpleNamespace(exists=lambda: False))
        monkeypatch.setattr(gateway, "get_systemd_linger_status", lambda username=None: (False, ""))
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/loginctl")
        monkeypatch.setattr("pwd.getpwnam", lambda name: SimpleNamespace(pw_uid=1001))

        run_calls = []

        def fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway.subprocess, "run", fake_run)
        waited = []
        monkeypatch.setattr(
            gateway, "_wait_for_user_dbus_socket", lambda timeout=3.0, uid=None: waited.append(uid) or True
        )

        assert gateway._ensure_linger_enabled("alice", system=True) is True

        assert run_calls == [["loginctl", "enable-linger", "alice"]]
        assert waited == [1001]
        out = capsys.readouterr().out
        assert "alice" in out and "logout" not in out
        assert f"sudo systemctl restart {gateway.get_service_name()}.service" not in out



class TestTargetUidSocketPaths:
    def test_explicit_uid_ignores_callers_runtime_env(self, monkeypatch):
        monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/0")
        assert gateway._user_dbus_socket_path(1001) == gateway.Path("/run/user/1001/bus")
        assert gateway._user_systemd_private_socket_path(1001) == gateway.Path("/run/user/1001/systemd/private")

    def test_wait_for_foreign_uid_does_not_adopt_env(self, monkeypatch):
        monkeypatch.setattr(gateway, "_user_systemd_socket_ready", lambda uid=None: True)
        adopted = []
        monkeypatch.setattr(gateway, "_ensure_user_systemd_env", lambda: adopted.append(True))

        assert gateway._wait_for_user_dbus_socket(timeout=0.1, uid=1001) is True
        assert adopted == []
        assert gateway._wait_for_user_dbus_socket(timeout=0.1) is True
        assert adopted == [True]


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


@pytest.mark.parametrize("user", ["alice", "root"])
def test_systemd_install_targets_linger_at_system_service_user(monkeypatch, tmp_path, user):
    """Fresh --system install enables linger for the unit's User= — root too, since restart-safe
    workers always cross systemd-run --user."""
    unit_path = tmp_path / "systemd" / "hermes-gateway.service"
    helper_calls = []

    monkeypatch.setattr(gateway, "_require_root_for_system_service", lambda action: None)
    monkeypatch.setattr(gateway, "has_legacy_hermes_units", lambda: False)
    monkeypatch.setattr(gateway, "get_systemd_unit_path", lambda system=False: unit_path)
    monkeypatch.setattr(
        gateway,
        "generate_systemd_unit",
        lambda system=False, run_as_user=None: f"[Service]\nUser={run_as_user}\n",
    )
    monkeypatch.setattr(gateway, "_run_systemctl", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gateway, "_ensure_linger_enabled",
        lambda username=None, system=False: helper_calls.append((username, system)) or False,
    )
    monkeypatch.setattr(gateway, "print_systemd_scope_conflict_warning", lambda: None)
    monkeypatch.setattr(gateway, "print_legacy_unit_warning", lambda: None)

    gateway.systemd_install(system=True, run_as_user=user)

    assert helper_calls == [(user, True)]


@pytest.mark.parametrize("unit_is_current", [True, False])
@pytest.mark.parametrize("running", [True, False])
def test_existing_system_install_repairs_linger_and_says_restart(
    monkeypatch, tmp_path, capsys, unit_is_current, running
):
    """Re-running install on an affected system service enables linger for User=; when the gateway is
    already running it must be told to restart (systemctl start on an active unit is a no-op)."""
    unit_path = tmp_path / "systemd" / "hermes-gateway.service"
    unit_path.parent.mkdir(parents=True)
    unit_path.write_text("[Service]\nUser=alice\n", encoding="utf-8")
    helper_calls = []

    monkeypatch.setattr(gateway, "_require_root_for_system_service", lambda action: None)
    monkeypatch.setattr(gateway, "has_legacy_hermes_units", lambda: False)
    monkeypatch.setattr(gateway, "get_systemd_unit_path", lambda system=False: unit_path)
    monkeypatch.setattr(gateway, "_sync_hermes_home_from_systemd_unit", lambda system=False: None)
    monkeypatch.setattr(gateway, "systemd_unit_is_current", lambda system=False: unit_is_current)
    monkeypatch.setattr(gateway, "refresh_systemd_unit_if_needed", lambda system=False: None)
    monkeypatch.setattr(gateway, "_run_systemctl", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway, "_systemd_unit_is_active", lambda system=False: running)
    monkeypatch.setattr(
        gateway, "_ensure_linger_enabled",
        lambda username=None, system=False: helper_calls.append((username, system)) or True,
    )

    gateway.systemd_install(system=True, run_as_user="alice")

    assert helper_calls == [("alice", True)]
    restart_hint = f"sudo systemctl restart {gateway.get_service_name()}.service"
    assert (restart_hint in capsys.readouterr().out) is running
