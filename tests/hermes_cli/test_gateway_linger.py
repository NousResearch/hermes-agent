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

    def test_system_scope_enables_target_user_without_logout_messaging(self, monkeypatch, capsys):
        monkeypatch.setattr(gateway, "is_linux", lambda: True)
        monkeypatch.setattr(gateway, "is_termux", lambda: False)
        monkeypatch.setattr(gateway, "Path", lambda _path: SimpleNamespace(exists=lambda: False))
        monkeypatch.setattr(gateway, "get_systemd_linger_status", lambda username=None: (False, ""))
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/loginctl")
        run_calls = []

        def fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway.subprocess, "run", fake_run)

        assert gateway._ensure_linger_enabled("alice", system=True) is True

        assert run_calls == [["loginctl", "enable-linger", "alice"]]
        out = capsys.readouterr().out
        assert "alice" in out and "logout" not in out

    def test_system_scope_warning_uses_system_restart(self, monkeypatch, capsys):
        monkeypatch.setattr(gateway, "is_linux", lambda: True)
        monkeypatch.setattr(gateway, "is_termux", lambda: False)
        monkeypatch.setattr(gateway, "Path", lambda _path: SimpleNamespace(exists=lambda: False))
        monkeypatch.setattr(gateway, "get_systemd_linger_status", lambda username=None: (False, ""))
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/loginctl")
        monkeypatch.setattr(
            gateway.subprocess, "run",
            lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="Permission denied"),
        )

        assert gateway._ensure_linger_enabled("alice", system=True) is False

        out = capsys.readouterr().out
        assert f"sudo systemctl restart {gateway.get_service_name()}.service" in out
        assert "systemctl --user" not in out


class TestEnsureSystemServiceLinger:
    @pytest.mark.parametrize("running", [True, False])
    def test_fresh_enable_waits_on_target_uid_and_hints_restart_only_when_running(
        self, monkeypatch, capsys, running
    ):
        """Root installing a system unit waits for the TARGET user's bus (never adopting its own env) and
        tells the operator to restart only when a bus-less gateway is already active."""
        monkeypatch.setattr(gateway, "_ensure_linger_enabled", lambda username, system=False: True)
        monkeypatch.setattr("pwd.getpwnam", lambda name: SimpleNamespace(pw_uid=1001))
        waited = []
        monkeypatch.setattr(gateway, "_wait_for_target_user_bus", lambda uid, timeout=5.0: waited.append(uid) or True)
        adopted = []
        monkeypatch.setattr(gateway, "_ensure_user_systemd_env", lambda: adopted.append(True))
        monkeypatch.setattr(gateway, "_systemd_unit_is_active", lambda system=False: running)

        gateway._ensure_system_service_linger("alice")

        assert waited == [1001] and adopted == []
        out = capsys.readouterr().out
        assert (f"sudo systemctl restart {gateway.get_service_name()}.service" in out) is running

    def test_already_enabled_skips_wait_and_activity_probe(self, monkeypatch):
        monkeypatch.setattr(gateway, "_ensure_linger_enabled", lambda username, system=False: False)
        monkeypatch.setattr(gateway, "_wait_for_target_user_bus", lambda uid, timeout=5.0: pytest.fail("waited"))
        monkeypatch.setattr(gateway, "_systemd_unit_is_active", lambda system=False: pytest.fail("probed"))

        gateway._ensure_system_service_linger("alice")


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
    monkeypatch.setattr(gateway, "_ensure_system_service_linger", lambda username: helper_calls.append(username))
    monkeypatch.setattr(gateway, "print_systemd_scope_conflict_warning", lambda: None)
    monkeypatch.setattr(gateway, "print_legacy_unit_warning", lambda: None)

    gateway.systemd_install(system=True, run_as_user=user)

    assert helper_calls == [user]


@pytest.mark.parametrize("unit_is_current", [True, False])
def test_existing_system_install_repairs_linger_for_configured_user(monkeypatch, tmp_path, unit_is_current):
    """Re-running install on an affected system service provisions linger for the unit's User=."""
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
    monkeypatch.setattr(gateway, "_ensure_system_service_linger", lambda username: helper_calls.append(username))

    gateway.systemd_install(system=True, run_as_user="alice")

    assert helper_calls == ["alice"]
