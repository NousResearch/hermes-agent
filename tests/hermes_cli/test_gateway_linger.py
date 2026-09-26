"""Tests for gateway linger auto-enable behavior on headless Linux installs."""
from gateway import service_identity
from gateway import systemd_identity
from gateway import systemd_legacy
from gateway import systemd_lifecycle
from gateway import systemd_linger
from gateway import systemd_runtime
from gateway import systemd_unit_render
from gateway import systemd_unit_state

from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_cli.gateway as gateway


def _stub_linger_file(monkeypatch, tmp_path, *, exists: bool) -> None:
    """Point the ``/var/lib/systemd/linger/<user>`` probe at a real file under tmp_path.

    ``gateway.Path`` stays ``pathlib.Path`` — every other ``Path(...)`` the code under test makes
    (``.expanduser()``, ``.resolve()`` in service-name resolution) keeps working. A ``SimpleNamespace``
    stand-in broke as soon as the code grew a call the stub did not implement (#119786).
    """
    linger_dir = Path("/var/lib/systemd/linger")
    fake_dir = tmp_path / "linger"
    fake_dir.mkdir()
    if exists:
        (fake_dir / "testuser").write_text("")

    def redirected_path(path, *more):
        candidate = Path(path, *more)
        return fake_dir / candidate.name if candidate.parent == linger_dir else candidate

    monkeypatch.setattr(systemd_linger, "Path", redirected_path)


def _root_with_system_unit_pinned_home(monkeypatch, tmp_path) -> None:
    """Take the root/system-unit branch of service-name resolution (``_bare_unit_pinned_home``), the
    path that calls ``Path(...).expanduser()`` while rendering the restart hint."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setattr(service_identity.sys, "platform", "linux")
    monkeypatch.setattr(service_identity.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(service_identity, "hermes_home_pinned_by_unit", lambda _unit_path: str(tmp_path / "system-home"))


class TestEnsureLingerEnabled:
    def test_linger_already_enabled_via_file(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setattr(systemd_runtime, "is_linux", lambda: True)
        monkeypatch.setattr(systemd_linger, "is_termux", lambda: False)
        monkeypatch.setattr("getpass.getuser", lambda: "testuser")
        _stub_linger_file(monkeypatch, tmp_path, exists=True)

        calls = []
        monkeypatch.setattr(gateway.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

        systemd_linger.ensure_linger_enabled()

        assert calls == []


    def test_loginctl_success_enables_linger(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setattr(systemd_runtime, "is_linux", lambda: True)
        monkeypatch.setattr(systemd_linger, "is_termux", lambda: False)
        monkeypatch.setattr("getpass.getuser", lambda: "testuser")
        _stub_linger_file(monkeypatch, tmp_path, exists=False)
        monkeypatch.setattr(systemd_runtime, "linger_status", lambda username=None: (False, ""))
        monkeypatch.setattr(systemd_linger.shutil, "which", lambda name: "/usr/bin/loginctl")

        run_calls = []

        def fake_run(cmd, capture_output=False, text=False, check=False, **kwargs):
            run_calls.append((cmd, capture_output, text, check))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(systemd_linger.subprocess, "run", fake_run)

        systemd_linger.ensure_linger_enabled()

        assert run_calls == [(["loginctl", "enable-linger", "testuser"], True, True, False)]


    def test_loginctl_failure_shows_manual_guidance(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setattr(systemd_runtime, "is_linux", lambda: True)
        monkeypatch.setattr(systemd_linger, "is_termux", lambda: False)
        monkeypatch.setattr("getpass.getuser", lambda: "testuser")
        _stub_linger_file(monkeypatch, tmp_path, exists=False)
        _root_with_system_unit_pinned_home(monkeypatch, tmp_path)
        monkeypatch.setattr(systemd_runtime, "linger_status", lambda username=None: (False, ""))
        monkeypatch.setattr(systemd_linger.shutil, "which", lambda name: "/usr/bin/loginctl")
        monkeypatch.setattr(
            systemd_linger.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="Permission denied"),
        )

        systemd_linger.ensure_linger_enabled()

        out = capsys.readouterr().out
        assert "sudo loginctl enable-linger testuser" in out
        assert "Permission denied" in out

    def test_system_scope_enables_target_user_without_logout_messaging(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setattr(systemd_runtime, "is_linux", lambda: True)
        monkeypatch.setattr(systemd_linger, "is_termux", lambda: False)
        _stub_linger_file(monkeypatch, tmp_path, exists=False)
        monkeypatch.setattr(systemd_runtime, "linger_status", lambda username=None: (False, ""))
        monkeypatch.setattr(systemd_linger.shutil, "which", lambda name: "/usr/bin/loginctl")
        run_calls = []

        def fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(systemd_linger.subprocess, "run", fake_run)

        assert systemd_linger.ensure_linger_enabled("alice", system=True) is True

        assert run_calls == [["loginctl", "enable-linger", "alice"]]
        out = capsys.readouterr().out
        assert "alice" in out and "logout" not in out

    def test_system_scope_warning_uses_system_restart(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setattr(systemd_runtime, "is_linux", lambda: True)
        monkeypatch.setattr(systemd_linger, "is_termux", lambda: False)
        _stub_linger_file(monkeypatch, tmp_path, exists=False)
        _root_with_system_unit_pinned_home(monkeypatch, tmp_path)
        monkeypatch.setattr(systemd_runtime, "linger_status", lambda username=None: (False, ""))
        monkeypatch.setattr(systemd_linger.shutil, "which", lambda name: "/usr/bin/loginctl")
        monkeypatch.setattr(
            systemd_linger.subprocess, "run",
            lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="Permission denied"),
        )

        assert systemd_linger.ensure_linger_enabled("alice", system=True) is False

        out = capsys.readouterr().out
        assert f"sudo systemctl restart {service_identity.service_name()}.service" in out
        assert "systemctl --user" not in out


class TestEnsureSystemServiceLinger:
    @pytest.mark.parametrize("running", [True, False])
    def test_fresh_enable_waits_on_target_uid_and_hints_restart_only_when_running(
        self, monkeypatch, capsys, running
    ):
        """Root installing a system unit waits for the TARGET user's bus (never adopting its own env) and
        tells the operator to restart only when a bus-less gateway is already active."""
        monkeypatch.setattr(systemd_linger, "ensure_linger_enabled", lambda username, system=False: True)
        monkeypatch.setattr(systemd_linger, "_lookup_uid", lambda name: 1001)
        waited = []
        monkeypatch.setattr(systemd_linger, "_target_bus_ready", lambda uid, timeout=5.0: waited.append(uid) or True)
        adopted = []
        monkeypatch.setattr(systemd_runtime, "ensure_user_env", lambda: adopted.append(True))
        monkeypatch.setattr(systemd_runtime, "unit_is_active", lambda system=False: running)

        systemd_linger.ensure_system_service_linger("alice")

        assert waited == [1001] and adopted == []
        out = capsys.readouterr().out
        assert (f"sudo systemctl restart {service_identity.service_name()}.service" in out) is running

    def test_already_enabled_skips_wait_and_activity_probe(self, monkeypatch):
        monkeypatch.setattr(systemd_linger, "ensure_linger_enabled", lambda username, system=False: False)
        monkeypatch.setattr(systemd_linger, "_target_bus_ready", lambda uid, timeout=5.0: pytest.fail("waited"))
        monkeypatch.setattr(systemd_runtime, "unit_is_active", lambda system=False: pytest.fail("probed"))

        systemd_linger.ensure_system_service_linger("alice")


def test_systemd_install_calls_linger_helper(monkeypatch, tmp_path, capsys):
    unit_path = tmp_path / "systemd" / "user" / "hermes-gateway.service"

    monkeypatch.setattr(systemd_identity, "unit_path", lambda system=False: unit_path)
    # Non-temp home so the temp-home write guard (which trips on the
    # hermetic test HERMES_HOME) stays out of the way.
    monkeypatch.setattr(
        systemd_unit_render,
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
    monkeypatch.setattr(
        systemd_runtime,
        "run_systemctl",
        lambda args, **kwargs: calls.append((args, kwargs.get("check", False)))
        or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(systemd_linger, "ensure_linger_enabled", lambda: helper_calls.append(True))

    systemd_lifecycle.install(force=False)

    assert unit_path.exists()
    assert [cmd for cmd, _ in calls] == [
        ["daemon-reload"],
        ["enable", service_identity.service_name()],
    ]
    assert helper_calls == [True]


@pytest.mark.parametrize("user", ["alice", "root"])
def test_systemd_install_targets_linger_at_system_service_user(monkeypatch, tmp_path, user):
    """Fresh --system install enables linger for the unit's User= — root too, since restart-safe
    workers always cross systemd-run --user."""
    unit_path = tmp_path / "systemd" / "hermes-gateway.service"
    helper_calls = []

    monkeypatch.setattr(systemd_identity, "require_root", lambda action: None)
    monkeypatch.setattr(systemd_legacy, "has_units", lambda: False)
    monkeypatch.setattr(systemd_identity, "unit_path", lambda system=False: unit_path)
    monkeypatch.setattr(
        systemd_unit_render,
        "generate_systemd_unit",
        lambda system=False, run_as_user=None: f"[Service]\nUser={run_as_user}\n",
    )
    monkeypatch.setattr(systemd_runtime, "run_systemctl", lambda *args, **kwargs: None)
    monkeypatch.setattr(systemd_linger, "ensure_system_service_linger", lambda username: helper_calls.append(username))
    monkeypatch.setattr(gateway, "print_systemd_scope_conflict_warning", lambda: None)
    monkeypatch.setattr(gateway, "print_legacy_unit_warning", lambda: None)

    systemd_lifecycle.install(system=True, run_as_user=user)

    assert helper_calls == [user]


@pytest.mark.parametrize("unit_is_current", [True, False])
def test_existing_system_install_repairs_linger_for_configured_user(monkeypatch, tmp_path, unit_is_current):
    """Re-running install on an affected system service provisions linger for the unit's User=."""
    unit_path = tmp_path / "systemd" / "hermes-gateway.service"
    unit_path.parent.mkdir(parents=True)
    unit_path.write_text("[Service]\nUser=alice\n", encoding="utf-8")
    helper_calls = []

    monkeypatch.setattr(systemd_identity, "require_root", lambda action: None)
    monkeypatch.setattr(systemd_legacy, "has_units", lambda: False)
    monkeypatch.setattr(systemd_identity, "unit_path", lambda system=False: unit_path)
    monkeypatch.setattr(systemd_runtime, "sync_home_from_unit", lambda system=False: None)
    monkeypatch.setattr(systemd_unit_state, "unit_is_current", lambda system=False: unit_is_current)
    monkeypatch.setattr(systemd_unit_state, "refresh_if_needed", lambda system=False: None)
    monkeypatch.setattr(systemd_runtime, "run_systemctl", lambda *args, **kwargs: None)
    monkeypatch.setattr(systemd_linger, "ensure_system_service_linger", lambda username: helper_calls.append(username))

    systemd_lifecycle.install(system=True, run_as_user="alice")

    assert helper_calls == ["alice"]


@pytest.mark.parametrize("system", [False, True])
def test_systemd_install_repair_path_keeps_linger_guarantee(monkeypatch, tmp_path, system):
    """Repairing a stale unit used to return before the linger step, so an upgraded headless
    user service died at logout (#12863). System scope never touches user linger."""
    unit_path = tmp_path / "hermes-gateway.service"
    unit_path.write_text("old unit\n", encoding="utf-8")
    monkeypatch.setattr(systemd_identity, "unit_path", lambda system=False: unit_path)
    monkeypatch.setattr(systemd_unit_state, "unit_is_current", lambda system=False: False)
    monkeypatch.setattr(systemd_legacy, "has_units", lambda: False)
    monkeypatch.setattr(systemd_identity, "require_root", lambda _action: None)
    monkeypatch.setattr(systemd_runtime, "sync_home_from_unit", lambda system=False: None)
    monkeypatch.setattr(systemd_identity, "read_unit_user", lambda path: None)
    monkeypatch.setattr(systemd_unit_state, "refresh_if_needed", lambda system=False: None)
    monkeypatch.setattr(systemd_runtime, "run_systemctl", lambda *args, **kwargs: None)
    helper_calls = []
    monkeypatch.setattr(systemd_linger, "ensure_linger_enabled", lambda: helper_calls.append(True))

    systemd_lifecycle.install(force=False, system=system)

    assert helper_calls == ([] if system else [True])
