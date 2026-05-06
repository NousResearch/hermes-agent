"""Tests for shared gateway service identity naming."""

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_cli.gateway as gateway_cli
from hermes_cli.gateway_service_identity import (
    gateway_launchd_plist_path,
    gateway_profile_arg,
    get_gateway_service_identity,
    is_custom_root_hermes_home,
    service_definition_matches_hermes_home,
)


def test_default_hermes_home_keeps_legacy_service_names(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)

    identity = get_gateway_service_identity()

    assert identity.suffix == ""
    assert identity.systemd_service_name == "hermes-gateway"
    assert identity.launchd_label == "ai.hermes.gateway"
    assert (
        gateway_launchd_plist_path(tmp_path)
        == tmp_path / "Library" / "LaunchAgents" / "ai.hermes.gateway.plist"
    )


def test_default_hermes_home_keeps_legacy_names_when_home_is_redirected(
    tmp_path, monkeypatch
):
    import pwd

    os_home = tmp_path / "os-home"
    hermes_home = os_home / ".hermes"
    redirected_home = hermes_home / "home"
    os_home.mkdir()
    redirected_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: redirected_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(
        pwd,
        "getpwuid",
        lambda uid: SimpleNamespace(pw_dir=str(os_home)),
    )

    identity = get_gateway_service_identity()

    assert identity.suffix == ""
    assert identity.systemd_service_name == "hermes-gateway"
    assert identity.launchd_label == "ai.hermes.gateway"
    assert is_custom_root_hermes_home() is False
    assert gateway_cli.get_service_name() == "hermes-gateway"


def test_custom_dot_hermes_root_with_redirected_home_uses_hash_suffix(
    tmp_path, monkeypatch
):
    import pwd

    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-root" / ".hermes"
    redirected_home = hermes_home / "home"
    os_home.mkdir()
    redirected_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: redirected_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(
        pwd,
        "getpwuid",
        lambda uid: SimpleNamespace(pw_dir=str(os_home)),
    )
    suffix = hashlib.sha256(str(hermes_home.resolve()).encode()).hexdigest()[:8]

    identity = get_gateway_service_identity()

    assert identity.suffix == suffix
    assert identity.systemd_service_name == f"hermes-gateway-{suffix}"
    assert identity.launchd_label == f"ai.hermes.gateway-{suffix}"
    assert is_custom_root_hermes_home() is True


def test_sudo_user_default_hermes_home_keeps_legacy_service_names(monkeypatch):
    import pwd

    monkeypatch.setattr(Path, "home", lambda: Path("/root"))
    monkeypatch.setenv("HERMES_HOME", "/home/alice/.hermes")
    monkeypatch.setenv("SUDO_USER", "alice")
    monkeypatch.setattr(
        pwd,
        "getpwnam",
        lambda name: SimpleNamespace(pw_dir="/home/alice"),
    )

    identity = get_gateway_service_identity()

    assert identity.suffix == ""
    assert identity.systemd_service_name == "hermes-gateway"
    assert identity.launchd_label == "ai.hermes.gateway"


def test_named_profile_uses_profile_suffix(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes" / "profiles" / "coder"
    hermes_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    identity = get_gateway_service_identity()

    assert identity.suffix == "coder"
    assert identity.systemd_service_name == "hermes-gateway-coder"
    assert identity.launchd_label == "ai.hermes.gateway-coder"
    assert gateway_profile_arg() == "--profile coder"


def test_custom_hermes_home_uses_hash_suffix(tmp_path, monkeypatch):
    hermes_home = tmp_path / "custom-hermes"
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "os-home")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    suffix = hashlib.sha256(str(hermes_home.resolve()).encode()).hexdigest()[:8]

    identity = get_gateway_service_identity()

    assert identity.suffix == suffix
    assert identity.systemd_service_name == f"hermes-gateway-{suffix}"
    assert identity.launchd_label == f"ai.hermes.gateway-{suffix}"
    assert gateway_profile_arg() == ""
    assert gateway_cli.get_service_name() == f"hermes-gateway-{suffix}"


def test_custom_root_profile_uses_profile_name(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-root" / "profiles" / "ops"
    hermes_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "os-home")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    identity = get_gateway_service_identity()

    assert identity.systemd_service_name == "hermes-gateway-ops"
    assert identity.launchd_label == "ai.hermes.gateway-ops"
    assert gateway_profile_arg() == "--profile ops"


def test_profile_arg_handles_target_user_profile_path(monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: Path("/root"))
    monkeypatch.setenv("HERMES_HOME", "/root/.hermes/profiles/coder")

    assert gateway_profile_arg("/home/alice/.hermes/profiles/coder") == "--profile coder"


def test_systemd_migrates_legacy_custom_root_unit(tmp_path, monkeypatch):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes"
    os_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    legacy_unit = os_home / ".config" / "systemd" / "user" / "hermes-gateway.service"
    legacy_unit.parent.mkdir(parents=True)
    legacy_unit.write_text(
        f'[Service]\nEnvironment="HERMES_HOME={hermes_home.resolve()}"\n',
        encoding="utf-8",
    )
    calls = []
    def run_systemctl(args, **kwargs):
        calls.append(args)
        stdout = "enabled\n" if args[:2] == ["is-enabled", "hermes-gateway"] else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(gateway_cli, "_run_systemctl", run_systemctl)
    monkeypatch.setattr(
        gateway_cli,
        "generate_systemd_unit",
        lambda system=False, run_as_user=None: "new scoped unit\n",
    )
    scoped_unit = gateway_cli.get_systemd_unit_path()

    assert gateway_cli._migrate_custom_root_systemd_service() is True

    assert not legacy_unit.exists()
    assert scoped_unit.read_text(encoding="utf-8") == "new scoped unit\n"
    assert calls == [
        ["is-enabled", "hermes-gateway"],
        ["daemon-reload"],
        ["is-enabled", gateway_cli.get_service_name()],
        ["enable", gateway_cli.get_service_name()],
        ["disable", "hermes-gateway"],
        ["daemon-reload"],
    ]


def test_systemd_migration_keeps_legacy_unit_when_enable_fails(
    tmp_path, monkeypatch
):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes"
    os_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    legacy_unit = os_home / ".config" / "systemd" / "user" / "hermes-gateway.service"
    legacy_unit.parent.mkdir(parents=True)
    legacy_unit.write_text(
        f'[Service]\nEnvironment="HERMES_HOME={hermes_home.resolve()}"\n',
        encoding="utf-8",
    )
    calls = []

    def run_systemctl(args, **kwargs):
        calls.append(args)
        if args[:2] == ["is-enabled", "hermes-gateway"]:
            return SimpleNamespace(returncode=0, stdout="enabled\n", stderr="")
        if args[:1] == ["enable"]:
            raise gateway_cli.subprocess.CalledProcessError(1, args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gateway_cli, "_run_systemctl", run_systemctl)
    monkeypatch.setattr(
        gateway_cli,
        "generate_systemd_unit",
        lambda system=False, run_as_user=None: "new scoped unit\n",
    )

    with pytest.raises(gateway_cli.subprocess.CalledProcessError):
        gateway_cli._migrate_custom_root_systemd_service()

    assert legacy_unit.exists()
    assert not gateway_cli.get_systemd_unit_path().exists()
    assert ["stop", "hermes-gateway"] not in calls
    assert ["disable", "hermes-gateway"] not in calls


def test_systemd_migrates_legacy_named_profile_unit(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes" / "profiles" / "coder"
    hermes_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    legacy_unit = (
        tmp_path
        / ".config"
        / "systemd"
        / "user"
        / "hermes-gateway.service"
    )
    legacy_unit.parent.mkdir(parents=True)
    legacy_unit.write_text(
        f'[Service]\nEnvironment="HERMES_HOME={hermes_home.resolve()}"\n',
        encoding="utf-8",
    )
    calls = []

    def run_systemctl(args, **kwargs):
        calls.append(args)
        stdout = "enabled\n" if args[:2] == ["is-enabled", "hermes-gateway"] else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(gateway_cli, "_run_systemctl", run_systemctl)
    monkeypatch.setattr(
        gateway_cli,
        "generate_systemd_unit",
        lambda system=False, run_as_user=None: "profile scoped unit\n",
    )

    assert gateway_cli.get_service_name() == "hermes-gateway-coder"
    assert gateway_cli._migrate_custom_root_systemd_service() is True

    scoped_unit = gateway_cli.get_systemd_unit_path()
    assert not legacy_unit.exists()
    assert scoped_unit.read_text(encoding="utf-8") == "profile scoped unit\n"
    assert ["enable", "hermes-gateway-coder"] in calls


def test_systemd_install_starts_scoped_service_when_legacy_unit_was_active(
    tmp_path, monkeypatch
):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes"
    os_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    legacy_unit = os_home / ".config" / "systemd" / "user" / "hermes-gateway.service"
    legacy_unit.parent.mkdir(parents=True)
    legacy_unit.write_text(
        f'[Service]\nEnvironment="HERMES_HOME={hermes_home.resolve()}"\n',
        encoding="utf-8",
    )
    calls = []

    def run_systemctl(args, **kwargs):
        calls.append(args)
        stdout = ""
        if args[:2] == ["is-enabled", "hermes-gateway"]:
            stdout = "enabled\n"
        elif args[:2] == ["is-active", "hermes-gateway"]:
            stdout = "active\n"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(gateway_cli, "_run_systemctl", run_systemctl)
    monkeypatch.setattr(gateway_cli, "has_legacy_hermes_units", lambda: False)
    monkeypatch.setattr(
        gateway_cli,
        "generate_systemd_unit",
        lambda system=False, run_as_user=None: "new scoped unit\n",
    )

    gateway_cli.systemd_install()

    assert ["start", gateway_cli.get_service_name()] in calls
    assert calls.index(["start", gateway_cli.get_service_name()]) > calls.index(
        ["enable", gateway_cli.get_service_name()]
    )
    assert not legacy_unit.exists()


def test_systemd_migration_restarts_legacy_unit_when_scoped_start_fails(
    tmp_path, monkeypatch
):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes"
    os_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    legacy_unit = os_home / ".config" / "systemd" / "user" / "hermes-gateway.service"
    legacy_unit.parent.mkdir(parents=True)
    legacy_unit.write_text(
        f'[Service]\nEnvironment="HERMES_HOME={hermes_home.resolve()}"\n',
        encoding="utf-8",
    )
    calls = []

    def run_systemctl(args, **kwargs):
        calls.append(args)
        if args[:2] == ["is-enabled", "hermes-gateway"]:
            return SimpleNamespace(returncode=0, stdout="enabled\n", stderr="")
        if args[:2] == ["is-active", "hermes-gateway"]:
            return SimpleNamespace(returncode=0, stdout="active\n", stderr="")
        if args == ["start", gateway_cli.get_service_name()]:
            raise gateway_cli.subprocess.CalledProcessError(1, args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gateway_cli, "_run_systemctl", run_systemctl)
    monkeypatch.setattr(
        gateway_cli,
        "generate_systemd_unit",
        lambda system=False, run_as_user=None: "new scoped unit\n",
    )

    with pytest.raises(gateway_cli.subprocess.CalledProcessError):
        gateway_cli._migrate_custom_root_systemd_service(start_new_if_active=True)

    assert legacy_unit.exists()
    assert not gateway_cli.get_systemd_unit_path().exists()
    assert ["start", "hermes-gateway"] in calls
    assert ["disable", gateway_cli.get_service_name()] in calls
    assert ["disable", "hermes-gateway"] not in calls


def test_systemd_migration_disables_scoped_unit_when_legacy_stop_fails(
    tmp_path, monkeypatch
):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes"
    os_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    legacy_unit = os_home / ".config" / "systemd" / "user" / "hermes-gateway.service"
    legacy_unit.parent.mkdir(parents=True)
    legacy_unit.write_text(
        f'[Service]\nEnvironment="HERMES_HOME={hermes_home.resolve()}"\n',
        encoding="utf-8",
    )
    calls = []

    def run_systemctl(args, **kwargs):
        calls.append(args)
        if args[:2] == ["is-enabled", "hermes-gateway"]:
            return SimpleNamespace(returncode=0, stdout="enabled\n", stderr="")
        if args[:2] == ["is-active", "hermes-gateway"]:
            return SimpleNamespace(returncode=0, stdout="active\n", stderr="")
        if args == ["stop", "hermes-gateway"]:
            raise gateway_cli.subprocess.CalledProcessError(1, args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gateway_cli, "_run_systemctl", run_systemctl)
    monkeypatch.setattr(
        gateway_cli,
        "generate_systemd_unit",
        lambda system=False, run_as_user=None: "new scoped unit\n",
    )

    with pytest.raises(gateway_cli.subprocess.CalledProcessError):
        gateway_cli._migrate_custom_root_systemd_service(start_new_if_active=True)

    assert legacy_unit.exists()
    assert not gateway_cli.get_systemd_unit_path().exists()
    assert ["disable", gateway_cli.get_service_name()] in calls
    assert ["disable", "hermes-gateway"] not in calls


def test_systemd_migration_refreshes_stale_scoped_unit_before_start(
    tmp_path, monkeypatch
):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes"
    os_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    legacy_unit = os_home / ".config" / "systemd" / "user" / "hermes-gateway.service"
    legacy_unit.parent.mkdir(parents=True)
    legacy_unit.write_text(
        f'[Service]\nEnvironment="HERMES_HOME={hermes_home.resolve()}"\n',
        encoding="utf-8",
    )
    scoped_unit = gateway_cli.get_systemd_unit_path()
    scoped_unit.write_text("stale scoped unit\n", encoding="utf-8")
    calls = []

    def run_systemctl(args, **kwargs):
        calls.append(args)
        if args == ["start", gateway_cli.get_service_name()]:
            assert scoped_unit.read_text(encoding="utf-8") == "fresh scoped unit\n"
        stdout = ""
        if args[:2] == ["is-enabled", "hermes-gateway"]:
            stdout = "enabled\n"
        elif args[:2] == ["is-active", "hermes-gateway"]:
            stdout = "active\n"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(gateway_cli, "_run_systemctl", run_systemctl)
    monkeypatch.setattr(
        gateway_cli,
        "generate_systemd_unit",
        lambda system=False, run_as_user=None: "fresh scoped unit\n",
    )

    assert (
        gateway_cli._migrate_custom_root_systemd_service(start_new_if_active=True)
        is True
    )

    assert scoped_unit.read_text(encoding="utf-8") == "fresh scoped unit\n"
    assert calls.index(["daemon-reload"]) < calls.index(
        ["start", gateway_cli.get_service_name()]
    )
    assert not legacy_unit.exists()


def test_systemd_migration_preserves_disabled_legacy_unit(tmp_path, monkeypatch):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes"
    os_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    legacy_unit = os_home / ".config" / "systemd" / "user" / "hermes-gateway.service"
    legacy_unit.parent.mkdir(parents=True)
    legacy_unit.write_text(
        f'[Service]\nEnvironment="HERMES_HOME={hermes_home.resolve()}"\n',
        encoding="utf-8",
    )
    calls = []

    def run_systemctl(args, **kwargs):
        calls.append(args)
        stdout = "disabled\n" if args[:2] == ["is-enabled", "hermes-gateway"] else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(gateway_cli, "_run_systemctl", run_systemctl)
    monkeypatch.setattr(
        gateway_cli,
        "generate_systemd_unit",
        lambda system=False, run_as_user=None: "new scoped unit\n",
    )

    assert gateway_cli._migrate_custom_root_systemd_service() is True

    assert ["enable", gateway_cli.get_service_name()] not in calls
    assert calls == [
        ["is-enabled", "hermes-gateway"],
        ["daemon-reload"],
        ["disable", "hermes-gateway"],
        ["daemon-reload"],
    ]


def test_systemd_migration_can_skip_enable_preservation(tmp_path, monkeypatch):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes"
    os_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    legacy_unit = os_home / ".config" / "systemd" / "user" / "hermes-gateway.service"
    legacy_unit.parent.mkdir(parents=True)
    legacy_unit.write_text(
        f'[Service]\nEnvironment="HERMES_HOME={hermes_home.resolve()}"\n',
        encoding="utf-8",
    )
    calls = []

    def run_systemctl(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="enabled\n", stderr="")

    monkeypatch.setattr(gateway_cli, "_run_systemctl", run_systemctl)
    monkeypatch.setattr(
        gateway_cli,
        "generate_systemd_unit",
        lambda system=False, run_as_user=None: "new scoped unit\n",
    )

    assert (
        gateway_cli._migrate_custom_root_systemd_service(preserve_enabled=False)
        is True
    )

    assert ["is-enabled", "hermes-gateway"] not in calls
    assert ["enable", gateway_cli.get_service_name()] not in calls


def test_systemd_stop_stops_scoped_and_legacy_units_when_both_exist(
    tmp_path, monkeypatch
):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes"
    os_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    scoped_unit = gateway_cli.get_systemd_unit_path()
    legacy_unit = os_home / ".config" / "systemd" / "user" / "hermes-gateway.service"
    scoped_unit.parent.mkdir(parents=True)
    scoped_unit.write_text("scoped\n", encoding="utf-8")
    legacy_unit.write_text(
        f'[Service]\nEnvironment="HERMES_HOME={hermes_home.resolve()}"\n',
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        gateway_cli,
        "_run_systemctl",
        lambda args, **kwargs: calls.append(args)
        or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    gateway_cli.systemd_stop()

    assert ["stop", gateway_cli.get_service_name()] in calls
    assert ["stop", "hermes-gateway"] in calls


def test_systemd_service_definition_detects_migratable_legacy_unit(tmp_path, monkeypatch):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes"
    os_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    legacy_unit = os_home / ".config" / "systemd" / "user" / "hermes-gateway.service"
    legacy_unit.parent.mkdir(parents=True)
    legacy_unit.write_text(
        f'[Service]\nEnvironment="HERMES_HOME={hermes_home.resolve()}"\n',
        encoding="utf-8",
    )

    assert gateway_cli._has_systemd_service_definition() is True
    assert gateway_cli.get_systemd_unit_path().exists() is False


def test_systemd_status_reports_legacy_without_migrating(tmp_path, monkeypatch):
    calls = []

    monkeypatch.setattr(gateway_cli, "_select_systemd_scope", lambda system=False: False)
    monkeypatch.setattr(
        gateway_cli,
        "get_systemd_unit_path",
        lambda system=False: tmp_path / "scoped.service",
    )
    monkeypatch.setattr(
        gateway_cli,
        "_has_migratable_custom_root_systemd_service",
        lambda system=False: True,
    )
    monkeypatch.setattr(
        gateway_cli,
        "_migrate_custom_root_systemd_service",
        lambda system=False: (_ for _ in ()).throw(AssertionError("status must not migrate")),
    )
    monkeypatch.setattr(
        gateway_cli,
        "_run_systemctl",
        lambda args, **kwargs: calls.append(args)
        or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    gateway_cli.systemd_status()

    assert calls == [["status", "hermes-gateway", "--no-pager"]]


def test_systemd_stop_uses_legacy_without_migrating(tmp_path, monkeypatch):
    calls = []
    legacy_unit = tmp_path / "legacy.service"
    legacy_unit.write_text("legacy\n", encoding="utf-8")

    monkeypatch.setattr(
        gateway_cli,
        "_select_systemd_scope",
        lambda system=False: False,
    )
    monkeypatch.setattr(
        gateway_cli,
        "get_systemd_unit_path",
        lambda system=False: tmp_path / "scoped.service",
    )
    monkeypatch.setattr(
        gateway_cli,
        "_systemd_service_targets",
        lambda system=False: [("hermes-gateway", legacy_unit)],
    )
    monkeypatch.setattr(
        gateway_cli,
        "_migrate_custom_root_systemd_service",
        lambda system=False: (_ for _ in ()).throw(
            AssertionError("stop must not migrate")
        ),
    )
    monkeypatch.setattr(
        gateway_cli,
        "_run_systemctl",
        lambda args, **kwargs: calls.append(args)
        or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    gateway_cli.systemd_stop()

    assert calls == [["stop", "hermes-gateway"]]


def test_systemd_uninstall_removes_legacy_without_migrating(tmp_path, monkeypatch):
    legacy_unit = tmp_path / "hermes-gateway.service"
    legacy_unit.write_text("legacy unit\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(gateway_cli, "_select_systemd_scope", lambda system=False: False)
    monkeypatch.setattr(
        gateway_cli,
        "_migrate_custom_root_systemd_service",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("uninstall must not migrate before removing")
        ),
    )
    monkeypatch.setattr(
        gateway_cli,
        "_systemd_service_targets",
        lambda system=False: [("hermes-gateway", legacy_unit)],
    )
    monkeypatch.setattr(
        gateway_cli,
        "_run_systemctl",
        lambda args, **kwargs: calls.append(args)
        or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    gateway_cli.systemd_uninstall()

    assert not legacy_unit.exists()
    assert calls == [
        ["stop", "hermes-gateway"],
        ["disable", "hermes-gateway"],
        ["daemon-reload"],
    ]


def test_systemd_migration_requires_exact_hermes_home_match(tmp_path, monkeypatch):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes"
    os_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    legacy_unit = os_home / ".config" / "systemd" / "user" / "hermes-gateway.service"
    legacy_unit.parent.mkdir(parents=True)
    legacy_unit.write_text(
        f'[Service]\nEnvironment="HERMES_HOME={hermes_home.resolve()}2"\n',
        encoding="utf-8",
    )

    assert gateway_cli._service_definition_matches_current_hermes_home(legacy_unit) is False
    assert gateway_cli._migrate_custom_root_systemd_service() is False
    assert legacy_unit.exists()


def test_systemd_detects_legacy_standalone_unit_with_matching_project_env(
    tmp_path, monkeypatch
):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes"
    project_dir = tmp_path / "project"
    script_path = project_dir / "scripts" / "hermes-gateway"
    os_home.mkdir()
    hermes_home.mkdir()
    script_path.parent.mkdir(parents=True)
    (project_dir / ".env").write_text(
        f"HERMES_HOME={hermes_home.resolve()}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    legacy_unit = os_home / ".config" / "systemd" / "user" / "hermes-gateway.service"
    legacy_unit.parent.mkdir(parents=True)
    legacy_unit.write_text(
        f"[Service]\nExecStart=/usr/bin/python {script_path} run\n",
        encoding="utf-8",
    )

    assert gateway_cli._has_migratable_custom_root_systemd_service() is True


def test_systemd_ignores_legacy_standalone_unit_without_matching_project_env(
    tmp_path, monkeypatch
):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes"
    project_dir = tmp_path / "project"
    script_path = project_dir / "scripts" / "hermes-gateway"
    os_home.mkdir()
    hermes_home.mkdir()
    script_path.parent.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    legacy_unit = os_home / ".config" / "systemd" / "user" / "hermes-gateway.service"
    legacy_unit.parent.mkdir(parents=True)
    legacy_unit.write_text(
        f"[Service]\nExecStart=/usr/bin/python {script_path} run\n",
        encoding="utf-8",
    )

    assert gateway_cli._has_migratable_custom_root_systemd_service() is False


def test_explicit_nonmatching_hermes_home_overrides_standalone_script_fallback(
    tmp_path, monkeypatch
):
    current_home = tmp_path / "current-hermes"
    other_home = tmp_path / "other-hermes"
    project_dir = tmp_path / "project"
    script_path = project_dir / "scripts" / "hermes-gateway"
    current_home.mkdir()
    other_home.mkdir()
    script_path.parent.mkdir(parents=True)
    (project_dir / ".env").write_text(
        f"HERMES_HOME={current_home.resolve()}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(current_home))
    unit = tmp_path / "hermes-gateway.service"
    unit.write_text(
        f'[Service]\nEnvironment="HERMES_HOME={other_home.resolve()}"\n'
        f"ExecStart=/usr/bin/python {script_path} run\n",
        encoding="utf-8",
    )

    assert service_definition_matches_hermes_home(unit) is False


def test_explicit_nonmatching_launchd_home_overrides_script_fallback(
    tmp_path, monkeypatch
):
    current_home = tmp_path / "current-hermes"
    other_home = tmp_path / "other-hermes"
    project_dir = tmp_path / "project"
    script_path = project_dir / "scripts" / "hermes-gateway"
    current_home.mkdir()
    other_home.mkdir()
    script_path.parent.mkdir(parents=True)
    (project_dir / ".env").write_text(
        f"HERMES_HOME={current_home.resolve()}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(current_home))
    plist = tmp_path / "ai.hermes.gateway.plist"
    plist.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<plist><dict>"
        "<key>ProgramArguments</key><array>"
        "<string>/usr/bin/python</string>"
        f"<string>{script_path}</string>"
        "<string>run</string>"
        "</array>"
        "<key>EnvironmentVariables</key><dict>"
        "<key>HERMES_HOME</key>"
        f"<string>{other_home.resolve()}</string>"
        "</dict></dict></plist>\n",
        encoding="utf-8",
    )

    assert service_definition_matches_hermes_home(plist) is False


def test_launchd_migration_requires_exact_hermes_home_match(tmp_path, monkeypatch):
    machine_home = tmp_path / "machine-home"
    hermes_home = tmp_path / "custom-hermes"
    machine_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: machine_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(gateway_cli, "_launchd_user_home", lambda: machine_home)
    legacy_plist = (
        machine_home
        / "Library"
        / "LaunchAgents"
        / "ai.hermes.gateway.plist"
    )
    legacy_plist.parent.mkdir(parents=True)
    legacy_plist.write_text(
        "<dict>\n"
        "<key>HERMES_HOME</key>\n"
        f"<string>{hermes_home.resolve()}2</string>\n"
        "</dict>\n",
        encoding="utf-8",
    )

    assert gateway_cli._service_definition_matches_current_hermes_home(legacy_plist) is False
    assert gateway_cli._migrate_custom_root_launchd_service() is False
    assert legacy_plist.exists()


def test_launchd_detects_legacy_standalone_plist_with_matching_project_env(
    tmp_path, monkeypatch
):
    machine_home = tmp_path / "machine-home"
    hermes_home = tmp_path / "custom-hermes"
    project_dir = tmp_path / "project"
    script_path = project_dir / "scripts" / "hermes-gateway"
    machine_home.mkdir()
    hermes_home.mkdir()
    script_path.parent.mkdir(parents=True)
    (project_dir / ".env").write_text(
        f"HERMES_HOME={hermes_home.resolve()}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: machine_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(gateway_cli, "_launchd_user_home", lambda: machine_home)
    legacy_plist = (
        machine_home
        / "Library"
        / "LaunchAgents"
        / "ai.hermes.gateway.plist"
    )
    legacy_plist.parent.mkdir(parents=True)
    legacy_plist.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0"><dict>'
        '<key>ProgramArguments</key><array>'
        '<string>/usr/bin/python</string>'
        f'<string>{script_path}</string>'
        '<string>run</string>'
        '</array>'
        '</dict></plist>\n',
        encoding="utf-8",
    )

    assert gateway_cli._has_migratable_custom_root_launchd_service() is True


def test_launchd_migrates_legacy_custom_root_plist(tmp_path, monkeypatch):
    machine_home = tmp_path / "machine-home"
    hermes_home = tmp_path / "custom-hermes"
    machine_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: machine_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(gateway_cli, "_launchd_user_home", lambda: machine_home)
    monkeypatch.setattr(gateway_cli, "generate_launchd_plist", lambda: "scoped plist\n")
    legacy_plist = (
        machine_home
        / "Library"
        / "LaunchAgents"
        / "ai.hermes.gateway.plist"
    )
    legacy_plist.parent.mkdir(parents=True)
    legacy_plist.write_text(
        "<dict>\n"
        "<key>HERMES_HOME</key>\n"
        f"<string>{hermes_home.resolve()}</string>\n"
        "</dict>\n",
        encoding="utf-8",
    )
    calls = []
    waits = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd == ["launchctl", "list", "ai.hermes.gateway"]:
            return SimpleNamespace(
                returncode=0,
                stdout="PID\tStatus\tLabel\n4321\t0\tai.hermes.gateway\n",
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        gateway_cli.subprocess,
        "run",
        run,
    )
    monkeypatch.setattr(
        gateway_cli,
        "_wait_for_pid_exit",
        lambda pid, **kwargs: waits.append((pid, kwargs)) or True,
    )
    monkeypatch.setattr(
        gateway_cli,
        "_wait_for_gateway_exit",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("must not wait on shared gateway PID")
        ),
    )

    assert gateway_cli._migrate_custom_root_launchd_service() is True

    scoped_plist = gateway_cli.get_launchd_plist_path()
    assert scoped_plist.exists()
    assert scoped_plist.read_text(encoding="utf-8") == "scoped plist\n"
    assert not legacy_plist.exists()
    assert calls == [
        ["launchctl", "list", "ai.hermes.gateway"],
        ["launchctl", "bootout", f"{gateway_cli._launchd_domain()}/ai.hermes.gateway"],
        ["launchctl", "bootstrap", gateway_cli._launchd_domain(), str(scoped_plist)],
    ]
    assert waits == [
        (
            4321,
            {
                "timeout": 10.0,
                "description": "legacy launchd service ai.hermes.gateway",
            },
        )
    ]


def test_launchd_migrates_legacy_named_profile_plist(tmp_path, monkeypatch):
    machine_home = tmp_path / "machine-home"
    hermes_home = tmp_path / ".hermes" / "profiles" / "coder"
    machine_home.mkdir()
    hermes_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(gateway_cli, "_launchd_user_home", lambda: machine_home)
    monkeypatch.setattr(gateway_cli, "generate_launchd_plist", lambda: "profile plist\n")
    legacy_plist = (
        machine_home
        / "Library"
        / "LaunchAgents"
        / "ai.hermes.gateway.plist"
    )
    legacy_plist.parent.mkdir(parents=True)
    legacy_plist.write_text(
        "<dict>\n"
        "<key>HERMES_HOME</key>\n"
        f"<string>{hermes_home.resolve()}</string>\n"
        "</dict>\n",
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        gateway_cli.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append(cmd)
        or SimpleNamespace(returncode=113, stdout="", stderr=""),
    )

    assert gateway_cli.get_launchd_label() == "ai.hermes.gateway-coder"
    assert gateway_cli._migrate_custom_root_launchd_service() is True

    scoped_plist = gateway_cli.get_launchd_plist_path()
    assert scoped_plist.exists()
    assert scoped_plist.read_text(encoding="utf-8") == "profile plist\n"
    assert not legacy_plist.exists()
    assert calls == [
        ["launchctl", "list", "ai.hermes.gateway"],
        ["launchctl", "bootout", f"{gateway_cli._launchd_domain()}/ai.hermes.gateway"]
    ]


def test_launchd_migration_refreshes_stale_scoped_plist(tmp_path, monkeypatch):
    machine_home = tmp_path / "machine-home"
    hermes_home = tmp_path / "custom-hermes"
    machine_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: machine_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(gateway_cli, "_launchd_user_home", lambda: machine_home)
    monkeypatch.setattr(gateway_cli, "generate_launchd_plist", lambda: "fresh plist\n")
    legacy_plist = (
        machine_home
        / "Library"
        / "LaunchAgents"
        / "ai.hermes.gateway.plist"
    )
    legacy_plist.parent.mkdir(parents=True)
    legacy_plist.write_text(
        "<dict>\n"
        "<key>HERMES_HOME</key>\n"
        f"<string>{hermes_home.resolve()}</string>\n"
        "</dict>\n",
        encoding="utf-8",
    )
    scoped_plist = gateway_cli.get_launchd_plist_path()
    scoped_plist.write_text("stale plist\n", encoding="utf-8")
    calls = []
    waits = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd == ["launchctl", "list", "ai.hermes.gateway"]:
            return SimpleNamespace(
                returncode=0,
                stdout="PID\tStatus\tLabel\n4321\t0\tai.hermes.gateway\n",
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gateway_cli.subprocess, "run", run)
    monkeypatch.setattr(
        gateway_cli,
        "_wait_for_pid_exit",
        lambda pid, **kwargs: waits.append((pid, kwargs)) or True,
    )
    monkeypatch.setattr(
        gateway_cli,
        "_wait_for_gateway_exit",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("must not wait on shared gateway PID")
        ),
    )

    assert gateway_cli._migrate_custom_root_launchd_service() is True

    assert scoped_plist.read_text(encoding="utf-8") == "fresh plist\n"
    assert not legacy_plist.exists()
    scoped_label = gateway_cli.get_launchd_label()
    assert calls == [
        ["launchctl", "list", scoped_label],
        ["launchctl", "list", "ai.hermes.gateway"],
        ["launchctl", "bootout", f"{gateway_cli._launchd_domain()}/ai.hermes.gateway"],
        ["launchctl", "bootout", f"{gateway_cli._launchd_domain()}/{scoped_label}"],
        ["launchctl", "bootstrap", gateway_cli._launchd_domain(), str(scoped_plist)],
    ]
    assert waits == [
        (
            4321,
            {
                "timeout": 10.0,
                "description": "legacy launchd service ai.hermes.gateway",
            },
        )
    ]


def test_launchd_migration_keeps_legacy_plist_when_bootout_fails(
    tmp_path, monkeypatch
):
    machine_home = tmp_path / "machine-home"
    hermes_home = tmp_path / "custom-hermes"
    machine_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: machine_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(gateway_cli, "_launchd_user_home", lambda: machine_home)
    legacy_plist = (
        machine_home
        / "Library"
        / "LaunchAgents"
        / "ai.hermes.gateway.plist"
    )
    legacy_plist.parent.mkdir(parents=True)
    legacy_plist.write_text(
        "<dict>\n"
        "<key>HERMES_HOME</key>\n"
        f"<string>{hermes_home.resolve()}</string>\n"
        "</dict>\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        gateway_cli.subprocess,
        "run",
        lambda cmd, **kwargs: SimpleNamespace(
            args=cmd,
            returncode=5,
            stdout="",
            stderr="bootout failed",
        ),
    )
    monkeypatch.setattr(
        gateway_cli,
        "_wait_for_pid_exit",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("must not wait after failed bootout")
        ),
    )

    with pytest.raises(gateway_cli.subprocess.CalledProcessError):
        gateway_cli._migrate_custom_root_launchd_service()

    assert legacy_plist.exists()
    assert not gateway_cli.get_launchd_plist_path().exists()


def test_launchd_migration_keeps_legacy_plist_when_gateway_does_not_exit(
    tmp_path, monkeypatch
):
    machine_home = tmp_path / "machine-home"
    hermes_home = tmp_path / "custom-hermes"
    machine_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: machine_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(gateway_cli, "_launchd_user_home", lambda: machine_home)
    legacy_plist = (
        machine_home
        / "Library"
        / "LaunchAgents"
        / "ai.hermes.gateway.plist"
    )
    legacy_plist.parent.mkdir(parents=True)
    legacy_plist.write_text(
        "<dict>\n"
        "<key>HERMES_HOME</key>\n"
        f"<string>{hermes_home.resolve()}</string>\n"
        "</dict>\n",
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        gateway_cli.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append(cmd)
        or SimpleNamespace(
            args=cmd,
            returncode=0,
            stdout="",
            stderr="",
        ),
    )
    waits = []
    monkeypatch.setattr(gateway_cli, "_launchd_label_pid", lambda label: 4321)
    monkeypatch.setattr(
        gateway_cli,
        "_wait_for_pid_exit",
        lambda pid, **kwargs: waits.append((pid, kwargs)) or False,
    )
    monkeypatch.setattr(
        gateway_cli,
        "_wait_for_gateway_exit",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("must not wait on shared gateway PID")
        ),
    )

    with pytest.raises(RuntimeError):
        gateway_cli._migrate_custom_root_launchd_service()

    assert legacy_plist.exists()
    assert not gateway_cli.get_launchd_plist_path().exists()
    assert ["launchctl", "bootstrap", gateway_cli._launchd_domain(), str(legacy_plist)] in calls
    assert [
        "launchctl",
        "kickstart",
        f"{gateway_cli._launchd_domain()}/ai.hermes.gateway",
    ] in calls
    assert waits == [
        (
            4321,
            {
                "timeout": 10.0,
                "description": "legacy launchd service ai.hermes.gateway",
            },
        )
    ]


def test_launchd_migration_removes_scoped_plist_when_bootstrap_fails(
    tmp_path, monkeypatch
):
    machine_home = tmp_path / "machine-home"
    hermes_home = tmp_path / "custom-hermes"
    machine_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: machine_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(gateway_cli, "_launchd_user_home", lambda: machine_home)
    monkeypatch.setattr(gateway_cli, "generate_launchd_plist", lambda: "scoped plist\n")
    legacy_plist = (
        machine_home
        / "Library"
        / "LaunchAgents"
        / "ai.hermes.gateway.plist"
    )
    legacy_plist.parent.mkdir(parents=True)
    legacy_plist.write_text(
        "<dict>\n"
        "<key>HERMES_HOME</key>\n"
        f"<string>{hermes_home.resolve()}</string>\n"
        "</dict>\n",
        encoding="utf-8",
    )
    scoped_plist = gateway_cli.get_launchd_plist_path()
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd == ["launchctl", "list", "ai.hermes.gateway"]:
            return SimpleNamespace(
                returncode=0,
                stdout="PID\tStatus\tLabel\n4321\t0\tai.hermes.gateway\n",
                stderr="",
            )
        if cmd == [
            "launchctl",
            "bootstrap",
            gateway_cli._launchd_domain(),
            str(scoped_plist),
        ]:
            raise gateway_cli.subprocess.CalledProcessError(5, cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gateway_cli.subprocess, "run", run)
    monkeypatch.setattr(gateway_cli, "_wait_for_pid_exit", lambda pid, **kwargs: True)
    monkeypatch.setattr(
        gateway_cli,
        "_wait_for_gateway_exit",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("must not wait on shared gateway PID")
        ),
    )

    with pytest.raises(gateway_cli.subprocess.CalledProcessError):
        gateway_cli._migrate_custom_root_launchd_service()

    assert legacy_plist.exists()
    assert not scoped_plist.exists()
    assert ["launchctl", "bootstrap", gateway_cli._launchd_domain(), str(legacy_plist)] in calls


def test_systemd_runtime_snapshot_probes_migratable_legacy_unit(tmp_path, monkeypatch):
    machine_home = tmp_path / "machine-home"
    hermes_home = tmp_path / "custom-hermes"
    machine_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: machine_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    legacy_unit = (
        machine_home
        / ".config"
        / "systemd"
        / "user"
        / "hermes-gateway.service"
    )
    legacy_unit.parent.mkdir(parents=True)
    legacy_unit.write_text(
        f'[Service]\nEnvironment="HERMES_HOME={hermes_home.resolve()}"\n',
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(gateway_cli, "find_gateway_pids", lambda: [])
    monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)
    monkeypatch.setattr(gateway_cli, "is_linux", lambda: False)
    monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: True)

    def fake_run_systemctl(args, *, system=False, **kwargs):
        calls.append((args, system))
        return SimpleNamespace(returncode=0, stdout="active\n", stderr="")

    monkeypatch.setattr(gateway_cli, "_run_systemctl", fake_run_systemctl)

    snapshot = gateway_cli.get_gateway_runtime_snapshot()

    assert snapshot.service_installed is True
    assert snapshot.service_running is True
    assert calls == [(["is-active", "hermes-gateway"], False)]


def test_launchd_service_definition_detects_migratable_legacy_plist(tmp_path, monkeypatch):
    machine_home = tmp_path / "machine-home"
    hermes_home = tmp_path / "custom-hermes"
    machine_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: machine_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(gateway_cli, "_launchd_user_home", lambda: machine_home)
    legacy_plist = (
        machine_home
        / "Library"
        / "LaunchAgents"
        / "ai.hermes.gateway.plist"
    )
    legacy_plist.parent.mkdir(parents=True)
    legacy_plist.write_text(
        "<dict>\n"
        "<key>HERMES_HOME</key>\n"
        f"<string>{hermes_home.resolve()}</string>\n"
        "</dict>\n",
        encoding="utf-8",
    )

    assert gateway_cli._has_launchd_service_definition() is True
    assert gateway_cli.get_launchd_plist_path().exists() is False


def test_launchd_runtime_snapshot_probes_migratable_legacy_plist(tmp_path, monkeypatch):
    machine_home = tmp_path / "machine-home"
    hermes_home = tmp_path / "custom-hermes"
    machine_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: machine_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(gateway_cli, "_launchd_user_home", lambda: machine_home)
    legacy_plist = (
        machine_home
        / "Library"
        / "LaunchAgents"
        / "ai.hermes.gateway.plist"
    )
    legacy_plist.parent.mkdir(parents=True)
    legacy_plist.write_text(
        "<dict>\n"
        "<key>HERMES_HOME</key>\n"
        f"<string>{hermes_home.resolve()}</string>\n"
        "</dict>\n",
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(gateway_cli, "find_gateway_pids", lambda: [])
    monkeypatch.setattr(gateway_cli, "is_termux", lambda: False)
    monkeypatch.setattr(gateway_cli, "is_linux", lambda: False)
    monkeypatch.setattr(gateway_cli, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(gateway_cli, "is_macos", lambda: True)
    monkeypatch.setattr(
        gateway_cli.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append(cmd)
        or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    snapshot = gateway_cli.get_gateway_runtime_snapshot()

    assert snapshot.service_installed is True
    assert snapshot.service_running is True
    assert calls == [["launchctl", "list", "ai.hermes.gateway"]]


def test_launchd_uninstall_removes_legacy_without_migrating(tmp_path, monkeypatch):
    legacy_plist = tmp_path / "legacy.plist"
    legacy_plist.write_text("legacy plist\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(
        gateway_cli,
        "_migrate_custom_root_launchd_service",
        lambda: (_ for _ in ()).throw(
            AssertionError("uninstall must not migrate before removing")
        ),
    )
    monkeypatch.setattr(
        gateway_cli,
        "_launchd_service_targets",
        lambda: [("ai.hermes.gateway", legacy_plist)],
    )
    monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: "gui/501")
    monkeypatch.setattr(
        gateway_cli.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append(cmd)
        or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    gateway_cli.launchd_uninstall()

    assert not legacy_plist.exists()
    assert calls == [["launchctl", "bootout", "gui/501/ai.hermes.gateway"]]


def test_launchd_install_bootstraps_migrated_scoped_plist(tmp_path, monkeypatch):
    plist_path = tmp_path / "ai.hermes.gateway-hash.plist"
    plist_path.write_text("scoped plist\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(gateway_cli, "_migrate_custom_root_launchd_service", lambda: True)
    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway-hash")
    monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: "gui/501")
    monkeypatch.setattr(
        gateway_cli.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append(cmd)
        or SimpleNamespace(returncode=1 if cmd[:2] == ["launchctl", "list"] else 0),
    )

    gateway_cli.launchd_install()

    assert ["launchctl", "list", "ai.hermes.gateway-hash"] in calls
    assert ["launchctl", "bootstrap", "gui/501", str(plist_path)] in calls


def test_launchd_install_refreshes_preexisting_migrated_scoped_plist(
    tmp_path, monkeypatch
):
    plist_path = tmp_path / "ai.hermes.gateway-hash.plist"
    plist_path.write_text("stale scoped plist\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(gateway_cli, "_migrate_custom_root_launchd_service", lambda: True)
    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway-hash")
    monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: "gui/501")
    monkeypatch.setattr(gateway_cli, "launchd_plist_is_current", lambda: False)
    monkeypatch.setattr(
        gateway_cli,
        "refresh_launchd_plist_if_needed",
        lambda: calls.append(["refresh"]) or True,
    )
    monkeypatch.setattr(
        gateway_cli.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append(cmd)
        or SimpleNamespace(returncode=1 if cmd[:2] == ["launchctl", "list"] else 0),
    )

    gateway_cli.launchd_install()

    assert ["refresh"] in calls
    assert ["launchctl", "bootstrap", "gui/501", str(plist_path)] in calls
    assert calls.index(["refresh"]) < calls.index(
        ["launchctl", "bootstrap", "gui/501", str(plist_path)]
    )


def test_launchd_status_reports_legacy_without_migrating(tmp_path, monkeypatch):
    calls = []

    monkeypatch.setattr(gateway_cli, "_has_migratable_custom_root_launchd_service", lambda: True)
    monkeypatch.setattr(
        gateway_cli,
        "_legacy_custom_root_launchd_plist_path",
        lambda: tmp_path / "legacy.plist",
    )
    monkeypatch.setattr(
        gateway_cli,
        "_migrate_custom_root_launchd_service",
        lambda: (_ for _ in ()).throw(AssertionError("status must not migrate")),
    )
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway-hash")
    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: tmp_path / "missing.plist")
    monkeypatch.setattr(
        gateway_cli.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append(cmd)
        or SimpleNamespace(returncode=1, stdout="", stderr=""),
    )

    gateway_cli.launchd_status()

    assert calls == [["launchctl", "list", "ai.hermes.gateway"]]


def test_launchd_stop_uses_legacy_without_migrating(tmp_path, monkeypatch):
    calls = []
    legacy_plist = tmp_path / "legacy.plist"
    legacy_plist.write_text("legacy\n", encoding="utf-8")

    monkeypatch.setattr(
        gateway_cli,
        "_has_migratable_custom_root_launchd_service",
        lambda: True,
    )
    monkeypatch.setattr(
        gateway_cli,
        "_legacy_custom_root_launchd_plist_path",
        lambda: legacy_plist,
    )
    monkeypatch.setattr(
        gateway_cli,
        "_migrate_custom_root_launchd_service",
        lambda: (_ for _ in ()).throw(AssertionError("stop must not migrate")),
    )
    monkeypatch.setattr(
        gateway_cli,
        "get_launchd_label",
        lambda: "ai.hermes.gateway-hash",
    )
    monkeypatch.setattr(
        gateway_cli,
        "get_launchd_plist_path",
        lambda: tmp_path / "missing.plist",
    )
    monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: "gui/501")
    monkeypatch.setattr(
        gateway_cli,
        "_wait_for_gateway_exit",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        gateway_cli.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append(cmd)
        or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    gateway_cli.launchd_stop()

    assert calls == [["launchctl", "bootout", "gui/501/ai.hermes.gateway"]]


def test_launchd_stop_boots_out_scoped_and_legacy_plists_when_both_exist(
    tmp_path, monkeypatch
):
    machine_home = tmp_path / "machine-home"
    hermes_home = tmp_path / "custom-hermes"
    machine_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: machine_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(gateway_cli, "_launchd_user_home", lambda: machine_home)
    monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: "gui/501")
    scoped_plist = gateway_cli.get_launchd_plist_path()
    legacy_plist = (
        machine_home
        / "Library"
        / "LaunchAgents"
        / "ai.hermes.gateway.plist"
    )
    scoped_plist.parent.mkdir(parents=True)
    scoped_plist.write_text("scoped\n", encoding="utf-8")
    legacy_plist.write_text(
        "<dict>\n"
        "<key>HERMES_HOME</key>\n"
        f"<string>{hermes_home.resolve()}</string>\n"
        "</dict>\n",
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        gateway_cli.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append(cmd)
        or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(gateway_cli, "_wait_for_gateway_exit", lambda **kwargs: True)

    gateway_cli.launchd_stop()

    assert ["launchctl", "bootout", f"gui/501/{gateway_cli.get_launchd_label()}"] in calls
    assert ["launchctl", "bootout", "gui/501/ai.hermes.gateway"] in calls


def test_launchd_restart_migrates_and_regenerates_missing_plist(tmp_path, monkeypatch):
    plist_path = tmp_path / "scoped.plist"
    calls = []
    state = {"migrated": False}

    monkeypatch.setattr(
        gateway_cli,
        "_migrate_custom_root_launchd_service",
        lambda: state.__setitem__("migrated", True) or True,
    )
    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(gateway_cli, "generate_launchd_plist", lambda: "plist\n")
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway-hash")
    monkeypatch.setattr(gateway_cli, "_request_gateway_self_restart", lambda pid: False)
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)

    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["launchctl", "kickstart"] and "-k" in cmd:
            raise gateway_cli.subprocess.CalledProcessError(3, cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gateway_cli.subprocess, "run", run)

    gateway_cli.launchd_restart()

    assert state["migrated"] is True
    assert plist_path.read_text(encoding="utf-8") == "plist\n"
    assert ["launchctl", "bootstrap", gateway_cli._launchd_domain(), str(plist_path)] in calls


def test_launchd_restart_bootstraps_migrated_service_after_self_restart(tmp_path, monkeypatch):
    plist_path = tmp_path / "scoped.plist"
    plist_path.write_text("plist\n", encoding="utf-8")
    calls = []
    waited = []

    monkeypatch.setattr(gateway_cli, "_migrate_custom_root_launchd_service", lambda: True)
    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway-hash")
    monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: "gui/501")
    monkeypatch.setattr(gateway_cli, "_launchd_label_is_loaded", lambda label: False)
    monkeypatch.setattr(gateway_cli, "_request_gateway_self_restart", lambda pid: True)
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: 12345)
    monkeypatch.setattr(os, "getpid", lambda: 99999)
    monkeypatch.setattr(
        gateway_cli,
        "_wait_for_gateway_exit",
        lambda **kwargs: waited.append(kwargs) or True,
    )

    monkeypatch.setattr(
        gateway_cli.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append(cmd)
        or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    gateway_cli.launchd_restart()

    assert waited
    assert ["launchctl", "bootstrap", "gui/501", str(plist_path)] in calls
    assert ["launchctl", "kickstart", "gui/501/ai.hermes.gateway-hash"] in calls


def test_launchd_restart_refreshes_migrated_scoped_plist_before_bootstrap(
    tmp_path, monkeypatch
):
    plist_path = tmp_path / "scoped.plist"
    plist_path.write_text("stale plist\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(gateway_cli, "_migrate_custom_root_launchd_service", lambda: True)
    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(gateway_cli, "get_launchd_label", lambda: "ai.hermes.gateway-hash")
    monkeypatch.setattr(gateway_cli, "_launchd_domain", lambda: "gui/501")
    monkeypatch.setattr(gateway_cli, "_launchd_label_is_loaded", lambda label: False)
    monkeypatch.setattr(gateway_cli, "_request_gateway_self_restart", lambda pid: True)
    monkeypatch.setattr(gateway_cli, "_wait_for_gateway_exit", lambda **kwargs: True)
    monkeypatch.setattr(gateway_cli, "launchd_plist_is_current", lambda: False)
    monkeypatch.setattr(gateway_cli, "generate_launchd_plist", lambda: "fresh plist\n")
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: 12345)
    monkeypatch.setattr(os, "getpid", lambda: 99999)
    monkeypatch.setattr(
        gateway_cli.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append(cmd)
        or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    gateway_cli.launchd_restart()

    assert plist_path.read_text(encoding="utf-8") == "fresh plist\n"
    assert ["launchctl", "bootstrap", "gui/501", str(plist_path)] in calls
    assert ["launchctl", "kickstart", "gui/501/ai.hermes.gateway-hash"] in calls
