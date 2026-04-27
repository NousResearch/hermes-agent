"""Regression tests for the standalone scripts/hermes-gateway entry point."""

import hashlib
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "hermes-gateway"


def load_gateway_script():
    loader = SourceFileLoader("hermes_gateway_script_under_test", str(SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    with patch("dotenv.load_dotenv"):
        loader.exec_module(module)
    return module


def test_launchd_plist_preserves_active_hermes_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes" / "profiles" / "coder"
    hermes_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    module = load_gateway_script()

    plist = module.generate_launchd_plist()

    assert module.get_launchd_label() == "ai.hermes.gateway-coder"
    assert (
        module.get_launchd_plist_path()
        == tmp_path / "Library" / "LaunchAgents" / "ai.hermes.gateway-coder.plist"
    )
    assert "<string>ai.hermes.gateway-coder</string>" in plist
    assert "<key>HERMES_HOME</key>" in plist
    assert f"<string>{hermes_home.resolve()}</string>" in plist
    assert f"<string>{hermes_home}/logs/gateway.log</string>" in plist
    assert f"<string>{hermes_home}/logs/gateway.error.log</string>" in plist


def test_launchd_install_creates_logs_under_active_hermes_home(tmp_path, monkeypatch):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes-home"
    os_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    module = load_gateway_script()
    monkeypatch.setattr(module.Path, "home", lambda: os_home)
    monkeypatch.setattr(
        module,
        "get_launchd_plist_path",
        lambda: tmp_path / "LaunchAgents" / "ai.hermes.gateway.plist",
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    module.install_launchd()

    assert (hermes_home / "logs").is_dir()
    assert not (os_home / ".hermes" / "logs").exists()


def test_systemd_unit_preserves_active_hermes_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes" / "profiles" / "coder"
    hermes_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    module = load_gateway_script()
    unit = module.generate_systemd_unit()

    assert module.get_systemd_service_name() == "hermes-gateway-coder"
    assert (
        module.get_systemd_unit_path()
        == tmp_path / ".config" / "systemd" / "user" / "hermes-gateway-coder.service"
    )
    assert f'Environment="HERMES_HOME={hermes_home.resolve()}"' in unit


def test_systemd_management_commands_use_profile_service(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes" / "profiles" / "coder"
    hermes_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    module = load_gateway_script()
    calls = []
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append(cmd)
        or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    module.systemd_start()
    module.systemd_stop()
    module.systemd_status()
    module.systemd_restart()

    assert calls == [
        ["systemctl", "--user", "start", "hermes-gateway-coder"],
        ["systemctl", "--user", "stop", "hermes-gateway-coder"],
        ["systemctl", "--user", "status", "hermes-gateway-coder"],
        ["systemctl", "--user", "restart", "hermes-gateway-coder"],
    ]


def test_systemd_management_commands_fall_back_to_matching_legacy_service(
    tmp_path, monkeypatch
):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes"
    os_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    module = load_gateway_script()
    legacy_unit = module.get_legacy_systemd_unit_path()
    legacy_unit.parent.mkdir(parents=True)
    legacy_unit.write_text(
        f'[Service]\nEnvironment="HERMES_HOME={hermes_home.resolve()}"\n',
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append(cmd)
        or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    module.systemd_start()
    module.systemd_stop()
    module.systemd_status()
    module.systemd_restart()

    assert calls == [
        ["systemctl", "--user", "start", "hermes-gateway"],
        ["systemctl", "--user", "stop", "hermes-gateway"],
        ["systemctl", "--user", "status", "hermes-gateway"],
        ["systemctl", "--user", "restart", "hermes-gateway"],
    ]


def test_systemd_recognizes_previous_standalone_unit_when_project_env_matches(
    tmp_path, monkeypatch
):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes"
    project_dir = tmp_path / "project"
    os_home.mkdir()
    hermes_home.mkdir()
    project_dir.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    module = load_gateway_script()
    monkeypatch.setattr(module, "PROJECT_DIR", project_dir)
    (project_dir / ".env").write_text(
        f"HERMES_HOME={hermes_home.resolve()}\n",
        encoding="utf-8",
    )
    legacy_unit = module.get_legacy_systemd_unit_path()
    legacy_unit.parent.mkdir(parents=True)
    legacy_unit.write_text(
        "[Service]\n"
        f"ExecStart={module.get_python_path()} {module.get_gateway_script_path()} run\n",
        encoding="utf-8",
    )

    assert module.has_legacy_systemd_service() is True


def test_systemd_ignores_explicit_nonmatching_home_before_script_fallback(
    tmp_path, monkeypatch
):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes"
    other_home = tmp_path / "other-hermes"
    project_dir = tmp_path / "project"
    os_home.mkdir()
    hermes_home.mkdir()
    other_home.mkdir()
    project_dir.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    module = load_gateway_script()
    monkeypatch.setattr(module, "PROJECT_DIR", project_dir)
    (project_dir / ".env").write_text(
        f"HERMES_HOME={hermes_home.resolve()}\n",
        encoding="utf-8",
    )
    legacy_unit = module.get_legacy_systemd_unit_path()
    legacy_unit.parent.mkdir(parents=True)
    legacy_unit.write_text(
        "[Service]\n"
        f'Environment="HERMES_HOME={other_home.resolve()}"\n'
        f"ExecStart={module.get_python_path()} {module.get_gateway_script_path()} run\n",
        encoding="utf-8",
    )

    assert module.legacy_service_matches_current_install(legacy_unit) is False
    assert module.has_legacy_systemd_service() is False


def test_systemd_ignores_script_path_only_legacy_unit_without_project_env(
    tmp_path, monkeypatch
):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes"
    project_dir = tmp_path / "project"
    os_home.mkdir()
    hermes_home.mkdir()
    project_dir.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    module = load_gateway_script()
    monkeypatch.setattr(module, "PROJECT_DIR", project_dir)
    legacy_unit = module.get_legacy_systemd_unit_path()
    legacy_unit.parent.mkdir(parents=True)
    legacy_unit.write_text(
        "[Service]\n"
        f"ExecStart={module.get_python_path()} {module.get_gateway_script_path()} run\n",
        encoding="utf-8",
    )

    assert module.has_legacy_systemd_service() is False


def test_systemd_uninstall_removes_matching_legacy_service(tmp_path, monkeypatch):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes"
    os_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    module = load_gateway_script()
    legacy_unit = module.get_legacy_systemd_unit_path()
    legacy_unit.parent.mkdir(parents=True)
    legacy_unit.write_text(
        f'[Service]\nEnvironment="HERMES_HOME={hermes_home.resolve()}"\n',
        encoding="utf-8",
    )
    calls = []
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append(cmd)
        or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    module.uninstall_systemd()

    assert not legacy_unit.exists()
    assert ["systemctl", "--user", "stop", "hermes-gateway"] in calls
    assert ["systemctl", "--user", "disable", "hermes-gateway"] in calls


def test_systemd_install_starts_scoped_service_when_legacy_service_was_active(
    tmp_path, monkeypatch
):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes"
    os_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    module = load_gateway_script()
    legacy_unit = module.get_legacy_systemd_unit_path()
    legacy_unit.parent.mkdir(parents=True)
    legacy_unit.write_text(
        f'[Service]\nEnvironment="HERMES_HOME={hermes_home.resolve()}"\n',
        encoding="utf-8",
    )
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        stdout = "active\n" if cmd == [
            "systemctl",
            "--user",
            "is-active",
            "hermes-gateway",
        ] else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(module.subprocess, "run", run)

    module.install_systemd()

    service_name = module.get_systemd_service_name()
    assert ["systemctl", "--user", "enable", service_name] in calls
    assert ["systemctl", "--user", "start", service_name] in calls
    assert not legacy_unit.exists()


def test_systemd_install_keeps_legacy_service_when_scoped_enable_fails(
    tmp_path, monkeypatch
):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes"
    os_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    module = load_gateway_script()
    legacy_unit = module.get_legacy_systemd_unit_path()
    legacy_unit.parent.mkdir(parents=True)
    legacy_unit.write_text(
        f'[Service]\nEnvironment="HERMES_HOME={hermes_home.resolve()}"\n',
        encoding="utf-8",
    )
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd == ["systemctl", "--user", "enable", module.get_systemd_service_name()]:
            raise module.subprocess.CalledProcessError(1, cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", run)

    with pytest.raises(module.subprocess.CalledProcessError):
        module.install_systemd()

    assert legacy_unit.exists()
    assert not module.get_systemd_unit_path().exists()
    assert ["systemctl", "--user", "stop", "hermes-gateway"] not in calls
    assert ["systemctl", "--user", "disable", "hermes-gateway"] not in calls


def test_systemd_install_disables_scoped_service_when_legacy_stop_fails(
    tmp_path, monkeypatch
):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes"
    os_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    module = load_gateway_script()
    legacy_unit = module.get_legacy_systemd_unit_path()
    legacy_unit.parent.mkdir(parents=True)
    legacy_unit.write_text(
        f'[Service]\nEnvironment="HERMES_HOME={hermes_home.resolve()}"\n',
        encoding="utf-8",
    )
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd == ["systemctl", "--user", "is-active", "hermes-gateway"]:
            return SimpleNamespace(returncode=0, stdout="active\n", stderr="")
        if cmd == ["systemctl", "--user", "stop", "hermes-gateway"]:
            raise module.subprocess.CalledProcessError(1, cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", run)

    with pytest.raises(module.subprocess.CalledProcessError):
        module.install_systemd()

    service_name = module.get_systemd_service_name()
    assert legacy_unit.exists()
    assert not module.get_systemd_unit_path().exists()
    assert ["systemctl", "--user", "disable", service_name] in calls
    assert ["systemctl", "--user", "disable", "hermes-gateway"] not in calls


def test_launchd_management_commands_use_profile_label(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes" / "profiles" / "coder"
    hermes_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    module = load_gateway_script()
    calls = []
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append(cmd)
        or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    module.launchd_start()
    module.launchd_stop()
    module.launchd_status()

    assert calls == [
        ["launchctl", "start", "ai.hermes.gateway-coder"],
        ["launchctl", "stop", "ai.hermes.gateway-coder"],
        ["launchctl", "list", "ai.hermes.gateway-coder"],
    ]


def test_launchd_management_commands_fall_back_to_matching_legacy_label(
    tmp_path, monkeypatch
):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes"
    os_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    module = load_gateway_script()
    legacy_plist = module.get_legacy_launchd_plist_path()
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
        module.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append(cmd)
        or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    module.launchd_start()
    module.launchd_stop()
    module.launchd_status()

    assert calls == [
        ["launchctl", "start", "ai.hermes.gateway"],
        ["launchctl", "stop", "ai.hermes.gateway"],
        ["launchctl", "list", "ai.hermes.gateway"],
    ]


def test_launchd_recognizes_previous_standalone_plist_when_project_env_matches(
    tmp_path, monkeypatch
):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes"
    project_dir = tmp_path / "project"
    os_home.mkdir()
    hermes_home.mkdir()
    project_dir.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    module = load_gateway_script()
    monkeypatch.setattr(module, "PROJECT_DIR", project_dir)
    (project_dir / ".env").write_text(
        f"HERMES_HOME={hermes_home.resolve()}\n",
        encoding="utf-8",
    )
    legacy_plist = module.get_legacy_launchd_plist_path()
    legacy_plist.parent.mkdir(parents=True)
    legacy_plist.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<plist><dict>"
        "<key>Label</key>"
        "<string>ai.hermes.gateway</string>"
        "<key>ProgramArguments</key>"
        "<array>"
        f"<string>{module.get_python_path()}</string>"
        f"<string>{module.get_gateway_script_path()}</string>"
        "<string>run</string>"
        "</array>"
        "</dict></plist>\n",
        encoding="utf-8",
    )

    assert module.has_legacy_launchd_service() is True


def test_launchd_ignores_explicit_nonmatching_home_before_script_fallback(
    tmp_path, monkeypatch
):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes"
    other_home = tmp_path / "other-hermes"
    project_dir = tmp_path / "project"
    os_home.mkdir()
    hermes_home.mkdir()
    other_home.mkdir()
    project_dir.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    module = load_gateway_script()
    monkeypatch.setattr(module, "PROJECT_DIR", project_dir)
    (project_dir / ".env").write_text(
        f"HERMES_HOME={hermes_home.resolve()}\n",
        encoding="utf-8",
    )
    legacy_plist = module.get_legacy_launchd_plist_path()
    legacy_plist.parent.mkdir(parents=True)
    legacy_plist.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<plist><dict>"
        "<key>Label</key>"
        "<string>ai.hermes.gateway</string>"
        "<key>ProgramArguments</key>"
        "<array>"
        f"<string>{module.get_python_path()}</string>"
        f"<string>{module.get_gateway_script_path()}</string>"
        "<string>run</string>"
        "</array>"
        "<key>EnvironmentVariables</key>"
        "<dict>"
        "<key>HERMES_HOME</key>"
        f"<string>{other_home.resolve()}</string>"
        "</dict>"
        "</dict></plist>\n",
        encoding="utf-8",
    )

    assert module.legacy_service_matches_current_install(legacy_plist) is False
    assert module.has_legacy_launchd_service() is False


def test_launchd_ignores_script_path_only_legacy_plist_without_project_env(
    tmp_path, monkeypatch
):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes"
    project_dir = tmp_path / "project"
    os_home.mkdir()
    hermes_home.mkdir()
    project_dir.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    module = load_gateway_script()
    monkeypatch.setattr(module, "PROJECT_DIR", project_dir)
    legacy_plist = module.get_legacy_launchd_plist_path()
    legacy_plist.parent.mkdir(parents=True)
    legacy_plist.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<plist><dict>"
        "<key>Label</key>"
        "<string>ai.hermes.gateway</string>"
        "<key>ProgramArguments</key>"
        "<array>"
        f"<string>{module.get_python_path()}</string>"
        f"<string>{module.get_gateway_script_path()}</string>"
        "<string>run</string>"
        "</array>"
        "</dict></plist>\n",
        encoding="utf-8",
    )

    assert module.has_legacy_launchd_service() is False


def test_launchd_remove_legacy_waits_for_matching_plist(tmp_path, monkeypatch):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes"
    os_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    module = load_gateway_script()
    legacy_plist = module.get_legacy_launchd_plist_path()
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
        module.subprocess,
        "run",
        run,
    )
    monkeypatch.setattr(
        module,
        "wait_for_pid_exit",
        lambda pid, **kwargs: waits.append((pid, kwargs)) or True,
    )

    assert module.remove_legacy_launchd_service() is True

    assert not legacy_plist.exists()
    assert ["launchctl", "list", "ai.hermes.gateway"] in calls
    assert ["launchctl", "unload", str(legacy_plist)] in calls
    assert waits == [(4321, {"timeout": 10.0})]


def test_launchd_install_keeps_legacy_plist_when_scoped_load_fails(
    tmp_path, monkeypatch
):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes"
    os_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    module = load_gateway_script()
    legacy_plist = module.get_legacy_launchd_plist_path()
    legacy_plist.parent.mkdir(parents=True)
    legacy_plist.write_text(
        "<dict>\n"
        "<key>HERMES_HOME</key>\n"
        f"<string>{hermes_home.resolve()}</string>\n"
        "</dict>\n",
        encoding="utf-8",
    )
    scoped_plist = module.get_launchd_plist_path()
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd == ["launchctl", "list", "ai.hermes.gateway"]:
            return SimpleNamespace(
                returncode=0,
                stdout="PID\tStatus\tLabel\n4321\t0\tai.hermes.gateway\n",
                stderr="",
            )
        if cmd == ["launchctl", "load", str(scoped_plist)]:
            raise module.subprocess.CalledProcessError(5, cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module, "wait_for_pid_exit", lambda pid, **kwargs: True)

    with pytest.raises(module.subprocess.CalledProcessError):
        module.install_launchd()

    assert legacy_plist.exists()
    assert not scoped_plist.exists()
    assert ["launchctl", "unload", str(legacy_plist)] in calls
    assert ["launchctl", "load", str(scoped_plist)] in calls
    assert ["launchctl", "load", str(legacy_plist)] in calls


def test_launchd_label_hashes_custom_hermes_home(tmp_path, monkeypatch):
    os_home = tmp_path / "os-home"
    hermes_home = tmp_path / "custom-hermes-a"
    os_home.mkdir()
    hermes_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: os_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    module = load_gateway_script()
    suffix = hashlib.sha256(str(hermes_home.resolve()).encode()).hexdigest()[:8]

    assert module.get_launchd_label() == f"ai.hermes.gateway-{suffix}"
    assert (
        module.get_launchd_plist_path()
        == os_home / "Library" / "LaunchAgents" / f"ai.hermes.gateway-{suffix}.plist"
    )


def test_launchd_plist_uses_absolute_log_paths_for_relative_hermes_home(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HERMES_HOME", "relative-hermes-home")

    module = load_gateway_script()
    plist = module.generate_launchd_plist()

    resolved = (tmp_path / "relative-hermes-home").resolve()
    assert f"<string>{resolved}/logs/gateway.log</string>" in plist
    assert f"<string>{resolved}/logs/gateway.error.log</string>" in plist
