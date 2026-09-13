"""Source launchers keep custom-home and selected-generation state at boot."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

from hermes_cli import _launchers
from hermes_cli.runtime_paths import install_state_dir, site_packages

ROOT = Path(__file__).resolve().parents[2]
BOOT_FILES = (
    "hermes_bootstrap.py", "hermes_constants.py", "hermes_cli/__init__.py", "hermes_cli/_launchers.py",
    "hermes_cli/runtime_paths.py", "hermes_cli/runtime_state.py",
    "hermes_cli/_early_recovery.py", "hermes_cli/_parser.py",
    "hermes_cli/venv_sync.py", "hermes_cli/steward.py",
    "hermes_cli/stderr_timestamp.py",
    "scripts/hermes-gateway",
)


def fixture_tree(tmp_path, monkeypatch):
    repo = tmp_path / "source 'café repo"
    for relative in BOOT_FILES:
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    (repo / "acp_adapter").mkdir()
    (repo / "acp_adapter" / "__init__.py").write_text("", encoding="utf-8")
    entry = (
        "import json, os, sys\n"
        "def main():\n"
        "    import selected_probe\n"
        "    print(json.dumps({'value': selected_probe.VALUE, 'argv': sys.argv[1:], "
        "'home': os.environ.get('HERMES_HOME'), 'exe': sys.executable}))\n"
        "    return 7\n"
        "if __name__ == '__main__':\n    sys.exit(main())\n"
    )
    for path in (repo / "hermes_cli/main.py", repo / "acp_adapter/entry.py"):
        path.write_text(entry, encoding="utf-8")
    home = tmp_path / "custom 'café home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    store = home / "tools"
    store.mkdir(parents=True)
    interpreter = Path(sys._base_executable).resolve()
    (store / "facts.json").write_text(json.dumps({"schema": 1, "packages": {"python": {
        "version": "fixture", "entry": str(interpreter.parent if os.name == "nt" else interpreter.parents[1])
    }}}), encoding="utf-8")
    return repo, home, interpreter


@pytest.mark.platforms("windows", "posix")
@pytest.mark.parametrize("form", ["native", "shell"])
def test_source_launchers_boot_selected_generation_from_custom_home(tmp_path, monkeypatch, form):
    repo, home, interpreter = fixture_tree(tmp_path, monkeypatch)
    out = tmp_path / "commands"
    out.mkdir()
    # No repo/venv or console script exists. PM selection lives outside the checkout.
    launchers = [Path(p) for p in _launchers.ensure_install_launchers(repo, out)]
    assert len(launchers) == len(_launchers.ENTRY_POINTS)
    if form == "shell":
        shell_out = tmp_path / "shell-commands"
        shell_out.mkdir()
        launchers = [
            _launchers._mint_shell_launcher(name, shell_out, interpreter,
                                            _launchers._launcher_script(name, repo, None))
            for name in _launchers.ENTRY_POINTS
        ]
    args = ['spaces and café', 'apostrophe\'s', r'one\two', '$HOME; echo no', '']
    for number in (1, 2):
        selected = install_state_dir(repo) / "environments" / str(number) / "venv"
        site = site_packages(selected)
        site.mkdir(parents=True)
        (selected / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
        (site / "selected_probe.py").write_text(f"VALUE = {number}\n", encoding="utf-8")
        (install_state_dir(repo) / "facts.json").write_text(
            json.dumps({"packages": {"venv": {"environment": str(selected)}}}), encoding="utf-8")
        env = dict(os.environ)
        env.pop("HERMES_HOME", None)
        env.pop("HERMES_RUNTIME_DIR", None)
        env["PYTHONHOME"] = str(tmp_path / "foreign-python")
        env["PYTHONPATH"] = str(tmp_path / "foreign-deps")
        for launcher in launchers:
            assert launcher is not None
            command = ["bash", "-s"] if form == "shell" else [str(launcher), *args]
            script = "exec " + shlex.join(["bash", str(launcher), *args]) + "\n" if form == "shell" else None
            result = subprocess.run(command, input=script, cwd=tmp_path, env=env,
                                    capture_output=True, text=True, encoding="utf-8", timeout=30)
            assert result.returncode == 7, result.stdout + result.stderr
            receipt = json.loads(result.stdout)
            assert receipt["value"] == number
            assert receipt["argv"] == args
            assert Path(receipt["home"]) == home
            assert Path(receipt["exe"]).samefile(interpreter)
    assert not (repo / "venv").exists()


@pytest.mark.platforms("posix")
def test_posix_materializer_publishes_only_executable_shell_launchers(tmp_path, monkeypatch):
    repo, _home, _interpreter = fixture_tree(tmp_path, monkeypatch)
    out = tmp_path / "bin"
    out.mkdir()
    launchers = [Path(p) for p in _launchers.ensure_install_launchers(repo, out)]
    assert {p.name for p in launchers} == set(_launchers.ENTRY_POINTS)
    assert all(os.access(p, os.X_OK) for p in launchers)
    assert set(out.iterdir()) == set(launchers)
    local = repo / ".hermes" / "bin"
    assert {p.name for p in local.iterdir()} == set(_launchers.ENTRY_POINTS)


def test_materializer_cli_refuses_missing_store_without_publishing(tmp_path, monkeypatch):
    repo, home, _interpreter = fixture_tree(tmp_path, monkeypatch)
    (home / "tools" / "facts.json").unlink()
    orphan = home / "tools" / "python-unrecorded" / ("python.exe" if os.name == "nt" else "bin/python3")
    orphan.parent.mkdir(parents=True)
    orphan.touch()  # uncommitted tool bytes are not an installed interpreter
    out = tmp_path / "bin"
    result = subprocess.run([sys.executable, "-I", str(repo / "hermes_cli/_launchers.py"), str(out)],
                            cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "store interpreter" in result.stderr
    assert not out.exists() or not list(out.iterdir())


def _command_survives_generation_collection(tmp_path, monkeypatch, surface):
    from hermes_cli.runtime_state import collect_generations

    repo, home, interpreter = fixture_tree(tmp_path, monkeypatch)
    out = tmp_path / "bin"
    out.mkdir()
    _launchers.ensure_install_launchers(repo, out)
    launcher = next(path for path in out.iterdir() if path.stem == "hermes")
    args = ["café ' quoted", "", "$HOME; not a shell"]
    command = []
    selected = install_state_dir(repo) / "environments" / "old" / "venv"
    for value in ("old", "new"):
        selected = selected.parent.parent / value / "venv"
        site = site_packages(selected)
        site.mkdir(parents=True)
        (selected / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
        (selected.parent / ".lease-managed").touch()
        (site / "selected_probe.py").write_text(f"VALUE = {value!r}\n", encoding="utf-8")
        (install_state_dir(repo) / "facts.json").write_text(
            json.dumps({"packages": {"venv": {"environment": str(selected)}}}), encoding="utf-8")
        if value == "old":
            if surface == "legacy":
                command = [sys.executable, "-I", str(repo / "scripts/hermes-gateway"), "--help"]
                args = ["gateway", "--help"]
            elif surface == "ssh":
                from hermes_cli.windows_ssh_runtime import _resolve_direct_command
                command = [*_resolve_direct_command(str(launcher)), *args]
            elif surface == "published":
                result = subprocess.run([str(launcher), "--print-runtime-command", "--", *args],
                                        capture_output=True, text=True, timeout=30)
                assert result.returncode == 0, result.stderr
                command = json.loads(result.stdout)
            else:
                from hermes_cli import gateway
                monkeypatch.setattr(gateway, "PROJECT_ROOT", repo)
                if surface == "launchd":
                    import plistlib
                    unit = gateway.generate_launchd_plist()
                    command = plistlib.loads(unit.encode())["ProgramArguments"]
                    args = ["gateway", "run", "--external-supervisor"]
                else:
                    unit = gateway.generate_systemd_unit()
                    command = shlex.split(next(line.removeprefix("ExecStart=") for line in unit.splitlines()
                                               if line.startswith("ExecStart=")))
                    args = ["gateway", "run"]
                assert str(selected.parent) not in unit
            if surface not in ("legacy", "systemd", "launchd"):
                assert Path(command[0]).samefile(interpreter)
    assert collect_generations(repo, min_age_seconds=0) == [selected.parent.parent / "old"]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 7, result.stderr
    assert json.loads(result.stdout)["value"] == "new"
    assert json.loads(result.stdout)["argv"] == args


@pytest.mark.parametrize("surface", ["published", "systemd", "launchd", "ssh", "legacy"])
@pytest.mark.platforms("posix")
@pytest.mark.spawns_gateway_lookalike
def test_posix_commands_survive_generation_collection(tmp_path, monkeypatch, surface):
    _command_survives_generation_collection(tmp_path, monkeypatch, surface)


@pytest.mark.parametrize("surface", ["published", "ssh"])
@pytest.mark.platforms("windows")
def test_windows_commands_survive_generation_collection(tmp_path, monkeypatch, surface):
    _command_survives_generation_collection(tmp_path, monkeypatch, surface)


@pytest.mark.platforms("windows")
def test_windows_repair_upgrades_healthy_old_pm_external_launchers(tmp_path, monkeypatch):
    from hermes_cli._install_repair import ensure_windows_bin_launchers

    repo, home, interpreter = fixture_tree(tmp_path, monkeypatch)
    managed = home / "hermes-agent"
    shutil.move(repo, managed)
    external = home / "bin"
    external.mkdir()
    for name in _launchers.ENTRY_POINTS:
        (external / f"{name}.exe").write_bytes(b"old PM launcher without a venv binding")
    assert ensure_windows_bin_launchers(managed, user_path_entries=[])
    local = managed / ".hermes" / "bin"
    launcher = local / "hermes.exe"
    if not launcher.exists():
        launcher = local / "hermes.cmd"
    result = subprocess.run([str(launcher), "--print-runtime-command"], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert Path(json.loads(result.stdout)[0]).samefile(interpreter)


def test_dashboard_action_boots_selected_dependencies(tmp_path, monkeypatch):
    from hermes_cli import web_server, web_server_gateway

    repo, home, _ = fixture_tree(tmp_path, monkeypatch)
    selected = install_state_dir(repo) / "environments" / "current" / "venv"
    site = site_packages(selected)
    site.mkdir(parents=True)
    (selected / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
    (site / "selected_probe.py").write_text("VALUE = 'selected'\n", encoding="utf-8")
    (install_state_dir(repo) / "facts.json").write_text(
        json.dumps({"packages": {"venv": {"environment": str(selected)}}}), encoding="utf-8")
    monkeypatch.setattr(web_server, "PROJECT_ROOT", repo)
    monkeypatch.setattr(web_server_gateway, "_ACTION_LOG_DIR", home / "logs")
    proc = web_server_gateway._spawn_hermes_action(["--version"], "gateway-restart")
    try:
        assert proc.wait(timeout=30) == 7
        log = (home / "logs" / web_server_gateway._ACTION_LOG_FILES["gateway-restart"]).read_text()
        assert json.loads(log.splitlines()[-1])["value"] == "selected"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)


@pytest.mark.parametrize("layout", ["legacy", "payload"])
def test_pre_pm_base_dependencies_activate_only_at_boot(tmp_path, monkeypatch, layout):
    repo, _, _ = fixture_tree(tmp_path, monkeypatch)
    environment = repo / "venv"
    if layout == "payload":
        environment = repo.parent / "payload-deps"
        (repo.parent / "manifest.json").write_text(
            json.dumps({"repo": repo.name, "venv": environment.name}), encoding="utf-8")
    site = site_packages(environment)
    site.mkdir(parents=True)
    editable = tmp_path / "editable"
    editable.mkdir()
    (editable / "selected_probe.py").write_text("VALUE = 'base-pth'\n", encoding="utf-8")
    (site / "member.pth").write_text(str(editable) + "\n", encoding="utf-8")
    command = _launchers.runtime_command(repo)
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 7, result.stderr
    assert json.loads(result.stdout)["value"] == "base-pth"


def test_external_interpreter_keeps_its_owned_dependencies(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "empty-store"))
    command = _launchers.runtime_command(ROOT, code="import ruamel.yaml; print('external-runtime-ready')")
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "external-runtime-ready"


@pytest.mark.platforms("posix")
@pytest.mark.spawns_gateway_lookalike
def test_service_survives_python_tool_replacement(tmp_path, monkeypatch):
    from hermes_cli import gateway

    repo, home, interpreter = fixture_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(gateway, "PROJECT_ROOT", repo)
    site = site_packages(repo / "venv")
    site.mkdir(parents=True)
    (site / "selected_probe.py").write_text("VALUE = 'ready'\n", encoding="utf-8")
    store = home / "tools"
    for version in ("python-A", "python-B"):
        python = store / version / "bin" / "python3"
        python.parent.mkdir(parents=True)
        python.symlink_to(interpreter)
        (store / "facts.json").write_text(json.dumps({"packages": {"python": {"entry": version}}}), encoding="utf-8")
        gateway._prepare_service_launcher()
        if version == "python-A":
            unit = gateway.generate_systemd_unit()
            assert str(store / version) not in unit
            command = shlex.split(next(line.split("=", 1)[1] for line in unit.splitlines() if line.startswith("ExecStart=")))
    shutil.rmtree(store / "python-A")
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 7, result.stderr
    assert json.loads(result.stdout)["value"] == "ready"


def test_update_import_probe_uses_selected_dependencies(tmp_path, monkeypatch):
    from hermes_cli import update_cmd, update_cmd_validation

    repo, _, _ = fixture_tree(tmp_path, monkeypatch)
    selected = install_state_dir(repo) / "environments" / "current" / "venv"
    site = site_packages(selected)
    site.mkdir(parents=True)
    (selected / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
    (site / "selected_probe.py").write_text("VALUE = 'selected'\n", encoding="utf-8")
    (install_state_dir(repo) / "facts.json").write_text(
        json.dumps({"packages": {"venv": {"environment": str(selected)}}}), encoding="utf-8")
    (repo / "hermes_integrity_probe.py").write_text("import selected_probe\n", encoding="utf-8")
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("hermes_integrity_probe",))
    assert update_cmd_validation._critical_module_import_failures(repo, report_runtime_errors=True) == {}
