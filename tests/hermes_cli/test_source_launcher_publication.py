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
from pm.environments import install_state_dir, site_packages

ROOT = Path(__file__).resolve().parents[2]
BOOT_FILES = (
    "hermes_bootstrap.py", "hermes_constants.py", "hermes_cli/__init__.py", "hermes_cli/_launchers.py",
    "pm/environments.py", "pm/filesystem.py", "pm/paths.py", "hermes_cli/runtime_state.py",
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
    # Windows resolves its default under LOCALAPPDATA, not HOME.
    if os.name == "nt":
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    home = tmp_path / ("hermes" if os.name == "nt" else ".hermes")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_RUNTIME_DIR", raising=False)
    store = home / "tools"
    store.mkdir(parents=True)
    interpreter = Path(sys._base_executable).resolve()
    (store / "facts.json").write_text(json.dumps({"schema": 1, "packages": {"python": {
        "version": "fixture", "entry": str(interpreter.parent if os.name == "nt" else interpreter.parents[1])
    }}}), encoding="utf-8")
    return repo, home, interpreter


def select_generation(repo, name, value):
    selected = install_state_dir(repo) / 'environments' / str(name) / 'venv'
    site = site_packages(selected)
    site.mkdir(parents=True)
    (selected / 'pyvenv.cfg').write_text('home = fixture\n', encoding='utf-8')
    (site / 'selected_probe.py').write_text(f'VALUE = {value!r}\n', encoding='utf-8')
    (install_state_dir(repo) / 'facts.json').write_text(
        json.dumps({'packages': {'venv': {'environment': str(selected)}}}), encoding='utf-8')
    return selected


@pytest.mark.platforms("windows", "posix")
@pytest.mark.parametrize("form", ["native", "shell"])
def test_source_launchers_boot_selected_generation_from_custom_home(tmp_path, monkeypatch, form, real_bash):
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
        select_generation(repo, number, number)
        env = dict(os.environ)
        env.pop("HERMES_HOME", None)
        env.pop("HERMES_RUNTIME_DIR", None)
        env["PYTHONHOME"] = str(tmp_path / "foreign-python")
        env["PYTHONPATH"] = str(tmp_path / "foreign-deps")
        for launcher in launchers:
            assert launcher is not None
            command = [real_bash, "-s"] if form == "shell" else [str(launcher), *args]
            script = "exec " + shlex.join([real_bash, str(launcher), *args]) + "\n" if form == "shell" else None
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
def test_launcher_resolves_default_home_at_use_not_publication(tmp_path, monkeypatch):
    repo, published_home, _ = fixture_tree(tmp_path, monkeypatch)
    launcher = Path(_launchers.ensure_install_launchers(repo, tmp_path / "commands")[0])
    new_user_home = tmp_path / "second-user"
    new_user_home.mkdir()
    monkeypatch.setenv("HOME", str(new_user_home))
    monkeypatch.setenv("HERMES_HOME", str(new_user_home / ".hermes"))
    select_generation(repo, "second", "from-second-user")
    env = dict(os.environ)
    env.pop("HERMES_HOME")
    result = subprocess.run([str(launcher)], cwd=tmp_path, env=env,
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 7, result.stdout + result.stderr
    assert json.loads(result.stdout)["home"] == str(new_user_home / ".hermes")
    assert json.loads(result.stdout)["value"] == "from-second-user"
    assert published_home != new_user_home / ".hermes"


@pytest.mark.parametrize("publisher", [
    pytest.param("boot", marks=pytest.mark.platforms("posix")),
    pytest.param("repair", marks=pytest.mark.platforms("posix")),
    pytest.param("native", marks=pytest.mark.platforms("windows")),
    pytest.param("cmd", marks=pytest.mark.platforms("windows")),
])
def test_profile_publication_preserves_shared_launcher_default_home(tmp_path, monkeypatch, publisher):
    from hermes_cli import boot_bootstrap, post_update

    repo, home, interpreter = fixture_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(repo))
    profile = home / "profiles" / "coder"
    profile.mkdir(parents=True)
    (home / "active_profile").write_text("default\n", encoding="utf-8")
    (repo / "install-stamp.json").write_text(json.dumps({
        "commit": "abcdef012345", "updateMechanism": "git", "runtimeDir": str(home / "tools"),
    }), encoding="utf-8")
    select_generation(repo, "shared", "ready")
    out = tmp_path / ".local" / "bin"
    if publisher == "cmd":
        monkeypatch.setattr(_launchers, "_load_script_maker", lambda: None)
    launchers = [Path(p) for p in _launchers.ensure_install_launchers(repo, out)]
    assert len(launchers) == len(_launchers.ENTRY_POINTS)
    if publisher == "cmd":
        assert all(launcher.suffix == ".cmd" for launcher in launchers)

    def assert_home(override, expected):
        env = dict(os.environ)
        env.pop("HERMES_HOME", None)
        if override is not None:
            env["HERMES_HOME"] = str(override)
        for launcher in launchers:
            result = subprocess.run([str(launcher)], cwd=tmp_path, env=env,
                                    capture_output=True, text=True, encoding="utf-8", timeout=30)
            assert result.returncode == 7, result.stdout + result.stderr
            receipt = json.loads(result.stdout)
            assert Path(receipt["home"]) == expected
            assert receipt["value"] == "ready"
            assert Path(receipt["exe"]).samefile(interpreter)

    assert_home(None, home)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    if publisher == "boot":
        # Keep the real per-profile boot gate and exposure step, not unrelated
        # migrations. A named profile's first boot republishes the shared files.
        monkeypatch.setattr(post_update, "BOOT_HOME_STEPS", tuple(
            step for step in post_update.BOOT_HOME_STEPS if step[0] == "expose_cli"
        ))
        result = boot_bootstrap.run_boot_bootstrap(repo)
        assert result["home"]["expose_cli"]["ok"], result
        assert boot_bootstrap.record_path(repo).is_file()
    elif publisher == "repair":
        assert _launchers.expose_cli(repo, create=False)["ok"]
    else:
        # Windows exposure is installer-owned. Both transports use this writer.
        assert _launchers.ensure_install_launchers(repo, out)
    assert_home(None, home)
    assert_home(profile, profile)
    other_home = tmp_path / "explicit custom home"
    other_home.mkdir()
    # A custom home is its own dependency root: it sees only generations committed there.
    monkeypatch.setenv("HERMES_HOME", str(other_home))
    select_generation(repo, "shared", "ready")
    assert_home(other_home, other_home)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("corruption", ["bom", "crlf"])
def test_posix_materializer_publishes_only_executable_shell_launchers(tmp_path, monkeypatch, corruption):
    repo, _home, _interpreter = fixture_tree(tmp_path, monkeypatch)
    out = tmp_path / "bin"
    out.mkdir()
    launchers = [Path(p) for p in _launchers.ensure_install_launchers(repo, out)]
    assert {p.name for p in launchers} == set(_launchers.ENTRY_POINTS)
    assert all(os.access(p, os.X_OK) for p in launchers)
    assert set(out.iterdir()) == set(launchers)
    local = repo / ".hermes" / "bin"
    assert {p.name for p in local.iterdir()} == set(_launchers.ENTRY_POINTS)
    launcher = local / "hermes"
    expected = launcher.read_bytes()
    launcher.write_bytes(b"\xef\xbb\xbf" + expected if corruption == "bom" else expected.replace(b"\n", b"\r\n"))
    assert _launchers.ensure_install_launchers(repo, out)
    # Shell executables need exact bytes: neither a BOM before #! nor CRLF is
    # interchangeable with the generated script, even if text decoding agrees.
    assert launcher.read_bytes() == expected
    result = subprocess.run([str(out / "hermes"), "--print-runtime-command"],
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    assert Path(json.loads(result.stdout)[0]).samefile(_interpreter)
    before = launcher.stat().st_mtime_ns
    assert _launchers.ensure_install_launchers(repo, out)
    assert launcher.stat().st_mtime_ns == before


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


@pytest.mark.platforms("posix")
def test_boot_migrates_legacy_conveniences_to_selected_runtime(tmp_path, monkeypatch):
    repo, home, interpreter = fixture_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(repo))
    select_generation(repo, 'current', 'migrated')
    out = home / ".local" / "bin"
    out.mkdir(parents=True)
    # Old venv and sibling-ACP wrappers, with an unrelated command sharing bin.
    (out / "hermes").write_text(f'#!/bin/sh\nexec "{repo}/venv/bin/python" "{repo}/hermes" "$@"\n', encoding="utf-8")
    (out / "hermes-acp").write_text(
        '#!/usr/bin/env bash\n# Hermes Agent — ACP launcher (written by `hermes update`).\n'
        f'exec "{out}/hermes" acp "$@"\n', encoding="utf-8")
    foreign = f'#!/bin/sh\n# user note about {repo}\nexit 19\n'
    (out / "hermes-agent").write_text(foreign, encoding="utf-8")

    result = _launchers.expose_cli()
    assert result["ok"], result
    assert set(result["written"]) == {"hermes", "hermes-acp"}
    for name in ("hermes", "hermes-acp"):
        run = subprocess.run([str(out / name), "quoted argument"], cwd=tmp_path,
                             capture_output=True, text=True, timeout=30, encoding="utf-8")
        assert run.returncode == 7, run.stderr
        receipt = json.loads(run.stdout)
        assert receipt["value"] == "migrated"
        assert receipt["argv"] == ["quoted argument"]
        assert Path(receipt["exe"]).samefile(interpreter)
    assert (out / "hermes-agent").read_text(encoding="utf-8-sig") == foreign
    before = {p: p.stat().st_mtime_ns for p in out.iterdir()}
    assert _launchers.expose_cli()["written"] == []
    assert before == {p: p.stat().st_mtime_ns for p in out.iterdir()}


def _command_survives_generation_collection(tmp_path, monkeypatch, surface):
    from hermes_cli.runtime_state import collect_generations

    repo, home, interpreter = fixture_tree(tmp_path, monkeypatch)
    out = tmp_path / "bin"
    out.mkdir()
    _launchers.ensure_install_launchers(repo, out)
    launcher = next(path for path in out.iterdir() if path.stem == "hermes")
    args = ["café ' quoted", "", "$HOME; not a shell"]
    command = []
    for value in ("old", "new"):
        selected = select_generation(repo, value, value)
        (selected.parent / ".lease-managed").touch()
        if value == "old":
            if surface == "legacy":
                command = [sys.executable, "-I", str(repo / "scripts/hermes-gateway"), "--help"]
                args = ["gateway", "--help"]
            elif surface == "ssh":
                from hermes_cli.windows_ssh_runtime import _resolve_direct_command
                command = [*_resolve_direct_command(str(launcher)), *args]
            elif surface == "published":
                result = subprocess.run([str(launcher), "--print-runtime-command", "--", *args],
                                        capture_output=True, text=True, timeout=30, encoding="utf-8")
                assert result.returncode == 0, result.stderr
                command = json.loads(result.stdout)
            else:
                from hermes_cli import gateway
                monkeypatch.setattr(gateway, "PROJECT_ROOT", repo)
                if surface == "launchd":
                    import plistlib
                    from tests.hermes_cli.test_gateway_service import _osascript_exec_argv
                    unit = gateway.generate_launchd_plist()
                    # The job runs through osascript (#71206); exec the child it would spawn, minus the
                    # `>> log 2>> log` tail that only means something to the shell.
                    command = _osascript_exec_argv(plistlib.loads(unit.encode())["ProgramArguments"])[:-4]
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
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30, encoding="utf-8")
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
@pytest.mark.parametrize("launcher_form", ["native", "cmd", "native-with-maker"])
def test_running_source_launcher_can_republish_itself(tmp_path, monkeypatch, launcher_form):
    repo, _home, interpreter = fixture_tree(tmp_path, monkeypatch)
    selected = select_generation(repo, "selected", "ready")
    if launcher_form == "native-with-maker":
        import distlib

        shutil.copytree(Path(distlib.__file__).parent, site_packages(selected) / "distlib")
    entry = repo / "hermes_cli/main.py"
    entry.write_text(
        "from pathlib import Path\n"
        "from hermes_cli._launchers import ensure_install_launchers, ENTRY_POINTS\n"
        "def main():\n"
        "    root = Path(__file__).resolve().parents[1]\n"
        "    written = ensure_install_launchers(root, root / '.hermes/bin')\n"
        "    print('published', len(written), len(ENTRY_POINTS), flush=True)\n"
        "    return 0 if len(written) == len(ENTRY_POINTS) else 1\n",
        encoding="utf-8",
    )
    out = repo / ".hermes/bin"
    if launcher_form == "cmd":
        monkeypatch.setattr(_launchers, "_load_script_maker", lambda: None)
    launchers = _launchers.ensure_install_launchers(repo, out)
    assert len(launchers) == len(_launchers.ENTRY_POINTS)
    command = next(Path(p) for p in launchers if Path(p).stem == "hermes")
    assert command.suffix == (".cmd" if launcher_form == "cmd" else ".exe")
    result = subprocess.run([str(command)], cwd=tmp_path, capture_output=True,
                            text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "published 2 2" in result.stdout
    assert _launchers.ensure_install_launchers(repo, out)
    selected_python = _launchers.resolve_store_python(repo)
    assert selected_python is not None and selected_python.samefile(interpreter)


@pytest.mark.platforms("windows")
def test_repin_without_distlib_retires_stale_native_launcher(tmp_path, monkeypatch):
    repo, home, _interpreter = fixture_tree(tmp_path, monkeypatch)
    local = repo / ".hermes/bin"
    original = Path(_launchers.ensure_install_launchers(repo, local)[0])
    assert original.suffix == ".exe"
    new_python = home / "tools" / "repinned" / "python.exe"
    new_python.parent.mkdir()
    new_python.touch()
    (home / "tools/facts.json").write_text(
        json.dumps({"packages": {"python": {"entry": "repinned"}}}), encoding="utf-8")
    monkeypatch.setattr(_launchers, "_load_script_maker", lambda: None)
    result = _launchers.stage_launcher("hermes", repo, local)
    assert result == local / "hermes.cmd"
    assert not original.exists()  # cmd.exe must not run the old exe first
    assert result is not None and str(new_python) in result.read_text(encoding="utf-8-sig")


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
                            text=True, timeout=30, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    assert Path(json.loads(result.stdout)[0]).samefile(interpreter)


def test_dashboard_action_boots_selected_dependencies(tmp_path, monkeypatch):
    from hermes_cli import web_server, web_server_gateway

    repo, home, _ = fixture_tree(tmp_path, monkeypatch)
    select_generation(repo, 'current', 'selected')
    monkeypatch.setattr(web_server, "PROJECT_ROOT", repo)
    monkeypatch.setattr(web_server_gateway, "_ACTION_LOG_DIR", home / "logs")
    proc = web_server_gateway._spawn_hermes_action(["--version"], "gateway-restart")
    try:
        assert proc.wait(timeout=30) == 7
        log = (home / "logs" / web_server_gateway._ACTION_LOG_FILES["gateway-restart"]).read_text(encoding="utf-8-sig")
        assert json.loads(log.splitlines()[-1])["value"] == "selected"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)


def test_pre_pm_base_dependencies_activate_only_at_boot(tmp_path, monkeypatch):
    # The in-tree pre-PM venv is never activated; only a sealed payload's own environment is.
    repo, home, _ = fixture_tree(tmp_path, monkeypatch)
    environment = repo.parent / "payload-deps"
    (repo.parent / "manifest.json").write_text(
        json.dumps({"repo": repo.name, "venv": environment.name,
                    "store": (home / "tools").relative_to(repo.parent).as_posix()}), encoding="utf-8")
    site = site_packages(environment)
    site.mkdir(parents=True)
    editable = tmp_path / "editable"
    editable.mkdir()
    (editable / "selected_probe.py").write_text("VALUE = 'base-pth'\n", encoding="utf-8")
    (site / "member.pth").write_text(str(editable) + "\n", encoding="utf-8")
    command = _launchers.runtime_command(repo)
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30, encoding="utf-8")
    assert result.returncode == 7, result.stderr
    assert json.loads(result.stdout)["value"] == "base-pth"


def test_external_interpreter_keeps_its_owned_dependencies(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "empty-store"))
    command = _launchers.runtime_command(ROOT, code="import ruamel.yaml; print('external-runtime-ready')")
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "external-runtime-ready"


@pytest.mark.platforms("posix")
@pytest.mark.spawns_gateway_lookalike
def test_service_survives_python_tool_replacement(tmp_path, monkeypatch):
    from hermes_cli import gateway

    repo, home, interpreter = fixture_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(gateway, "PROJECT_ROOT", repo)
    select_generation(repo, "shared", "ready")
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
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30, encoding="utf-8")
    assert result.returncode == 7, result.stderr
    assert json.loads(result.stdout)["value"] == "ready"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("create", [True, False])
def test_sync_migrates_old_store_wrapper_before_python_collection(tmp_path, monkeypatch, create):
    import builtins

    from hermes_cli.venv_sync import publish_launchers

    repo, home, interpreter = fixture_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(Path, "home", lambda: home)
    select_generation(repo, "shared", "ready")
    out = home / ".local/bin"
    out.mkdir(parents=True)
    if not create:
        original_import = builtins.__import__

        def without_config(name, *args, **kwargs):
            assert name != "hermes_cli.config", "bootstrap publication imported application config"
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", without_config)
    store = home / "tools"
    for version in ("python-A", "python-B"):
        python = store / version / "bin/python3"
        python.parent.mkdir(parents=True)
        python.symlink_to(interpreter)
        (store / "facts.json").write_text(json.dumps({"schema": 1, "packages": {"python": {"entry": version}}}), encoding="utf-8")
        if version == "python-A":
            _launchers.mint_launcher("hermes", repo, out, python, None)
        else:
            if create:
                publish_launchers(repo)
            else:
                publish_launchers(repo, create=False)
                assert set(out.iterdir()) == {out / "hermes"}
                assert not (home / "bin").exists()
                assert not (home / "config.yaml").exists()
                assert not (home / "skills").exists()
    shutil.rmtree(store / "python-A")
    result = subprocess.run([str(out / "hermes")], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30, encoding="utf-8")
    assert result.returncode == 7, result.stderr
    assert json.loads(result.stdout)["value"] == "ready"
    assert Path(json.loads(result.stdout)["exe"]) == store / "python-B/bin/python3"


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
