"""Startup consumes one real PM verdict; activation never serves a partial toolchain."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

import pm
from pm import paths, registry
from pm.lock import Lockfile
from pm.packages import BinaryPackage
from tests.pm._fixtures import make_tar, served as served  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import


@pytest.fixture
def checked_store(tmp_path, monkeypatch, served):
    engine = importlib.import_module("pm.install")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "store"))
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path / "repo")
    monkeypatch.setattr(paths, "lockfile_path", lambda: tmp_path / "lock.json")
    monkeypatch.setattr(registry, "_packages", {})
    docroot, url = served
    lock = Lockfile(paths.lockfile_path())
    binaries = []
    for name in ("first", "second", "third"):
        package = BinaryPackage()
        package.name = name
        package.probe_version = False
        package.binary_rel = {"posix": f"bin/{name}", "win32": f"bin/{name}"}
        registry._packages[name] = package
        archive, digest = make_tar(docroot, f"{name}.tar.gz", {f"bin/{name}": "fixture tool"})
        lock.set_pin(name, "1.0", {"any": {"url": f"{url}/{archive}", "sha256": digest}})
        lock.save()
        engine.ensure(name, explicit=True)
        installed = pm.installed_package(name)
        assert installed is not None and installed.binary is not None
        binaries.append(installed.binary)

    checks = []
    real_check = engine.check

    def counted_check(**kwargs):
        problems = real_check(**kwargs)
        checks.append(problems)
        return problems

    monkeypatch.setattr(engine, "check", counted_check)
    monkeypatch.setattr(pm, "check", counted_check)
    return binaries, checks


def test_activate_moves_store_dirs_ahead_of_user_dirs(checked_store, monkeypatch):
    binaries, checks = checked_store
    store = [str(binary.parent) for binary in binaries]
    user_dir = os.path.join(os.sep, "user", "bin")
    # A store dir already on PATH behind a user dir must still end up in front of it.
    monkeypatch.setenv("PATH", os.pathsep.join([user_dir, store[1]]))

    assert pm.activate() == []
    assert checks == [[]]
    assert os.environ["PATH"].split(os.pathsep) == [*store, user_dir]


def test_activate_keeps_the_path_this_process_inherited(checked_store, monkeypatch):
    environments = importlib.import_module("pm.environments")
    # As in a process whose activate_dependencies() returned before putting anything first.
    monkeypatch.setattr(environments, "_inherited_path", None)
    inherited = os.path.join(os.sep, "user", "bin")
    monkeypatch.setenv("PATH", inherited)

    assert pm.activate() == []

    assert os.environ["PATH"] != inherited
    assert environments.inherited_path() == inherited


def test_drift_activates_intact_chains_and_reports_the_live_verdict(checked_store, monkeypatch):
    binaries, checks = checked_store
    user_dir = os.path.join(os.sep, "user", "bin")
    monkeypatch.setattr(registry._packages["third"], "deps", ("second",))
    assert pm.activate() == []

    # A second invocation must observe fresh damage, even after healthy activation.
    monkeypatch.setenv("PATH", user_dir)
    binaries[1].unlink()
    problems = pm.activate()
    assert problems == ["second: not installed or outdated"]
    assert checks == [[], problems]
    # Intact first stays ahead of the user's copies; damaged second and third, which
    # depends on it, stay off PATH so no toolchain is served half from the store.
    assert os.environ["PATH"].split(os.pathsep) == [str(binaries[0].parent), user_dir]


def test_managed_startup_can_run_before_yaml_is_available(tmp_path, monkeypatch):
    from hermes_cli.venv_sync import check_runtime
    from pm.environments import install_state_dir, runtime_facts_path
    from pm.install import venv_is_current
    from pm.lock import Facts
    from pm.packages import Venv

    source = paths.repo_root()
    root = tmp_path / "checkout"
    root.mkdir()
    (root / ".git").mkdir()
    (root / "install-stamp.json").write_text('{"updateMechanism":"self"}', encoding="utf-8")
    (root / "uv.lock").write_text("", encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("plugins: {}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "store"))
    monkeypatch.delenv("HERMES_INSTALL_STATE_DIR", raising=False)
    monkeypatch.setattr(paths, "repo_root", lambda: root)
    monkeypatch.setattr(paths, "lockfile_path", lambda: root / "lock.json")
    venv = install_state_dir(root) / "environments" / "current" / "venv"
    venv.mkdir(parents=True)
    (venv / "pyvenv.cfg").touch()
    facts = Facts(runtime_facts_path(root))
    facts.record_state("venv", Venv(root).expected_stamp([]), [], environment=venv)
    assert venv_is_current()
    assert check_runtime(root) is None

    probe = """
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from hermes_cli.venv_sync import check_runtime
from pm import paths

root = Path(sys.argv[2])
paths.repo_root = lambda: root
paths.lockfile_path = lambda: root / "lock.json"
assert check_runtime(root) is None
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", probe, str(source), str(root)],
        capture_output=True, text=True, timeout=15, check=False,
    )
    assert result.returncode == 0, result.stderr

    facts.record_state("venv", "stale", [], environment=venv)
    problem = check_runtime(root)
    assert problem is not None and "venv: out of sync with uv.lock" in problem


@pytest.mark.parametrize("surface", ["gateway", "cli"])
@pytest.mark.parametrize("damaged", [False, True])
def test_startup_uses_one_verdict(checked_store, monkeypatch, capsys, caplog, surface, damaged):
    binaries, checks = checked_store
    if surface == "gateway":
        from gateway import run

        async def started(*args, **kwargs):
            return True

        monkeypatch.setattr(run, "start_gateway", started)
        monkeypatch.setattr(run, "_exit_after_graceful_shutdown", lambda code: None)
        monkeypatch.setattr("hermes_cli.boot_bootstrap.maybe_run_boot_bootstrap", lambda _root: None)
        monkeypatch.setattr(sys, "argv", ["gateway"])
        start = run.main
    else:
        from hermes_cli import main

        # Stop at --help: exercise the actual startup block, not an agent session.
        monkeypatch.setattr("hermes_cli.boot_bootstrap.maybe_run_boot_bootstrap", lambda _root: None)
        for name in ("_set_process_title", "_advertise_agent_env",
                     "_sweep_stale_bytecode_if_checkout_changed",
                     "_try_termux_fast_tui_launch", "_try_termux_fast_cli_launch",
                     "_try_fast_serve_launch", "_try_fast_chat_launch"):
            monkeypatch.setattr(main, name, lambda: None)
        monkeypatch.setattr(sys, "argv", ["hermes", "--help"])

        def start():
            with pytest.raises(SystemExit) as exit_info:
                main.main()
            assert exit_info.value.code == 0

    if damaged:
        binaries[1].unlink()
    original_path = os.environ["PATH"]
    monkeypatch.setenv("PATH", original_path)
    start()

    expected = ["second: not installed or outdated"] if damaged else []
    assert checks == [expected]
    diagnostics = capsys.readouterr().err + caplog.text
    if damaged:
        assert os.environ["PATH"].split(os.pathsep)[0] == str(binaries[0].parent)
        assert str(binaries[1].parent) not in os.environ["PATH"].split(os.pathsep)
        assert "install out of sync (second: not installed or outdated)" in diagnostics
    else:
        assert all(str(binary.parent) in os.environ["PATH"].split(os.pathsep) for binary in binaries)
        assert "install out of sync" not in diagnostics
