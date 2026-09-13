"""Startup consumes one real PM verdict without activating a partial store."""

import importlib
import os
import sys
from pathlib import Path

import pytest

import pm
from pm import paths, registry
from pm.lock import Lockfile
from pm.packages import BinaryPackage
from tests.pm._fixtures import make_tar, served as served


@pytest.fixture
def checked_store(tmp_path, monkeypatch, served):
    engine = importlib.import_module("pm.ensure")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "store"))
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path / "repo")
    monkeypatch.setattr(paths, "lockfile_path", lambda: tmp_path / "lock.json")
    monkeypatch.setattr(registry, "_packages", {})
    docroot, url = served
    lock = Lockfile(paths.lockfile_path())
    binaries = []
    for name in ("first", "second"):
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

    def counted_check():
        problems = real_check()
        checks.append(problems)
        return problems

    monkeypatch.setattr(engine, "check", counted_check)
    monkeypatch.setattr(pm, "check", counted_check)
    return binaries, checks


def test_activate_returns_live_verdict_without_partial_environment(checked_store, monkeypatch):
    binaries, checks = checked_store
    original_path = os.environ["PATH"]
    monkeypatch.setenv("PATH", original_path)

    assert pm.activate() == []
    assert checks == [[]]
    active_path = os.environ["PATH"].split(os.pathsep)
    assert all(str(binary.parent) in active_path for binary in binaries)

    # A second invocation must observe fresh damage, even after healthy activation.
    monkeypatch.setenv("PATH", original_path)
    binaries[1].unlink()
    before = dict(os.environ)
    problems = pm.activate()
    assert problems == ["second: not installed or outdated"]
    assert checks == [[], problems]
    assert dict(os.environ) == before


@pytest.mark.parametrize("surface", ["gateway", "cli"])
@pytest.mark.parametrize("damaged", [False, True])
def test_startup_uses_one_verdict(checked_store, monkeypatch, capsys, caplog, surface, damaged):
    binaries, checks = checked_store
    if surface == "gateway":
        from gateway.run import _run_pm_startup
        start = _run_pm_startup
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
        assert os.environ["PATH"] == original_path
        assert "install out of sync (second: not installed or outdated)" in diagnostics
    else:
        assert all(str(binary.parent) in os.environ["PATH"].split(os.pathsep) for binary in binaries)
        assert "install out of sync" not in diagnostics