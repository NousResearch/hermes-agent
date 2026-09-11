"""Bootstrap entrypoints must hand dependency operations to PM's own interpreter."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_cli_dispatches_before_calling_engine(monkeypatch):
    from pm import cli, runtime

    calls = []
    monkeypatch.setattr(runtime, "is_runtime", lambda: False)
    monkeypatch.setattr(runtime, "run_cli", lambda argv: calls.append(argv) or 19)
    monkeypatch.setattr(cli, "cmd_install", lambda args: pytest.fail("caller imported the engine"))
    argv = ["install", "venv"]
    assert cli.main(argv) == 19
    assert calls == [argv]


def test_cli_runtime_executes_without_redispatch(monkeypatch):
    from pm import cli, runtime

    monkeypatch.setattr(runtime, "is_runtime", lambda: True)
    monkeypatch.setattr(runtime, "run_cli", lambda argv: pytest.fail("recursive PM dispatch"))
    monkeypatch.setattr(cli, "cmd_install", lambda args: 7 if args.names == ["venv"] else 1)
    assert cli.main(["install", "venv"]) == 7


def test_ci_dependency_phase_uses_isolated_runtime(tmp_path, monkeypatch):
    from pm import runtime
    from scripts.ci import setup_toolchain
    import subprocess

    monkeypatch.setattr("pm._uv._toolchain", lambda: pytest.fail("dependency work ran in bootstrap Python"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    calls = []
    python = tmp_path / "pm-runtime" / "python"
    monkeypatch.setattr(runtime, "is_runtime", lambda: False)
    monkeypatch.setattr(runtime, "runtime_python", lambda: python)
    monkeypatch.setattr(subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "app-deps"))
    args = SimpleNamespace(home=tmp_path, extras=["dev"], toolchain="all")
    setup_toolchain.dependencies(args)
    command, options = calls.pop()
    assert command[:3] == [str(python), "-I", "-B"]
    assert Path(command[3]).name == "setup_toolchain.py"
    assert command[4:] == ["dependencies", "--home", str(tmp_path), "--toolchain", "all", "--extras", json.dumps(["dev"])]
    assert "PYTHONPATH" not in options["env"]
    assert options["check"] is True
    assert calls == []


@pytest.mark.parametrize("distribution", ["nix", "docker"])
def test_packaged_runtime_uses_explicit_stamp_without_tool_downloads(tmp_path, monkeypatch, distribution):
    from pm import runtime, paths

    monkeypatch.setattr("pm._uv._toolchain", lambda **kw: pytest.fail("packaged PM tried to download tools"))
    project = tmp_path / "app"
    project.mkdir()
    monkeypatch.setattr(paths, "repo_root", lambda: project)
    install_root = tmp_path / "package"
    install_root.mkdir()
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(install_root))
    pm = install_root / "pm-runtime"
    site = pm / "site-packages"
    site.mkdir(parents=True)
    python = install_root / "python"
    python.touch()
    (pm / "pm-runtime.json").write_text(json.dumps({"python": str(python), "sitePackages": str(site)}))
    stamp = {"distribution": distribution, "pmRuntime": str(pm)}
    (install_root / "install-stamp.json").write_text(json.dumps(stamp))
    command = runtime.runtime_command(project / "worker.py", ["argument"])
    assert command[:4] == [str(python), "-I", "-S", "-B"]
    assert str(site) in command
    assert command[-2:] == [str(project / "worker.py"), "argument"]
    (pm / "pm-runtime.json").unlink()
    from pm.package import InstallError
    with pytest.raises(InstallError, match="PM runtime"):
        runtime.runtime_command(project / "worker.py")
