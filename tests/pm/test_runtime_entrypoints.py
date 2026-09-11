"""Bootstrap entrypoints must hand dependency operations to PM's own interpreter."""
import json
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
    import pm.client as client
    from pm import paths
    from scripts.ci import setup_toolchain

    project = tmp_path / "source"
    project.mkdir()
    (project / "pyproject.toml").write_text('[project]\nname="ci-proof"\nversion="1"\n[project.optional-dependencies]\ndev=[]\n')
    monkeypatch.setattr(paths, "repo_root", lambda: project)
    monkeypatch.setattr(client, "is_runtime", lambda: False)
    calls = []

    def request(operation, arguments, **kwargs):
        calls.append((operation, arguments))
        return str(tmp_path / "test-environment/bin/python") if operation == "build_environment" else None

    monkeypatch.setattr(client, "_request", request)
    monkeypatch.setattr(setup_toolchain, "python3_alias", lambda _: None)
    monkeypatch.setattr(setup_toolchain, "file_commands", lambda *args: None)
    monkeypatch.setattr(setup_toolchain, "add_path", lambda *args: None)
    setup_toolchain.dependencies(SimpleNamespace(home=tmp_path, extras=["dev"], toolchain="all"))
    assert [operation for operation, _ in calls] == ["check_project_lock", "build_environment"]
    assert calls[0][1]["source"] == str(project)
    assert calls[1][1]["extras"] == ["dev"]
    assert calls[1][1]["groups"] == ["test"]
    assert all(arguments["explicit"] for _, arguments in calls)


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
