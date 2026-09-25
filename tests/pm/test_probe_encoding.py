"""PM probe pipes retain diagnostics on legacy locales (part of #122470)."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from pm.package import InstallError


@pytest.mark.parametrize("boundary", ["validate", "stage", "marker", "version", "npm"])
@pytest.mark.parametrize("failed", [False, True])
def test_probe_decoding_preserves_result_and_diagnostics(tmp_path, monkeypatch, boundary, failed):
    from pm import extras, runtime, runtime_stage, plugin_eviction, packages
    import pm.environment
    import pm.install
    import pm._uv

    # Exercise real Popen/TextIOWrapper pipes, not a mock that merely checks kwargs.
    # Only substitute the command: never install packages or invoke a real toolchain.
    run = subprocess.run
    stderr = "诊断 — invalid byte: ".encode("utf-8") + b"\xff\n"
    stdout = b"3.14.7\n" if boundary == "version" else b"0\n"
    code = f"import os; os.write(1, {stdout!r}); os.write(2, {stderr!r}); raise SystemExit({int(failed)})"
    seen = []

    def run_fixture(command, **kwargs):
        result = run([sys.executable, "-I", "-c", code], **kwargs)
        seen.append(result)
        return result

    monkeypatch.setattr(subprocess, "_text_encoding", lambda: "cp936")
    monkeypatch.setattr(subprocess, "run", run_fixture)
    monkeypatch.setattr(runtime, "runtime_environment", lambda: dict(os.environ))
    python = Path(sys.executable)
    if boundary == "validate":
        value = runtime._validate(python, dict(os.environ))
        assert value == (stderr.decode("utf-8", "replace").strip() if failed else "")
    elif boundary == "marker":
        monkeypatch.setattr(runtime, "runtime_command", lambda *a: [str(python)])
        assert extras._evaluate_in_runtime("sys_platform == 'never'", {}) is failed
    elif boundary == "version":
        monkeypatch.setattr(pm._uv, "_toolchain", lambda **kw: (python, python))
        if failed:
            with pytest.raises(subprocess.CalledProcessError) as error:
                plugin_eviction._interpreter_version()
            assert error.value.stderr == stderr.decode("utf-8", "replace")
            return
        assert plugin_eviction._interpreter_version() == "3.14.7"
    else:
        if boundary == "stage":
            project = tmp_path / "project"
            project.mkdir()
            for name in ("pyproject.toml", "uv.lock"):
                (project / name).write_text("", encoding="utf-8")
            environment = SimpleNamespace(executable=python, create=lambda: None,
                                          sync=lambda *a, **kw: None)
            monkeypatch.setattr(pm.environment, "PythonEnvironment", lambda **kw: environment)
            invoke = lambda: runtime_stage.stage_runtime(python, python, tmp_path / "runtime",
                                                         project=project, cache=tmp_path / "cache")
        else:
            node = tmp_path / "node"
            node.mkdir()
            binary = node / ("node.exe" if os.name == "nt" else "bin/node")
            binary.parent.mkdir(exist_ok=True)
            binary.touch()
            target = "win32-x64" if os.name == "nt" else "linux-x64"
            cli = node / ("node_modules/npm/bin/npm-cli.js" if os.name == "nt"
                          else "lib/node_modules/npm/bin/npm-cli.js")
            cli.parent.mkdir(parents=True)
            cli.touch()
            store = SimpleNamespace(entry=lambda key: node)
            monkeypatch.setattr(pm.install, "_lockfile", lambda: None)
            monkeypatch.setattr(pm.install, "_installed_location", lambda *a: ({"node": {"entry": "node"}}, store))
            invoke = lambda: packages.Npm().unpack(tmp_path / "npm.tgz", tmp_path / "staged", target)
        if failed:
            with pytest.raises(InstallError) as error:
                invoke()
            assert stderr.decode("utf-8", "replace").strip() in str(error.value)
        else:
            invoke()
    assert seen and seen[0].stderr == stderr.decode("utf-8", "replace")


@pytest.mark.parametrize("failure", [OSError("unavailable"), subprocess.TimeoutExpired("probe", 30)])
def test_probe_fallbacks_preserve_spawn_failures(monkeypatch, failure):
    from pm import extras, runtime

    def fail(*a, **kw):
        raise failure

    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(runtime, "runtime_command", lambda *a: ["fixture"])
    monkeypatch.setattr(runtime, "runtime_environment", lambda: {})
    assert runtime._validate(Path("fixture"), {}) == str(failure)
    assert extras._evaluate_in_runtime("invalid", {}) is True
