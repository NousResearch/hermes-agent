"""Offline plugin command fixtures; only tool acquisition is substituted.

The actual client, worker, resolver, lock, publication callbacks and receipts
run unchanged. The prepared test interpreter hosts PM in a separate process;
fresh selected interpreters never inherit its site-packages.
"""
from __future__ import annotations

import json
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


import pytest
import hermes_yaml as yaml

from tests.pm._fixtures import _wheel, isolated_python as isolated_python


def worker_command(worker: Path, uv: str, python: str, *, prelude: str = "", runtime_python: str | None = None) -> list[str]:
    script = (
        "import runpy, sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(worker.parent.parent)!r}); import pm._uv; "
        f"pm._uv._toolchain = lambda **kwargs: (Path({uv!r}), Path({python!r}))\n"
        + prelude + "\n"
        f"runpy.run_path({str(worker)!r}, run_name='__main__')"
    )
    return [runtime_python or python, "-I", "-B", "-c", script]


def git(repo: Path, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if k not in {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"}}
    env.update(GIT_AUTHOR_NAME="fixture", GIT_AUTHOR_EMAIL="fixture@example.invalid",
               GIT_COMMITTER_NAME="fixture", GIT_COMMITTER_EMAIL="fixture@example.invalid")
    result = subprocess.run(["git", *args], cwd=repo, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def version(repo: Path, *, name="plugin-worker-proof", surface="python_dependencies", suffix="yaml", pin="1.0", capabilities=(), dependency="plugin-proof-dep") -> str:
    manifest: dict = {"name": name, "version": pin}
    if capabilities:
        manifest["capabilities"] = list(capabilities)
    specs = [f"{dependency}=={pin}"]
    if surface and surface != "pyproject":
        manifest[surface] = specs
    (repo / f"plugin.{suffix}").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (repo / "__init__.py").write_text(
        f"import {dependency.replace('-', '_')} as plugin_proof_dep\nVERSION = {pin!r}\n"
        "def register(ctx):\n    return (VERSION, plugin_proof_dep.__version__)\n", encoding="utf-8")
    if surface == "pyproject":
        (repo / "pyproject.toml").write_text(
            f'[project]\nname={json.dumps(name)}\nversion="{pin}"\nrequires-python=">=3.11"\n'
            f'dependencies={json.dumps(specs)}\n[tool.uv]\npackage=false\n', encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", f"fixture {pin}")
    return git(repo, "rev-parse", "HEAD")


class PluginWorld:
    runtime_python: str

    def __init__(self, root: Path):
        self.root = root
        self.home = root / "home"
        self.core = root / "core"

    def origin(self, **kwargs) -> tuple[Path, str]:
        repo = self.root / (kwargs.get("name", "plugin-worker-proof") + "-origin")
        repo.mkdir()
        git(repo, "init", "-qb", "main")
        return repo, version(repo, **kwargs)

    def command(self, action: str, **kwargs) -> None:
        from hermes_cli.plugins_cmd import plugins_command
        from hermes_cli.subcommands.plugins import build_plugins_parser

        parser = argparse.ArgumentParser()
        build_plugins_parser(parser.add_subparsers(), cmd_plugins=plugins_command)
        positional = {"install": ("identifier",), "enable": ("name",), "disable": ("name",),
                      "pack": ("pack_action", "source")}[action]
        argv = ["plugins", action, *(str(kwargs.pop(key)) for key in positional)]
        for key, value in kwargs.items():
            if value is True:
                argv.append("--" + key.replace("_", "-"))
            elif value is not None and value is not False:
                argv.extend(["--" + key.replace("_", "-"), str(value)])
        plugins_command(parser.parse_args(argv))

    def selected(self) -> Path:
        from hermes_cli.runtime_paths import selected_venv
        return selected_venv(self.core)

    def enabled(self) -> list[str]:
        return yaml.safe_load((self.home / "config.yaml").read_text(encoding="utf-8"))["plugins"]["enabled"]

    def imports(self, name="plugin-worker-proof", pin="1.0") -> None:
        selected = self.selected()
        python = selected / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        code = (
            "import importlib.util, pathlib; import plugin_core_dep, plugin_proof_dep; "
            f"assert pathlib.Path(plugin_core_dep.__file__).is_relative_to({str(selected)!r}); "
            f"assert pathlib.Path(plugin_proof_dep.__file__).is_relative_to({str(selected)!r}); "
            f"spec = importlib.util.spec_from_file_location('loaded_plugin', {str(self.home / 'plugins' / name / '__init__.py')!r}); "
            "plugin = importlib.util.module_from_spec(spec); spec.loader.exec_module(plugin); "
            f"assert plugin.register(None) == ({pin!r}, {pin!r})"
        )
        result = subprocess.run([str(python), "-I", "-B", "-c", code], cwd=self.root,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr


@pytest.fixture
def plugin_world(tmp_path, monkeypatch, isolated_python):
    from pm import client, paths

    world = PluginWorld(tmp_path)
    uv = shutil.which("uv")
    assert uv, "real uv is required for the plugin lifecycle"
    for key in ("HERMES_MANAGED_MODE", "HERMES_INSTALL_ROOT", "VIRTUAL_ENV"):
        monkeypatch.delenv(key, raising=False)
    for key, value in {"HOME": tmp_path, "USERPROFILE": tmp_path, "HERMES_HOME": world.home,
                       "HERMES_RUNTIME_DIR": tmp_path / "store", "UV_CACHE_DIR": tmp_path / "cache",
                       "XDG_CONFIG_HOME": tmp_path / "config", "XDG_CONFIG_DIRS": tmp_path / "config",
                       "HERMES_DISABLE_LAZY_INSTALLS": "1", "UV_OFFLINE": "1"}.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(paths, "repo_root", lambda: world.core)
    monkeypatch.setattr(paths, "lockfile_path", lambda: tmp_path / "lock.json")
    world.runtime_python = str(isolated_python)
    monkeypatch.setattr(client, "runtime_command", lambda worker, **kwargs:
                        worker_command(worker, uv, sys.executable, runtime_python=world.runtime_python))
    assert not client.is_runtime(), "fixture must exercise the client/worker boundary"
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _wheel(wheels, "plugin_core_dep")
    _wheel(wheels, "plugin_proof_other")
    for pin in ("1.0", "2.0"):
        _wheel(wheels, "plugin_proof_dep", pin)
    world.home.mkdir()
    (world.home / "config.yaml").write_text("plugins:\n  enabled: []\n  disabled: []\n", encoding="utf-8")
    world.core.mkdir()
    (world.core / "pyproject.toml").write_text(
        '[project]\nname="plugin-proof-core"\nversion="1"\nrequires-python=">=3.11"\n'
        'dependencies=["plugin-core-dep==1.0"]\n[tool.uv]\npackage=false\nno-index=true\n'
        f'find-links=[{json.dumps(wheels.as_posix())}]\n', encoding="utf-8")
    result = subprocess.run([uv, "lock", "--python", sys.executable], cwd=world.core,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return world