"""A sealed payload enters its venvs on its own interpreter, never through their redirectors.

A venv's ``Scripts\\python.exe`` sits outside the MSIX and starts the package's interpreter,
which Windows refuses (WinError 5) to a process without package identity. ``venv_command``
must therefore run the payload's interpreter with the venv attached, and the result must
behave like the venv's own interpreter for scripts, ``-c`` and ``-m``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from pm.environments import site_packages, venv_command, venv_python


def _venv(root: Path) -> Path:
    """A venv whose interpreter must never run, with a module and a .pth-added directory."""
    venv = root / "venv"
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    venv.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(f"home = /nonexistent\nversion = {version}.0\n", encoding="utf-8")
    trap = venv_python(venv)
    trap.parent.mkdir(parents=True)
    trap.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    trap.chmod(trap.stat().st_mode | stat.S_IEXEC)
    site = site_packages(venv)
    site.mkdir(parents=True)
    (site / "venv_only_dep.py").write_text("VALUE = 'from the venv'\n", encoding="utf-8")
    (site / "venv_only_cli.py").write_text(
        "import json, sys, venv_only_dep\n"
        "print(json.dumps([venv_only_dep.VALUE, sys.argv]))\n", encoding="utf-8")
    extra = root / "pth-added"
    extra.mkdir()
    (extra / "pth_only_dep.py").write_text("VALUE = 'from a .pth'\n", encoding="utf-8")
    (site / "extra.pth").write_text(str(extra) + "\n", encoding="utf-8")
    return venv


def _payload(root: Path, python: str) -> Path:
    """A sealed payload layout whose recorded interpreter is *python*."""
    payload = root / "payload"
    repo = payload / "hermes-agent"
    repo.mkdir(parents=True)
    tools = payload / "tools" / "python"
    tools.mkdir(parents=True)
    store_python = tools / "python"
    store_python.write_text(f"#!/bin/sh\nexec {python!s} \"$@\"\n", encoding="utf-8")
    store_python.chmod(store_python.stat().st_mode | stat.S_IEXEC)
    (payload / "manifest.json").write_text(json.dumps({
        "schema": 1, "target": "test", "repo": "hermes-agent", "venv": "venv", "store": "tools",
        "runtime": {"repoDir": "hermes-agent", "toolsDir": "tools", "storePython": "tools/python/python",
                    "sitePackages": "venv/site", "commands": {}},
    }), encoding="utf-8")
    return repo


def _run(command: list[str], cwd: Path) -> list:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=60,
                            env={k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_unsealed_trees_run_the_venv_interpreter(tmp_path):
    venv = _venv(tmp_path)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    assert venv_command(checkout, venv, ("-I", "-u")) == [str(venv_python(venv)), "-I", "-u"]


@pytest.mark.platforms("linux", "macos")
def test_sealed_payload_runs_scripts_code_and_modules_without_the_redirector(tmp_path):
    venv = _venv(tmp_path)
    repo = _payload(tmp_path, sys.executable)
    prefix = venv_command(repo, venv, ("-I", "-u"))
    assert str(venv_python(venv)) not in prefix
    assert prefix[0] == str((repo.parent / "tools/python/python").resolve())
    assert "-S" in prefix  # the payload interpreter's own site must not load

    script = tmp_path / "work" / "script.py"
    script.parent.mkdir()
    script.write_text("import json, sys, venv_only_dep, pth_only_dep\n"
                      "print(json.dumps([venv_only_dep.VALUE, pth_only_dep.VALUE, sys.argv]))\n",
                      encoding="utf-8")
    assert _run([*prefix, str(script), "a", "--flag"], tmp_path) == [
        "from the venv", "from a .pth", [str(script), "a", "--flag"]]

    code = ("import json, sys, venv_only_dep, pth_only_dep; "
            "print(json.dumps([venv_only_dep.VALUE, pth_only_dep.VALUE, sys.argv, __name__]))")
    assert _run([*prefix, "-c", code, "x"], tmp_path) == [
        "from the venv", "from a .pth", ["-c", "x"], "__main__"]

    value, argv = _run([*prefix, "-m", "venv_only_cli", "y"], tmp_path)
    assert value == "from the venv"
    assert argv[1:] == ["y"] and argv[0].endswith("venv_only_cli.py")


def test_entry_keeps_isolation_and_pm_off_the_import_path(tmp_path):
    """Under -I neither the caller's cwd nor PM's own directory shadows the venv's modules."""
    venv = _venv(tmp_path)
    (tmp_path / "venv_only_dep.py").write_text("VALUE = 'cwd shadow'\n", encoding="utf-8")
    entry = Path(__import__("pm").__file__).with_name("_venv_entry.py")
    code = ("import json, sys, venv_only_dep; "
            f"print(json.dumps([venv_only_dep.VALUE, {str(entry.parent)!r} in sys.path]))")
    assert _run([sys.executable, "-I", "-S", str(entry), str(site_packages(venv)), "-c", code],
                tmp_path) == ["from the venv", False]
