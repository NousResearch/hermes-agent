"""Activation exports a child PYTHONPATH that carries the selected tree's .pth dirs.

A child never runs site.addsitedir(), so exporting site-packages alone drops every
directory its .pth files add: the parent resolves pywintypes, the child raises
ModuleNotFoundError (regression for #121692).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

# Lives in the dir pywin32.pth adds, so it is importable ONLY when that dir is on
# the child's path. Stands in for pywintypes, which cannot be probed off-Windows.
_PROBE_MODULE = "pth_probe"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _generation(tmp_path, monkeypatch, project_root: Path) -> tuple[Path, Path]:
    """A committed generation whose site tree ships a pywin32-style .pth."""
    import pm.environments as environments

    home = tmp_path / "home"
    (home / ".hermes").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home / ".hermes"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "store"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    environment = environments.install_state_dir(project_root) / "environments" / "gen1" / "venv"
    site = environments.site_packages(environment)
    (site / "win32" / "lib").mkdir(parents=True)
    (site / "pywin32_system32").mkdir()
    (environment / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
    (site / "pywin32.pth").write_text(
        "# .pth file for the PyWin32 extensions\nwin32\nwin32\\lib\nPythonwin\n"
        "import pywin32_bootstrap\n",
        encoding="utf-8",
    )
    (site / "win32" / "lib" / f"{_PROBE_MODULE}.py").write_text("VALUE = 42\n", encoding="utf-8")
    environments.runtime_facts_path(project_root).write_text(
        json.dumps({"packages": {"venv": {"environment": str(environment)}}}), encoding="utf-8")
    return environment, site


def _imports_with(pythonpath: str, cwd: Path) -> subprocess.CompletedProcess:
    """Import the probe under ONLY this PYTHONPATH: no addsitedir, no inherited path."""
    return subprocess.run(
        [sys.executable, "-c", f"import {_PROBE_MODULE} as probe; print(probe.VALUE)"],
        cwd=cwd, capture_output=True, text=True, timeout=60,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": pythonpath,
             "PYTHONDONTWRITEBYTECODE": "1"},
    )


def test_activation_environment_exports_pth_directories(tmp_path, monkeypatch):
    """The env composed for a spawned child resolves what .pth processing adds."""
    project_root = tmp_path / "checkout"
    project_root.mkdir()
    _, site = _generation(tmp_path, monkeypatch, project_root)
    from pm.environments import activation_environment

    entries = activation_environment(project_root)["PYTHONPATH"].split(os.pathsep)

    assert str(site) in entries, "the selected site tree itself must stay exported"
    assert str(site / "win32") in entries
    assert str(site / "win32" / "lib") in entries
    # An `import` line is code, not a path: exporting it would put a bogus entry
    # on every child's PYTHONPATH.
    assert not any("pywin32_bootstrap" in entry for entry in entries)
    assert not any(entry.endswith("Pythonwin") for entry in entries)
    # The dir .pth names but that does not exist is not exported.
    assert _imports_with(os.pathsep.join(entries), tmp_path).stdout.strip() == "42"


def test_activate_dependencies_exports_pth_directories_to_children(tmp_path, monkeypatch):
    """The process-wide export children inherit resolves what .pth processing adds."""
    project_root = tmp_path / "checkout"
    project_root.mkdir()
    _, site = _generation(tmp_path, monkeypatch, project_root)

    probe = tmp_path / "export.py"
    probe.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(_REPO_ROOT)!r})\n"
        "from pm.environments import activate_dependencies\n"
        f"activate_dependencies(Path({str(project_root)!r}))\n"
        "print(json.dumps(os.environ['PYTHONPATH'].split(os.pathsep)))\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(probe)], cwd=tmp_path, capture_output=True, text=True, timeout=120,
        env={"PATH": os.environ.get("PATH", ""),
             "HERMES_HOME": os.environ["HERMES_HOME"], "HERMES_RUNTIME_DIR": os.environ["HERMES_RUNTIME_DIR"],
             "HOME": os.environ.get("HOME", str(tmp_path)), "USERPROFILE": os.environ.get("USERPROFILE", str(tmp_path)),
             "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr
    entries = json.loads(result.stdout)

    assert str(site) in entries
    assert str(site / "win32" / "lib") in entries
    assert _imports_with(os.pathsep.join(entries), tmp_path).stdout.strip() == "42"


def test_exported_pth_directories_stay_hermes_owned(tmp_path, monkeypatch):
    """A terminal/code child must not inherit the generation's pywin32 dirs.

    The exported overlay is Hermes's, so the ownership stripper has to recognise it:
    leaking ``win32\\lib`` into a foreign interpreter is the cross-ABI crash the
    strip exists to prevent (#74817).
    """
    project_root = tmp_path / "checkout"
    project_root.mkdir()
    environment, site = _generation(tmp_path, monkeypatch, project_root)
    from tools.environments import local_pythonpath

    # Provenance is proven by the caller's VIRTUAL_ENV/facts record; this asserts the
    # ownership of what activation exports, so the proven runtime tree is supplied.
    monkeypatch.setattr(local_pythonpath, "_validated_runtime_venv", lambda env: environment)
    monkeypatch.setattr(local_pythonpath, "_state", lambda: type(
        "_S", (), {"_hermes_site_packages": None, "_in_venv": False,
                   "_hermes_repo_root_aliases": ()})())
    env = {"PYTHONPATH": os.pathsep.join([str(site), str(site / "win32" / "lib")])}

    local_pythonpath._strip_hermes_owned_pythonpath(env)

    assert "PYTHONPATH" not in env, env
    # Control: the site tree alone does NOT resolve the probe, so the strip is what
    # keeps the generation's pywin32 out of a foreign interpreter's path.
    assert _imports_with(str(site), tmp_path).returncode != 0


def test_pth_entries_survive_a_non_utf8_file_and_an_egg_line(tmp_path, monkeypatch):
    """Mirror ``site.addpackage`` on the two shapes that diverge from it.

    A ``.pth`` that is not UTF-8 (cp936, or UTF-16 out of PowerShell 5.1) decodes
    with the locale encoding instead of raising, and a line counts when its target
    *exists*: an ``.egg``/``.zip`` is importable through zipimport, not a directory.
    """
    import locale as locale_module

    from pm.environments import pth_dirs

    site = tmp_path / "site-packages"
    (site / "win32").mkdir(parents=True)
    (site / "zegg-1.0-py3.9.egg.zip").write_bytes(b"PK\x03\x04")
    (site / "zegg.pth").write_text("zegg-1.0-py3.9.egg.zip\n", encoding="utf-8")
    (site / "legacy.pth").write_bytes("# caf\xe9\nwin32\n".encode("latin-1"))

    monkeypatch.setattr(locale_module, "getencoding", lambda: "latin-1")

    assert pth_dirs(site) == [
        str(site / "win32"), str(site / "zegg-1.0-py3.9.egg.zip"),
    ]
