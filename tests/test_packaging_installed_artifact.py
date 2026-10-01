"""Installed-artifact smoke coverage for extracted profile packaging."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _build_nix_wheel(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "dist"
    artifact_dir.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    extra_cfg = tmp_path / "dist-extra.cfg"
    extra_cfg.write_text(
        f"[build]\nbuild_base = {scratch / 'build'}\n\n"
        f"[egg_info]\negg_base = {scratch}\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["HERMES_NIX_BUILD"] = "1"
    env["DIST_EXTRA_CONFIG"] = str(extra_cfg)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from setuptools.build_meta import build_wheel; "
                f"build_wheel({str(artifact_dir)!r})"
            ),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    wheels = list(artifact_dir.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def test_nix_wheel_installs_and_imports_extracted_profiles(tmp_path: Path) -> None:
    wheel = _build_nix_wheel(tmp_path)
    venv = tmp_path / "venv"

    created = subprocess.run(
        [sys.executable, "-m", "venv", str(venv)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert created.returncode == 0, created.stderr

    python = _venv_python(venv)
    installed = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            str(wheel),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert installed.returncode == 0, installed.stderr

    artifact_modules = [
        "profiles",
        "profiles.current",
        "profiles.paths",
        "profiles.registry",
    ]
    artifact_modules.extend(
        package
        for package in ("runtime", "storage")
        if (PROJECT_ROOT / package / "__init__.py").is_file()
    )
    imported = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            (
                "import importlib, json; "
                f"mods = {artifact_modules!r}; "
                "loaded = {name: importlib.import_module(name).__file__ for name in mods}; "
                "print(json.dumps(loaded))"
            ),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert imported.returncode == 0, imported.stderr

    loaded = json.loads(imported.stdout)
    assert set(loaded) == set(artifact_modules)
    checkout = PROJECT_ROOT.resolve()
    for module_path in loaded.values():
        assert checkout not in Path(module_path).resolve().parents
