"""``hermes update`` installs ``.[all]`` (or ``.[termux-all]``), not ``[dev]``.

setuptools 79.0.1 remains in some venvs from older builds. A pin that lives
only on ``[dev]`` or ``[build-system]`` does not move it. The updater's
install group must request setuptools at the advisory floor, and installing
that spec must replace 79.0.1.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[2]
_SETUPTOOLS_ADVISORY_FLOOR = Version("83.0.0")
_STALE_SETUPTOOLS = "79.0.1"


def _distribution_name(requirement: str) -> str:
    spec = requirement.split(";", 1)[0]
    spec = spec.split("@", 1)[0]
    spec = spec.split("[", 1)[0]
    return spec.split("=", 1)[0].split(">", 1)[0].split("<", 1)[0].split("~", 1)[0].split("!", 1)[0].strip().lower()


def _extra_closure(extras: dict, name: str) -> set[str]:
    seen, todo = set(), [name]
    while todo:
        cur = todo.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for spec in extras.get(cur, ()):
            if _distribution_name(spec) == "hermes-agent":
                todo.extend(spec.split("[", 1)[1].split("]", 1)[0].split(","))
    return seen


def _install_group_specs(group: str) -> list[str]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    specs = list(data["project"].get("dependencies") or [])
    for extra in _extra_closure(extras, group):
        specs.extend(extras.get(extra, ()))
    return specs


def _setuptools_pins_for_install_group(group: str) -> set[str]:
    pins: set[str] = set()
    for spec in _install_group_specs(group):
        if _distribution_name(spec) != "setuptools":
            continue
        assert "==" in spec, f".[{group}] must exact-pin setuptools, got {spec!r}"
        pins.add(spec.split("==", 1)[1].split(";", 1)[0].strip())
    return pins


def test_updater_install_groups_pin_setuptools_at_advisory_floor():
    for group in ("all", "termux-all"):
        pins = _setuptools_pins_for_install_group(group)
        assert pins, (
            f".[{group}] does not request setuptools; hermes update leaves "
            f"setuptools=={_STALE_SETUPTOOLS} installed"
        )
        below = sorted(v for v in pins if Version(v) < _SETUPTOOLS_ADVISORY_FLOOR)
        assert not below, (
            f".[{group}] requests setuptools {sorted(pins)}, below advisory "
            f"floor {_SETUPTOOLS_ADVISORY_FLOOR}"
        )


def _venv_python(venv: Path) -> Path:
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _setuptools_version(python: Path) -> str:
    return subprocess.check_output(
        [str(python), "-c", "from importlib.metadata import version; print(version('setuptools'))"],
        text=True,
    ).strip()


def test_updater_setuptools_pin_upgrades_stale_venv(tmp_path):
    pins = _setuptools_pins_for_install_group("all")
    assert pins, (
        ".[all] does not request setuptools; cannot prove a stale venv upgrades"
    )
    spec = f"setuptools=={max(pins, key=Version)}"
    venv = tmp_path / "venv"
    subprocess.run(["uv", "venv", str(venv)], check=True, cwd=REPO_ROOT)
    python = _venv_python(venv)
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), f"setuptools=={_STALE_SETUPTOOLS}"],
        check=True,
        cwd=REPO_ROOT,
    )
    assert _setuptools_version(python) == _STALE_SETUPTOOLS
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), spec],
        check=True,
        cwd=REPO_ROOT,
    )
    assert Version(_setuptools_version(python)) >= _SETUPTOOLS_ADVISORY_FLOOR
