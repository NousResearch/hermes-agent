"""Regression coverage for single-file runtime modules shipped in the wheel."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _root_py_modules() -> set[str]:
    """Mirrors tests/test_packaging_py_modules.py's loader: setup.py's py-modules
    list is derived from the tree at build time (see setup.py::_root_py_modules()),
    not declared statically in pyproject.toml -- a static list drifted from the
    tree every time the layout changed and broke installed wheels."""
    spec = importlib.util.spec_from_file_location("_hermes_setup_py", REPO_ROOT / "setup.py")
    mod = importlib.util.module_from_spec(spec)
    saved = sys.argv
    sys.argv = ["setup.py", "--name"]  # setup() must not try to build anything on import
    try:
        try:
            spec.loader.exec_module(mod)
        except SystemExit:
            pass
    finally:
        sys.argv = saved
    return set(mod._root_py_modules())


def test_every_state_runtime_is_declared_as_a_packaged_module() -> None:
    packaged = _root_py_modules()
    state_modules = {path.stem for path in REPO_ROOT.glob("hermes_state*.py")}

    assert state_modules <= packaged
