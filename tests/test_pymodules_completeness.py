"""py-modules must cover every top-level hermes_state* module (#101147).

hermes_state.py unconditionally re-exports from hermes_state_registry at
import time. A sealed venv that ships hermes_state.py without
hermes_state_registry.py dies on import — sessions silently stop being
indexed. This pins the packaging list against exactly that regression.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PYPROJECT = REPO / "pyproject.toml"


def _py_modules() -> list[str]:
    src = PYPROJECT.read_text(encoding="utf-8")
    m = re.search(r"py-modules\s*=\s*\[", src)
    assert m, "py-modules list must exist in pyproject.toml"
    end = src.index("]", m.end())
    return re.findall(r'"([a-z_0-9]+)"', src[m.start():end])


def test_hermes_state_registry_is_declared():
    """The #101147 one-liner: the registry module must ship in wheels."""
    assert "hermes_state_registry" in _py_modules()


def test_every_hermes_state_sibling_module_is_declared():
    """Guard against the whole class: every top-level hermes_state*.py in the
    source tree must appear in py-modules — a sibling added by a future
    split without a packaging line reproduces #101147 silently."""
    declared = set(_py_modules())
    tree_top = {
        p.stem
        for p in REPO.glob("hermes_state*.py")
        if p.is_file()
    }
    missing = sorted(tree_top - declared)
    assert not missing, (
        f"top-level hermes_state* modules missing from py-modules: {missing}"
    )


def test_declared_modules_exist_in_the_tree():
    """The inverse direction: a py-modules entry pointing at a file that no
    longer exists breaks the wheel build (setuptools fails hard)."""
    declared = set(_py_modules())
    for name in sorted(n for n in declared if n.startswith("hermes_")):
        assert (REPO / f"{name}.py").is_file(), (
            f"py-modules declares {name!r} but {name}.py is absent from the tree"
        )
