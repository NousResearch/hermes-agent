"""Phase 6: HT-branded module aliases resolve to the canonical hermes_* modules.

Each ``ht_<name>`` top-level module is a thin shim that replaces itself in
``sys.modules`` with the real ``hermes_<name>`` module, so the two names are the
exact same object — no duplicate module state, singletons, or isinstance
surprises. ``hermes_<name>`` stays canonical; the alias just makes the new brand
name importable too.
"""

import importlib

import pytest

_ALIASES = [
    ("ht_constants", "hermes_constants"),
    ("ht_state", "hermes_state"),
    ("ht_time", "hermes_time"),
    ("ht_logging", "hermes_logging"),
    ("ht_bootstrap", "hermes_bootstrap"),
]


@pytest.mark.parametrize("new,old", _ALIASES)
def test_alias_is_the_canonical_module(new, old):
    new_mod = importlib.import_module(new)
    old_mod = importlib.import_module(old)
    # Same object — not a copy — so attributes and module-level state agree.
    assert new_mod is old_mod


def test_alias_exposes_real_attributes():
    import ht_constants

    assert callable(ht_constants.get_hermes_home)
    assert callable(ht_constants.maybe_migrate_home)


def test_aliases_declared_in_py_modules():
    """The shim files must be packaged, or `import ht_constants` breaks in the
    wheel/Docker image (same failure mode as the ht_compat py-modules bug)."""
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as handle:
        py_modules = tomllib.load(handle)["tool"]["setuptools"]["py-modules"]
    for new, _ in _ALIASES:
        assert new in py_modules, f"{new} shim is not declared in py-modules"
