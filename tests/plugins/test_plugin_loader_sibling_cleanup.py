"""Regression tests for plugins/plugin_loader.py sibling-exec failure cleanup.

Regression for #112096: a sibling `*.py` whose `exec_module` raises used to keep its
half-initialized module in ``sys.modules``, so a later relative import raised
``ImportError: cannot import name ...`` (uncatchable for plugins matching
``ModuleNotFoundError``) instead of the expected ``ModuleNotFoundError``.
"""

import pytest

from plugins import plugin_loader


@pytest.fixture
def plugin_tree(tmp_path):
    """Plugin dir whose ``_boom`` sibling fails at exec (import of a missing module)."""
    d = tmp_path / "omh"
    d.mkdir()
    (d / "__init__.py").write_text("")
    # One healthy sibling and one that raises during exec.
    (d / "_safe.py").write_text("BAKED_COOKIE = 3\n")
    (d / "_boom.py").write_text("import _this_module_does_not_exist_anywhere\n")
    return d


def _fresh_logger():
    import logging

    return logging.getLogger("test-plugin-loader")


def test_failed_sibling_not_left_in_sys_modules(plugin_tree):
    """A failed sibling exec must be evicted from sys.modules."""
    mod = plugin_loader.load_plugin_module(
        "scoped._hermes_user_memory.fixture",
        plugin_tree,
        parents=(),
        logger=_fresh_logger(),
        synthetic_namespace="scoped._hermes_user_memory",
    )
    assert mod is not None
    full = "scoped._hermes_user_memory.fixture._boom"
    assert full not in __import__("sys").modules


def test_failed_sibling_later_import_raises_modulenotfounderror(plugin_tree, monkeypatch):
    """The plugin-visible contract: a relative import of the failed sibling's name
    must raise ModuleNotFoundError (catchable), not ImportError with name=None."""
    import importlib
    import sys

    mod = plugin_loader.load_plugin_module(
        "scoped._hermes_user_memory.fixture",
        plugin_tree,
        parents=(),
        logger=_fresh_logger(),
        synthetic_namespace="scoped._hermes_user_memory",
    )
    assert mod is not None
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("scoped._hermes_user_memory.fixture._boom")
    # And the full name is now gone so a retry starts clean:
    assert "scoped._hermes_user_memory.fixture._boom" not in sys.modules


def test_healthy_sibling_still_loaded(plugin_tree):
    """No behavioral change for the happy path: healthy siblings load and bind."""
    mod = plugin_loader.load_plugin_module(
        "scoped2._hermes_user_memory.fixture",
        plugin_tree,
        parents=(),
        logger=_fresh_logger(),
        synthetic_namespace="scoped2._hermes_user_memory",
    )
    assert mod is not None
    assert mod._safe.BAKED_COOKIE == 3


def test_init_failure_still_returns_none(plugin_tree, monkeypatch):
    """The pre-existing main-module path (pop + return None) is unchanged."""
    (plugin_tree / "__init__.py").write_text("import _this_module_does_not_exist_anywhere\n")
    mod = plugin_loader.load_plugin_module(
        "scoped3._hermes_user_memory.fixture",
        plugin_tree,
        parents=(),
        logger=_fresh_logger(),
        synthetic_namespace="scoped3._hermes_user_memory",
    )
    assert mod is None
    import sys

    assert "scoped3._hermes_user_memory.fixture" not in sys.modules
