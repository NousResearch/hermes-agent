"""plugins.memory: recognising a memory provider that re-exports the contract.

A catalog wrapper carries no implementation of its own: it imports ``register``
and ``register_memory_provider`` from the implementation package and re-exports
them, so the loader calls the imported ``register``, which calls
``register_memory_provider`` with the collector. That is the same runtime
contract as a direct call — but the install-time detector only recognised the
direct call, so such a wrapper loaded as a provider at runtime yet was never
recognised as one when installed, and the install left it inactive.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from plugins.memory import _defines_memory_provider, _is_memory_provider_dir


def _write(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    return path


def test_wrapper_that_re_exports_the_contract_is_a_provider(tmp_path):
    """The catalog-wrapper shape: the contract is imported, not called here."""
    plugin = tmp_path / "wrapped"
    plugin.mkdir()
    _write(plugin / "__init__.py", (
        '"""Wrapper."""\n'
        'from mnemosyne_hermes import register, register_memory_provider  # noqa: F401\n'
        '\n'
        '__all__ = ["register", "register_memory_provider"]\n'
    ))
    assert _defines_memory_provider(plugin) is True


def test_wrapper_reexport_is_recognised_like_a_direct_call(tmp_path):
    """The two detectors must agree: the loose one already saw this as a provider."""
    plugin = tmp_path / "wrapped"
    plugin.mkdir()
    _write(plugin / "__init__.py", (
        'from mnemosyne_hermes import register, register_memory_provider  # noqa: F401\n'
    ))
    assert _defines_memory_provider(plugin) == _is_memory_provider_dir(plugin) is True


def test_a_direct_call_is_still_a_provider(tmp_path):
    """The original shape keeps working."""
    plugin = tmp_path / "direct"
    plugin.mkdir()
    _write(plugin / "__init__.py", (
        'def register(ctx):\n'
        '    register_memory_provider(ctx)\n'
    ))
    assert _defines_memory_provider(plugin) is True


def test_a_memory_provider_subclass_is_still_a_provider(tmp_path):
    """The subclass shape keeps working."""
    plugin = tmp_path / "subclass"
    plugin.mkdir()
    _write(plugin / "__init__.py", (
        'from agent.memory_provider import MemoryProvider\n'
        '\n'
        'class MyProvider(MemoryProvider):\n'
        '    pass\n'
    ))
    assert _defines_memory_provider(plugin) is True


def test_a_mention_in_a_docstring_is_not_a_provider(tmp_path):
    """A docstring naming the contract does not count — that is the point of the AST check."""
    plugin = tmp_path / "docs"
    plugin.mkdir()
    _write(plugin / "__init__.py", (
        '"""This module mentions register_memory_provider and MemoryProvider in prose only."""\n'
    ))
    assert _defines_memory_provider(plugin) is False


def test_an_import_without_the_register_half_is_not_a_provider(tmp_path):
    """``register_memory_provider`` alone is half the contract: without ``register``
    the loader never calls it, so the provider would never register."""
    plugin = tmp_path / "half"
    plugin.mkdir()
    _write(plugin / "__init__.py", (
        'from mnemosyne_hermes import register_memory_provider  # noqa: F401\n'
    ))
    assert _defines_memory_provider(plugin) is False


def test_a_plugin_that_is_not_a_provider_is_not_one(tmp_path):
    """An ordinary plugin stays out of the provider set."""
    plugin = tmp_path / "plain"
    plugin.mkdir()
    _write(plugin / "__init__.py", 'def register(ctx):\n    ctx.register_cli_command(name="x")\n')
    assert _defines_memory_provider(plugin) is False


@pytest.mark.parametrize("module", ["__init__.py", "provider.py"])
def test_a_re_export_in_any_top_level_module_counts(tmp_path, module):
    """The loader reads the package's top-level modules, so any of them may carry it."""
    plugin = tmp_path / "any"
    plugin.mkdir()
    _write(plugin / module, (
        'from mnemosyne_hermes import register, register_memory_provider  # noqa: F401\n'
    ))
    assert _defines_memory_provider(plugin) is True
