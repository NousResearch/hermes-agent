"""A memory provider's cli.py is bound on its parent package in either load order, and unbound on failure.

The provider is user-installed (outside ``_MEMORY_PLUGINS_DIR``) and no parent package is
planted: every module these tests read is one the loader itself registered.
"""

import sys

import pytest

import plugins.memory as pm

PROVIDER = "heldtestprovider"


@pytest.fixture
def provider(tmp_path, monkeypatch):
    plugin_dir = tmp_path / "user_plugins" / PROVIDER
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text("pass\n")
    (plugin_dir / "plugin.yaml").write_text(f"name: {PROVIDER}\n")
    assert not pm._is_bundled(plugin_dir)
    monkeypatch.setattr(pm, "_get_active_memory_provider", lambda: PROVIDER)
    monkeypatch.setattr(pm, "find_provider_dir", lambda name: plugin_dir if name == PROVIDER else None)
    parent_name = pm._module_name(plugin_dir, PROVIDER)
    assert parent_name not in sys.modules
    yield plugin_dir, parent_name
    for name in [n for n in sys.modules if n == parent_name or n.startswith(parent_name + ".")]:
        sys.modules.pop(name, None)


def _write_cli(plugin_dir, body="def register_cli(subparser):\n    pass\n"):
    (plugin_dir / "cli.py").write_text(body)


def test_cli_loaded_before_provider_is_bound_on_the_real_parent(provider):
    plugin_dir, parent_name = provider
    _write_cli(plugin_dir)

    assert [c["name"] for c in pm.discover_plugin_cli_commands()] == [PROVIDER]
    package = pm._load_package(plugin_dir, PROVIDER)

    assert package is sys.modules[parent_name]
    assert getattr(package, "cli", None) is sys.modules[f"{parent_name}.cli"]


def test_cli_loaded_after_provider_is_bound_on_the_real_parent(provider):
    plugin_dir, parent_name = provider
    _write_cli(plugin_dir)

    package = pm._load_package(plugin_dir, PROVIDER)
    assert [c["name"] for c in pm.discover_plugin_cli_commands()] == [PROVIDER]

    assert package is sys.modules[parent_name]
    assert getattr(package, "cli", None) is sys.modules[f"{parent_name}.cli"]


def test_failed_cli_module_leaves_nothing_bound(provider):
    plugin_dir, parent_name = provider
    _write_cli(plugin_dir, "raise RuntimeError('boom')\n")

    assert pm.discover_plugin_cli_commands() == []

    assert f"{parent_name}.cli" not in sys.modules
    assert not hasattr(sys.modules[parent_name], "cli")
