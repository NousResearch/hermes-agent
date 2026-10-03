"""A memory provider's cli.py is bound on its parent package, and unbound on failure."""

import sys
import types

import pytest

import plugins.memory as pm

PROVIDER = "heldtestprovider"
PARENT = f"plugins.memory.{PROVIDER}"
CLI = f"{PARENT}.cli"


@pytest.fixture
def provider(tmp_path, monkeypatch):
    plugin_dir = tmp_path / PROVIDER
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").write_text("pass\n")
    (plugin_dir / "plugin.yaml").write_text(f"name: {PROVIDER}\n")
    monkeypatch.setattr(pm, "_MEMORY_PLUGINS_DIR", tmp_path)
    monkeypatch.setattr(pm, "_get_active_memory_provider", lambda: PROVIDER)
    parent = types.ModuleType(PARENT)
    monkeypatch.setitem(sys.modules, PARENT, parent)
    sys.modules.pop(CLI, None)
    yield plugin_dir, parent
    sys.modules.pop(CLI, None)


def test_loaded_cli_module_is_bound_on_its_parent(provider):
    plugin_dir, parent = provider
    (plugin_dir / "cli.py").write_text("def register_cli(subparser):\n    pass\n")

    cmds = pm.discover_plugin_cli_commands()

    assert [c["name"] for c in cmds] == [PROVIDER]
    assert getattr(parent, "cli", None) is sys.modules[CLI]


def test_failed_cli_module_leaves_nothing_bound(provider):
    plugin_dir, parent = provider
    (plugin_dir / "cli.py").write_text("raise RuntimeError('boom')\n")

    assert pm.discover_plugin_cli_commands() == []

    assert CLI not in sys.modules
    assert not hasattr(parent, "cli")
