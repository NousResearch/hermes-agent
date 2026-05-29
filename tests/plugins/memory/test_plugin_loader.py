"""Regression tests for memory provider plugin loading."""

from __future__ import annotations

import sys

from plugins.memory import _load_provider_from_dir


def test_user_memory_plugin_loader_does_not_execute_setup_py(tmp_path):
    """Packaging helpers like setup.py must not run during provider load."""
    plugin_dir = tmp_path / "mnemosyne_like"
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").write_text(
        "from agent.memory_provider import MemoryProvider\n"
        "class DummyProvider(MemoryProvider):\n"
        "    @property\n"
        "    def name(self):\n"
        "        return 'dummy'\n"
        "    def is_available(self):\n"
        "        return True\n"
        "    def initialize(self, session_id, **kwargs):\n"
        "        pass\n"
        "    def get_tool_schemas(self):\n"
        "        return []\n"
        "def register(ctx):\n"
        "    ctx.register_memory_provider(DummyProvider())\n"
    )
    (plugin_dir / "setup.py").write_text(
        "raise SystemExit('setup.py was executed during plugin load')\n"
    )
    (plugin_dir / "provider.py").write_text("VALUE = 1\n")

    provider = _load_provider_from_dir(plugin_dir)

    assert provider is not None
    assert provider.name == "dummy"
    assert f"_hermes_user_memory.{plugin_dir.name}.setup" not in sys.modules
