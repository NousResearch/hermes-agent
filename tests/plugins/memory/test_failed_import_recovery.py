"""A setup dependency becoming available must unblock the next discovery in-process."""

import importlib
import sys

import pytest

import plugins.memory as memory_plugins


@pytest.mark.parametrize("failure_site", ["sibling", "package"])
def test_discovery_recovers_after_dependency_install(tmp_path, monkeypatch, failure_site):
    home = tmp_path / "home"
    plugin = home / "plugins" / "retry_memory"
    plugin.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(memory_plugins, "_MEMORY_PLUGINS_DIR", tmp_path / "bundled")
    monkeypatch.setattr(memory_plugins, "_iter_entry_points", lambda: [])
    monkeypatch.chdir(tmp_path)
    dependency_dir = tmp_path / "dependencies"
    dependency_dir.mkdir()
    monkeypatch.syspath_prepend(str(dependency_dir))
    dependency_name = f"hermes_retry_sdk_{failure_site}"
    (plugin / "stable.py").write_text("TOKEN = object()\n", encoding="utf-8")
    dependency_import = f"from {dependency_name} import READY\n"
    if failure_site == "sibling":
        (plugin / "backend.py").write_text(dependency_import, encoding="utf-8")
        dependency_import = "from .backend import READY\n"
    (plugin / "__init__.py").write_text(
        "from .stable import TOKEN\n"
        + dependency_import
        + "from agent.memory_provider import MemoryProvider\n"
        "class Provider(MemoryProvider):\n"
        "    name = 'retry_memory'\n"
        "    token = TOKEN\n"
        "    def is_available(self): return READY\n"
        "    def initialize(self, session_id, **kwargs): pass\n"
        "    def get_tool_schemas(self): return []\n"
        "def register(ctx): ctx.register_memory_provider(Provider())\n",
        encoding="utf-8",
    )
    module_name = memory_plugins._module_name(plugin, "retry_memory")
    try:
        assert memory_plugins.discover_memory_providers() == [("retry_memory", "", False)]
        stable = sys.modules[f"{module_name}.stable"]
        (dependency_dir / f"{dependency_name}.py").write_text("READY = True\n", encoding="utf-8")
        importlib.invalidate_caches()

        assert memory_plugins.discover_memory_providers() == [("retry_memory", "", True)]
        provider = memory_plugins.load_memory_provider("retry_memory", register_skills=False)
        assert provider is not None
        assert provider.is_available()
        assert provider.token is stable.TOKEN
        assert sys.modules[f"{module_name}.stable"] is stable
    finally:
        for name in list(sys.modules):
            if name == module_name or name.startswith(f"{module_name}.") or name == dependency_name:
                sys.modules.pop(name, None)
