"""Category-owned provider compatibility reporting and removal gates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from hermes_cli import plugin_compat as pc


OLD_IMPORT = "from tools.web_tools import prefers_gateway\n"


def _write_category_plugins(home: Path, *, model: bool = True, memory: bool = True) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if model:
        plugin = home / "plugins" / "model-providers" / "oldmodel"
        plugin.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text(
            "name: oldmodel\nkind: model-provider\nversion: 1.0.0\n",
            encoding="utf-8",
        )
        (plugin / "__init__.py").write_text(
            OLD_IMPORT
            + "from pathlib import Path\n"
            + "from providers import register_provider\n"
            + "from providers.base import ProviderProfile\n"
            + f"Path({str(plugin / 'LOADED')!r}).write_text('loaded')\n"
            + "register_provider(ProviderProfile(name='oldmodel', base_url='https://oldmodel.invalid/v1'))\n",
            encoding="utf-8",
        )
        paths["model"] = plugin
    if memory:
        plugin = home / "plugins" / "oldmemory"
        plugin.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text(
            "name: oldmemory\nversion: 1.0.0\n",
            encoding="utf-8",
        )
        (plugin / "__init__.py").write_text(
            OLD_IMPORT
            + "from pathlib import Path\n"
            + "from agent.memory_provider import MemoryProvider\n"
            + f"Path({str(plugin / 'LOADED')!r}).write_text('loaded')\n"
            + "class Provider(MemoryProvider):\n"
            + "    @property\n"
            + "    def name(self): return 'oldmemory'\n"
            + "    def is_available(self): return True\n"
            + "    def initialize(self, **kwargs): pass\n"
            + "    def get_tool_schemas(self): return []\n"
            + "def register(ctx):\n"
            + "    ctx.register_memory_provider(Provider())\n",
            encoding="utf-8",
        )
        paths["memory"] = plugin
    return paths


def _clear_provider_modules() -> None:
    import providers

    providers._REGISTRY.clear()
    providers._ALIASES.clear()
    providers._PROVIDER_LIST_CACHE = None
    providers._discovered = False
    for name in list(sys.modules):
        if name.startswith("_hermes_user_provider"):
            del sys.modules[name]


def test_category_plugins_are_in_compat_report_after_real_discovery(tmp_path, monkeypatch):
    """The manager's report includes model and memory category plugins it does not load itself."""
    home = tmp_path / ".hermes"
    paths = _write_category_plugins(home)
    plain_model = home / "plugins" / "model-providers" / "plainmodel"
    plain_model.mkdir(parents=True)
    (plain_model / "__init__.py").write_text(OLD_IMPORT, encoding="utf-8")
    installed_model = home / "plugins" / "installedmodel"
    installed_model.mkdir()
    (installed_model / "plugin.yaml").write_text(
        "name: installedmodel\nkind: model-provider\nversion: 1.0.0\n",
        encoding="utf-8",
    )
    (installed_model / "__init__.py").write_text(OLD_IMPORT, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(pc, "report_file_path", lambda: home / pc.REPORT_FILE)

    from hermes_cli.plugins import PluginManager

    PluginManager(scope_key=str(home)).discover_and_load()

    payload = json.loads((home / pc.REPORT_FILE).read_text(encoding="utf-8"))
    assert set(payload["plugins"]) == {
        "installedmodel", "oldmemory", "oldmodel", "plainmodel",
    }
    assert not (paths["model"] / "LOADED").exists()
    assert not (paths["memory"] / "LOADED").exists()


@pytest.mark.parametrize("category", ["model", "memory"])
def test_category_plugins_are_skipped_after_removal_and_force_load_with_override(
    tmp_path, monkeypatch, category
):
    """Real category loaders skip old imports after removal, but accept the explicit override."""
    home = tmp_path / ".hermes"
    paths = _write_category_plugins(home, model=category == "model", memory=category == "memory")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(pc, "removal_in_effect", lambda today=None: True)
    monkeypatch.setattr(pc, "allow_deprecated_imports", lambda config=None: False)

    if category == "model":
        _clear_provider_modules()
        from providers import get_provider_profile

        assert get_provider_profile("oldmodel") is None
    else:
        import plugins.memory as memory_plugins

        monkeypatch.setattr(memory_plugins, "_get_user_plugins_dir", lambda: home / "plugins")
        assert memory_plugins.load_memory_provider("oldmemory") is None
    assert not (paths[category] / "LOADED").exists()

    monkeypatch.setattr(pc, "allow_deprecated_imports", lambda config=None: True)
    if category == "model":
        _clear_provider_modules()
        from providers import get_provider_profile

        assert get_provider_profile("oldmodel") is not None
    else:
        assert memory_plugins.load_memory_provider("oldmemory") is not None
    assert (paths[category] / "LOADED").exists()
