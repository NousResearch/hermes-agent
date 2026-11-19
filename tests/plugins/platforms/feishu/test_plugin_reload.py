"""A Feishu plugin load must run the card helpers from the plugin directory it loads.

The directory loader imports a plugin as ``hermes_plugins.<slug>`` and evicts that namespace on
reload; a helper imported by absolute path lands under ``plugins.platforms.feishu.*`` instead, so
neither a reload nor a plugin copy elsewhere ever reaches its code.
"""

import asyncio
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.plugins import PluginManager
from hermes_cli.plugins_loader import _evict_modules
from hermes_cli.plugins_manifest import PluginManifest

_FEISHU_DIR = Path(__file__).resolve().parents[4] / "plugins" / "platforms" / "feishu"
# The p2p branch of the chat_mode -> chat type mapping, wherever the plugin keeps it.
_P2P_BRANCH = re.compile(r'(if normalized == "p2p":\n\s+return )"dm"')


@pytest.fixture
def plugin_copy(tmp_path):
    plugin_dir = tmp_path / "feishu"
    shutil.copytree(_FEISHU_DIR, plugin_dir, ignore=shutil.ignore_patterns("__pycache__"))
    manager = PluginManager(scope_key=str(tmp_path / "hermes"))
    manifest = PluginManifest(name="feishu-platform", key="feishu-platform", kind="platform",
                              source="bundled", path=str(plugin_dir))
    loaded: set[str] = set()

    def load():
        module = manager._load_directory_module(manifest)
        loaded.add(module.__name__)
        return sys.modules[f"{module.__name__}.adapter"]

    yield plugin_dir, load
    for name in loaded:
        _evict_modules(name)


def _chat_type_for_p2p(adapter_module) -> str:
    adapter = object.__new__(adapter_module.FeishuAdapter)
    adapter._chat_info_cache = {}
    adapter._build_get_chat_request = lambda _cid: object()
    data = SimpleNamespace(name="Chat", chat_type="private", chat_mode="p2p")
    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(chat=SimpleNamespace(
        get=lambda _req: SimpleNamespace(success=lambda: True, data=data)))))

    async def _run_blocking(func, *args):
        return func(*args)

    adapter._run_blocking = _run_blocking
    return asyncio.run(adapter.get_chat_info("oc_x"))["type"]


def test_reload_runs_the_edited_chat_type_helper(plugin_copy):
    plugin_dir, load = plugin_copy
    assert _chat_type_for_p2p(load()) == "dm"

    edited = 0
    for path in plugin_dir.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        new_source, count = _P2P_BRANCH.subn(r'\1"edited"', source)
        if count:
            path.write_text(new_source, encoding="utf-8")
            edited += count
    assert edited == 1

    assert _chat_type_for_p2p(load()) == "edited"
