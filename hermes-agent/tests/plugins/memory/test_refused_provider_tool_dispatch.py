"""A memory provider refused for a profile must not serve that profile through the tools its
``register(ctx)`` put in the tool registry (#147659).

The provider below is the reported shape (mnemosyne-hermes 0.7.1): one process-wide instance, handed
to every profile's manager, whose ``register(ctx)`` also registers its tools through ``ctx`` with a
handler bound to that instance. The second profile's manager refuses the instance, but the registry
tool registered during that profile's load still points at it, and so at the first profile's store.
"""
from __future__ import annotations

import contextlib
import json
import shutil
import sys

import pytest

from agent.memory_manager import MemoryManager
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

_TOOL = "sharedmem_recall"

_SHARED_MODULE = '''
import json
from functools import partial

from agent.memory_provider import MemoryProvider

_provider = None


class SharedProvider(MemoryProvider):
    stores = {}
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def name(self):
        return "sharedmem"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.home = kwargs["hermes_home"]

    def get_tool_schemas(self):
        return [{"name": "sharedmem_recall", "description": "recall",
                 "parameters": {"type": "object", "properties": {}}}]

    def handle_tool_call(self, tool_name, args, **kwargs):
        return json.dumps({"rows": type(self).stores.get(self.home, [])})


def register(ctx):
    global _provider
    if _provider is None:
        _provider = SharedProvider()
    ctx.register_memory_provider(_provider)
    for schema in _provider.get_tool_schemas():
        ctx.register_tool(name=schema["name"], toolset="memory", schema=schema,
                          handler=partial(_provider.handle_tool_call, schema["name"]))
'''


@pytest.fixture
def homes(tmp_path, monkeypatch):
    shared_dir = tmp_path / "site"
    shared_dir.mkdir()
    (shared_dir / "sharedmem_singleton.py").write_text(_SHARED_MODULE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(shared_dir))
    root = tmp_path / ".hermes"
    alpha, beta = root / "profiles" / "alpha", root / "profiles" / "beta"
    for home in (alpha, beta):
        plugin_dir = home / "plugins" / "sharedmem"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text("name: sharedmem\nkind: exclusive\n", encoding="utf-8")
        (plugin_dir / "__init__.py").write_text(
            "# MemoryProvider\nfrom sharedmem_singleton import register\n", encoding="utf-8")
        (home / "config.yaml").write_text("memory:\n  provider: sharedmem\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    from agent.secret_scope import reset_multiplex_context, set_multiplex_context
    from hermes_cli.plugins import _reset_plugin_managers_for_tests
    _reset_plugin_managers_for_tests()
    token = set_multiplex_context(True)
    try:
        yield alpha, beta
    finally:
        reset_multiplex_context(token)
        _reset_plugin_managers_for_tests()
        sys.modules.pop("sharedmem_singleton", None)


@contextlib.contextmanager
def _scoped(home):
    token = set_hermes_home_override(str(home))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _build_manager(home):
    from plugins.memory import load_memory_provider
    with _scoped(home):
        provider = load_memory_provider("sharedmem")
        manager = MemoryManager()
        manager.add_provider(provider)
        manager.initialize_all(session_id=f"s-{home.name}", platform="api_server", hermes_home=str(home))
    return provider, manager


def _registry_call(home):
    from tools.registry import registry
    with _scoped(home):
        return json.loads(registry.dispatch(_TOOL, {}))


def test_refused_profile_cannot_read_the_owning_profiles_store_through_registry_tools(homes):
    alpha_home, beta_home = homes
    owner, _ = _build_manager(beta_home)
    type(owner).stores[str(beta_home)] = ["beta-secret"]
    shared, alpha = _build_manager(alpha_home)
    assert shared is owner and alpha.providers == []  # refused at init: the guard held there

    result = _registry_call(alpha_home)

    assert "beta-secret" not in json.dumps(result), result
    assert "refused" in result.get("error", ""), result
    assert str(beta_home) not in result["error"] and str(alpha_home) not in result["error"]


def test_owning_profile_still_reaches_its_provider_through_registry_tools(homes):
    alpha_home, beta_home = homes
    owner, _ = _build_manager(beta_home)
    type(owner).stores[str(beta_home)] = ["beta-row"]
    _build_manager(alpha_home)  # the refused profile's load re-registers the tools in its own scope

    assert _registry_call(beta_home) == {"rows": ["beta-row"]}


@pytest.mark.parametrize("loader", ["user_dir", "entry_point"])
def test_subclass_fallback_ctx_tool_still_refuses_shared_instance(homes, loader):
    """A register(ctx) that only adds tools may let the loader find the provider subclass."""
    alpha_home, beta_home = homes
    site = alpha_home.parents[2] / "site"
    fallback_source = _SHARED_MODULE.replace("    ctx.register_memory_provider(_provider)\n", "")
    assert fallback_source != _SHARED_MODULE
    (site / "sharedmem_singleton.py").write_text(fallback_source, encoding="utf-8")
    for home in (alpha_home, beta_home):
        plugin_dir = home / "plugins" / "sharedmem"
        if loader == "entry_point":  # a bare-module entry point goes through _load_provider_from_entry_point
            shutil.rmtree(plugin_dir)
            continue
        (plugin_dir / "__init__.py").write_text(
            "# MemoryProvider\nfrom sharedmem_singleton import SharedProvider, register\n",
            encoding="utf-8",
        )
    if loader == "entry_point":
        dist = site / "sharedmem_ep-0.1.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: sharedmem-ep\nVersion: 0.1\n", encoding="utf-8")
        (dist / "entry_points.txt").write_text(
            "[hermes_agent.memory_providers]\nsharedmem = sharedmem_singleton\n", encoding="utf-8")

    owner, _ = _build_manager(beta_home)
    type(owner).stores[str(beta_home)] = ["beta-private"]
    shared, refused_manager = _build_manager(alpha_home)
    assert shared is owner and refused_manager.providers == []

    denied = _registry_call(alpha_home)
    assert "beta-private" not in json.dumps(denied), denied
    assert "refused" in denied.get("error", ""), denied
    assert _registry_call(beta_home) == {"rows": ["beta-private"]}
