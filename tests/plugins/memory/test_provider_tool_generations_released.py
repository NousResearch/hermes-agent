"""Re-collecting a memory provider on every agent build must not pin every generation.

The reported shape (mnemosyne-hermes 0.7.x): ``register(ctx)`` builds a NEW provider instance per
call and registers its tools through ``ctx`` with a handler bound to that instance. ``load_memory_provider``
runs once per agent build, so a long-lived gateway re-registers the same tools hundreds of times a day.
Each registration used to stay leased in the replacement coordinator and the plugin ledger, pinning its
handler -> collector -> provider (and any SQLite connection the provider held open) for the life of the
process. Observed live: ~55 orphaned providers / ~110 open ``.db`` descriptors per day on one gateway.
"""
from __future__ import annotations

import contextlib
import gc
import json
import sys
import weakref

import pytest

from agent.memory_manager import MemoryManager
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

_TOOL = "pergen_recall"

_MODULE = '''
import json
from functools import partial

from agent.memory_provider import MemoryProvider

built = []


class PerGenProvider(MemoryProvider):
    @property
    def name(self):
        return "pergen"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.session_id = session_id

    def get_tool_schemas(self):
        return [{"name": "pergen_recall", "description": "recall",
                 "parameters": {"type": "object", "properties": {}}}]

    def handle_tool_call(self, tool_name, args, **kwargs):
        return json.dumps({"generation": self.generation})


def register(ctx):
    provider = PerGenProvider()
    provider.generation = len(built)
    built.append(None)  # count only -- never hold the instance
    ctx.register_memory_provider(provider)
    for schema in provider.get_tool_schemas():
        ctx.register_tool(name=schema["name"], toolset="memory", schema=schema,
                          handler=partial(provider.handle_tool_call, schema["name"]))
'''


@pytest.fixture
def home(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    (site / "pergen_provider.py").write_text(_MODULE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(site))
    root = tmp_path / ".hermes"
    plugin_dir = root / "plugins" / "pergen"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: pergen\nkind: exclusive\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text("# MemoryProvider\nfrom pergen_provider import register\n",
                                           encoding="utf-8")
    (root / "config.yaml").write_text("memory:\n  provider: pergen\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    from hermes_cli.plugins import _reset_plugin_managers_for_tests
    _reset_plugin_managers_for_tests()
    try:
        yield root
    finally:
        _reset_plugin_managers_for_tests()
        sys.modules.pop("pergen_provider", None)


@contextlib.contextmanager
def _scoped(home):
    token = set_hermes_home_override(str(home))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _build_and_finish(home, i):
    """One agent build: load, initialize, then shut down -- the agent is done with it."""
    from plugins.memory import load_memory_provider
    with _scoped(home):
        provider = load_memory_provider("pergen")
        assert provider is not None
        manager = MemoryManager()
        manager.add_provider(provider)
        manager.initialize_all(session_id=f"s{i}", platform="api_server", hermes_home=str(home))
        manager.shutdown_all()
    return weakref.ref(provider)


def test_finished_provider_generations_are_collectable(home):
    refs = [_build_and_finish(home, i) for i in range(6)]
    gc.collect()
    pinned = [i for i, ref in enumerate(refs[:-1]) if ref() is not None]
    assert pinned == [], f"superseded provider generations still pinned: {pinned}"


def test_registry_tool_still_routes_to_the_newest_generation(home):
    for i in range(4):
        _build_and_finish(home, i)
    from tools.registry import registry
    with _scoped(home):
        result = registry.dispatch(_TOOL, {})
        assert json.loads(result if isinstance(result, str) else json.dumps(result)) == {"generation": 3}


def test_lease_and_ledger_bookkeeping_stays_bounded(home):
    from hermes_cli.plugins import get_plugin_manager
    from registration_lifecycle import replacement_coordinator

    for i in range(8):
        _build_and_finish(home, i)
    with _scoped(home):
        manager = get_plugin_manager()
        slot = ("tool", manager.scope_key, _TOOL)
        tool_regs = [r for r in manager._registration_order if r.kind == "tool" and r.key == _TOOL]
    assert len(replacement_coordinator._active.get(slot, [])) == 1
    assert len(tool_regs) == 1


def test_throwaway_manager_load_leaves_the_live_tool_and_pins_nothing(home):
    """A one-shot load into a throwaway plugin manager (``transient_plugin_host``, the profile
    lifecycle hook) must neither revoke the active provider's registry tool nor stay recorded as a
    generation once that manager is torn down."""
    import pergen_provider
    from hermes_cli.plugins import PluginManager, get_plugin_manager
    from plugins.memory import _MEMORY_PROVIDER_TOOL_REGISTRATIONS, _ProviderCollector
    from tools.registry import registry

    first = _build_and_finish(home, 0)
    current = _build_and_finish(home, 1)
    with _scoped(home):
        scope = get_plugin_manager().scope_key
        expected = registry.snapshot_registration(_TOOL, scope=scope)
        assert expected is not None
        recorded = dict(_MEMORY_PROVIDER_TOOL_REGISTRATIONS)
        throwaway = PluginManager()
        throwaway.scope_key = f"{throwaway.scope_key}#transient"  # what transient_plugin_host() yields
        collector = _ProviderCollector("pergen", register_skills=False, manager=throwaway)
        collector.collect(pergen_provider.register)
        assert collector.provider is not None
        transient = weakref.ref(collector.provider)
        del collector
        throwaway.unload()
        assert registry.snapshot_registration(_TOOL, scope=scope) is expected
        assert json.loads(registry.dispatch(_TOOL, {})) == {"generation": 1}
        assert _MEMORY_PROVIDER_TOOL_REGISTRATIONS == recorded
    gc.collect()
    assert first() is None, "a superseded normal generation is still retained"
    assert transient() is None, "the finished throwaway-manager provider is still retained"
    assert current() is not None
