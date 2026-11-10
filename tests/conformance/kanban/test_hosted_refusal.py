"""The local store's own hosted refusal: every path that reaches the shared root fails there.

A provider can hand one store to whoever asks and another to its callers, so hosted resolution
never opens a store to inspect it (``tests/conformance/kanban/test_resolve.py`` refuses the
local backend by its name, its module and the one shared-board-root declaration). What is left —
a wrapper delegating to the local provider, a forwarding proxy around the local store, the local
provider with the declaration overridden away on a subclass, a provider that opens the local
backend only from its second open — all reach the same local store, and the store itself refuses
to open in a hosted process, whoever constructs it. No card of one agent's board is exposed
under another agent's scope, and on a single-user machine nothing changes.
"""
import sys
import types

import pytest

from agent.runtime_scope import bind_profile_scope
from hermes_kanban_port import KanbanStorage, KanbanStorageProvider, KanbanStore, resolve_kanban_storage
from hermes_kanban_port_local import LocalKanbanStorageProvider, LocalKanbanStore
import plugins.kanban_storage as loader

_REFUSAL = "refuses to open in a hosted process"


def _serve_hosted_tenants(monkeypatch):
    """This process reports itself hosted the way the boot profile does
    (``hermes_cli.hosted_mode.is_hosted`` — absent from a checkout without that module, which is
    why the store's guarded import answers False there): the store's refusal must hold for
    whoever constructs it, not only for callers that pass the answer explicitly."""
    module = types.ModuleType("hermes_cli.hosted_mode")
    module.is_hosted = lambda: True
    monkeypatch.setitem(sys.modules, "hermes_cli.hosted_mode", module)


class _Proxy:
    """Forwards every attribute to the store it wraps: not a subclass of anything."""
    LOCAL_SHARED_BOARD = False

    def __init__(self, store):
        self.store = store

    def __getattr__(self, name):
        return getattr(self.store, name)


def _isolating_stub():
    """A concrete store stub that declares nothing shared and answers nothing."""
    def refused(self, *args, **kwargs):
        raise NotImplementedError
    ops = {name: refused for name in KanbanStore.__abstractmethods__}
    return KanbanStorage(type("IsolatingStubStore", (KanbanStore,), ops)())


def _seed_a_card(tmp_path, monkeypatch, title="a card of the first agent"):
    """One agent's card in the machine's one shared root, seeded while the process is not
    hosted: a second profile reading it is today's single-user handoff, and what a hosted
    process must never serve."""
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "board"))
    (tmp_path / "home-a").mkdir()
    (tmp_path / "home-b").mkdir()
    with bind_profile_scope(tmp_path / "home-a", hydrate_secrets=False):
        store = LocalKanbanStorageProvider().open().kanban
        return store.create_task(title=title, board="default")


def test_the_local_store_refuses_to_open_in_a_hosted_process_whoever_reaches_it(
        tmp_path, monkeypatch):
    """The refusal sits in the store's own open, before any pin or path is touched, so it
    holds for a direct construction, for a provider that knows nothing, and under another
    agent's bound scope alike."""
    _serve_hosted_tenants(monkeypatch)
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "board"))
    with pytest.raises(LookupError, match=_REFUSAL):
        LocalKanbanStore()
    with pytest.raises(LookupError, match=_REFUSAL):
        LocalKanbanStorageProvider().open()
    with bind_profile_scope(tmp_path / "home-b", hydrate_secrets=False):
        with pytest.raises(LookupError, match=_REFUSAL):
            LocalKanbanStorageProvider().open()


def test_the_resolvers_explicit_hosted_answer_carries_to_the_local_store(tmp_path, monkeypatch):
    """The resolver passes its explicit ``hosted=`` answer to the local provider it hands out
    and the provider carries it to the store, so the refusal holds without the boot profile's
    module — and an explicit not-hosted answer opens today's store, as a local caller needs."""
    provider = resolve_kanban_storage("local", hosted=False)
    assert provider.hosted is False
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "board"))
    assert provider.open().kanban.list_boards()  # the explicit answer says this is not hosted
    with pytest.raises(LookupError, match=_REFUSAL):
        LocalKanbanStorageProvider(hosted=True).open()
    with pytest.raises(LookupError, match=_REFUSAL):
        LocalKanbanStore(hosted=True)


def test_an_un_declaring_subclass_still_cannot_open_the_shared_root_hosted(tmp_path, monkeypatch):
    """The declaration is what the conformance suite asserts, never what hosted mode trusts:
    a subclass can un-declare it (the suite fails it there), but the store it opens still
    refuses a hosted process outright — and on a single-user machine it still opens."""
    _serve_hosted_tenants(monkeypatch)
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "board"))
    hiding = type("Hiding", (LocalKanbanStore,), {"LOCAL_SHARED_BOARD": False})
    with pytest.raises(LookupError, match=_REFUSAL):
        hiding()  # no explicit answer: the hosted process itself refuses
    assert hiding(hosted=False).list_boards()  # explicit not-hosted: today's store opens
    silent = type("Silent", (LocalKanbanStore,), {})
    assert silent.LOCAL_SHARED_BOARD is True
    assert silent(hosted=False).list_boards()  # today's local behaviour, unchanged


@pytest.mark.parametrize("kind", ["proxy", "second-open"])
def test_hosted_a_provider_handing_out_the_local_store_fails_at_the_stores_own_refusal(
        tmp_path, monkeypatch, kind):
    """The ported bypass probes: a forwarding proxy that is no subclass of anything, and a
    provider that answers one isolating open and only hands out the local store from its
    second. Hosted resolution does not open either store to check; both providers resolve,
    and both fail at the local store's own refusal — no card of the first agent's board is
    ever exposed under the second agent's scope."""
    card = _seed_a_card(tmp_path, monkeypatch)

    class Provider(KanbanStorageProvider):
        name = "review"
        opens = 0

        def assert_isolating(self):
            return None

        def open(self):
            self.opens += 1
            if kind == "proxy":
                return KanbanStorage(_Proxy(LocalKanbanStorageProvider().open().kanban))
            if self.opens == 1:
                return _isolating_stub()
            return LocalKanbanStorageProvider().open()

    provider = Provider()
    monkeypatch.setattr(loader, "load_kanban_storage", lambda name: provider)
    _serve_hosted_tenants(monkeypatch)
    resolved = resolve_kanban_storage("review", hosted=True)  # nothing inspects a store
    assert resolved is provider
    if kind == "proxy":
        with pytest.raises(LookupError, match=_REFUSAL):
            resolved.open()  # the store the provider hands out refuses, whoever calls it
    else:
        # The one storage it does hand out answers nothing: the first agent's card is not
        # readable through it under the second agent's scope (or any scope).
        stub = resolved.open()
        with bind_profile_scope(tmp_path / "home-b", hydrate_secrets=False):
            with pytest.raises(NotImplementedError):
                stub.kanban.get_task(card, board="default")
        with pytest.raises(LookupError, match=_REFUSAL):
            resolved.open()  # the second open reaches the local store and refuses
    with bind_profile_scope(tmp_path / "home-b", hydrate_secrets=False):
        with pytest.raises(LookupError, match=_REFUSAL):
            LocalKanbanStorageProvider().open()  # the shared root stays unreachable, for any agent


#: Delegates everything to the local provider while declaring nothing itself.
_WRAPPER = '''
from hermes_kanban_port import KanbanStorageProvider
from hermes_kanban_port_local import LocalKanbanStorageProvider


class Wrapper(KanbanStorageProvider):
    @property
    def name(self):
        return "wrappedlocal"

    def open(self):
        return LocalKanbanStorageProvider().open()

    def assert_isolating(self):
        return LocalKanbanStorageProvider().assert_isolating()


def register(ctx):
    ctx.register_kanban_storage(Wrapper())
'''

#: The local provider with its shared-board-root declaration overridden away on a subclass.
_HIDDEN = '''
from hermes_kanban_port_local import LocalKanbanStorageProvider


class Hidden(LocalKanbanStorageProvider):
    LOCAL_SHARED_BOARD = False

    @property
    def name(self):
        return "hiddenlocal"


def register(ctx):
    ctx.register_kanban_storage(Hidden())
'''


@pytest.mark.parametrize("plugin", ["wrappedlocal", "hiddenlocal"])
def test_hosted_a_wrapper_or_un_declaring_provider_fails_at_the_local_store_itself(
        tmp_path, monkeypatch, plugin):
    """Neither probe is the local backend by name, module or declaration, so both resolve
    hosted — resolution cannot soundly inspect what they open. The local store they open
    refuses, so neither serves the shared root, and no card of the first agent's board is
    exposed under the second agent's scope."""
    _seed_a_card(tmp_path, monkeypatch)
    home = tmp_path / "plugins-home"
    for name, source in (("wrappedlocal", _WRAPPER), ("hiddenlocal", _HIDDEN)):
        directory = home / "plugins" / name
        directory.mkdir(parents=True)
        (directory / "__init__.py").write_text(source)
    (home / "config.yaml").write_text("plugins:\\n  isolation: in_process\\n")
    _serve_hosted_tenants(monkeypatch)
    with bind_profile_scope(home, hydrate_secrets=False):
        provider = resolve_kanban_storage(plugin, hosted=True)  # structural checks do not fire
        assert provider.LOCAL_SHARED_BOARD is False
        with pytest.raises(LookupError, match=_REFUSAL):
            provider.open()  # the store it opens refuses the hosted process itself
    with bind_profile_scope(tmp_path / "home-b", hydrate_secrets=False):
        with pytest.raises(LookupError, match=_REFUSAL):
            LocalKanbanStorageProvider().open()  # the seeded card stays unreachable


def test_locally_the_same_wrapper_still_serves_the_shared_root_across_profiles(
        tmp_path, monkeypatch):
    """The control: on a single-user machine — not hosted — a wrapper provider opens the one
    shared root and another profile reads the first agent's card through it, exactly as
    today's dispatcher handoff needs. The refusal is hosted-only; local behaviour is
    unchanged."""
    card = _seed_a_card(tmp_path, monkeypatch)

    class Wrapper(KanbanStorageProvider):
        name = "wrappedlocal"

        def open(self):
            return LocalKanbanStorageProvider().open()

        def assert_isolating(self):
            return LocalKanbanStorageProvider().assert_isolating()

    monkeypatch.setattr(loader, "load_kanban_storage", lambda name: Wrapper())
    provider = resolve_kanban_storage("wrappedlocal")
    with bind_profile_scope(tmp_path / "home-b", hydrate_secrets=False):
        store = provider.open().kanban
        assert store.get_task(card, board="default") is not None
