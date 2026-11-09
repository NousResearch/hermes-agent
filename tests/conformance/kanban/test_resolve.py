"""Resolving a kanban storage backend by name, and the one provider hosted mode refuses.

``resolve_kanban_storage`` takes the backend's name explicitly and never falls back. A provider
that declares the single-user machine's one shared board root (``LOCAL_SHARED_BOARD``) — the
local backend does, and a user plugin may not paper over it — is refused wherever the process
runs hosted: a shared deployment's tenants each get their own agents' boards, behind a backend
that pins the agent too. The declaration is checked where it cannot be hidden: the provider's
own class first, then — hosted — the store the provider actually opens, so a wrapper that
delegates to the local provider and the local provider with the declaration overridden away on
a subclass are refused too, and the store's own class refuses to be subclassed into hiding it.
"""
import textwrap

import pytest

from agent.runtime_scope import bind_profile_scope
from hermes_kanban_port import resolve_kanban_storage

_ISOLATING_STORE = '''
from hermes_kanban_port import KanbanStorage, KanbanStore


def _isolating_store():
    """A concrete store stub that declares nothing shared and answers nothing: resolution
    opens the store it would serve, and this one passes that inspection."""
    def refused(self, *args, **kwargs):
        raise NotImplementedError
    ops = {name: refused for name in KanbanStore.__abstractmethods__}
    return KanbanStorage(type("IsolatingProbeStore", (KanbanStore,), ops)())
'''

_PROBE = '''
from hermes_kanban_port import KanbanStorageProvider


class Probe(KanbanStorageProvider):
    LOCAL_SHARED_BOARD = {shared!r}

    @property
    def name(self):
        return "sharedprobe"

    def open(self):
        return _isolating_store()

    def assert_isolating(self):
        return


def register(ctx):
    ctx.register_kanban_storage(Probe())
'''

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
from hermes_kanban_port import KanbanStorageProvider
from hermes_kanban_port_local import LocalKanbanStorageProvider


class Hidden(LocalKanbanStorageProvider):
    LOCAL_SHARED_BOARD = False

    @property
    def name(self):
        return "hiddenlocal"


def register(ctx):
    ctx.register_kanban_storage(Hidden())
'''


def _install(home, plugin: str, source: str):
    directory = home / "plugins" / plugin
    directory.mkdir(parents=True)
    (directory / "__init__.py").write_text(textwrap.dedent(source))
    (home / "config.yaml").write_text("plugins:\n  isolation: in_process\n")
    return directory


def _install_probe(home, *, shared: bool):
    return _install(home, "sharedprobe", _ISOLATING_STORE + _PROBE.format(shared=shared))


def test_the_local_backend_resolves_by_name_and_refuses_to_serve_hosted():
    provider = resolve_kanban_storage("local")
    assert provider.name == "local" and provider.LOCAL_SHARED_BOARD is True
    assert resolve_kanban_storage("").name == "local"
    assert resolve_kanban_storage("local", hosted=False).name == "local"
    with pytest.raises(LookupError, match="single-user machine"):
        resolve_kanban_storage("local", hosted=True)


def test_a_user_provider_that_declares_a_shared_board_root_is_refused_hosted(tmp_path):
    home = tmp_path / "home"
    _install_probe(home, shared=True)
    with bind_profile_scope(home, hydrate_secrets=False):
        provider = resolve_kanban_storage("sharedprobe", hosted=False)
        assert provider.LOCAL_SHARED_BOARD is True
        with pytest.raises(LookupError, match="single-user machine"):
            resolve_kanban_storage("sharedprobe", hosted=True)


def test_a_provider_without_the_declaration_serves_hosted(tmp_path):
    home = tmp_path / "home"
    _install_probe(home, shared=False)
    with bind_profile_scope(home, hydrate_secrets=False):
        provider = resolve_kanban_storage("sharedprobe", hosted=True)
        assert provider.name == "sharedprobe" and provider.LOCAL_SHARED_BOARD is False
        store = provider.open().kanban  # opened and inspected, not refused
        assert store.LOCAL_SHARED_BOARD is False


@pytest.mark.parametrize("plugin", ["wrappedlocal", "hiddenlocal"])
def test_a_wrapper_or_overriding_provider_cannot_hide_the_shared_board_root_from_hosted_mode(
        tmp_path, plugin):
    """The declaration is checked on the store the provider actually opens, not only the
    provider's own class: both probes declare nothing shared yet hand out the local backend's
    one shared board root, and hosted mode refuses them for the store's sake."""
    home = tmp_path / "home"
    _install(home, "wrappedlocal", _WRAPPER)
    _install(home, "hiddenlocal", _HIDDEN)
    with bind_profile_scope(home, hydrate_secrets=False):
        provider = resolve_kanban_storage(plugin, hosted=False)
        assert provider.LOCAL_SHARED_BOARD is False  # the provider's class says no
        assert provider.open().kanban.LOCAL_SHARED_BOARD is True  # the store it opens says yes
        with pytest.raises(LookupError, match="single-user machine"):
            resolve_kanban_storage(plugin, hosted=True)


def test_the_shared_board_root_declaration_cannot_be_unset_on_a_subclass():
    """A store class that declares the one shared board root cannot be subclassed into
    hiding it: the declaration survives the subclass, or the subclass is refused outright."""
    from hermes_kanban_port_local import LocalKanbanStore
    with pytest.raises(TypeError, match="cannot declare LOCAL_SHARED_BOARD false"):
        type("Hiding", (LocalKanbanStore,), {"LOCAL_SHARED_BOARD": False})
    silent = type("Silent", (LocalKanbanStore,), {})
    assert silent.LOCAL_SHARED_BOARD is True


def test_a_missing_provider_is_an_error_never_a_fallback():
    with pytest.raises(LookupError, match="no such kanban storage provider"):
        resolve_kanban_storage("never-installed")
