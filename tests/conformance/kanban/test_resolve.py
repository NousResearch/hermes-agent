"""Resolving a kanban storage backend by name, and the providers hosted mode refuses.

``resolve_kanban_storage`` takes the backend's name explicitly and never falls back. Wherever
the process runs hosted, the single-user machine's local backend cannot serve: a shared
deployment's tenants each get their own agents' boards, behind a backend that pins the agent
too. The refusal here is structural — the name that selects the local backend, the module a
provider class is defined in when that module is the local backend's, and the one
shared-board-root declaration (``LOCAL_SHARED_BOARD``) a provider cannot make and still isolate
tenants — because a provider can hand one store to whoever asks and another to its callers:
resolution never opens a store to check. Every other path that reaches the local board is
refused at the board's own access boundary (``tests/conformance/kanban/test_hosted_refusal.py``).
"""
import textwrap

import pytest

from agent.runtime_scope import bind_profile_scope
from hermes_kanban_port import resolve_kanban_storage

_ISOLATING_STORE = '''
from hermes_kanban_port import KanbanStorage, KanbanStore


def _isolating_store():
    """A concrete store stub that declares nothing shared and answers nothing."""
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
    with pytest.raises(LookupError, match="single-user machine"):
        resolve_kanban_storage("", hosted=True)


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
        store = provider.open().kanban  # nothing about it is the local backend
        assert store.LOCAL_SHARED_BOARD is False


def test_a_provider_class_defined_in_the_local_backends_module_is_refused_hosted(monkeypatch):
    """The module a provider class is defined in cannot be lied about the way a store's
    behaviour can: a class that lives in the local backend's module is that backend, whatever
    name it answers to."""
    from hermes_kanban_port_local import LocalKanbanStorageProvider
    from plugins import kanban_storage as loader

    class Sibling(LocalKanbanStorageProvider):
        #: The declaration is hidden — the module the class lives in is not.
        LOCAL_SHARED_BOARD = False

        @property
        def name(self):
            return "siblinglocal"

    Sibling.__module__ = LocalKanbanStorageProvider.__module__
    monkeypatch.setattr(loader, "load_kanban_storage", lambda name: Sibling())
    assert resolve_kanban_storage("siblinglocal", hosted=False).name == "siblinglocal"
    with pytest.raises(LookupError, match="single-user machine"):
        resolve_kanban_storage("siblinglocal", hosted=True)


def test_a_missing_provider_is_an_error_never_a_fallback():
    with pytest.raises(LookupError, match="no such kanban storage provider"):
        resolve_kanban_storage("never-installed")
