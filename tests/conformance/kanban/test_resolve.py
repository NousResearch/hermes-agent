"""Resolving a kanban storage backend by name, and the one provider hosted mode refuses.

``resolve_kanban_storage`` takes the backend's name explicitly and never falls back. A provider
that declares the single-user machine's one shared board root (``LOCAL_SHARED_BOARD``) — the
local backend does, and a user plugin may not paper over it — is refused wherever the process
runs hosted: a shared deployment's tenants each get their own agents' boards, behind a backend
that pins the agent too.
"""
import textwrap

import pytest

from agent.runtime_scope import bind_profile_scope
from hermes_kanban_port import resolve_kanban_storage

_PROVIDER = '''
from hermes_kanban_port import KanbanStorageProvider


class Probe(KanbanStorageProvider):
    LOCAL_SHARED_BOARD = {shared!r}

    @property
    def name(self):
        return "sharedprobe"

    def open(self):
        raise NotImplementedError

    def assert_isolating(self):
        return


def register(ctx):
    ctx.register_kanban_storage(Probe())
'''


def _install(home, *, shared: bool):
    plugin = home / "plugins" / "sharedprobe"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(textwrap.dedent(_PROVIDER.format(shared=shared)))
    (home / "config.yaml").write_text("plugins:\n  isolation: in_process\n")
    return plugin


def test_the_local_backend_resolves_by_name_and_refuses_to_serve_hosted():
    provider = resolve_kanban_storage("local")
    assert provider.name == "local" and provider.LOCAL_SHARED_BOARD is True
    assert resolve_kanban_storage("").name == "local"
    assert resolve_kanban_storage("local", hosted=False).name == "local"
    with pytest.raises(LookupError, match="single-user machine"):
        resolve_kanban_storage("local", hosted=True)


def test_a_user_provider_that_declares_a_shared_board_root_is_refused_hosted(tmp_path):
    home = tmp_path / "home"
    _install(home, shared=True)
    with bind_profile_scope(home, hydrate_secrets=False):
        provider = resolve_kanban_storage("sharedprobe", hosted=False)
        assert provider.LOCAL_SHARED_BOARD is True
        with pytest.raises(LookupError, match="single-user machine"):
            resolve_kanban_storage("sharedprobe", hosted=True)


def test_a_provider_without_the_declaration_serves_hosted(tmp_path):
    home = tmp_path / "home"
    _install(home, shared=False)
    with bind_profile_scope(home, hydrate_secrets=False):
        provider = resolve_kanban_storage("sharedprobe", hosted=True)
        assert provider.name == "sharedprobe" and provider.LOCAL_SHARED_BOARD is False


def test_a_missing_provider_is_an_error_never_a_fallback():
    with pytest.raises(LookupError, match="no such kanban storage provider"):
        resolve_kanban_storage("never-installed")
