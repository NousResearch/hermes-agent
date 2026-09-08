"""Scoped plugin registration and cached desktop authority are one lifecycle."""

import json
from contextlib import contextmanager

import pytest

from agent.computer_use_provider import ComputerUseProvider
from agent import computer_use_registry as providers
from hermes_cli.plugins import PluginContext, PluginManager
from hermes_cli.plugins_manifest import PluginManifest
from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override
from tools.computer_use import tool as cu


@contextmanager
def home_scope(home):
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


class RecordingProvider(ComputerUseProvider):
    def __init__(self, name="leased", events=None):
        self._name = name
        self.events = events if events is not None else []
        self.backends = []

    @property
    def name(self):
        return self._name

    def is_available(self):
        return True

    def create_backend(self, session_id, permission_mode):
        events = self.events

        class Backend(cu._NoopBackend):
            def stop(self):
                events.append(("stop", hermes_home_key(), self))

        backend = Backend()
        self.backends.append(backend)
        return backend

    def emergency_cleanup(self):
        self.events.append(("cleanup", hermes_home_key(), self))


@pytest.fixture
def plugins(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_COMPUTER_USE_BACKEND", raising=False)
    cu.reset_backend_for_tests()
    managers = []

    def context(home, plugin="lease-plugin"):
        home.mkdir(exist_ok=True)
        manager = PluginManager(scope_key=hermes_home_key(home))
        managers.append(manager)
        return PluginContext(PluginManifest(name=plugin), manager)

    yield context
    cu.reset_backend_for_tests()
    for manager in managers:
        manager.unload()


@pytest.mark.parametrize("name", ["local", "host", "cua", "cua-driver", "builtin", "noop", "remote", " LOCAL "])
def test_scoped_plugins_cannot_shadow_builtin_names_or_aliases(plugins, tmp_path, name):
    context = plugins(tmp_path)
    before = providers.resolve_provider("local")
    try:
        handle = context.register_computer_use_provider(RecordingProvider(name))
    except ValueError as exc:
        assert "reserved" in str(exc)
    else:
        assert handle is None, "the real PluginContext must refuse builtin-name registrations"
    with home_scope(tmp_path):
        assert providers.resolve_provider("local") is before
        assert providers.snapshot_registration(name, scope=hermes_home_key()) is None


def test_builtin_factories_still_register_normally():
    from tools.computer_use.host_provider import HostCuaProvider, NoopCuaProvider
    from tools.computer_use.remote_provider import RemoteCuaProvider
    assert isinstance(providers.get_provider("local"), HostCuaProvider)
    assert isinstance(providers.get_provider("noop"), NoopCuaProvider)
    assert isinstance(providers.get_provider("remote"), RemoteCuaProvider)


def dispatch(home, sid="same-session"):
    with home_scope(home):
        return json.loads(cu.handle_computer_use({"action": "list_apps"}, session_id=sid))


def register(context, provider):
    handle = context.register_computer_use_provider(provider)
    assert handle is not None
    return handle


@pytest.mark.parametrize("transition", ["replace", "unload"])
def test_cached_session_rechecks_provider_ownership(plugins, tmp_path, transition):
    a, b = tmp_path / "a", tmp_path / "b"
    ca, cb = plugins(a), plugins(b)
    for home in (a, b):
        (home / "config.yaml").write_text("computer_use:\n  provider: leased\n")
    first, sibling, replacement = RecordingProvider(), RecordingProvider(), RecordingProvider()
    handle = register(ca, first)
    register(cb, sibling)
    assert "error" not in dispatch(a)
    assert "error" not in dispatch(b)
    if transition == "replace":
        register(plugins(a, "replacement"), replacement)
    else:
        handle.dispose()
    assert "error" in dispatch(a), "cached authority must not survive provider unload/replacement"
    assert len(first.backends[0].calls) == 1
    assert [e[0] for e in first.events] == ["stop"]
    assert "error" not in dispatch(b), "another profile's same-name provider must remain usable"
    # An unrelated registration in the same profile must not revoke the selected provider.
    register(plugins(b, "unrelated"), RecordingProvider("other"))
    assert "error" not in dispatch(b)
    if transition == "replace":
        assert "error" not in dispatch(a, "new-session")
        assert len(replacement.backends) == 1


@pytest.mark.parametrize("transition", ["replace", "unload", "replace-restore"])
@pytest.mark.parametrize("stage", ["create", "start"])
def test_provider_transition_fences_blocked_acquisition(plugins, tmp_path, transition, stage):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    entered, resume = threading.Event(), threading.Event()
    context = plugins(tmp_path)
    (tmp_path / "config.yaml").write_text("computer_use:\n  provider: leased\n")

    class BlockingProvider(RecordingProvider):
        def create_backend(self, sid, mode):
            backend = super().create_backend(sid, mode)

            def block():
                entered.set()
                assert resume.wait(5)

            if stage == "create":
                block()
            else:
                backend.start = block
            return backend

    first, replacement = BlockingProvider(), RecordingProvider()
    handle = register(context, first)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(dispatch, tmp_path)
        try:
            assert entered.wait(5)
            if transition == "unload":
                handle.dispose()
            else:
                replacement_handle = register(plugins(tmp_path, "replacement"), replacement)
                if transition == "replace-restore":
                    replacement_handle.dispose()
        finally:
            resume.set()
        assert "error" in pending.result(timeout=5), "obsolete startup must not publish or dispatch"
    assert not first.backends[0].calls
    assert [e[0] for e in first.events] == ["stop"]
    with home_scope(tmp_path):
        owner = cu._backend_owner_key("same-session")
        assert owner not in cu._backends
        assert owner not in cu._backend_start_locks


@pytest.mark.parametrize("shared_provider", [False, True])
def test_emergency_cleanup_captures_home_and_cleans_displaced_resources_once(plugins, tmp_path, shared_provider):
    a, b = tmp_path / "a", tmp_path / "b"
    ca, cb = plugins(a), plugins(b)
    events = []
    first = RecordingProvider(events=events)
    sibling = first if shared_provider else RecordingProvider(events=events)
    replacement, dormant = RecordingProvider(events=events), RecordingProvider("dormant", events)
    register(ca, first)
    register(cb, sibling)
    register(ca, dormant)
    for home in (a, b):
        (home / "config.yaml").write_text("computer_use:\n  provider: leased\n")
    assert "error" not in dispatch(a)
    backend_a = first.backends[-1]
    assert "error" not in dispatch(b)
    backend_b = sibling.backends[-1]
    register(plugins(a, "replacement"), replacement)
    assert "error" not in dispatch(a, "replacement-session")
    # Teardown runs under neither owner: it must restore each captured home itself.
    cu._shutdown_backend_atexit()
    stops = [(home, resource) for kind, home, resource in events if kind == "stop"]
    cleanups = [(home, resource) for kind, home, resource in events if kind == "cleanup"]
    assert set(stops) == {(hermes_home_key(a), backend_a), (hermes_home_key(b), backend_b),
                          (hermes_home_key(a), replacement.backends[0])}
    assert set(cleanups) == {(hermes_home_key(a), first), (hermes_home_key(b), sibling),
                             (hermes_home_key(a), replacement)}
    assert [kind for kind, _, _ in events] == ["stop"] * 3 + ["cleanup"] * 3
    cu._shutdown_backend_atexit()
    assert len(events) == 6, "emergency teardown must be idempotent, including displaced providers"


@pytest.mark.parametrize("transition", ["replace", "unload", "replace-restore"])
def test_provider_revocation_after_lookup_prevents_dispatch(plugins, tmp_path, monkeypatch, transition):
    context = plugins(tmp_path)
    (tmp_path / "config.yaml").write_text("computer_use:\n  provider: leased\n")
    first = RecordingProvider()
    handle = register(context, first)
    real_lookup = cu._get_backend

    def revoke_after_lookup(session_id=""):
        backend = real_lookup(session_id)
        if transition == "unload":
            handle.dispose()
        else:
            replacement = register(plugins(tmp_path, "replacement"), RecordingProvider())
            if transition == "replace-restore":
                replacement.dispose()
        return backend

    monkeypatch.setattr(cu, "_get_backend", revoke_after_lookup)
    result = dispatch(tmp_path)
    assert not first.backends[0].calls, "lookup must not carry revoked plugin authority into dispatch"
    assert "error" in result
