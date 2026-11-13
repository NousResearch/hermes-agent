"""Home-scoped process listeners survive replacement without crossing profiles."""
import threading

from hermes_cli import plugins
from hermes_cli.plugins_lifecycle import reserve_plugin_manager_for_home, unload_plugin_manager_for_home
from tui_gateway import methods_tools


def test_listener_replacement_unsubscribe_and_stale_dispatch(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    old = plugins.get_plugin_manager()
    seen = []
    cancel = plugins.subscribe_plugin_loaded_for_home(old, lambda source, event: seen.append((source, event)))
    stale = old._plugin_loaded_listeners[-1]
    assert unload_plugin_manager_for_home(tmp_path)
    new = plugins.get_plugin_manager()
    assert not old._plugin_loaded_listeners
    stale(["stale"])
    new._plugin_loaded_listeners[-1](["new"])
    assert seen == [(new, ["new"])]
    cancel()
    cancel()
    assert not new._plugin_loaded_listeners
    assert not plugins._scoped_plugin_listeners.get(new.home_path)


def test_registration_waits_for_retirement_and_attaches_replacement(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    old = plugins.get_plugin_manager()
    reserved = threading.Event()
    release = threading.Event()
    registered = threading.Event()
    seen = []
    cleanup = []

    def retire():
        with reserve_plugin_manager_for_home(tmp_path):
            reserved.set()
            assert release.wait(5)

    def register():
        cleanup.append(plugins.subscribe_plugin_loaded_for_home(old, lambda manager, event: seen.append(manager)))
        registered.set()

    a = threading.Thread(target=retire)
    b = threading.Thread(target=register)
    a.start()
    assert reserved.wait(5)
    b.start()
    assert not registered.wait(.1)
    release.set()
    a.join(5)
    b.join(5)
    assert not a.is_alive() and not b.is_alive() and registered.is_set()
    new = plugins.get_plugin_manager()
    new._plugin_loaded_listeners[-1](["new"])
    assert seen == [new]
    cleanup[0]()


def test_tui_profile_activation_isolation_and_replacement(monkeypatch, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(a))
    methods_tools._ensure_plugin_activation_listener()
    old = plugins.get_plugin_manager()
    old._plugin_loaded_listeners[-1]([{"name": "same", "key": "same", "source": "a"}])
    assert methods_tools._with_activation({}, "same")["activation"]["source"] == "a"
    monkeypatch.setenv("HERMES_HOME", str(b))
    methods_tools._ensure_plugin_activation_listener()
    assert "activation" not in methods_tools._with_activation({}, "same")
    monkeypatch.setenv("HERMES_HOME", str(a))
    assert unload_plugin_manager_for_home(a)
    new = plugins.get_plugin_manager()
    assert "activation" not in methods_tools._with_activation({}, "same")
    new._plugin_loaded_listeners[-1]([{"name": "same", "key": "same", "source": "new"}])
    assert methods_tools._with_activation({}, "same")["activation"]["source"] == "new"
    for scope in (old.scope_key, str(b.resolve())):
        methods_tools._plugin_activation_subscribed.pop(scope)()
        methods_tools._plugin_activations.pop(scope, None)


def test_gateway_queued_retired_event_is_not_rewired(monkeypatch, tmp_path):
    from gateway.run_plugin_rewire import GatewayPluginRewireMixin

    class Loop:
        def __init__(self):
            self.pending = []

        def is_closed(self):
            return False

        def call_soon_threadsafe(self, callback):
            self.pending.append(callback)

    class Runner(GatewayPluginRewireMixin):
        def __init__(self):
            self.calls = []

        def _rewire_plugin_handlers(self, name, home):
            self.calls.append((name, home))

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    old = plugins.get_plugin_manager()
    runner, loop = Runner(), Loop()
    runner._subscribe_plugin_rewire(old, loop=loop)
    old._plugin_loaded_listeners[-1](["old"])
    assert unload_plugin_manager_for_home(tmp_path)
    new = plugins.get_plugin_manager()
    new._plugin_loaded_listeners[-1](["new"])
    for callback in loop.pending:
        callback()
    assert runner.calls == [(None, None)]
    runner._plugin_rewire_unsubscribe[old.scope_key]()
    assert old.scope_key not in runner._plugin_rewire_unsubscribe
    runner._subscribe_plugin_rewire(new, loop=loop)
    assert len(new._plugin_loaded_listeners) == 1
    runner._plugin_rewire_unsubscribe[old.scope_key]()
