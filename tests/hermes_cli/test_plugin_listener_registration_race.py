"""Registration completion must not let a retired manager win a scope."""
import threading

import pytest

from gateway.run_plugin_rewire import GatewayPluginRewireMixin
from hermes_cli import plugins
from hermes_cli.plugins_lifecycle import unload_plugin_manager_for_home
from tui_gateway import methods_tools


@pytest.mark.parametrize("kind", ["gateway", "tui"])
@pytest.mark.parametrize("first", ["old", "new"])
def test_retired_registration_cannot_displace_current(monkeypatch, tmp_path, kind, first):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    old = plugins.get_plugin_manager()
    registered = {name: threading.Event() for name in ("old", "new")}
    release = {name: threading.Event() for name in ("old", "new")}
    original = plugins.subscribe_plugin_loaded_for_home
    errors = []

    class Loop:
        def is_closed(self):
            return False

        def call_soon_threadsafe(self, callback):
            callback()

    class Runner(GatewayPluginRewireMixin):
        def __init__(self):
            self.calls = []

        def _rewire_plugin_handlers(self, *args):
            self.calls.append(args)

    runner = Runner()

    def subscribe(manager, callback):
        cancel = original(manager, callback)
        name = "old" if manager is old else "new"
        if not registered[name].is_set():
            registered[name].set()
            assert release[name].wait(10)
        return cancel

    monkeypatch.setattr(plugins, "subscribe_plugin_loaded_for_home", subscribe)

    def register(manager):
        try:
            if kind == "gateway":
                runner._subscribe_plugin_rewire(manager, loop=Loop())
            else:
                methods_tools._ensure_plugin_activation_listener()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=register, args=(old,))]
    try:
        threads[0].start()
        assert registered["old"].wait(10)
        assert unload_plugin_manager_for_home(tmp_path)
        new = plugins.get_plugin_manager()
        threads.append(threading.Thread(target=register, args=(new,)))
        threads[1].start()
        assert registered["new"].wait(10)
        release[first].set()
        threads[0 if first == "old" else 1].join(10)
        release["new" if first == "old" else "old"].set()
        for thread in threads:
            thread.join(10)
            assert not thread.is_alive()
        assert len(errors) == 1 and str(errors[0]) == "cannot subscribe a retired manager"
        subscriptions = (runner._plugin_rewire_unsubscribe if kind == "gateway"
                         else methods_tools._plugin_activation_subscribed)
        assert subscriptions[new.scope_key].manager is new
        assert len(new._plugin_loaded_listeners) == 1
        new._plugin_loaded_listeners[0]([{"name": "test", "key": "test"}])
        if kind == "gateway":
            assert len(runner.calls) == 1
        else:
            assert methods_tools._with_activation({}, "test")["activation"]["name"] == "test"
    finally:
        for event in release.values():
            event.set()
        for thread in threads:
            thread.join(10)
        subscriptions = (runner._plugin_rewire_unsubscribe if kind == "gateway"
                         else methods_tools._plugin_activation_subscribed)
        if subscriptions:
            cancel = subscriptions.pop(old.scope_key, None)
            if cancel:
                cancel()
        methods_tools._plugin_activations.pop(old.scope_key, None)
        plugins._reset_plugin_managers_for_tests()


@pytest.mark.parametrize("kind", ["gateway", "tui"])
@pytest.mark.parametrize("failure", ["retired", "unrelated"])
def test_retirement_before_subscribe(monkeypatch, tmp_path, kind, failure):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    old = plugins.get_plugin_manager()
    original = plugins.subscribe_plugin_loaded_for_home
    cached = {"kept": {"name": "kept"}}
    methods_tools._plugin_activations[old.scope_key] = (old, cached)

    class Loop:
        def is_closed(self):
            return False

    def subscribe(manager, callback):
        if manager is not old:
            return original(manager, callback)
        assert unload_plugin_manager_for_home(tmp_path)
        if failure == "unrelated":
            raise RuntimeError("unrelated failure")
        return original(manager, callback)

    monkeypatch.setattr(plugins, "subscribe_plugin_loaded_for_home", subscribe)
    runner = GatewayPluginRewireMixin()

    def register():
        if kind == "gateway":
            runner._subscribe_plugin_rewire(old, loop=Loop())
        else:
            methods_tools._ensure_plugin_activation_listener()

    try:
        if failure == "unrelated":
            with pytest.raises(RuntimeError, match="unrelated failure"):
                register()
        else:
            with pytest.raises(RuntimeError, match="retired manager"):
                register()
        assert methods_tools._plugin_activations[old.scope_key] == (old, cached)
        monkeypatch.setattr(plugins, "subscribe_plugin_loaded_for_home", original)
        new = plugins.get_plugin_manager()
        if kind == "gateway":
            runner._subscribe_plugin_rewire(new, loop=Loop())
            subscriptions = runner._plugin_rewire_unsubscribe
        else:
            methods_tools._ensure_plugin_activation_listener()
            subscriptions = methods_tools._plugin_activation_subscribed
        assert subscriptions[new.scope_key].manager is new
        assert len(new._plugin_loaded_listeners) == 1
    finally:
        subscriptions = (runner._plugin_rewire_unsubscribe if kind == "gateway"
                         else methods_tools._plugin_activation_subscribed)
        if subscriptions:
            cancel = subscriptions.pop(old.scope_key, None)
            if cancel:
                cancel()
        methods_tools._plugin_activations.pop(old.scope_key, None)
        plugins._reset_plugin_managers_for_tests()


@pytest.mark.parametrize("kind", ["gateway", "tui"])
def test_retirement_between_subscription_and_real_discovery(monkeypatch, tmp_path, kind):
    from gateway.run_plugin_rewire import reload_plugins_verb
    from hermes_cli.plugins_activation import load_and_go_live

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    manager = plugins.get_plugin_manager()
    runner = GatewayPluginRewireMixin()
    if kind == "gateway":
        original = runner._subscribe_plugin_rewire
    else:
        original = methods_tools._ensure_plugin_activation_listener

    def retire_after_subscribe(*args, **kwargs):
        original(*args, **kwargs)
        assert unload_plugin_manager_for_home(tmp_path)

    if kind == "gateway":
        monkeypatch.setattr(runner, "_subscribe_plugin_rewire", retire_after_subscribe)
        result = reload_plugins_verb(runner, object())({})
        assert result["reloaded"] is False
        assert result["error"] == "plugin manager retired"
    else:
        monkeypatch.setattr(methods_tools, "_ensure_plugin_activation_listener", retire_after_subscribe)
        assert load_and_go_live("missing") is None
    assert manager._retired
    assert not manager._discovered
    assert not plugins._plugin_managers_by_home
    subscriptions = runner._plugin_rewire_unsubscribe if kind == "gateway" else methods_tools._plugin_activation_subscribed
    cancel = (subscriptions or {}).pop(manager.scope_key, None)
    if cancel:
        cancel()
    plugins._reset_plugin_managers_for_tests()


@pytest.mark.parametrize("kind", ["gateway", "tui"])
@pytest.mark.parametrize("operation", ["delete", "rename"])
def test_retired_subscription_does_not_recreate_profile(monkeypatch, tmp_path, kind, operation):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = plugins.get_plugin_manager()
    original = plugins.subscribe_plugin_loaded_for_home
    runner = GatewayPluginRewireMixin()

    def retire(manager, callback):
        cancel = original(manager, callback)
        assert unload_plugin_manager_for_home(home)
        if operation == "rename":
            home.rename(tmp_path / "renamed")
        else:
            home.rmdir()
        return cancel

    monkeypatch.setattr(plugins, "subscribe_plugin_loaded_for_home", retire)
    with pytest.raises(RuntimeError, match="retired manager"):
        if kind == "gateway":
            runner._subscribe_plugin_rewire(manager, loop=object())
        else:
            methods_tools._ensure_plugin_activation_listener(manager)
    assert not home.exists()
    assert not plugins._plugin_managers_by_home
    assert not manager._plugin_loaded_listeners
    plugins._reset_plugin_managers_for_tests()
