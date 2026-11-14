import threading

import pytest

from hermes_cli import plugins, plugins_activation, plugins_activation_live, plugins_discovery


def test_go_live_notifies_unlocked_and_reads_one_locked_snapshot(monkeypatch):
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(plugins_discovery, "join_background_discovery", lambda: False)
    monkeypatch.setattr(manager, "_discover_and_load_inner", lambda: None)
    monkeypatch.setattr(manager, "_refresh_secret_sources_after_discovery", lambda: None)
    monkeypatch.setattr(manager, "_re_register_config_hooks_after_force", lambda: None)
    monkeypatch.setattr(plugins_activation, "activation_summaries", lambda _: [
        {"name": "new", "key": "new", "activated_now": {}, "deferred": {}}
    ])
    monkeypatch.setattr(plugins_activation_live, "plugin_skills", lambda _: [])
    monkeypatch.setattr(plugins_activation_live, "connect_plugin_mcp", lambda *_: [])
    callback_acquired = threading.Event()
    contender_started = threading.Event()
    contender_acquired = threading.Event()
    workers = []
    callback_completed = []

    def acquire(event):
        with manager._discovery_lock:
            event.set()

    def listener(_summaries):
        worker = threading.Thread(target=acquire, args=(callback_acquired,))
        workers.append(worker)
        worker.start()
        callback_completed.append(callback_acquired.wait(1))

    def portable():
        def contend():
            contender_started.set()
            acquire(contender_acquired)
        worker = threading.Thread(target=contend)
        workers.append(worker)
        worker.start()
        assert contender_started.wait(1)
        assert not contender_acquired.wait(0.1), "snapshot read without discovery lock"
        return {}

    manager.on_plugin_loaded(listener)
    monkeypatch.setattr(manager, "get_portable_mcp_servers", portable)
    result = plugins_activation.load_and_go_live("new")
    for worker in workers:
        worker.join(2)
        assert not worker.is_alive()
    assert callback_completed == [True]
    assert callback_acquired.is_set()
    assert contender_acquired.is_set()
    assert result is not None
    assert result["key"] == "new"
    assert result["live_now"] == {"mcp_servers": [], "skills": []}


def test_go_live_snapshot_belongs_to_its_forced_pass(monkeypatch):
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(plugins_discovery, "join_background_discovery", lambda: False)
    generation = []
    monkeypatch.setattr(manager, "_discover_and_load_inner", lambda: generation.append(len(generation) + 1))
    monkeypatch.setattr(manager, "_refresh_secret_sources_after_discovery", lambda: None)
    monkeypatch.setattr(manager, "_re_register_config_hooks_after_force", lambda: None)
    monkeypatch.setattr(plugins_activation, "activation_summaries", lambda _: [
        {"name": "new", "key": "new", "generation": generation[-1], "activated_now": {}, "deferred": {}}
    ])
    monkeypatch.setattr(manager, "get_portable_mcp_servers", lambda: {"generation": generation[-1]})
    monkeypatch.setattr(plugins_activation_live, "plugin_skills", lambda _: [str(generation[-1])])
    monkeypatch.setattr(plugins_activation_live, "connect_plugin_mcp", lambda _, portable: [portable])
    monkeypatch.setattr(plugins_activation_live, "live_notice", lambda _: "")

    def notify(_):
        if len(generation) == 1:
            worker = threading.Thread(target=lambda: manager.discover_and_load(force=True))
            worker.start()
            worker.join(2)
            assert not worker.is_alive(), "listener ran under discovery lock"

    monkeypatch.setattr(manager, "_notify_plugin_loaded", notify)
    result = plugins_activation.load_and_go_live("new")
    assert generation == [1, 2]
    assert result["generation"] == 1
    assert result["live_now"] == {"mcp_servers": [{"generation": 1}], "skills": ["1"]}


def test_snapshot_failure_still_notifies_outside_lock(monkeypatch):
    manager = plugins.PluginManager()
    monkeypatch.setattr(manager, "_discover_and_load_inner", lambda: None)
    monkeypatch.setattr(manager, "_refresh_secret_sources_after_discovery", lambda: None)
    notified = []
    error = RuntimeError("snapshot failed")

    def snapshot():
        raise error

    def notify(loaded_before):
        acquired = threading.Event()
        def acquire():
            with manager._discovery_lock:
                acquired.set()
        worker = threading.Thread(target=acquire)
        worker.start()
        worker.join(2)
        assert acquired.is_set(), "notification ran under discovery lock"
        notified.append(loaded_before)

    monkeypatch.setattr(manager, "_notify_plugin_loaded", notify)
    with pytest.raises(RuntimeError) as caught:
        manager.discover_and_load(snapshot=snapshot)
    assert caught.value is error
    assert manager._discovered
    assert notified == [frozenset()]
    manager.discover_and_load()
    assert notified == [frozenset()]
