import threading

from agent import plugin_stream_hooks
from hermes_cli import plugins, plugins_activation


def manager_with_empty_sweep(monkeypatch):
    manager = plugins.PluginManager()
    monkeypatch.setattr(manager, "_discover_and_load_inner", lambda: None)
    monkeypatch.setattr(manager, "_refresh_secret_sources_after_discovery", lambda: None)
    monkeypatch.setattr(manager, "_re_register_config_hooks_after_force", lambda: None)
    return manager


def test_lazy_discovery_listener_can_read_manager_from_other_thread(monkeypatch):
    manager = manager_with_empty_sweep(monkeypatch)
    monkeypatch.setattr(plugins_activation, "activation_summaries", lambda _: [{"key": "new"}])
    done = threading.Event()
    read = threading.Event()
    results = []

    def listener(_):
        def reader():
            with manager._discovery_lock:
                read.set()
        worker = threading.Thread(target=reader)
        worker.start()
        results.append(read.wait(2))
        worker.join(2)
        done.set()

    manager.on_plugin_loaded(listener)
    plugin_stream_hooks._schedule_lazy_discovery(manager, manager.discover_and_load)
    assert done.wait(5)
    assert results == [True]


def test_notifications_keep_their_discovery_generation(monkeypatch):
    manager = manager_with_empty_sweep(monkeypatch)
    generation = []
    monkeypatch.setattr(manager, "_discover_and_load_inner", lambda: generation.append(len(generation) + 1))
    monkeypatch.setattr(plugins_activation, "activation_summaries", lambda _: [{"key": "new", "generation": generation[-1]}])
    received = []
    manager.on_plugin_loaded(lambda summaries: received.append(summaries))
    notify = manager._notify_plugin_loaded

    def interleave(summaries):
        if len(generation) == 1:
            worker = threading.Thread(target=lambda: manager.discover_and_load(force=True))
            worker.start()
            worker.join(5)
            assert not worker.is_alive()
        notify(summaries)

    monkeypatch.setattr(manager, "_notify_plugin_loaded", interleave)
    manager.discover_and_load()
    assert [entry[0]["generation"] for entry in received] == [2, 1]


def test_force_reload_waits_for_retired_worker_outside_discovery_lock(monkeypatch):
    manager = manager_with_empty_sweep(monkeypatch)
    started = threading.Event()
    proceed = threading.Event()
    finished = threading.Event()

    def observer():
        started.set()
        assert proceed.wait(5)
        with manager._discovery_lock:
            finished.set()

    worker = threading.Thread(target=observer)
    worker.start()
    assert started.wait(2)
    monkeypatch.setattr(plugin_stream_hooks, "retire_plugin_observer_dispatchers", lambda *a, **kw: [worker])

    def stop(dispatcher, *, timeout, discard_pending):
        proceed.set()
        dispatcher.join(timeout)

    monkeypatch.setattr(plugin_stream_hooks, "_stop_dispatcher", stop)
    try:
        manager.discover_and_load(force=True)
        assert finished.is_set()
    finally:
        proceed.set()
        worker.join(5)
