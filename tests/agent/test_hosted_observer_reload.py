import threading


def test_multiple_hosted_callbacks_share_worker_cap(monkeypatch, tmp_path):
    from agent import plugin_stream_hooks as observers
    from hermes_cli import plugins
    from hermes_cli.plugin_host import PluginHost

    manager = plugins.PluginManager(scope_key=str(tmp_path / "profile"))
    manager._discovered = True
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    release = threading.Event()
    started = [threading.Event() for _ in range(3)]
    host = object.__new__(PluginHost)
    host._generation = 1

    def invoke(ref, args, kwargs, *, generation):
        started[ref].set()
        release.wait(timeout=30)

    monkeypatch.setattr(host, "invoke", invoke)
    manifest = plugins.PluginManifest(name="hosted-observer")
    context = plugins.PluginContext(manifest, manager)
    try:
        for ref in range(3):
            context.register_hook("memory_prefetch", host._callable_proxy({"__callable__": ref}))
        assert observers.enqueue_plugin_observer_hook("memory_prefetch", event_id=0)
        assert all(event.wait(timeout=5) for event in started[:2])
        manager.unload(manifest)
        assert sum(d.thread is not None and d.thread.is_alive() for d in observers._dispatchers.values()) == 2
    finally:
        release.set()
        manager.unload(manifest)
        observers.shutdown_plugin_observer_dispatcher(timeout=5)


def test_hosted_reload_bounds_blocked_workers_and_recovers(monkeypatch, tmp_path):
    from agent import plugin_stream_hooks as observers
    from hermes_cli import plugins
    from hermes_cli.plugin_host import PluginHost

    manager = plugins.PluginManager(scope_key=str(tmp_path / "profile"))
    manager._discovered = True
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    release = threading.Event()
    started = [threading.Event() for _ in range(2)]
    recovered = threading.Event()

    def invoke(ref, args, kwargs, *, generation):
        if ref < 2:
            started[ref].set()
            release.wait(timeout=30)
        else:
            recovered.set()

    host = object.__new__(PluginHost)
    host._generation = 1
    monkeypatch.setattr(host, "invoke", invoke)
    manifest = plugins.PluginManifest(name="hosted-observer")

    def register(ref):
        proxy = PluginHost._callable_proxy(host, {"__callable__": ref})
        plugins.PluginContext(manifest, manager).register_hook("memory_prefetch", proxy)

    try:
        for ref in range(2):
            register(ref)
            assert observers.enqueue_plugin_observer_hook("memory_prefetch", event_id=ref)
            assert started[ref].wait(timeout=5)
            manager.unload(manifest)
        for ref in range(2, 7):
            register(ref)
            assert not observers.enqueue_plugin_observer_hook("memory_prefetch", event_id=ref)
            manager.unload(manifest)
        other_manager = plugins.PluginManager(scope_key=str(tmp_path / "other-profile"))
        other_manager._discovered = True
        other_proxy = host._callable_proxy({"__callable__": 8})
        plugins.PluginContext(manifest, other_manager).register_hook("memory_prefetch", other_proxy)
        monkeypatch.setattr(plugins, "get_plugin_manager", lambda: other_manager)
        assert observers.enqueue_plugin_observer_hook("memory_prefetch", event_id="other")
        assert recovered.wait(timeout=5)
        recovered.clear()
        other_manager.unload(manifest)
        monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
        release.set()
        for dispatcher in list(observers._dispatchers.values()):
            if dispatcher.thread:
                dispatcher.thread.join(timeout=5)
        register(7)
        assert observers.enqueue_plugin_observer_hook("memory_prefetch", event_id=7)
        assert recovered.wait(timeout=5)
    finally:
        release.set()
        manager.unload(manifest)
        observers.shutdown_plugin_observer_dispatcher(timeout=5)
