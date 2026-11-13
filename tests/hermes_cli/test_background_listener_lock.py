import threading

from hermes_cli import plugins, plugins_discovery


def test_background_listener_runs_without_discovery_lock(monkeypatch):
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, 'get_plugin_manager', lambda: manager)
    monkeypatch.setattr(plugins, '_background_discovery_thread', None)
    monkeypatch.setattr(plugins, '_persist_plugin_toolset_keys', lambda **kwargs: None)
    monkeypatch.setattr(manager, '_discover_and_load_inner', lambda: None)
    monkeypatch.setattr(manager, '_refresh_secret_sources_after_discovery', lambda: None)
    monkeypatch.setattr('hermes_cli.plugins_activation.activation_summaries',
                        lambda manager: [{'key': 'new'}])
    acquired = []

    def listener(_summaries):
        def inspect():
            with manager._discovery_lock:
                acquired.append(True)
        thread = threading.Thread(target=inspect)
        thread.start()
        thread.join(1)
        acquired.append('completed' if not thread.is_alive() else 'blocked')

    manager.on_plugin_loaded(listener)
    plugins_discovery.start_background_plugin_discovery()
    plugins._background_discovery_thread.join(5)
    assert acquired == [True, 'completed']
