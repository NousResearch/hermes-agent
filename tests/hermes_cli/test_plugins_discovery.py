"""Safe workers must not join or start plugin discovery."""

from agent import safe_worker_policy
from hermes_cli import plugins, plugins_discovery


def test_safe_worker_skips_plugin_discovery(monkeypatch):
    monkeypatch.setattr(safe_worker_policy, "safe_worker_enabled", lambda: True)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("safe worker attempted plugin discovery")

    monkeypatch.setattr(plugins, "get_plugin_manager", unexpected)
    monkeypatch.setattr(plugins_discovery, "join_background_discovery", unexpected)

    plugins.discover_plugins()
    plugins_discovery.start_background_plugin_discovery()
