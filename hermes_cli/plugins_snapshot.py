"""Read the loaded-plugin map without racing a late plugin loader.

``PluginManager._plugins`` is mutated by plugin loading, which can run on a deadline worker
that was abandoned when a load timed out but keeps running (``run_with_load_deadline``). A
reader that iterates the live dict can then raise
``RuntimeError: dictionary changed size during iteration``, which surfaced as a bogus plugin
failure ("Failed to load plugin 'openai'/'web-perplexity'") even though the plugin was fine.

``PluginManager.snapshot_plugins()`` copies the map under ``_plugins_lock``. Every reader
outside the manager should go through these helpers instead of touching ``_plugins``; they
also stay tolerant of the test doubles that stand in for a manager in unit tests (#132886).
"""

from typing import Any


def snapshot_plugin_items(manager: Any) -> list[tuple[str, Any]]:
    """Point-in-time ``(key, LoadedPlugin)`` pairs from any manager-like object."""
    snapshot = getattr(manager, "snapshot_plugins", None)
    if callable(snapshot):
        return list(snapshot())
    fallback: Any = getattr(manager, "_plugins", None) or {}
    return list(fallback.items())


def snapshot_plugin_values(manager: Any) -> list[Any]:
    """Point-in-time ``LoadedPlugin`` values from any manager-like object."""
    return [loaded for _key, loaded in snapshot_plugin_items(manager)]


def snapshot_plugin_keys(manager: Any) -> list[str]:
    """Point-in-time plugin keys from any manager-like object."""
    return [key for key, _loaded in snapshot_plugin_items(manager)]
