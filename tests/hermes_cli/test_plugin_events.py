"""Public event bridge for plugin backends (``plugin_api.py``).

Regression for #116305 item 8: a plugin backend pushes events to its own
desktop half through ``hermes_cli.plugin_events`` instead of importing
``tui_gateway.server._broadcast_global_event``.
"""
from __future__ import annotations

import pytest

from hermes_cli import plugin_events


def test_event_name_namespaces_under_the_plugin():
    assert plugin_events.plugin_event_name("rss-reader", "items") == "plugin.rss-reader.items"


def test_broadcast_reaches_the_global_stream_with_the_namespaced_name(monkeypatch):
    seen: list[tuple[str, object]] = []

    import tui_gateway.server as server

    monkeypatch.setattr(server, "_broadcast_global_event", lambda event, payload=None: seen.append((event, payload)))

    plugin_events.broadcast_plugin_event("rss-reader", "items", {"count": 3})

    assert seen == [("plugin.rss-reader.items", {"count": 3})]


def test_broadcast_defaults_the_payload_to_an_empty_dict(monkeypatch):
    seen: list[tuple[str, object]] = []

    import tui_gateway.server as server

    monkeypatch.setattr(server, "_broadcast_global_event", lambda event, payload=None: seen.append((event, payload)))

    plugin_events.broadcast_plugin_event("kanban", "changed")

    assert seen == [("plugin.kanban.changed", {})]


@pytest.mark.parametrize(
    ("plugin_id", "event"),
    [
        ("", "items"),
        ("Bad Id", "items"),
        ("has/slash", "items"),
        ("ok", ""),
        ("ok", "bad name"),
        ("ok", "with.dot"),
        ("ok", "plugin.other.items"),
    ],
)
def test_invalid_names_raise(plugin_id, event):
    with pytest.raises(ValueError):
        plugin_events.plugin_event_name(plugin_id, event)


def test_payload_must_be_a_dict():
    with pytest.raises(TypeError):
        plugin_events.broadcast_plugin_event("ok", "items", ["not", "a", "dict"])  # type: ignore[arg-type]
