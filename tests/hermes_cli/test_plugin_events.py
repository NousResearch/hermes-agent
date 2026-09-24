"""Public event bridge for plugin backends (#116305 item 8, salvage of #116419).

A plugin backend pushes events to its own desktop half through
``hermes_cli.plugin_events`` instead of importing
``tui_gateway.server._broadcast_global_event``.
"""
from __future__ import annotations

import pytest

from hermes_cli import plugin_events


class _Peer:
    """A connected client as the gateway sees it (``tui_gateway.transport.Transport``)."""

    def __init__(self):
        self.frames: list[dict] = []

    def write(self, obj: dict) -> bool:
        self.frames.append(obj)
        return True

    def close(self) -> None:
        pass


def test_broadcast_reaches_a_registered_client_as_a_namespaced_global_event():
    import tui_gateway.server as server

    peer = _Peer()
    server.register_live_transport(peer)
    try:
        plugin_events.broadcast_plugin_event("rss-reader", "feed.updated", {"count": 3})
        plugin_events.broadcast_plugin_event("kanban", "changed")
    finally:
        server.unregister_live_transport(peer)

    assert peer.frames == [
        {"jsonrpc": "2.0", "method": "event",
         "params": {"type": "plugin.rss-reader.feed.updated", "session_id": "", "payload": {"count": 3}}},
        {"jsonrpc": "2.0", "method": "event",
         "params": {"type": "plugin.kanban.changed", "session_id": "", "payload": {}}},
    ]


@pytest.mark.parametrize(
    ("plugin_id", "event", "payload", "exc"),
    [
        ("", "items", None, ValueError),
        ("Bad Id", "items", None, ValueError),
        ("has/slash", "items", None, ValueError),
        ("other.x", "items", None, ValueError),  # a dotted id would spell plugin ``other``'s namespace
        ("ok", "", None, ValueError),
        ("ok", "bad name", None, ValueError),
        ("ok", "../x", None, ValueError),
        ("ok", ".leading", None, ValueError),
        ("ok", "a..b", None, ValueError),
        ("ok", "items", ["not", "a", "dict"], TypeError),
    ],
)
def test_names_that_cannot_form_a_namespaced_event_are_refused_before_emit(monkeypatch, plugin_id, event, payload, exc):
    import tui_gateway.server as server

    monkeypatch.setattr(server, "_broadcast_global_event", lambda *_a, **_k: pytest.fail("must not emit"))
    with pytest.raises(exc):
        plugin_events.broadcast_plugin_event(plugin_id, event, payload)  # type: ignore[arg-type]
