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


def test_broadcast_from_a_turn_isolation_child_reaches_the_parent_gateways_clients(monkeypatch):
    """``dashboard.turn_isolation`` runs plugin agent-side code (tools, hooks, slash commands) in the
    compute-host child, where no client is connected: the frame must ride the host pipe to the parent
    ``hermes serve`` and fan out there instead of dying at ``logger.debug``."""
    import io
    import json
    import os
    import threading

    import tui_gateway.server as server
    from tui_gateway import compute_host

    monkeypatch.setenv("HERMES_COMPUTE_HOST_HEARTBEAT_SECS", "0")
    monkeypatch.setenv("HERMES_COMPUTE_HOST_CHILD", "0")
    child_stdout = io.StringIO()
    stdin_r, stdin_w = os.pipe()
    child = threading.Thread(
        target=compute_host.run_host, kwargs={"stdin": os.fdopen(stdin_r), "stdout": child_stdout}, daemon=True)
    child.start()
    try:
        deadline = threading.Event()
        for _ in range(100):  # the child registers its transport right after ``hello``
            if any(isinstance(t, compute_host._HostTransport) for t in list(server._live_transports)):
                break
            deadline.wait(0.05)
        plugin_events.broadcast_plugin_event("rss-reader", "feed.updated", {"count": 3})
    finally:
        os.close(stdin_w)
        child.join(timeout=10)
    assert not child.is_alive()
    assert not any(isinstance(t, compute_host._HostTransport) for t in list(server._live_transports))

    piped = [json.loads(line) for line in child_stdout.getvalue().splitlines()]
    relayed = [f for f in piped if f.get("type") == "rpc"]
    assert len(relayed) == 1 and relayed[0]["sid"] == ""

    # Parent side: the supervisor hands the child's frame to the bridge, which must broadcast it.
    peer = _Peer()
    server.register_live_transport(peer)
    try:
        server._relay_compute_host_rpc(relayed[0]["message"])
    finally:
        server.unregister_live_transport(peer)
    assert [f["params"]["type"] for f in peer.frames] == ["plugin.rss-reader.feed.updated"]
    assert peer.frames[0]["params"]["payload"] == {"count": 3}


def test_broadcast_without_a_gateway_module_is_a_logged_no_op(monkeypatch, caplog):
    """A plugin-only process (``hermes plugins validate`` importing the backend) has no
    ``tui_gateway.server``; the documented "safe from any handler" promise must hold there too."""
    import builtins
    import logging

    real_import = builtins.__import__

    def _no_gateway(name, *args, **kwargs):
        if name == "tui_gateway.server":
            raise ImportError("No module named 'tui_gateway.server'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_gateway)
    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugin_events"):
        plugin_events.broadcast_plugin_event("rss-reader", "feed.updated")
    assert "plugin.rss-reader.feed.updated" in caplog.text


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
