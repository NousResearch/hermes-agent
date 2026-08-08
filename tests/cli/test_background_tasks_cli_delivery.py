"""CLI delivery integration for external background-task completions.

A plugin-registered external task completed in a CLI parent session must reach
the owning CLI window exactly once through the existing idle completion drain
(``HermesCLI._drain_process_notifications``) and be durably acknowledged.
"""

import queue

from cli import HermesCLI
from types import SimpleNamespace

from agent.background_tasks import _create_external_background_tasks_service
from agent.host_context import bind_host_parent
from tools.process_registry import process_registry
from tools.async_delegation import get_durable_delegation


def _drain_one():
    import time

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not process_registry.completion_queue.empty():
            return process_registry.completion_queue.get_nowait()
        time.sleep(0.01)
    return None


def test_cli_delivery_reaches_bound_parent_once():
    parent = SimpleNamespace(session_id="cli-session-1")
    svc = _create_external_background_tasks_service(
        plugin_id="cli-plugin",
        parent_agent_resolver=lambda: parent,
    )
    with bind_host_parent(parent):
        handle = svc.register_external(external_id="run-1", label="cli run")

    assert svc.complete(handle, event_id="e1", summary="cli summary").accepted is True
    evt = _drain_one()
    assert evt is not None
    assert evt["session_key"] == "cli-session-1"
    process_registry.completion_queue.put(evt)  # restore for the drain under test

    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "cli-session-1"
    cli._session_db = None
    cli._pending_input = queue.Queue()

    cli._drain_process_notifications("test-cli")
    message = cli._pending_input.get_nowait()
    assert "cli summary" in message
    assert "cli run" in message
    assert cli._pending_input.empty()

    # Durable row is acknowledged as delivered.
    info = get_durable_delegation(evt["delegation_id"])
    assert info is not None
    assert info["delivery_state"] == "delivered"

    # A second drain delivers nothing (delivered once).
    cli._drain_process_notifications("test-cli")
    assert cli._pending_input.empty()


def test_cli_delivery_rejects_foreign_session():
    """A different CLI window cannot claim the completion."""
    parent = SimpleNamespace(session_id="cli-session-a")
    svc = _create_external_background_tasks_service(
        plugin_id="cli-plugin",
        parent_agent_resolver=lambda: parent,
    )
    with bind_host_parent(parent):
        handle = svc.register_external(external_id="run-1")

    assert svc.complete(handle, event_id="e1", summary="for session a").accepted is True
    evt = _drain_one()
    process_registry.completion_queue.put(evt)  # restore for the drain under test

    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "cli-session-b"
    cli._session_db = None
    cli._pending_input = queue.Queue()

    cli._drain_process_notifications("test-cli")
    assert cli._pending_input.empty()

    # The owning session can still drain it.
    owner = HermesCLI.__new__(HermesCLI)
    owner.session_id = "cli-session-a"
    owner._session_db = None
    owner._pending_input = queue.Queue()
    owner._drain_process_notifications("test-cli")
    assert not owner._pending_input.empty()
    assert owner._pending_input.get_nowait()
    info = get_durable_delegation(evt["delegation_id"])
    assert info["delivery_state"] == "delivered"
