import threading
import time

import pytest

from gateway.browser_control_broker import (
    BrowserControlBroker,
    ControllerCancelled,
    ControllerScope,
    ControllerTicketInvalid,
)


def _scope(**overrides):
    values = {
        "principal_id": "principal-fixture",
        "profile_id": "default",
        "session_id": "session-fixture",
        "controller_id": "controller-fixture",
        "browser_profile_id": "browser-profile-fixture",
        "transport_family": "local-api",
        "capabilities": frozenset({"controller.noop"}),
    }
    values.update(overrides)
    return ControllerScope(**values)


def test_registration_ticket_is_short_lived_single_use_and_identity_bound():
    now = [100.0]
    broker = BrowserControlBroker(ticket_ttl=30.0, clock=lambda: now[0])
    scope = _scope()

    ticket = broker.mint_ticket(scope)
    assert len(ticket.value) >= 32
    assert ticket.expires_at == 130.0
    assert broker.consume_ticket(ticket.value) == scope
    with pytest.raises(ControllerTicketInvalid, match="unknown|consumed"):
        broker.consume_ticket(ticket.value)

    expired = broker.mint_ticket(scope)
    now[0] = 131.0
    with pytest.raises(ControllerTicketInvalid, match="expired"):
        broker.consume_ticket(expired.value)


def test_controller_selection_requires_exact_principal_profile_session_controller_and_browser_profile():
    broker = BrowserControlBroker()
    scope = _scope()
    broker.attach(scope, lambda _frame: None)

    assert broker.select(scope, "controller.noop") is not None
    for field, value in (
        ("principal_id", "other-principal"),
        ("profile_id", "other-profile"),
        ("session_id", "other-session"),
        ("controller_id", "other-controller"),
        ("browser_profile_id", "other-browser-profile"),
        ("transport_family", "remote-api"),
    ):
        assert broker.select(_scope(**{field: value}), "controller.noop") is None
    assert broker.select(scope, "browser_navigate") is None


def test_noop_round_trip_uses_controller_and_returns_result_without_enabling_browser_actions():
    broker = BrowserControlBroker(command_timeout=1.0)
    scope = _scope()

    def send(frame):
        assert frame["method"] == "browser.controller.command"
        assert frame["params"]["action"] == "controller.noop"
        broker.complete(
            frame["params"]["command_id"],
            ok=True,
            result={"echo": frame["params"]["arguments"]["echo"]},
        )

    broker.attach(scope, send)
    result = broker.dispatch(
        scope,
        action="controller.noop",
        arguments={"echo": "phase-4"},
        tool_call_id="tool-call-fixture",
    )
    assert result == {"echo": "phase-4"}
    assert broker.pending_count == 0
    assert broker.select(scope, "browser_navigate") is None


def test_cancellation_targets_only_the_matching_pending_command_and_cleans_up():
    broker = BrowserControlBroker(command_timeout=2.0)
    scope = _scope()
    frames = []
    command_ready = threading.Event()

    def send(frame):
        frames.append(frame)
        if frame["method"] == "browser.controller.command":
            command_ready.set()

    broker.attach(scope, send)
    outcome = {}

    def run_dispatch():
        try:
            broker.dispatch(
                scope,
                action="controller.noop",
                arguments={},
                tool_call_id="tool-call-cancelled",
            )
        except Exception as exc:  # asserted below
            outcome["error"] = exc

    thread = threading.Thread(target=run_dispatch)
    thread.start()
    assert command_ready.wait(timeout=1.0)

    assert broker.cancel(scope, tool_call_id="wrong-tool-call") is False
    assert broker.cancel(scope, tool_call_id="tool-call-cancelled") is True
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), ControllerCancelled)
    cancel_frames = [frame for frame in frames if frame["method"] == "browser.controller.cancel"]
    assert len(cancel_frames) == 1
    assert cancel_frames[0]["params"]["command_id"] == frames[0]["params"]["command_id"]
    assert broker.pending_count == 0


def test_detach_fails_pending_work_closed_and_late_completion_is_ignored():
    broker = BrowserControlBroker(command_timeout=2.0)
    scope = _scope()
    command_id = []
    command_ready = threading.Event()

    def send(frame):
        if frame["method"] == "browser.controller.command":
            command_id.append(frame["params"]["command_id"])
            command_ready.set()

    broker.attach(scope, send)
    outcome = {}

    def run_dispatch():
        try:
            broker.dispatch(scope, action="controller.noop", arguments={})
        except Exception as exc:  # asserted below
            outcome["error"] = exc

    thread = threading.Thread(target=run_dispatch)
    thread.start()
    assert command_ready.wait(timeout=1.0)
    broker.detach(scope)
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), ControllerCancelled)
    assert broker.complete(command_id[0], ok=True, result={}) is False
    assert broker.pending_count == 0
