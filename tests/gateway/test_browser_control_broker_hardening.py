import threading

import pytest

from gateway.browser_control_broker import (
    BrowserControlBroker,
    browser_control_enabled,
    ControllerCancelled,
    ControllerScope,
    ControllerRejected,
    ControllerTimeout,
    ControllerUnavailable,
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


def _start_pending(broker, scope, *, tool_call_id="tool-call-fixture"):
    outcome = {}
    ready = threading.Event()
    frames = []

    def send(frame):
        frames.append(frame)
        if frame["method"] == "browser.controller.command":
            ready.set()

    broker.attach(scope, send, owner="owner-fixture")

    def run():
        try:
            outcome["result"] = broker.dispatch(
                scope,
                action="controller.noop",
                arguments={},
                tool_call_id=tool_call_id,
            )
        except Exception as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    assert ready.wait(timeout=1.0)
    return thread, outcome, frames


def test_detach_emits_cancel_before_controller_is_removed():
    broker = BrowserControlBroker(command_timeout=1.0)
    scope = _scope()
    thread, outcome, frames = _start_pending(broker, scope)

    broker.detach(scope)
    thread.join(timeout=1.0)

    assert isinstance(outcome.get("error"), ControllerCancelled)
    assert [frame["method"] for frame in frames] == [
        "browser.controller.command",
        "browser.controller.cancel",
    ]


def test_dispatch_revalidates_selected_controller_after_detach_race():
    broker = BrowserControlBroker(command_timeout=0.2)
    scope = _scope()
    broker.attach(scope, lambda _frame: None)
    selected = threading.Event()
    resume = threading.Event()
    original_select = broker.select

    def paused_select(candidate_scope, capability):
        controller = original_select(candidate_scope, capability)
        selected.set()
        assert resume.wait(timeout=1.0)
        return controller

    broker.select = paused_select
    outcome = {}

    def run():
        try:
            broker.dispatch(scope, action="controller.noop", arguments={})
        except Exception as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    assert selected.wait(timeout=1.0)
    broker.detach(scope)
    resume.set()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), ControllerUnavailable)
    assert broker.pending_count == 0


def test_completion_requires_the_same_scope_as_the_pending_command():
    broker = BrowserControlBroker(command_timeout=1.0)
    scope = _scope()
    thread, outcome, frames = _start_pending(broker, scope)
    command_id = frames[0]["params"]["command_id"]

    assert broker.complete(
        command_id,
        scope=_scope(principal_id="other-principal"),
        ok=True,
        result={"unsafe": True},
    ) is False
    assert thread.is_alive()
    assert broker.complete(
        command_id,
        scope=scope,
        ok=True,
        result={"safe": True},
    ) is True
    thread.join(timeout=1.0)

    assert outcome.get("result") == {"safe": True}
    assert broker.pending_count == 0


def test_reattach_cancels_pending_work_from_the_previous_controller_generation():
    broker = BrowserControlBroker(command_timeout=1.0)
    scope = _scope()
    thread, outcome, frames = _start_pending(broker, scope)
    old_command_id = frames[0]["params"]["command_id"]

    broker.attach(scope, lambda _frame: None, owner="replacement-owner")
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), ControllerCancelled)
    assert broker.complete(old_command_id, scope=scope, ok=True, result={}) is False


def test_session_lookup_fails_closed_on_ambiguity_and_owner_detach_is_scoped():
    broker = BrowserControlBroker()
    first = _scope(controller_id="controller-one")
    second = _scope(controller_id="controller-two")
    other = _scope(
        session_id="other-session",
        controller_id="controller-other",
        transport_family="cloud-ticket-ws",
    )
    broker.attach(first, lambda _frame: None, owner="owner-shared")
    broker.attach(second, lambda _frame: None, owner="owner-shared")
    broker.attach(other, lambda _frame: None, owner="owner-other")

    assert broker.scope_for_session(
        session_id="session-fixture",
        principal_id="principal-fixture",
        transport_family="local-api",
    ) is None
    assert broker.scope_for_session(
        session_id="other-session",
        principal_id="principal-fixture",
        transport_family="cloud-ticket-ws",
    ) == other

    assert broker.detach_owner("owner-shared") == 2
    assert broker.scope_for_session(
        session_id="other-session",
        principal_id="principal-fixture",
        transport_family="cloud-ticket-ws",
    ) == other
    assert broker.detach_owner("missing-owner") == 0

    broker.reset()
    assert broker.scope_for_session(
        session_id="other-session",
        principal_id="principal-fixture",
        transport_family="cloud-ticket-ws",
    ) is None
    assert broker.pending_count == 0


def test_session_lookup_requires_exact_server_principal_and_transport_family():
    broker = BrowserControlBroker()
    local = _scope(
        principal_id="principal:api:local",
        controller_id="controller-local",
        transport_family="local-api",
    )
    remote = _scope(
        principal_id="principal:api:remote",
        controller_id="controller-remote",
        transport_family="remote-api",
    )
    broker.attach(local, lambda _frame: None, owner="owner-local")
    broker.attach(remote, lambda _frame: None, owner="owner-remote")

    assert broker.scope_for_session(session_id="session-fixture") is None
    assert broker.scope_for_session(
        session_id="session-fixture",
        principal_id="principal:api:local",
        transport_family="local-api",
    ) == local
    assert broker.scope_for_session(
        session_id="session-fixture",
        principal_id="principal:api:local",
        transport_family="remote-api",
    ) is None
    assert broker.scope_for_session(
        session_id="session-fixture",
        principal_id="principal:api:attacker",
        transport_family="local-api",
    ) is None


def test_feature_flag_requires_literal_boolean_true():
    assert browser_control_enabled({}) is False
    assert browser_control_enabled(
        {"browser": {"extension_control": {"enabled": False}}}
    ) is False
    assert browser_control_enabled(
        {"browser": {"extension_control": {"enabled": True}}}
    ) is True
    for ambiguous in ("true", "false", "yes", 1, [], {}):
        assert browser_control_enabled(
            {"browser": {"extension_control": {"enabled": ambiguous}}}
        ) is False


def test_transport_teardown_can_cancel_waiters_without_writing_to_closing_peer():
    broker = BrowserControlBroker(command_timeout=1.0)
    scope = _scope()
    thread, outcome, frames = _start_pending(broker, scope)

    assert broker.detach_owner("owner-fixture", notify_controller=False) == 1
    thread.join(timeout=1.0)

    assert isinstance(outcome.get("error"), ControllerCancelled)
    assert [frame["method"] for frame in frames] == ["browser.controller.command"]
    assert broker.pending_count == 0


def test_stale_owner_teardown_cannot_detach_replacement_controller_generation():
    broker = BrowserControlBroker()
    scope = _scope()
    first_owner = object()
    live_owner = object()
    broker.attach(scope, lambda frame: None, owner=first_owner)
    broker.attach(scope, lambda frame: None, owner=live_owner)

    broker.detach(
        scope,
        owner=first_owner,
        notify_controller=False,
    )

    selected = broker.select(scope, "controller.noop")
    assert selected is not None
    assert selected.owner is live_owner


def test_completion_winning_at_timeout_boundary_is_not_misreported_as_timeout():
    broker = BrowserControlBroker(command_timeout=0.01)
    scope = _scope()

    class BoundaryEvent:
        def set(self):
            pass

        def wait(self, timeout):
            assert broker.complete(
                command_id,
                scope=scope,
                ok=True,
                result={"boundary": "completed"},
            )
            return False

    def send(frame):
        nonlocal command_id
        command_id = frame["params"]["command_id"]
        broker._pending[command_id].event = BoundaryEvent()

    command_id = ""
    broker.attach(scope, send)
    assert broker.dispatch(scope, action="controller.noop") == {
        "boundary": "completed"
    }


def test_timeout_marks_terminal_and_emits_cancel_to_controller():
    broker = BrowserControlBroker(command_timeout=0.01)
    scope = _scope()
    frames = []
    broker.attach(scope, frames.append)

    with pytest.raises(ControllerTimeout):
        broker.dispatch(
            scope,
            action="controller.noop",
            tool_call_id="tool-timeout",
        )

    assert [frame["method"] for frame in frames] == [
        "browser.controller.command",
        "browser.controller.cancel",
    ]
    assert broker.pending_count == 0


def test_non_boolean_success_values_fail_closed():
    broker = BrowserControlBroker(command_timeout=1.0)
    scope = _scope()

    def send(frame):
        assert broker.complete(
            frame["params"]["command_id"],
            scope=scope,
            ok="false",
            result={"spoofed": True},
        )

    broker.attach(scope, send)
    with pytest.raises(ControllerRejected):
        broker.dispatch(scope, action="controller.noop")
