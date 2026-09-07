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


def test_same_identity_reattach_preserves_pending_work_and_completes_on_new_owner():
    broker = BrowserControlBroker(command_timeout=1.0)
    scope = _scope()
    thread, outcome, frames = _start_pending(broker, scope)
    command_id = frames[0]["params"]["command_id"]
    replacement_frames = []

    broker.attach(scope, replacement_frames.append, owner="replacement-owner")
    assert thread.is_alive()
    assert broker.pending_count == 1
    assert broker.complete(
        command_id,
        scope=scope,
        ok=True,
        result={"reconnected": True},
    ) is True
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert outcome.get("result") == {"reconnected": True}
    assert "error" not in outcome
    assert replacement_frames == []


def test_same_stable_identity_can_renegotiate_capabilities_without_ambiguity():
    broker = BrowserControlBroker(command_timeout=1.0)
    original = _scope()
    thread, outcome, frames = _start_pending(broker, original)
    command_id = frames[0]["params"]["command_id"]
    refreshed = _scope(capabilities=frozenset({"controller.noop", "browser_snapshot"}))

    broker.attach(refreshed, lambda _frame: None, owner="replacement-owner")

    assert broker.scope_for_session(
        session_id="session-fixture",
        principal_id="principal-fixture",
        transport_family="local-api",
    ) == refreshed
    assert broker.complete(
        command_id,
        scope=refreshed,
        ok=True,
        result={"capabilities": "refreshed"},
    ) is True
    thread.join(timeout=1.0)
    assert outcome.get("result") == {"capabilities": "refreshed"}


def test_transport_disconnect_parks_pending_and_reconnect_completes_it():
    broker = BrowserControlBroker(command_timeout=1.0)
    scope = _scope()
    thread, outcome, frames = _start_pending(broker, scope)
    command_id = frames[0]["params"]["command_id"]

    assert broker.disconnect_owner("owner-fixture") == 1
    assert thread.is_alive()
    assert broker.pending_count == 1
    assert broker.select(scope, "controller.noop") is None
    with pytest.raises(ControllerUnavailable):
        broker.dispatch(scope, action="controller.noop")

    broker.attach(scope, lambda _frame: None, owner="replacement-owner")
    assert broker.complete(
        command_id,
        scope=scope,
        ok=True,
        result={"after": "reconnect"},
    ) is True
    thread.join(timeout=1.0)
    assert outcome.get("result") == {"after": "reconnect"}


def test_timeout_while_disconnected_flushes_cancel_before_new_dispatch():
    broker = BrowserControlBroker(command_timeout=0.02)
    scope = _scope()
    thread, outcome, frames = _start_pending(broker, scope)
    command_id = frames[0]["params"]["command_id"]

    assert broker.disconnect_owner("owner-fixture") == 1
    thread.join(timeout=1.0)
    assert isinstance(outcome.get("error"), ControllerTimeout)
    assert broker.pending_count == 0

    replacement_frames = []
    broker.attach(scope, replacement_frames.append, owner="replacement-owner")
    assert replacement_frames == [
        {
            "method": "browser.controller.cancel",
            "params": {
                "command_id": command_id,
                "tool_call_id": "tool-call-fixture",
            },
        }
    ]

    second = {}

    def run_second():
        try:
            second["result"] = broker.dispatch(
                scope,
                action="controller.noop",
                tool_call_id="tool-call-second",
            )
        except Exception as exc:
            second["error"] = exc

    second_thread = threading.Thread(target=run_second)
    second_thread.start()
    while len(replacement_frames) < 2:
        second_thread.join(timeout=0.01)
    assert replacement_frames[1]["method"] == "browser.controller.command"
    assert broker.complete(
        replacement_frames[1]["params"]["command_id"],
        scope=scope,
        ok=True,
        result={"second": True},
    ) is True
    second_thread.join(timeout=1.0)
    assert second.get("result") == {"second": True}


def test_old_transport_owner_cannot_complete_after_same_identity_reconnect():
    broker = BrowserControlBroker(command_timeout=1.0)
    scope = _scope()
    thread, outcome, frames = _start_pending(broker, scope)
    command_id = frames[0]["params"]["command_id"]
    refreshed_scope = _scope(capabilities=frozenset({"controller.noop", "browser_snapshot"}))

    broker.attach(refreshed_scope, lambda _frame: None, owner="replacement-owner")

    assert broker.is_owner(refreshed_scope, "owner-fixture") is False
    assert broker.is_owner(refreshed_scope, "replacement-owner") is True
    assert broker.complete(
        command_id,
        scope=scope,
        ok=True,
        result={"stale": True},
    ) is False
    assert broker.complete(
        command_id,
        scope=refreshed_scope,
        ok=True,
        result={"fresh": True},
    ) is True
    thread.join(timeout=1.0)
    assert outcome.get("result") == {"fresh": True}


def test_cancel_with_pre_reconnect_scope_still_cancels_same_stable_identity():
    broker = BrowserControlBroker(command_timeout=1.0)
    original = _scope()
    thread, outcome, frames = _start_pending(
        broker,
        original,
        tool_call_id="tool-call-before-reconnect",
    )
    refreshed = _scope(capabilities=frozenset({"controller.noop", "browser_snapshot"}))
    refreshed_frames = []
    broker.attach(refreshed, refreshed_frames.append, owner="replacement-owner")

    assert broker.cancel(
        original,
        tool_call_id="tool-call-before-reconnect",
    ) is True
    thread.join(timeout=1.0)
    assert isinstance(outcome.get("error"), ControllerCancelled)
    assert refreshed_frames == [
        {
            "method": "browser.controller.cancel",
            "params": {
                "command_id": frames[0]["params"]["command_id"],
                "tool_call_id": "tool-call-before-reconnect",
            },
        }
    ]


def test_different_stable_identity_hard_replaces_and_cancels_pending_work():
    broker = BrowserControlBroker(command_timeout=1.0)
    original = _scope()
    thread, outcome, frames = _start_pending(broker, original)
    command_id = frames[0]["params"]["command_id"]
    replacement = _scope(controller_id="different-controller")

    broker.attach(replacement, lambda _frame: None, owner="replacement-owner")

    assert broker.complete(
        command_id,
        scope=replacement,
        ok=True,
        result={"unsafe": True},
    ) is False
    assert broker.complete(
        command_id,
        scope=original,
        ok=True,
        result={"original": True},
    ) is False
    thread.join(timeout=1.0)
    assert isinstance(outcome.get("error"), ControllerCancelled)


def test_different_controller_identity_hard_replaces_parked_session_controller():
    broker = BrowserControlBroker(command_timeout=1.0)
    original = _scope()
    thread, outcome, _frames = _start_pending(broker, original)
    assert broker.disconnect_owner("owner-fixture") == 1

    replacement = ControllerScope(
        principal_id=original.principal_id,
        profile_id=original.profile_id,
        session_id=original.session_id,
        controller_id="replacement-controller",
        browser_profile_id=original.browser_profile_id,
        transport_family=original.transport_family,
        capabilities=original.capabilities,
    )
    broker.attach(replacement, lambda _frame: None, owner="replacement-owner")

    thread.join(timeout=1.0)
    assert isinstance(outcome.get("error"), ControllerCancelled)
    assert broker.scope_for_session(
        session_id=original.session_id,
        principal_id=original.principal_id,
        transport_family=original.transport_family,
    ) == replacement
    assert broker.select(original, "controller.noop") is None
    assert broker.select(replacement, "controller.noop") is not None


def test_failed_deferred_cancel_flush_keeps_controller_offline():
    broker = BrowserControlBroker(command_timeout=0.01)
    scope = _scope()
    thread, outcome, _frames = _start_pending(broker, scope)
    assert broker.disconnect_owner("owner-fixture") == 1
    thread.join(timeout=1.0)
    assert isinstance(outcome.get("error"), ControllerTimeout)

    def fail_send(_frame):
        raise ConnectionError("fixture flush failed")

    with pytest.raises(ConnectionError, match="flush"):
        broker.attach(scope, fail_send, owner="replacement-owner")
    assert broker.select(scope, "controller.noop") is None


def test_session_lane_replacement_and_owner_detach_are_scoped():
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
    ) == second
    assert broker.scope_for_session(
        session_id="other-session",
        principal_id="principal-fixture",
        transport_family="cloud-ticket-ws",
    ) == other

    broker.detach(second, owner="owner-shared")
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

    broker.detach(scope, owner="owner-fixture", notify_controller=False)
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
