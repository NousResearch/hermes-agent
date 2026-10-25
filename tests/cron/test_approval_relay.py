"""Contracts for presentation-only cron approval lifetimes."""

import importlib
import pytest


def relay_module():
    spec = importlib.util.find_spec("cron.approval_relay")
    assert spec is not None, "Cron invocations need bounded approval presentation"
    return importlib.import_module("cron.approval_relay")


def test_original_deadline_and_cancellation_are_latched():
    relay = relay_module()
    clock = [100.0]
    live = [True]
    lifetime = relay.ApprovalLifetime(110.0, lambda: live[0], clock=lambda: clock[0])
    assert lifetime.remaining_seconds == 10.0
    clock[0] = 110.0
    with pytest.raises(relay.ApprovalUnavailable):
        lifetime.check()
    clock[0] = 101.0
    with pytest.raises(relay.ApprovalUnavailable):
        lifetime.check()

    cancelled = relay.ApprovalLifetime(120.0, lambda: live[0], clock=lambda: clock[0])
    live[0] = False
    with pytest.raises(relay.ApprovalUnavailable):
        cancelled.request(lambda: pytest.fail("Cancelled wait ran side effect"))
    live[0] = True
    with pytest.raises(relay.ApprovalUnavailable):
        cancelled.check()


def test_presentation_registration_is_profile_scoped_and_disposable(tmp_path):
    relay = relay_module()
    first = relay.PresenterRegistration(str(tmp_path / "first"), "guard", lambda *_: None)
    second = relay.PresenterRegistration(str(tmp_path / "second"), "guard", lambda *_: None)
    try:
        assert relay.presenters_for(first.home) == {"guard": first}
        assert relay.presenters_for(second.home) == {"guard": second}
        first.dispose()
        assert relay.presenters_for(first.home) == {}
        assert relay.presenters_for(second.home) == {"guard": second}
    finally:
        first.dispose()
        second.dispose()


def test_request_identity_and_deadline_are_immutable():
    relay = relay_module()
    request = relay.ApprovalRequest(
        profile_home="/profiles/research", job_id="daily", execution_id="exec-1",
        session_id="cron-daily", external_session_id="hm-session", call_id="call-1",
        invocation_id="invocation-1", deadline=200.0,
    )
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        request.profile_home = "/profiles/default"
    assert request.execution_id == "exec-1"


def test_close_wakes_presenter_without_extending_deadline():
    relay = relay_module()
    lifetime = relay.ApprovalLifetime(20.0, lambda: True, clock=lambda: 10.0)
    lifetime.close("cancelled")
    assert lifetime.ended.wait(2)
    assert lifetime.reason == "cancelled"
    lifetime.close("completed")
    assert lifetime.reason == "cancelled"
    with pytest.raises(relay.ApprovalUnavailable):
        lifetime.check()


def test_disconnect_during_request_discards_result():
    relay = relay_module()
    lifetime = relay.ApprovalLifetime(20.0, lambda: True, clock=lambda: 10.0)

    def operation():
        lifetime.close("disconnected")
        return "approve"

    with pytest.raises(relay.ApprovalUnavailable):
        lifetime.request(operation)
