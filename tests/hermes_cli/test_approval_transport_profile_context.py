"""Approval transport workers retain the caller's routed profile Context."""

import asyncio

import pytest

from agent.secret_scope import get_secret, reset_secret_scope, set_secret_scope
from hermes_cli.approval_transport import ApprovalRequest, invoke_approval_transport
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)


def _request() -> ApprovalRequest:
    return ApprovalRequest.create(
        command="echo ok",
        description="profile-context regression",
        pattern_key="profile_context",
        pattern_keys=("profile_context",),
        session_key="session-a",
        surface="gateway",
        allow_session=False,
        allow_permanent=False,
        timeout_seconds=5,
    )


@pytest.mark.parametrize("async_callback", [False, True])
def test_approval_transport_worker_keeps_routed_profile_context(tmp_path, monkeypatch, async_callback):
    launch_home = (tmp_path / "launch").resolve()
    served_home = (tmp_path / "served").resolve()
    launch_home.mkdir()
    served_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.setenv("APPROVAL_CONTEXT_TEST_TOKEN", "launch-only")

    observed = []

    def present(request: ApprovalRequest):
        observed.append((get_hermes_home(), get_secret("APPROVAL_CONTEXT_TEST_TOKEN")))
        return request.respond("once")

    async def present_async(request: ApprovalRequest):
        await asyncio.sleep(0)  # exercise the coroutine path after an event-loop handoff
        return present(request)

    callback = present_async if async_callback else present

    assert invoke_approval_transport(callback, _request(), timeout_seconds=5).choice == "once"

    token = set_hermes_home_override(served_home)
    secret_token = set_secret_scope({"APPROVAL_CONTEXT_TEST_TOKEN": "served-only"})
    try:
        assert invoke_approval_transport(callback, _request(), timeout_seconds=5).choice == "once"
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(token)

    assert invoke_approval_transport(callback, _request(), timeout_seconds=5).choice == "once"
    assert observed == [
        (launch_home, "launch-only"),
        (served_home, "served-only"),
        (launch_home, "launch-only"),
    ]
