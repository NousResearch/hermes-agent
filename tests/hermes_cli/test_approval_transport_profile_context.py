"""Approval transport workers retain the caller's routed profile Context."""

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
        timeout_seconds=1,
    )


def test_approval_transport_worker_keeps_routed_profile_context(tmp_path, monkeypatch):
    launch_home = (tmp_path / "launch").resolve()
    served_home = (tmp_path / "served").resolve()
    launch_home.mkdir()
    served_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch_home))

    observed = []

    def present(request: ApprovalRequest):
        observed.append(get_hermes_home())
        return request.respond("once")

    request = _request()
    assert invoke_approval_transport(present, request, timeout_seconds=1).choice == "once"

    token = set_hermes_home_override(served_home)
    try:
        assert invoke_approval_transport(present, request, timeout_seconds=1).choice == "once"
    finally:
        reset_hermes_home_override(token)

    assert invoke_approval_transport(present, request, timeout_seconds=1).choice == "once"
    assert observed == [launch_home, served_home, launch_home]
