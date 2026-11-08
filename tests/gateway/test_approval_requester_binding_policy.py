"""Under ``approvals.approvers: []`` every surface binds an approval to the same request and the same
admin scope: a native tap is checked against the request it settles (never a substituted one), a DM
admin is not a group admin on buttons any more than on ``/approve``, a coalesced follower is not
answered by another requester, and a relayed answer is checked against the bot-owning profile."""

import threading

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource
from tests.gateway.test_approval_requester_binding import pending_from, write_approvers


@pytest.fixture(autouse=True)
def _clean_queues():
    from tools import approval as approval_mod

    approval_mod._gateway_queues.clear()
    yield
    approval_mod._gateway_queues.clear()


def _group_pending(user_id, user_name, session_key):
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools import approval as approval_mod
    from tools.approval_gateway_wait import _ApprovalEntry

    tokens = set_session_vars(platform="telegram", chat_type="group", user_id=user_id, user_name=user_name,
                              session_key=session_key)
    try:
        entry = _ApprovalEntry({"command": "rm -rf build"})
    finally:
        clear_session_vars(tokens)
    approval_mod._gateway_queues.setdefault(session_key, []).append(entry)
    return entry


def test_a_dm_admin_is_not_a_group_admin_on_buttons_either():
    """``allow_admin_from`` is the DM scope (slash_access.policy_from_extra): a native button in a group
    refused it exactly as ``/approve`` does, instead of unioning both scopes."""
    from gateway.approval_actor import approval_refusal
    from gateway.slash_access import policy_for_runner_source
    from types import SimpleNamespace

    write_approvers([])
    extra = {"allow_admin_from": ["bob"]}
    _group_pending("alice", "Alice", "g")
    runner = SimpleNamespace(config=GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(extra=extra)}))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="g", chat_type="group", user_id="bob")
    native = approval_refusal("g", "bob", extra=extra)
    text = approval_refusal("g", "bob", admin_ids=policy_for_runner_source(runner, source).admin_user_ids)
    assert native == text == "Alice"
    assert approval_refusal("g", "carol", extra={"group_allow_admin_from": ["carol"]}) is None
