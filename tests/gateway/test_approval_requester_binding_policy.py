"""Under ``approvals.approvers: []`` every surface binds an approval to the same request and the same
admin scope: a native tap is checked against the request it settles (never a substituted one), a DM
admin is not a group admin on buttons any more than on ``/approve``, a coalesced follower is not
answered by another requester, and a relayed answer is checked against the bot-owning profile."""

import threading

import pytest

from hermes_agent.gateway.config import GatewayConfig, Platform, PlatformConfig
from hermes_agent.gateway.session import SessionSource
from tests.gateway.test_approval_requester_binding import pending_from, write_approvers


@pytest.fixture(autouse=True)
def _clean_queues():
    from hermes_agent.tools import approval as approval_mod

    approval_mod._gateway_queues.clear()
    yield
    approval_mod._gateway_queues.clear()


def _group_pending(user_id, user_name, session_key):
    from hermes_agent.gateway.session_context import clear_session_vars, set_session_vars
    from hermes_agent.tools import approval as approval_mod
    from hermes_agent.tools.approval_gateway_wait import _ApprovalEntry

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
    from hermes_agent.gateway.approval_actor import approval_refusal
    from hermes_agent.gateway.slash_access import policy_for_runner_source
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


def test_an_id_less_tap_settles_nothing_even_for_the_requester():
    """A card tap without a request id cannot say which request it answers, so it resolves nothing,
    whoever taps (#62359); the requester then answers their own card by id. Bob's request expiring
    never lets his id-less tap reach Alice's, and Bob may not answer Alice's by id either."""
    from hermes_agent.gateway.approval_actor import approval_refusal
    from hermes_agent.tools import approval

    write_approvers([])
    bob = pending_from("bob", "Bob", "s")
    alice = pending_from("alice", "Alice", "s")
    assert approval_refusal("s", "bob") is None
    approval.withdraw_gateway_approval("s", bob.data["request_id"], "timeout")
    assert approval.resolve_gateway_approval_card("s", "once", None, actor_id="bob") == 0
    assert approval.resolve_gateway_approval_card("s", "once", None, actor_id="alice") == 0
    assert approval.resolve_gateway_approval_card("s", "once", alice.data["request_id"], actor_id="bob") == 0
    assert alice.result is None and not alice.event.is_set()
    assert approval.resolve_gateway_approval_card("s", "once", alice.data["request_id"], actor_id="alice") == 1
    assert alice.result == "once"


@pytest.mark.asyncio
async def test_a_telegram_tap_settles_only_what_the_tapper_may_answer(monkeypatch):
    """The production callback: the tap's request expires between the refusal check and the settle."""
    from unittest.mock import AsyncMock, MagicMock

    from tests.gateway.test_telegram_approval_buttons import _make_adapter
    from hermes_agent.tools import approval

    write_approvers([])
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    bob = pending_from("bob", "Bob", "agent:main:telegram:group:1:9")
    alice = pending_from("alice", "Alice", "agent:main:telegram:group:1:9")
    class _ExpiresBobOnClaim(dict):  # Bob's wait times out right after his tap passed the check
        def pop(self, key, *default):
            value = super().pop(key, *default)
            approval.withdraw_gateway_approval("agent:main:telegram:group:1:9", bob.data["request_id"], "timeout")
            return value

    adapter = _make_adapter()
    adapter._approval_state = _ExpiresBobOnClaim({4: "agent:main:telegram:group:1:9"})
    query = AsyncMock(data="ea:once:4", message=MagicMock(chat_id=1, text_html="x"),
                      from_user=MagicMock(first_name="Bob", id="bob"))
    await adapter._handle_callback_query(MagicMock(callback_query=query), MagicMock())
    assert alice.result is None and not alice.event.is_set()


def _wait_in_thread(user, session_key, notify, outcomes):
    from hermes_agent.gateway.session_context import clear_session_vars, set_session_vars
    from hermes_agent.tools.approval_gateway_wait import _await_gateway_decision

    tokens = set_session_vars(platform="telegram", user_id=user, user_name=user.title(), session_key=session_key)
    try:
        outcomes[user] = _await_gateway_decision(session_key, notify, {"command": "echo same", "pattern_keys": ["k"]})
    finally:
        clear_session_vars(tokens)


def test_another_requesters_deny_does_not_answer_a_coalesced_wait(monkeypatch):
    """Identical requests from two requesters do not share one prompt under the opt-in: Bob's deny
    answers Bob, and Alice gets her own prompt that only she (or an approver) answers."""
    from hermes_agent.tools import approval

    write_approvers([])
    prompts, outcomes, cond = [], {}, threading.Condition()

    def notify(data):
        with cond:
            prompts.append(data["request_id"])
            cond.notify_all()

    bob = threading.Thread(target=_wait_in_thread, args=("bob", "c", notify, outcomes), daemon=True)
    bob.start()
    with cond:
        assert cond.wait_for(lambda: len(prompts) == 1, 10)
    alice = threading.Thread(target=_wait_in_thread, args=("alice", "c", notify, outcomes), daemon=True)
    alice.start()
    with cond:
        cond.wait_for(lambda: len(prompts) == 2, 5)
    approval.resolve_gateway_approval("c", "deny", request_id=prompts[0])
    bob.join(10)
    assert outcomes["bob"]["choice"] == "deny"
    assert len(prompts) == 2 and "alice" not in outcomes, "Alice's wait was answered by Bob's deny"
    approval.resolve_gateway_approval("c", "once", request_id=prompts[1])
    alice.join(10)
    assert outcomes["alice"]["choice"] == "once" and not outcomes["alice"].get("coalesced")


def test_same_requester_identical_requests_still_coalesce():
    from hermes_agent.tools import approval

    write_approvers([])
    prompts, outcomes, cond = [], {}, threading.Condition()

    def notify(data):
        with cond:
            prompts.append(data["request_id"])
            cond.notify_all()

    first = threading.Thread(target=_wait_in_thread, args=("alice", "d", notify, outcomes), daemon=True)
    first.start()
    with cond:
        assert cond.wait_for(lambda: len(prompts) == 1, 10)
    results = {}
    second = threading.Thread(target=_wait_in_thread, args=("alice", "d", notify, results), daemon=True)
    second.start()
    with cond:
        cond.wait_for(lambda: len(prompts) == 2, 1)
    approval.resolve_gateway_approval("d", "deny", request_id=prompts[0])
    first.join(10)
    second.join(10)
    assert len(prompts) == 1 and results["alice"]["coalesced"] is True
