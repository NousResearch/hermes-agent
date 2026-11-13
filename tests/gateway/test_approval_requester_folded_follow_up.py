"""Another member's steer or redirect folded into a running turn leaves that turn with no single actor:
an approval raised after the fold fails closed to approvers and admins, instead of naming the member who
opened the turn while it now runs the other member's request. One member's own follow-ups stay bound."""

from contextvars import copy_context

import pytest

from hermes_agent.gateway.config import GatewayConfig
from hermes_agent.gateway.run import GatewayRunner
from hermes_agent.gateway.session import build_session_context, build_session_key
from tests.gateway.test_approval_requester_binding import write_approvers
from tests.gateway.test_approval_requester_binding_shared_turns import (  # _clean_state: autouse fixture
    _clean_state, _observe_adapter, _observed_event)


class _Agent:
    _supports_active_turn_redirect = True

    def redirect(self, text):
        return True

    def steer(self, text):
        return True


def _raise_approval(key):
    from hermes_agent.tools import approval as approval_mod
    from hermes_agent.tools.approval_gateway_wait import _ApprovalEntry

    approval_mod._gateway_queues.setdefault(key, []).append(_ApprovalEntry({"command": "rm -rf notes"}))


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["interrupt", "steer"])
@pytest.mark.parametrize(("follow_uid", "follow_name", "alice_answers", "bob_answers"), [
    (222, "Bob", False, False),           # Bob's input joined Alice's turn: neither member answers
    (111, "Alice Example", True, False),  # Alice's own follow-up: still bound to Alice
], ids=["other-member", "same-member"])
async def test_a_folded_follow_up_rebinds_the_running_turns_approvals(
        mode, follow_uid, follow_name, alice_answers, bob_answers):
    from hermes_agent.gateway.approval_actor import approval_refusal

    write_approvers([])
    adapter = _observe_adapter()
    alice = _observed_event(adapter, 111, "Alice Example", "summarize the repo")
    follow = _observed_event(adapter, follow_uid, follow_name, "actually delete the notes dir")
    key = build_session_key(alice.source)
    runner = GatewayRunner(config=GatewayConfig())
    agent = _Agent()
    turn = runner._session_state(key).turn
    turn.agent, turn.event = agent, alice
    tokens = runner._set_session_env(build_session_context(alice.source, GatewayConfig(), event=alice))
    try:
        worker = copy_context()  # the turn's worker copied its context when Alice's turn started
    finally:
        runner._clear_session_env(tokens)

    outcome = await runner._resolve_busy_steer_or_redirect(follow, key, mode, agent)
    assert outcome.redirected or outcome.steered
    worker.run(_raise_approval, key)

    assert (approval_refusal(key, "111") is None) is alice_answers
    assert (approval_refusal(key, "222") is None) is bob_answers
    assert approval_refusal(key, "444", extra={"group_allow_admin_from": ["444"]}) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(("follow_uid", "follow_name", "actor"), [
    (222, "Bob", ("", "")),                       # mixed turn: a child names no single actor
    (111, "Alice Example", ("111", "Alice Example")),  # one member's turn: the child keeps the sender
], ids=["other-member", "same-member"])
async def test_a_child_process_started_after_a_fold_names_the_turns_actor(follow_uid, follow_name, actor):
    from hermes_agent.tools.environments.local import _inject_session_context_env

    adapter = _observe_adapter()
    alice = _observed_event(adapter, 111, "Alice Example", "summarize the repo")
    follow = _observed_event(adapter, follow_uid, follow_name, "actually delete the notes dir")
    key = build_session_key(alice.source)
    runner = GatewayRunner(config=GatewayConfig())
    agent = _Agent()
    turn = runner._session_state(key).turn
    turn.agent, turn.event = agent, alice
    tokens = runner._set_session_env(build_session_context(alice.source, GatewayConfig(), event=alice))
    try:
        worker = copy_context()
    finally:
        runner._clear_session_env(tokens)

    assert (await runner._resolve_busy_steer_or_redirect(follow, key, "steer", agent)).steered
    env = {}
    worker.run(_inject_session_context_env, env)  # the env a terminal / execute_code child starts with

    assert (env["HERMES_SESSION_ACTOR_ID"], env["HERMES_SESSION_ACTOR_NAME"]) == actor
