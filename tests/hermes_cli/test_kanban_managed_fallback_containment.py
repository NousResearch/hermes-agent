"""Fallback containment for managed Kanban attempts (design §5: "Disable
inherited model fallback chains for these runs" / "no silent escape through
parent inheritance, global fallback chains").

``cli.py``'s ``_init_agent`` always wires the process-level
``fallback_model=self._fallback_model`` (``get_fallback_chain(CLI_CONFIG)``)
into every ``AIAgent`` regardless of whether this turn is a receipted managed
attempt. That inherited chain was never approved by the routing policy for
THIS receipted route, so a real provider failure on a managed attempt must
never silently transmit content to that unapproved fallback
provider/model — the attempt must end instead (existing claim/retry/re-route
picks a NEW receipted route). This exercises the exact seam
``cli._enforce_kanban_routing_receipt`` calls
(``_disable_inherited_fallback_for_managed_run``) against a real ``AIAgent``
instance carrying a real, non-empty fallback chain — not a stand-in mock of
the guard boundary.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _requirements(**overrides):
    base = {
        "schema_version": 1, "role": "builder", "execution_kind": "kanban",
        "execution_id": "t_fallback_containment", "attempt_id": "1", "slot_id": "",
        "task_class": "cross-component", "required_capabilities": [],
        "input_tokens": 1000, "reserve_tokens": 8192, "reasoning": "high",
        "provenance": {"frozen_sha": "deadbeef", "verified_by": "test",
                       "complete": True, "contributors": []},
    }
    base.update(overrides)
    return base


def _policy():
    return {
        "schema_version": 1, "policy_id": "kanban-default", "revision": 1,
        "approval_ref": "operator:test",
        "routes": [{
            "route_id": "openai-gpt5", "route_revision": 1, "provider": "openai",
            "model": "gpt-5", "endpoint": "https://api.openai.com/v1", "maker": "openai",
            "model_family": "gpt-5", "status": "approved", "allowed_roles": ["builder"],
            "capabilities": [], "verified_input_budget": 200000,
            "allowed_reasoning": ["low", "medium", "high"],
            "qualifications": ["shallow", "deep"], "assessment": "reviewed", "evidence": {},
        }],
        "rankings": {"builder": {"deep": ["openai-gpt5"], "shallow": ["openai-gpt5"]}},
    }


@pytest.fixture
def routing_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def _persist_receipt(routing_home):
    from agent.model_selection import select
    from agent.model_selection_store import activate_policy, persist_receipt, publish_policy

    record = publish_policy(routing_home, _policy(), approval_ref="operator:test")
    activate_policy(routing_home, "kanban-default", record["revision"])
    decision = select(_requirements(), _policy(), {}, now=1000)
    return persist_receipt(routing_home, decision)


class _FakeAgent:
    """Minimal real-shaped stand-in carrying exactly the attributes
    ``enforce_worker_route``/``_disable_inherited_fallback_for_managed_run``
    read/write — real ``AIAgent`` construction needs live credentials this
    isolated test must not depend on, but the fallback-chain SHAPE mirrors
    ``agent.agent_init._init_fallback_chain`` exactly (list of
    provider/model dicts, index, legacy single-entry mirror)."""

    def __init__(self, *, provider, model, base_url, fallback_chain):
        self.provider = self.requested_provider = provider
        self.model = model
        self.base_url = base_url
        self._fallback_chain = list(fallback_chain)
        self._fallback_index = 0
        self._fallback_model = self._fallback_chain[0] if self._fallback_chain else None


def test_matched_route_with_inherited_fallback_chain_is_cleared(routing_home):
    from cli import _disable_inherited_fallback_for_managed_run
    from hermes_cli.kanban_model_routing import enforce_worker_route

    receipt_id = _persist_receipt(routing_home)
    inherited_chain = [{"provider": "anthropic", "model": "claude-unapproved-fallback"}]
    agent = _FakeAgent(
        provider="openai", model="gpt-5", base_url="https://api.openai.com/v1",
        fallback_chain=inherited_chain,
    )
    # The receipt/route check itself passes (matched route) ...
    enforce_worker_route(
        routing_home, receipt_id,
        actual_provider=agent.provider, actual_model=agent.model,
        actual_endpoint=agent.base_url, actual_reasoning="high",
    )
    assert agent._fallback_chain == inherited_chain, "sanity: chain still present pre-containment"

    # ... but the managed-run seam must strip the inherited chain before any
    # inference, so a later provider failure cannot silently escape to it.
    _disable_inherited_fallback_for_managed_run(agent, receipt_id)

    assert agent._fallback_chain == []
    assert agent._fallback_index == 0
    assert agent._fallback_model is None


def test_unmanaged_agent_fallback_chain_is_never_touched(routing_home):
    """An agent this seam is never called for (no receipt / unmanaged task,
    exercised via cli._enforce_kanban_routing_receipt's no-receipt no-op
    branch) must keep its configured chain exactly as constructed — this is
    NOT a global fallback-disabling change, only a managed-attempt one."""
    inherited_chain = [{"provider": "anthropic", "model": "claude-configured-default"}]
    agent = _FakeAgent(
        provider="openai", model="gpt-5", base_url="https://api.openai.com/v1",
        fallback_chain=inherited_chain,
    )
    # No call to _disable_inherited_fallback_for_managed_run at all (mirrors
    # cli._enforce_kanban_routing_receipt's early `return True` when
    # HERMES_KANBAN_ROUTING_RECEIPT is unset for a non-managed run).
    assert agent._fallback_chain == inherited_chain
    assert agent._fallback_model == inherited_chain[0]


def test_empty_fallback_chain_is_a_harmless_noop(routing_home):
    from cli import _disable_inherited_fallback_for_managed_run

    agent = _FakeAgent(
        provider="openai", model="gpt-5", base_url="https://api.openai.com/v1",
        fallback_chain=[],
    )
    _disable_inherited_fallback_for_managed_run(agent, "rr_whatever")
    assert agent._fallback_chain == []
    assert agent._fallback_model is None


def test_enforcement_success_path_calls_containment_before_returning_true(routing_home, monkeypatch):
    """End-to-end through the actual guard entry point
    ``cli._enforce_kanban_routing_receipt``: a matched route with a live
    inherited fallback chain on ``cli.agent`` must come out cleared, and the
    call must still report success (True) — containment is not a failure
    mode, it is what makes success actually safe."""
    import os

    import cli as cli_module

    receipt_id = _persist_receipt(routing_home)
    inherited_chain = [{"provider": "anthropic", "model": "claude-unapproved-fallback"}]
    fake_cli = type("FakeCLI", (), {})()
    fake_cli.agent = _FakeAgent(
        provider="openai", model="gpt-5", base_url="https://api.openai.com/v1",
        fallback_chain=inherited_chain,
    )
    fake_cli.reasoning_config = "high"

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_ROUTING_ORIGIN_HOME", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_ROUTING_RECEIPT", receipt_id)
    monkeypatch.setattr(
        cli_module, "requested_effort_for_kanban_guard", lambda cli: "high",
    )

    assert cli_module._enforce_kanban_routing_receipt(fake_cli) is True
    assert fake_cli.agent._fallback_chain == []
    assert fake_cli.agent._fallback_model is None


def _recovery_policy():
    policy = _policy()
    policy["routes"].append({
        **policy["routes"][0],
        "route_id": "anthropic-alternate", "provider": "anthropic",
        "model": "claude-alternate", "endpoint": "https://api.anthropic.com",
        "maker": "anthropic", "model_family": "claude",
    })
    policy["rankings"]["builder"]["deep"] = ["openai-gpt5", "anthropic-alternate"]
    policy["rankings"]["builder"]["shallow"] = ["openai-gpt5", "anthropic-alternate"]
    return policy


def _fail_current_run(conn, task):
    from hermes_cli import kanban_db_dispatch as kbd

    kbd._record_task_failure(
        conn, task.id, "fixture provider failure", outcome="crashed",
        failure_limit=10, release_claim=True, end_run=True,
        expected_run_id=task.current_run_id,
    )


def _health_payload(home, receipt_id, *, status, replay_safe):
    import time
    from agent.model_selection_store import get_receipt

    route = get_receipt(home, receipt_id)["selected"]
    now = int(time.time())
    return {
        "status": status, "replay_safe": replay_safe, "target_profile": "alice",
        "route_revision": route["route_revision"], "endpoint": route["endpoint"],
        "observed_at": now, "retry_after": now + 600 if status != "healthy" else 0,
    }


def test_managed_recovery_allows_one_safe_alternate_then_holds(
    routing_home, all_assignees_spawnable,
):
    """A known pre-output provider refusal may select one alternate route, but
    a third automatic worker attempt in the same failure episode is blocked."""
    from agent.model_selection_store import activate_policy, append_outcome, publish_policy
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    kb.init_db()
    policy = _recovery_policy()
    record = publish_policy(routing_home, policy, approval_ref="operator:test")
    activate_policy(routing_home, "kanban-default", record["revision"])
    spawned = []

    def _spawn(task, workspace):
        spawned.append(task.model_override)
        return None

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="bounded replacement", assignee="alice", routing_role="builder",
            routing_requirements={"input_tokens": 1000, "reserve_tokens": 8192},
            max_retries=10,
        )
        kbd.dispatch_once(conn, spawn_fn=_spawn)
        first = kb.get_task(conn, tid)
        append_outcome(routing_home, first.routing_receipt_id, "routing_started", {})
        append_outcome(
            routing_home, first.routing_receipt_id, "routing_health",
            _health_payload(
                routing_home, first.routing_receipt_id,
                status="denied_model", replay_safe=True,
            ),
        )
        _fail_current_run(conn, first)

        kbd.dispatch_once(conn, spawn_fn=_spawn)
        second = kb.get_task(conn, tid)
        append_outcome(routing_home, second.routing_receipt_id, "routing_started", {})
        append_outcome(
            routing_home, second.routing_receipt_id, "routing_health",
            _health_payload(
                routing_home, second.routing_receipt_id,
                status="denied_model", replay_safe=True,
            ),
        )
        _fail_current_run(conn, second)

        final = kbd.dispatch_once(conn, spawn_fn=_spawn)
        task = kb.get_task(conn, tid)

    assert spawned == ["gpt-5", "claude-alternate"]
    assert final.spawned == []
    assert task.status == "blocked"
    assert "alternate" in (task.last_failure_error or "").lower()


def test_managed_recovery_holds_when_prior_contact_is_uncertain(
    routing_home, all_assignees_spawnable,
):
    """A request that may have produced output or tool effects is never
    automatically replayed by a replacement worker."""
    from agent.model_selection_store import activate_policy, append_outcome, publish_policy
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    kb.init_db()
    policy = _recovery_policy()
    record = publish_policy(routing_home, policy, approval_ref="operator:test")
    activate_policy(routing_home, "kanban-default", record["revision"])
    spawned = []

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="uncertain replay", assignee="alice", routing_role="builder",
            routing_requirements={"input_tokens": 1000, "reserve_tokens": 8192},
            max_retries=10,
        )
        kbd.dispatch_once(conn, spawn_fn=lambda task, workspace: spawned.append(task.model_override))
        first = kb.get_task(conn, tid)
        append_outcome(routing_home, first.routing_receipt_id, "routing_started", {})
        append_outcome(
            routing_home, first.routing_receipt_id, "routing_health",
            _health_payload(
                routing_home, first.routing_receipt_id,
                status="healthy", replay_safe=False,
            ),
        )
        _fail_current_run(conn, first)

        result = kbd.dispatch_once(conn, spawn_fn=lambda task, workspace: spawned.append(task.model_override))
        task = kb.get_task(conn, tid)

    assert spawned == ["gpt-5"]
    assert result.spawned == []
    assert task.status == "blocked"
    assert "uncertain" in (task.last_failure_error or "").lower()


def test_operator_reconciliation_allows_one_explicit_replay(
    routing_home, all_assignees_spawnable,
):
    """A replay hold is recoverable only through an auditable operator
    disposition; unblocking alone is not treated as reconciliation."""
    from agent.model_selection_store import (
        activate_policy, append_outcome, authorize_replay, publish_policy,
    )
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    kb.init_db()
    policy = _recovery_policy()
    record = publish_policy(routing_home, policy, approval_ref="operator:test")
    activate_policy(routing_home, "kanban-default", record["revision"])
    spawned = []

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="reconciled replay", assignee="alice", routing_role="builder",
            routing_requirements={"input_tokens": 1000, "reserve_tokens": 8192},
            max_retries=10,
        )
        kbd.dispatch_once(conn, spawn_fn=lambda task, workspace: spawned.append(task.model_override))
        first = kb.get_task(conn, tid)
        append_outcome(routing_home, first.routing_receipt_id, "routing_started", {})
        append_outcome(
            routing_home, first.routing_receipt_id, "routing_health",
            _health_payload(
                routing_home, first.routing_receipt_id,
                status="healthy", replay_safe=False,
            ),
        )
        _fail_current_run(conn, first)
        kbd.dispatch_once(conn, spawn_fn=lambda task, workspace: spawned.append(task.model_override))
        assert kb.get_task(conn, tid).status == "blocked"

        authorize_replay(
            routing_home, first.routing_receipt_id,
            reason="verified no external effect", approval_ref="operator:test-reconcile",
        )
        assert kb.unblock_task(conn, tid)
        result = kbd.dispatch_once(
            conn, spawn_fn=lambda task, workspace: spawned.append(task.model_override),
        )

    assert result.spawned
    assert spawned == ["gpt-5", "gpt-5"]
