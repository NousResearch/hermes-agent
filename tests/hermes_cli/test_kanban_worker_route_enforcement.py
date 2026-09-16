"""Worker-side guided-routing enforcement (design §4 step 7, §12 "Claim/start/
crash sequence" step 5): the actual Kanban worker process must validate its
own constructed route against the SAME receipted decision the dispatcher
persisted at claim time, immediately before its first real inference call.

Isolated: real temp HERMES_HOME, real model_routing.db sqlite file (via
agent.model_selection_store), no provider clients, no network. This exercises
``hermes_cli.kanban_model_routing.enforce_worker_route`` — the exact function
``cli.py``'s single-query worker bootstrap calls — not a stand-in mock of the
enforcement boundary.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent.model_selection_types import RoutingBlocked


def _requirements(**overrides):
    base = {
        "schema_version": 1, "role": "builder", "execution_kind": "kanban",
        "execution_id": "t_worker_1", "attempt_id": "1", "slot_id": "",
        "task_class": "cross-component", "required_capabilities": [],
        "input_tokens": 0, "reserve_tokens": 0, "reasoning": "high",
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

    # A real dispatcher only ever resolves a decision against a policy that
    # is (still) the active revision — publish/activate it here too so this
    # helper mirrors resolve_task_route's real path, including the worker-side
    # revocation check (design §12) that later validates the SAME revision is
    # still active before first inference.
    record = publish_policy(routing_home, _policy(), approval_ref="operator:test")
    activate_policy(routing_home, "kanban-default", record["revision"])
    decision = select(_requirements(), _policy(), {}, now=1000)
    receipt_id = persist_receipt(routing_home, decision)
    return receipt_id


def test_worker_matching_actual_route_enforces_cleanly(routing_home):
    from hermes_cli.kanban_model_routing import enforce_worker_route

    receipt_id = _persist_receipt(routing_home)
    enforce_worker_route(
        routing_home, receipt_id,
        actual_provider="openai", actual_model="gpt-5",
        actual_endpoint="https://api.openai.com/v1", actual_reasoning="high",
    )


def test_worker_constructed_a_different_model_is_blocked(routing_home):
    """The core integration gap the parent flagged: a worker that actually got
    constructed with a different model than the receipted decision must be
    stopped here, not merely have the dispatcher's kwargs recorded."""
    from hermes_cli.kanban_model_routing import enforce_worker_route

    receipt_id = _persist_receipt(routing_home)
    with pytest.raises(RoutingBlocked, match="stale_or_revoked_decision"):
        enforce_worker_route(
            routing_home, receipt_id,
            actual_provider="openai", actual_model="gpt-4o-legacy",
            actual_endpoint="https://api.openai.com/v1", actual_reasoning="high",
        )


def test_worker_missing_receipt_fails_closed_not_silently(routing_home):
    """A receipt id that does not resolve (bad env, wrong profile scope, a
    revoked/never-persisted decision) must block, never silently proceed with
    whatever the worker happened to construct."""
    from hermes_cli.kanban_model_routing import enforce_worker_route

    with pytest.raises(RoutingBlocked, match="stale_or_revoked_decision"):
        enforce_worker_route(
            routing_home, "rr_does_not_exist",
            actual_provider="openai", actual_model="gpt-5",
            actual_endpoint="https://api.openai.com/v1", actual_reasoning="high",
        )


def test_resolve_task_route_carries_receipt_id_for_worker_enforcement(routing_home):
    """The dispatcher-side resolver must hand back the same receipt id the
    worker-side enforcement call loads — the wire between claim time and
    first-inference time."""
    from agent.model_selection_store import activate_policy, publish_policy
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli.kanban_model_routing import enforce_worker_route, resolve_task_route

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    record = publish_policy(routing_home, _policy(), approval_ref="operator:test")
    activate_policy(routing_home, "kanban-default", record["revision"])

    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="managed", assignee="alice", routing_role="builder")
        kb.claim_task(conn, tid, claimer="alice")
        task = kb.get_task(conn, tid)
        kwargs = resolve_task_route(
            routing_home, conn, task, now=1000, frozen_sha="deadbeef", verified_by="test",
        )
    finally:
        conn.close()

    assert kwargs["receipt_id"]
    # The worker enforcement boundary must accept the exact route the
    # dispatcher resolved — this is the "real worker construction" the
    # spawned Kanban process is expected to have actually built.
    enforce_worker_route(
        routing_home, kwargs["receipt_id"],
        actual_provider=kwargs["provider"], actual_model=kwargs["model"],
        actual_endpoint=kwargs["endpoint"], actual_reasoning=kwargs["reasoning_effort"],
    )
    # And a worker that diverges from that same receipt is blocked.
    with pytest.raises(RoutingBlocked, match="stale_or_revoked_decision"):
        enforce_worker_route(
            routing_home, kwargs["receipt_id"],
            actual_provider="anthropic", actual_model="claude-x",
            actual_endpoint=kwargs["endpoint"], actual_reasoning=kwargs["reasoning_effort"],
        )


def test_routine_policy_edit_does_not_block_a_matching_worker(routing_home):
    """A ROUTINE policy edit (publish + activate a new revision, no explicit revocation)
    AFTER the receipt was persisted must NOT block a worker whose actual construction still
    matches the receipted route — routine edits affect only NEW attempts, never an already-
    receipted in-flight one (design §12: "Routine policy edits affect new attempts, not active
    conversations")."""
    from agent.model_selection_store import activate_policy, publish_policy
    from hermes_cli.kanban_model_routing import enforce_worker_route

    receipt_id = _persist_receipt(routing_home)
    edited = _policy()
    edited["revision"] = 2
    record = publish_policy(routing_home, edited, approval_ref="operator:routine-edit")
    activate_policy(routing_home, "kanban-default", record["revision"])

    enforce_worker_route(
        routing_home, receipt_id,
        actual_provider="openai", actual_model="gpt-5",
        actual_endpoint="https://api.openai.com/v1", actual_reasoning="high",
    )


def test_explicit_revocation_blocks_worker_even_with_matching_route(routing_home):
    """An EXPLICIT emergency revocation (not a routine publish/activate) of the receipted
    route must still block a worker even though its actual construction matches the receipted
    route byte-for-byte (design §12 "Availability, budget, reasoning and revocation")."""
    from agent.model_selection_store import revoke_route
    from hermes_cli.kanban_model_routing import enforce_worker_route

    receipt_id = _persist_receipt(routing_home)
    revoke_route(
        routing_home, "kanban-default", route_id="openai-gpt5",
        reason="incident", approval_ref="operator:revoke",
    )

    with pytest.raises(RoutingBlocked, match="stale_or_revoked_decision"):
        enforce_worker_route(
            routing_home, receipt_id,
            actual_provider="openai", actual_model="gpt-5",
            actual_endpoint="https://api.openai.com/v1", actual_reasoning="high",
        )
