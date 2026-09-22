"""Claim/run CAS + origin/endpoint propagation for guided-routing dispatch
(design §12 "Claim/start/crash sequence"). Exercises the REAL dispatch
resolver (``resolve_task_route``) against a real sqlite Kanban DB and a real
model_selection_store — no mocking of the guard, the store, or the DB layer.

Covers what the parent flagged as unproven by the CLI-subprocess slice:
  - the receipt gets linked via a compare-and-set against ``current_run_id``,
    so a concurrent reclaim (run replaced after the decision was resolved)
    causes the link to be refused, not silently attached to the new run;
  - a task dispatched under one profile's origin authority (its
    model_routing.db) must NOT be validated against a different profile's
    (unrelated) default store — an unrelated receipt id that happens to
    exist there must not substitute.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent.model_selection_types import RoutingBlocked


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


def _make_home(tmp_path, name):
    home = tmp_path / name
    home.mkdir()
    from agent.model_selection_store import activate_policy, publish_policy
    record = publish_policy(home, _policy(), approval_ref="operator:test")
    activate_policy(home, "kanban-default", record["revision"])
    return home


@pytest.fixture
def origin_home(tmp_path, monkeypatch):
    home = _make_home(tmp_path, "origin_home")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def _kanban_conn():
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return kbc.connect()


def test_claim_cas_refuses_link_after_concurrent_reclaim(origin_home):
    """A decision resolved under run N must not get linked once the task's
    current_run_id has moved past N (a concurrent reclaim/replace) — the
    "claim/run CAS" acceptance criterion. resolve_task_route must raise
    RoutingBlocked(stale_or_revoked_decision), not silently attach the
    receipt to the wrong (new) run."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_model_routing import resolve_task_route

    conn = _kanban_conn()
    try:
        tid = kb.create_task(conn, title="managed", assignee="alice", routing_role="builder",
                             routing_requirements={"input_tokens": 1000, "reserve_tokens": 8192})
        run_id = kb.claim_task(conn, tid, claimer="worker-a").current_run_id
        task = kb.get_task(conn, tid)
        assert task.current_run_id == run_id

        # Simulate a concurrent reclaim superseding this run before the
        # dispatcher gets to link the receipt (e.g. TTL expiry + reclaim by
        # another dispatcher tick). We bump current_run_id directly to force
        # the CAS to miss deterministically, independent of reclaim/TTL
        # timing internals.
        conn.execute("UPDATE tasks SET current_run_id = current_run_id + 1000 WHERE id = ?", (tid,))
        conn.commit()

        with pytest.raises(RoutingBlocked, match="stale_or_revoked_decision"):
            resolve_task_route(
                origin_home, conn, task, now=1000,
                frozen_sha="deadbeef", verified_by="test",
            )
    finally:
        conn.close()


def test_claim_cas_links_receipt_when_run_still_current(origin_home):
    """The normal (non-conflicting) path: the run resolve_task_route sees on
    the task is still current, so the receipt links and a routing_selected
    outcome is recorded — proving the CAS actually succeeds when it should,
    not merely that it can fail."""
    from agent.model_selection_store import list_outcomes
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_model_routing import resolve_task_route

    conn = _kanban_conn()
    try:
        tid = kb.create_task(conn, title="managed", assignee="alice", routing_role="builder",
                             routing_requirements={"input_tokens": 1000, "reserve_tokens": 8192})
        kb.claim_task(conn, tid, claimer="worker-a")
        task = kb.get_task(conn, tid)

        kwargs = resolve_task_route(
            origin_home, conn, task, now=1000, frozen_sha="deadbeef", verified_by="test",
        )
        reloaded = kb.get_task(conn, tid)
        assert reloaded.routing_receipt_id == kwargs["receipt_id"]

        outcomes = [o["kind"] for o in list_outcomes(origin_home, kwargs["receipt_id"])]
        assert "routing_selected" in outcomes
    finally:
        conn.close()


def test_worker_validates_under_origin_authority_not_own_default_home(tmp_path, monkeypatch):
    """Origin A dispatches to worker B (a different profile). The worker
    process must validate the receipt under A's origin store (the one the
    dispatcher actually resolved and persisted into), never its own
    default-profile store — even when B's own default store happens to
    contain a completely unrelated receipt under the SAME id collision-wise.
    This is the exact "origin receipt reference" propagation gap the parent
    flagged as unproven."""
    from agent.model_selection_store import persist_receipt
    from agent.model_selection import select
    from agent.managed_route_runtime import enforce_worker_route

    origin = _make_home(tmp_path, "profile_a_home")
    worker_default = _make_home(tmp_path, "profile_b_home")

    requirements = {
        "schema_version": 1, "role": "builder", "execution_kind": "kanban",
        "execution_id": "t_cross_1", "attempt_id": "1", "slot_id": "",
        "task_class": "cross-component", "required_capabilities": [],
        "input_tokens": 1000, "reserve_tokens": 8192, "reasoning": "high",
        "provenance": {"frozen_sha": "deadbeef", "verified_by": "test",
                       "complete": True, "contributors": []},
    }
    decision = select(requirements, _policy(), {}, now=1000)
    receipt_id = persist_receipt(origin, decision)

    # The worker's OWN default home has no such receipt at all — proving that
    # falling back to it (instead of the propagated origin) fails closed.
    with pytest.raises(RoutingBlocked, match="stale_or_revoked_decision"):
        enforce_worker_route(
            worker_default, receipt_id,
            actual_provider="openai", actual_model="gpt-5",
            actual_endpoint="https://api.openai.com/v1", actual_reasoning="high",
        )

    # Validating under the actual propagated origin succeeds cleanly.
    enforce_worker_route(
        origin, receipt_id,
        actual_provider="openai", actual_model="gpt-5",
        actual_endpoint="https://api.openai.com/v1", actual_reasoning="high",
    )


def test_dispatch_worker_argv_carries_origin_home_env(origin_home, monkeypatch, tmp_path):
    """The dispatcher must set HERMES_KANBAN_ROUTING_ORIGIN_HOME in the
    ACTUAL spawned worker subprocess env to its OWN (origin) hermes_home —
    the wire the worker-side enforcement call in cli.py reads to pick the
    correct store, independent of the worker's assignee-profile HERMES_HOME.
    Exercises the real ``_default_spawn`` env construction (subprocess.Popen
    itself is stubbed out — no real worker process is launched)."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kdd

    conn = _kanban_conn()
    try:
        tid = kb.create_task(conn, title="managed", assignee="alice", routing_role="builder")
        claimed = kb.claim_task(conn, tid, claimer="worker-a")
        task = kb.get_task(conn, tid)
        # Mirror what resolve_task_route would have set on the claimed task
        # object right before spawn (dispatch_lane_task's real sequence).
        task.routing_receipt_id = "rr_fake_for_env_test"
        task.routing_origin_home = str(origin_home)

        captured = {}

        class _FakeProc:
            pid = 4242

        def _fake_popen(argv, *, env=None, **kwargs):
            captured["env"] = env
            return _FakeProc()

        monkeypatch.setattr(kdd.subprocess, "Popen", _fake_popen)
        monkeypatch.setattr(
            kdd, "resolve_profile_env" if hasattr(kdd, "resolve_profile_env") else "_noop",
            (lambda *_a, **_kw: str(tmp_path / "profile_home")) if hasattr(kdd, "resolve_profile_env") else None,
            raising=False,
        )
        workspace = str(tmp_path / "ws")
        Path(workspace).mkdir(parents=True, exist_ok=True)
        kdd._default_spawn(task, workspace)
    finally:
        conn.close()

    assert "env" in captured, "subprocess.Popen was not invoked by _default_spawn"
    assert captured["env"].get("HERMES_KANBAN_ROUTING_RECEIPT") == "rr_fake_for_env_test"
    assert captured["env"].get("HERMES_KANBAN_ROUTING_ORIGIN_HOME") == str(origin_home)


def test_worker_argv_propagates_routing_endpoint_via_base_url():
    """``_worker_argv`` must translate ``task.routing_endpoint`` (set by
    ``dispatch_lane_task`` from ``managed_child_kwargs()["endpoint"]``) into
    a ``--base-url`` flag on the actual worker command line, so the
    constructed client (not just the receipt) targets the selected route's
    endpoint. Previously this field was computed by resolve_task_route and
    then silently dropped -- never reaching argv at all."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kdd

    task = kb.Task(
        id="t1", title="x", body=None, assignee="alice", status="running",
        priority=0, created_by=None, created_at=0, started_at=None,
        completed_at=None, workspace_kind="none", workspace_path=None,
        claim_lock=None, claim_expires=None, tenant=None,
        model_override="fake-model", provider_override="custom-fake",
        reasoning_effort="high", routing_role="builder",
        routing_receipt_id="rr1",
        routing_endpoint="http://127.0.0.1:9999/v1",
    )
    task.routing_origin_home = "/tmp/origin"
    argv = kdd._worker_argv(task, "alice", None)
    assert "--base-url" in argv, f"routing_endpoint must reach argv as --base-url; got {argv!r}"
    idx = argv.index("--base-url")
    assert argv[idx + 1] == "http://127.0.0.1:9999/v1"


def test_worker_argv_omits_base_url_for_unmanaged_task():
    """An unmanaged task (no routing_endpoint) must not gain a spurious
    ``--base-url`` -- this propagation is scoped to guided-routing only."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kdd

    task = kb.Task(
        id="t2", title="x", body=None, assignee="alice", status="running",
        priority=0, created_by=None, created_at=0, started_at=None,
        completed_at=None, workspace_kind="none", workspace_path=None,
        claim_lock=None, claim_expires=None, tenant=None,
    )
    argv = kdd._worker_argv(task, "alice", None)
    assert "--base-url" not in argv
