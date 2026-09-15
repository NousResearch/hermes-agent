"""E2E: guided-routing Kanban vertical slice (plans/2026-09-15_141016-
guided-model-routing.md). Isolated temp Hermes home + real sqlite files
(kanban.db, model_routing.db) + a real dispatch_once() tick — no mocks of
the routing library itself, only the spawn subprocess is stubbed (a real
child process is out of scope for this test and is exercised by the
existing kanban worker-argv tests).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent.model_selection_store import get_receipt, publish_policy, activate_policy
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


def _policy():
    return {
        "schema_version": 1,
        "policy_id": "kanban-default",
        "revision": 1,
        "routes": [
            {
                "route_id": "openai-gpt5", "route_revision": 1, "provider": "openai",
                "model": "gpt-5", "endpoint": "https://api.openai.com/v1", "maker": "openai",
                "model_family": "gpt-5", "status": "approved",
                "allowed_roles": ["builder"], "capabilities": [],
                "verified_input_budget": 200000, "allowed_reasoning": ["low", "medium", "high"],
                "qualifications": ["shallow", "deep"], "assessment": "reviewed", "evidence": {},
            },
        ],
        "rankings": {"builder": {"deep": ["openai-gpt5"], "shallow": ["openai-gpt5"]}},
    }


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_managed_task_resolves_receipted_route_and_spawns_with_it(
    kanban_home, all_assignees_spawnable,
):
    from hermes_constants import get_hermes_home

    record = publish_policy(get_hermes_home(), _policy(), approval_ref="operator:test-e2e")
    activate_policy(get_hermes_home(), "kanban-default", record["revision"])

    captured = {}

    def _fake_spawn(task, workspace):
        captured["model"] = task.model_override
        captured["provider"] = task.provider_override
        captured["reasoning"] = task.reasoning_effort
        return 4242

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="managed card", assignee="alice", routing_role="builder",
        )
        res = kbd.dispatch_once(conn, dry_run=False, spawn_fn=_fake_spawn)
        task = kb.get_task(conn, tid)

    assert res.spawned == [(tid, "alice", "")] or res.spawned[0][0] == tid
    # The worker was actually handed the receipted route, not a default.
    assert captured["provider"] == "openai"
    assert captured["model"] == "gpt-5"
    # A real receipt id was persisted on the task row and resolves to the
    # exact decision that was applied.
    assert task.routing_receipt_id
    decision = get_receipt(get_hermes_home(), task.routing_receipt_id)
    assert decision["selected"]["provider"] == "openai"
    assert decision["selected"]["model"] == "gpt-5"
    assert decision["requirements"]["role"] == "builder"


def test_unmanaged_task_is_never_touched_by_routing(kanban_home, all_assignees_spawnable):
    """No routing_role -> zero guided-routing involvement; existing
    model_override/provider_override behavior is untouched."""
    captured = {}

    def _fake_spawn(task, workspace):
        captured["model"] = task.model_override
        return 99

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="plain card", assignee="alice")
        kbd.dispatch_once(conn, dry_run=False, spawn_fn=_fake_spawn)
        task = kb.get_task(conn, tid)

    assert captured["model"] is None
    assert task.routing_receipt_id is None


def test_managed_task_with_no_active_policy_fails_closed_not_default_route(
    kanban_home, all_assignees_spawnable,
):
    """No published/active policy -> spawn fails via the normal breaker path;
    the task must NOT silently fall back to an unmanaged default route
    (design §5)."""
    spawned = []

    def _fake_spawn(task, workspace):
        spawned.append(task.id)
        return 1

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="managed card, no policy", assignee="alice", routing_role="builder",
            max_retries=1,
        )
        res = kbd.dispatch_once(conn, dry_run=False, spawn_fn=_fake_spawn)
        task = kb.get_task(conn, tid)

    assert spawned == []
    assert res.spawned == []
    assert task.routing_receipt_id is None
    assert task.status in ("blocked", "ready")  # auto-blocked by the breaker, or requeued for retry
