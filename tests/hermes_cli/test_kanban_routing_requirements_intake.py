"""Genuine per-task routing requirements intake + validation (design §3.B):
replaces the hardcoded resolve_task_route placeholders (fixed task_class,
empty capabilities, zero token budgets, fabricated complete=True provenance)
with real operator-supplied intake stored on the task row and read back at
claim time. Isolated: real temp HERMES_HOME, real kanban.db + model_routing.db
sqlite files, no mocks of the routing library.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent.model_selection_store import activate_policy, get_receipt, publish_policy
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_model_routing import (
    RoutingBlocked,
    parse_routing_requirements_json,
    resolve_task_route,
    validate_routing_requirements,
)


def _policy_two_routes():
    """Two approved builder routes from different makers, plus a reviewer
    route from a THIRD maker so an independent review can actually exclude
    the builder's maker without exhausting the roster."""
    return {
        "schema_version": 1,
        "policy_id": "kanban-default",
        "revision": 1,
        "approval_ref": "operator:test",
        "routes": [
            {
                "route_id": "openai-shallow", "route_revision": 1, "provider": "openai",
                "model": "gpt-5-mini", "endpoint": "https://api.openai.com/v1", "maker": "openai",
                "model_family": "gpt-5", "status": "approved", "allowed_roles": ["builder"],
                "capabilities": [], "verified_input_budget": 50000,
                "allowed_reasoning": ["low", "medium", "high"],
                "qualifications": ["shallow"], "assessment": "reviewed", "evidence": {},
            },
            {
                "route_id": "openai-deep", "route_revision": 1, "provider": "openai",
                "model": "gpt-5", "endpoint": "https://api.openai.com/v1", "maker": "openai",
                "model_family": "gpt-5", "status": "approved", "allowed_roles": ["builder"],
                "capabilities": ["tool_use"], "verified_input_budget": 200000,
                "allowed_reasoning": ["low", "medium", "high"],
                "qualifications": ["shallow", "deep"], "assessment": "reviewed", "evidence": {},
            },
            {
                "route_id": "anthropic-review", "route_revision": 1, "provider": "anthropic",
                "model": "claude-opus", "endpoint": "https://api.anthropic.com", "maker": "anthropic",
                "model_family": "claude", "status": "approved", "allowed_roles": ["reviewquality"],
                "capabilities": ["tool_use"], "verified_input_budget": 200000,
                "allowed_reasoning": ["high"],
                "qualifications": ["shallow", "deep"], "assessment": "reviewed", "evidence": {},
            },
            {
                "route_id": "openai-review", "route_revision": 1, "provider": "openai",
                "model": "gpt-5", "endpoint": "https://api.openai.com/v1", "maker": "openai",
                "model_family": "gpt-5", "status": "approved", "allowed_roles": ["reviewquality"],
                "capabilities": ["tool_use"], "verified_input_budget": 200000,
                "allowed_reasoning": ["high"],
                "qualifications": ["shallow", "deep"], "assessment": "reviewed", "evidence": {},
            },
        ],
        "rankings": {
            "builder": {
                "deep": ["openai-deep", "openai-shallow"],
                "shallow": ["openai-shallow", "openai-deep"],
            },
            "reviewquality": {"deep": ["anthropic-review", "openai-review"]},
        },
    }


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _activate(hermes_home):
    from hermes_constants import get_hermes_home

    record = publish_policy(get_hermes_home(), _policy_two_routes(), approval_ref="operator:test")
    activate_policy(get_hermes_home(), "kanban-default", record["revision"])


# --- validate_routing_requirements / parse_routing_requirements_json ---

def test_validate_rejects_unknown_task_class():
    with pytest.raises(ValueError, match="task_class"):
        validate_routing_requirements({"task_class": "nonsense"})


def test_validate_rejects_negative_token_counts():
    with pytest.raises(ValueError, match="input_tokens"):
        validate_routing_requirements({"input_tokens": -5})


def test_validate_rejects_non_int_token_counts():
    with pytest.raises(ValueError, match="reserve_tokens"):
        validate_routing_requirements({"reserve_tokens": "lots"})


def test_validate_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown fields"):
        validate_routing_requirements({"bogus_field": 1})


def test_validate_rejects_malformed_provenance_missing_field():
    with pytest.raises(ValueError, match="provenance missing fields"):
        validate_routing_requirements({"provenance": {"frozen_sha": "a" * 40}})


def test_validate_rejects_contributor_without_maker():
    with pytest.raises(ValueError, match="maker"):
        validate_routing_requirements({
            "provenance": {
                "frozen_sha": "a" * 40, "verified_by": "parent", "complete": True,
                "contributors": [{"evidence": "run:1"}],
            },
        })


def test_validate_accepts_none_and_valid_full_shape():
    assert validate_routing_requirements(None) is None
    out = validate_routing_requirements({
        "task_class": "high-consequence", "required_capabilities": ["tool_use", "tool_use"],
        "input_tokens": 5000, "reserve_tokens": 2000,
        "provenance": {
            "frozen_sha": "a" * 40, "verified_by": "parent", "complete": True,
            "contributors": [{"maker": "openai"}],
        },
    })
    assert out["task_class"] == "high-consequence"
    assert out["required_capabilities"] == ["tool_use"]  # deduped
    assert out["input_tokens"] == 5000
    assert out["reserve_tokens"] == 2000
    assert out["provenance"]["contributors"] == [{"maker": "openai"}]


def test_parse_json_rejects_invalid_json():
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_routing_requirements_json("{not json")


def test_parse_json_none_and_empty_are_no_intake():
    assert parse_routing_requirements_json(None) is None
    assert parse_routing_requirements_json("") is None
    assert parse_routing_requirements_json("   ") is None


def test_create_task_rejects_malformed_requirements_at_creation_not_claim(kanban_home):
    """A bad --routing-requirements payload must fail at hermes_cli.kanban_db.create_task
    (creation time), never be silently stored and only discovered later at claim time."""
    with kbc.connect() as conn:
        with pytest.raises(ValueError, match="task_class"):
            kb.create_task(
                conn, title="bad intake", assignee="alice", routing_role="builder",
                routing_requirements={"task_class": "not-a-real-class"},
            )


# --- requirements affect task-specific route eligibility ---

def test_requirements_task_class_alone_cannot_attest_shallow_scope(
    kanban_home, all_assignees_spawnable,
):
    """Model-facing intake is a proposal, not a persisted authority record."""
    _activate(kanban_home)
    captured = {}

    def _fake_spawn(task, workspace):
        captured["model"] = task.model_override
        return 1

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="typo fix", assignee="alice", routing_role="builder",
            routing_requirements={"task_class": "established-pattern", "input_tokens": 1000, "reserve_tokens": 8192},
        )
        kbd.dispatch_once(conn, dry_run=False, spawn_fn=_fake_spawn)
        task = kb.get_task(conn, tid)

    decision = get_receipt(kanban_home, task.routing_receipt_id)
    assert decision["requirements"]["task_class"] == "established-pattern"
    assert decision["requirements"]["classification"] is None
    assert decision["requirements"]["quality"] == "deep"
    assert decision["selected"]["route_id"] == "openai-deep"


def test_requirements_unclassified_task_class_defaults_deep(kanban_home, all_assignees_spawnable):
    """No task_class supplied at all -> the documented conservative deep
    default applies (design §3.B: 'unclassified... scope defaults to deep'),
    never the arbitrary fixed 'cross-component' placeholder string."""
    _activate(kanban_home)

    def _fake_spawn(task, workspace):
        return 1

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="vague task", assignee="alice", routing_role="builder",
                             routing_requirements={"input_tokens": 1000, "reserve_tokens": 8192})
        kbd.dispatch_once(conn, dry_run=False, spawn_fn=_fake_spawn)
        task = kb.get_task(conn, tid)

    decision = get_receipt(kanban_home, task.routing_receipt_id)
    assert decision["requirements"]["quality"] == "deep"
    assert decision["selected"]["route_id"] == "openai-deep"
    # Never the old hardcoded literal — the field genuinely reflects "no
    # class was supplied", not a fabricated cross-component classification.
    assert decision["requirements"]["task_class"] != "cross-component"


def test_requirements_insufficient_input_budget_rejects_route(kanban_home, all_assignees_spawnable):
    """A genuinely large declared input_tokens/reserve_tokens must actually
    reject a route whose verified_input_budget can't fit it — proving real
    budget/input sizing flows from per-task intake, not a hardcoded 0."""
    _activate(kanban_home)
    captured = {}

    def _fake_spawn(task, workspace):
        captured["model"] = task.model_override
        return 1

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="huge context task", assignee="alice", routing_role="builder",
            routing_requirements={
                # 250k tokens exceeds EVERY approved builder route's
                # verified_input_budget (50k shallow, 200k deep) — proving
                # real per-task input sizing actually constrains eligibility,
                # not a hardcoded 0 that would silently satisfy any budget.
                "input_tokens": 200000, "reserve_tokens": 50000,
            },
            max_retries=1,
        )
        res = kbd.dispatch_once(conn, dry_run=False, spawn_fn=_fake_spawn)
        task = kb.get_task(conn, tid)

    # No approved builder route can fit the declared budget -> no candidate
    # qualifies -> spawn fails closed, never silently proceeds anyway.
    assert "model" not in captured
    assert task.routing_receipt_id is None
    assert task.status in ("blocked", "ready")


def test_requirements_capabilities_affect_eligibility(kanban_home, all_assignees_spawnable):
    """A declared required_capabilities that only the deep route satisfies
    must select that route even under a shallow task_class ranking that
    would otherwise prefer the shallow-only route."""
    _activate(kanban_home)
    captured = {}

    def _fake_spawn(task, workspace):
        captured["model"] = task.model_override
        return 1

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="needs tools", assignee="alice", routing_role="builder",
            routing_requirements={
                "task_class": "established-pattern", "required_capabilities": ["tool_use"],
                "input_tokens": 1000, "reserve_tokens": 8192,
            },
        )
        kbd.dispatch_once(conn, dry_run=False, spawn_fn=_fake_spawn)
        task = kb.get_task(conn, tid)

    assert captured["model"] == "gpt-5"  # only openai-deep has tool_use
    decision = get_receipt(kanban_home, task.routing_receipt_id)
    assert decision["selected"]["route_id"] == "openai-deep"
    assert decision["rejections"].get("openai-shallow") == ["missing_capabilities"]


# --- independent review provenance: real evidence required, forged/absent doesn't select same maker ---

def test_review_role_missing_provenance_fails_closed(kanban_home, all_assignees_spawnable):
    """A reviewquality task with NO --routing-requirements provenance intake
    must fail closed (provenance_incomplete), never manufacture a fabricated
    complete=True/empty-contributors shape to let it through."""
    _activate(kanban_home)

    def _fake_spawn(task, workspace):
        return 1

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="review with no provenance", assignee="alice",
            routing_role="reviewquality", max_retries=1, reasoning_effort="high",
        )
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (tid,))
        res = kbd.dispatch_once(conn, dry_run=False, spawn_fn=_fake_spawn)
        task = kb.get_task(conn, tid)

    assert res.spawned == []
    assert task.routing_receipt_id is None
    assert task.status in ("blocked", "ready")


def test_review_role_with_real_provenance_excludes_contributing_maker(
    kanban_home, all_assignees_spawnable,
):
    """A reviewquality task with genuine contributor provenance naming
    'openai' must select the non-openai (anthropic) reviewer route — proving
    requirements-driven independence, not a static maker table."""
    _activate(kanban_home)
    captured = {}

    def _fake_spawn(task, workspace):
        captured["provider"] = task.provider_override
        return 1

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="review PR", assignee="alice", routing_role="reviewquality",
            reasoning_effort="high",
            routing_requirements={
                "input_tokens": 1000, "reserve_tokens": 8192,
                "provenance": {
                    "frozen_sha": "a" * 40, "verified_by": "parent", "complete": True,
                    "contributors": [{"maker": "openai", "evidence": "run:builder-attempt"}],
                },
            },
        )
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (tid,))
        res = kbd.dispatch_once(conn, dry_run=False, spawn_fn=_fake_spawn)
        task = kb.get_task(conn, tid)

    assert captured["provider"] == "anthropic"
    decision = get_receipt(kanban_home, task.routing_receipt_id)
    assert decision["selected"]["route_id"] == "anthropic-review"
    assert decision["rejections"].get("openai-review") == ["contributing_maker"]


def test_ordinary_builder_role_never_blocked_for_lacking_review_provenance(
    kanban_home, all_assignees_spawnable,
):
    """An ordinary (non-review) role with zero provenance intake must NOT be
    blocked — the mandatory-review-provenance fail-closed rule is scoped to
    independent-review roles only, per the explicit task requirement."""
    _activate(kanban_home)
    captured = {}

    def _fake_spawn(task, workspace):
        captured["model"] = task.model_override
        return 1

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="plain build", assignee="alice", routing_role="builder",
                             routing_requirements={"input_tokens": 1000, "reserve_tokens": 8192})
        res = kbd.dispatch_once(conn, dry_run=False, spawn_fn=_fake_spawn)
        task = kb.get_task(conn, tid)

    assert "model" in captured
    assert task.routing_receipt_id


def test_shadow_routing_records_recommendation_without_changing_worker_route(
    kanban_home, all_assignees_spawnable,
):
    """Shadow is additive observation for an otherwise legacy launch.  The
    recommended route is inspectable, but the worker keeps its explicit legacy
    model/provider and receives no enforcement receipt."""
    _activate(kanban_home)
    captured = {}

    def _fake_spawn(task, workspace):
        captured.update(
            provider=task.provider_override,
            model=task.model_override,
            receipt=task.routing_receipt_id,
        )
        return 1

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="observe legacy worker", assignee="alice",
            model_override="legacy-model", provider_override="legacy-provider",
            routing_role="builder", routing_mode="shadow",
            routing_requirements={"input_tokens": 1000, "reserve_tokens": 8192},
        )
        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=_fake_spawn)
        task = kb.get_task(conn, tid)
        events = kb.list_events(conn, tid)

    assert result.spawned and captured == {
        "provider": "legacy-provider", "model": "legacy-model", "receipt": None,
    }
    assert task.routing_receipt_id is None
    assert task.routing_shadow_receipt_id
    assert any(event.kind == "routing_shadow_selected" for event in events)


def test_shadow_routing_failure_never_blocks_legacy_worker(
    kanban_home, all_assignees_spawnable,
):
    """A missing policy is an observation failure in shadow mode, not
    permission to delay, reroute, retry, or block the legacy launch."""
    captured = {}

    def _fake_spawn(task, workspace):
        captured["model"] = task.model_override
        return 1

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="shadow without policy", assignee="alice",
            model_override="legacy-model", provider_override="legacy-provider",
            routing_role="builder", routing_mode="shadow",
        )
        result = kbd.dispatch_once(conn, dry_run=False, spawn_fn=_fake_spawn)
        task = kb.get_task(conn, tid)
        events = kb.list_events(conn, tid)

    assert result.spawned and captured["model"] == "legacy-model"
    assert task.routing_receipt_id is None
    assert task.routing_shadow_receipt_id is None
    assert any(event.kind == "routing_shadow_failed" for event in events)
