"""Focused tests for the CH-385 /autopilot controller entrypoint."""

from __future__ import annotations

from datetime import datetime, timezone
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.autopilot import (
    ACTION_DISABLE,
    ACTION_DRY_RUN,
    ACTION_ENABLE,
    ACTION_ONE_SHOT,
    ACTION_STATUS,
    AutopilotParseError,
    AutopilotStateStore,
    autopilot_controller_tick,
    classify_linear_target,
    classify_work_state_target,
    evaluate_autopilot_pr_closeout_gate,
    evaluate_dry_run_admission,
    evaluate_kanban_closeout_gate,
    handle_autopilot_command,
    handle_autopilot_runtime_command,
    ingest_autopilot_executor_result,
    parse_autopilot_args,
    resolve_autopilot_integration_branch,
    run_autopilot_once,
)
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


class _FakeLinearClient:
    def __init__(self, payloads, *, queue=None):
        self.payloads = payloads
        self.queue = queue
        self.calls = []
        self.queue_calls = []

    def fetch_issue(self, identifier: str):
        self.calls.append(identifier)
        return self.payloads[identifier]

    def list_execution_ready_issues(self, *, team_key="CH", state_name="Execution Ready", limit=100):
        self.queue_calls.append(
            {"team_key": team_key, "state_name": state_name, "limit": limit}
        )
        if isinstance(self.queue, dict):
            return self.queue
        return {"status": "ok", "issues": list(self.queue or [])}


class _FakeWorkStateStore:
    def __init__(self, records=None):
        self.records = records or []
        self.resolve_calls = []
        self.upsert_calls = []
        self.update_calls = []

    def list_records(self):
        return self.records

    def upsert(self, record):
        self.upsert_calls.append(record)
        self.records = [
            existing
            for existing in self.records
            if not (
                existing.work_id == record.work_id
                and existing.owner_session_id == record.owner_session_id
            )
        ]
        self.records.append(record)

    def update_record(self, work_id: str, owner_session_id: str, **updates):
        self.update_calls.append(
            {"work_id": work_id, "owner_session_id": owner_session_id, "updates": updates}
        )
        for record in self.records:
            if record.work_id == work_id and record.owner_session_id == owner_session_id:
                for key, value in updates.items():
                    setattr(record, key, value)
                return True
        return False

    def resolve_delegated_signal_candidate(self, *, work_id: str, live_only: bool):
        self.resolve_calls.append({"work_id": work_id, "live_only": live_only})
        return {"status": "no_match", "reason": "not_found", "matches": []}


class _BrokenWorkStateStore:
    def list_records(self):
        raise RuntimeError("corrupt work_state")

    def resolve_delegated_signal_candidate(self, *, work_id: str, live_only: bool):
        raise RuntimeError("corrupt work_state")


def _issue(
    identifier: str,
    *,
    state_name: str = "Execution Ready",
    state_type: str = "started",
    parent=None,
    children=None,
    description: str | None = None,
    priority: int = 3,
    project: str | None = "Autopilot",
):
    return {
        "status": "ok",
        "identifier": identifier,
        "title": f"{identifier} title",
        "description": description
        if description is not None
        else "## Done when\n- implementation complete\n\n## Verification\n- focused tests pass",
        "priority": priority,
        "url": f"https://linear.app/chriskim12/issue/{identifier.lower()}",
        "state": {"name": state_name, "type": state_type},
        "parent": parent,
        "children": {"nodes": children or []},
        "project": {"name": project} if project else None,
        "labels": {"nodes": []},
    }


def _state_store(tmp_path):
    return AutopilotStateStore(tmp_path / "autopilot-state.json")


def _autopilot_work_record(identifier="CH-401", **overrides):
    from gateway.work_state import WorkRecord

    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    values = {
        "work_id": identifier,
        "title": f"{identifier} title",
        "objective": f"Execute Linear {identifier}",
        "owner": "hermes",
        "executor": "hermes",
        "mode": "autopilot",
        "owner_session_id": f"autopilot:{identifier}",
        "state": "completed",
        "started_at": now,
        "last_progress_at": now,
        "next_action": "Await review-ready PR evidence",
        "proof": "executor_completed",
        "close_authority": "autopilot_pr_closeout_gate",
        "cleanup_required": True,
        "cleanup_proof": "pending_autopilot_pr_closeout_cleanup",
        "review_closeout": {"status": "pending", "work_id": identifier},
    }
    values.update(overrides)
    return WorkRecord(**values)


@pytest.mark.parametrize(
    "raw,expected_action,expected_target",
    [
        ("", ACTION_STATUS, None),
        ("status", ACTION_STATUS, None),
        ("status ch-385", ACTION_STATUS, "CH-385"),
        ("dry-run", ACTION_DRY_RUN, None),
        ("dry-run ch-385", ACTION_DRY_RUN, "CH-385"),
        ("ON", ACTION_ENABLE, None),
        ("on", ACTION_ENABLE, None),
        ("OFF", ACTION_DISABLE, None),
        ("off", ACTION_DISABLE, None),
        ("ch-385", ACTION_ONE_SHOT, "CH-385"),
    ],
)
def test_parse_autopilot_accepts_only_contract_shapes(raw, expected_action, expected_target):
    command = parse_autopilot_args(raw)

    assert command.action == expected_action
    assert command.target_id == expected_target


@pytest.mark.parametrize(
    "raw",
    [
        "ON CH-385",
        "OFF CH-385",
        "status CH-385 extra",
        "dry-run CH-385 extra",
        "status not-a-card",
        "dry-run not-a-card",
        "CH-nope",
        "CH-385 CH-386",
        "start CH-385",
    ],
)
def test_parse_autopilot_rejects_invalid_shapes_fail_closed(raw):
    with pytest.raises(AutopilotParseError):
        parse_autopilot_args(raw)


def test_status_and_dry_run_are_read_only_no_state_write_no_executor(tmp_path):
    store = _state_store(tmp_path)
    work_state = _FakeWorkStateStore()
    spawner = MagicMock()

    status = handle_autopilot_command(
        "status",
        state_store=store,
        work_state_store=work_state,
        executor_spawner=spawner,
    )
    dry_run = handle_autopilot_command(
        "dry-run",
        state_store=store,
        work_state_store=work_state,
        executor_spawner=spawner,
    )

    assert status.decision["side_effects"] == {
        "state_written": False,
        "executor_spawned": False,
        "linear_done_mutated": False,
    }
    assert dry_run.decision["side_effects"] == {
        "state_written": False,
        "executor_spawned": False,
        "linear_done_mutated": False,
    }
    assert status.decision["admission"]["admission_bypassed"] is False
    assert dry_run.decision["admission"]["admission_bypassed"] is False
    assert store.path.exists() is False
    spawner.assert_not_called()


def test_targeted_status_and_dry_run_are_read_only_admission_classification_only(tmp_path):
    store = _state_store(tmp_path)
    work_state = _FakeWorkStateStore()
    spawner = MagicMock()
    linear = _FakeLinearClient({"CH-385": _issue("CH-385")})

    status = handle_autopilot_command(
        "status CH-385",
        state_store=store,
        work_state_store=work_state,
        linear_client=linear,
        executor_spawner=spawner,
    )
    dry_run = handle_autopilot_command(
        "dry-run CH-385",
        state_store=store,
        work_state_store=work_state,
        linear_client=linear,
        executor_spawner=spawner,
    )

    assert status.decision["side_effects"] == {
        "state_written": False,
        "executor_spawned": False,
        "linear_done_mutated": False,
    }
    assert dry_run.decision["side_effects"] == {
        "state_written": False,
        "executor_spawned": False,
        "linear_done_mutated": False,
    }
    assert status.decision["linear"]["shape"] == "standalone"
    assert dry_run.decision["linear"]["shape"] == "standalone"
    assert status.decision["admission"]["admission_bypassed"] is False
    assert dry_run.decision["admission"]["admission_bypassed"] is False
    assert linear.calls == ["CH-385", "CH-385"]
    assert work_state.resolve_calls == [
        {"work_id": "CH-385", "live_only": True},
        {"work_id": "CH-385", "live_only": True},
    ]
    assert store.path.exists() is False
    spawner.assert_not_called()


@pytest.mark.parametrize("enabled_value", ["false", "true", "0", "1", 0, 1, None])
def test_state_store_loads_malformed_enabled_values_disabled_fail_closed(tmp_path, enabled_value):
    store = _state_store(tmp_path)
    store.path.write_text(
        json.dumps({"schema_version": 1, "enabled": enabled_value}),
        encoding="utf-8",
    )

    state = store.status()

    assert state["enabled"] is False
    assert state["state_error"] == "invalid_enabled_type"


def test_state_store_loads_boolean_enabled_values_without_coercion(tmp_path):
    store = _state_store(tmp_path)
    store.path.write_text('{"schema_version": 1, "enabled": true}', encoding="utf-8")

    assert store.status()["enabled"] is True


def test_on_off_persist_controller_intent_only(tmp_path):
    store = _state_store(tmp_path)
    work_state = _FakeWorkStateStore()
    now = datetime(2026, 5, 4, 12, 0, tzinfo=timezone.utc)

    enabled = handle_autopilot_command(
        "ON",
        actor="tester",
        state_store=store,
        work_state_store=work_state,
        now=now,
    )
    disabled = handle_autopilot_command(
        "OFF",
        actor="tester",
        state_store=store,
        work_state_store=work_state,
        now=now,
    )

    assert enabled.decision["side_effects"]["state_written"] is True
    assert enabled.decision["side_effects"]["executor_spawned"] is False
    assert enabled.decision["side_effects"]["linear_done_mutated"] is False
    assert enabled.decision["admission"] == {
        "status": "controller_intent_enabled",
        "reason": "future_automatic_starts_still_require_admission",
        "admission_bypassed": False,
    }
    assert disabled.decision["state"]["enabled"] is False
    assert disabled.decision["state"]["updated_by"] == "tester"
    assert disabled.decision["side_effects"] == {
        "state_written": True,
        "executor_spawned": False,
        "linear_done_mutated": False,
    }
    assert store.status()["enabled"] is False


@pytest.mark.parametrize(
    "payload,expected_shape",
    [
        (
            _issue(
                "CH-173",
                children=[{"identifier": "CH-385", "state": {"name": "Execution Ready"}}],
            ),
            "parent",
        ),
        (
            _issue(
                "CH-385",
                parent={"identifier": "CH-173", "state": {"name": "Backlog"}},
            ),
            "child",
        ),
        (_issue("CH-999"), "standalone"),
    ],
)
def test_classify_linear_target_shapes(payload, expected_shape):
    classified = classify_linear_target(payload, payload["identifier"])

    assert classified["shape"] == expected_shape
    assert classified["status"] == "ok"


def test_one_shot_classifies_target_without_executor_or_linear_mutation(tmp_path):
    store = _state_store(tmp_path)
    work_state = _FakeWorkStateStore()
    spawner = MagicMock()
    linear = _FakeLinearClient(
        {
            "CH-385": _issue(
                "CH-385",
                parent={"identifier": "CH-173", "state": {"name": "Backlog"}},
            )
        }
    )

    result = handle_autopilot_command(
        "ch-385",
        state_store=store,
        work_state_store=work_state,
        linear_client=linear,
        executor_spawner=spawner,
    )

    assert result.decision["linear"]["shape"] == "child"
    assert result.decision["linear"]["execution_ready"] is True
    assert result.decision["admission"] == {
        "status": "eligible_for_admission",
        "reason": "one_shot_target_requires_admission_before_executor_start",
        "admission_bypassed": False,
    }
    assert result.decision["side_effects"] == {
        "state_written": False,
        "executor_spawned": False,
        "linear_done_mutated": False,
    }
    assert linear.calls == ["CH-385"]
    assert work_state.resolve_calls == [{"work_id": "CH-385", "live_only": True}]
    assert store.path.exists() is False
    spawner.assert_not_called()


def test_parent_target_requires_child_selection_without_spawning(tmp_path):
    store = _state_store(tmp_path)
    spawner = MagicMock()
    linear = _FakeLinearClient(
        {
            "CH-173": _issue(
                "CH-173",
                children=[{"identifier": "CH-385", "state": {"name": "Execution Ready"}}],
            )
        }
    )

    result = handle_autopilot_command(
        "CH-173",
        state_store=store,
        work_state_store=_FakeWorkStateStore(),
        linear_client=linear,
        executor_spawner=spawner,
    )

    assert result.decision["linear"]["shape"] == "parent"
    assert result.decision["admission"]["status"] == "requires_child_selection"
    assert result.decision["admission"]["admission_bypassed"] is False
    assert result.decision["side_effects"]["executor_spawned"] is False
    spawner.assert_not_called()


def test_unavailable_linear_target_blocks_fail_closed(tmp_path):
    store = _state_store(tmp_path)
    spawner = MagicMock()
    linear = _FakeLinearClient(
        {
            "CH-404": {
                "status": "unavailable",
                "reason": "LINEAR_API_KEY_missing",
                "identifier": "CH-404",
            }
        }
    )

    result = handle_autopilot_command(
        "CH-404",
        state_store=store,
        work_state_store=_FakeWorkStateStore(),
        linear_client=linear,
        executor_spawner=spawner,
    )

    assert result.decision["linear"]["shape"] == "unknown"
    assert result.decision["admission"] == {
        "status": "blocked",
        "reason": "linear_target_unavailable_fail_closed",
        "admission_bypassed": False,
    }
    assert result.decision["side_effects"]["executor_spawned"] is False
    assert result.decision["side_effects"]["linear_done_mutated"] is False
    spawner.assert_not_called()


def test_dry_run_enabled_queue_selects_one_eligible_task_with_would_artifacts(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    spawner = MagicMock()
    issue = _issue(
        "CH-401",
        parent={"identifier": "CH-173", "title": "parent", "state": {"name": "Backlog", "type": "backlog"}},
    )
    linear = _FakeLinearClient({}, queue=[issue])

    result = handle_autopilot_command(
        "dry-run",
        state_store=store,
        work_state_store=_FakeWorkStateStore(),
        linear_client=linear,
        executor_spawner=spawner,
    )

    dry_run = result.decision["dry_run"]
    assert dry_run["status"] == "would_run"
    assert dry_run["selected_issue"]["identifier"] == "CH-401"
    assert dry_run["would_create_work_state"] == {
        "work_id": "CH-401",
        "owner": "hermes",
        "executor": "hermes",
        "mode": "autopilot",
        "state": "created",
        "parent_id": "CH-173",
    }
    assert dry_run["would_goal_contract"]["command"].startswith("/goal ")
    assert "CH-401" in dry_run["would_goal_contract"]["summary"]
    assert result.decision["admission"] == {
        "status": "would_run",
        "reason": "dry_run_selected_execution_ready_issue",
        "admission_bypassed": False,
    }
    assert result.decision["side_effects"] == {
        "state_written": False,
        "executor_spawned": False,
        "linear_done_mutated": False,
    }
    assert linear.queue_calls == [
        {"team_key": "CH", "state_name": "Execution Ready", "limit": 100}
    ]
    assert store.status()["enabled"] is True
    spawner.assert_not_called()


def test_targeted_dry_run_opt_in_includes_kanban_payload_contract(tmp_path):
    store = _state_store(tmp_path)
    work_state = _FakeWorkStateStore()
    spawner = MagicMock()
    issue = _issue(
        "CH-410",
        parent={"identifier": "CH-173", "title": "parent", "state": {"name": "Backlog", "type": "backlog"}},
    )
    issue["comments"] = {"nodes": [{"body": "latest context", "createdAt": "2026-05-07T00:00:00Z"}]}
    linear = _FakeLinearClient({"CH-410": issue})

    result = handle_autopilot_command(
        "dry-run CH-410",
        state_store=store,
        work_state_store=work_state,
        linear_client=linear,
        executor_spawner=spawner,
        repo_policy={
            "kanban_payload_dry_run": True,
            "github_repo": "chriskim12/hermes-agent",
            "integration_branch": "develop",
            "executor": "kanban-dispatcher",
            "tenant": "hermes",
            "profile": "coder",
            "skills": ["kanban-worker", "linear"],
        },
    )

    dry_run = result.decision["dry_run"]
    payload = dry_run["would_kanban_payload"]
    assert dry_run["status"] == "would_run"
    assert payload["status"] == "would_create"
    assert payload["source_card"]["identifier"] == "CH-410"
    assert payload["goal_contract"] == {
        "summary": "Execute Linear CH-410: CH-410 title",
        "mode": "single-card",
    }
    assert payload["repo_intent"]["repo_full_name"] == "chriskim12/hermes-agent"
    assert payload["repo_intent"]["base_branch"] == "develop"
    assert payload["repo_intent"]["worktree_branch"].startswith("autopilot/ch-410")
    assert payload["execution_hints"] == {
        "executor": "kanban-dispatcher",
        "profile": "coder",
        "skills": ["kanban-worker", "linear"],
    }
    assert payload["task"]["tenant"] == "hermes"
    assert payload["task"]["idempotency_key"] == "linear:CH-410"
    assert "```json source_payload" in payload["task"]["body"]
    assert "metadata" not in payload["task"]
    assert payload["dependencies"] == [{"source": "linear_parent", "identifier": "CH-173"}]
    assert payload["comments"] == [{"body": "latest context", "created_at": "2026-05-07T00:00:00Z"}]
    assert payload["events"][0]["kind"] == "autopilot_kanban_payload_dry_run"
    assert payload["task_runs_metadata"]["idempotency_key"] == "linear:CH-410"
    assert payload["side_effects"] == {
        "kanban_task_written": False,
        "executor_spawned": False,
        "linear_done_mutated": False,
        "kanban_done_projected_to_linear": False,
    }
    assert work_state.upsert_calls == []
    spawner.assert_not_called()


def test_kanban_enabled_flag_alone_does_not_opt_into_payload_dry_run(tmp_path):
    store = _state_store(tmp_path)
    issue = _issue("CH-410")
    linear = _FakeLinearClient({"CH-410": issue})

    result = handle_autopilot_command(
        "dry-run CH-410",
        state_store=store,
        work_state_store=_FakeWorkStateStore(),
        linear_client=linear,
        repo_policy={"kanban": {"enabled": True}, "tenant": "hermes"},
    )

    dry_run = result.decision["dry_run"]
    assert dry_run["status"] == "would_run"
    assert "would_kanban_payload" not in dry_run


def test_nested_payload_dry_run_flag_is_explicit_opt_in(tmp_path):
    store = _state_store(tmp_path)
    issue = _issue("CH-410")
    linear = _FakeLinearClient({"CH-410": issue})

    result = handle_autopilot_command(
        "dry-run CH-410",
        state_store=store,
        work_state_store=_FakeWorkStateStore(),
        linear_client=linear,
        repo_policy={
            "kanban": {"payload_dry_run": True, "executor": "kanban-dispatcher", "tenant": "hermes"},
            "github_repo": "chriskim12/hermes-agent",
        },
    )

    dry_run = result.decision["dry_run"]
    assert dry_run["status"] == "would_run"
    assert dry_run["would_kanban_payload"]["task"]["idempotency_key"] == "linear:CH-410"


def test_kanban_payload_contract_missing_repo_or_executor_fails_closed(tmp_path):
    store = _state_store(tmp_path)
    issue = _issue("CH-410")
    linear = _FakeLinearClient({"CH-410": issue})

    result = handle_autopilot_command(
        "dry-run CH-410",
        state_store=store,
        work_state_store=_FakeWorkStateStore(),
        linear_client=linear,
        repo_policy={"kanban_payload_dry_run": True, "tenant": "hermes"},
    )

    dry_run = result.decision["dry_run"]
    payload = dry_run["would_kanban_payload"]
    assert dry_run["status"] == "blocked"
    assert dry_run["reason"] == "kanban_payload_contract_missing_required_fields"
    assert dry_run["would_create_work_state"] is None
    assert payload["status"] == "blocked"
    assert payload["missing"] == ["repo_full_name", "executor"]
    assert payload["side_effects"]["kanban_task_written"] is False
    assert result.decision["side_effects"]["executor_spawned"] is False


def test_kanban_native_candidate_selection_picks_ready_task_without_linear(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    task = SimpleNamespace(
        id="t_kanban1",
        public_id="BO-101",
        title="Native queue task",
        body="## Done when\n- done\n\n## Verification\n- tests",
        status="ready",
        tenant="brain-os",
        review_phase=None,
    )
    linear = _FakeLinearClient({}, queue={"status": "unavailable", "reason": "linear_should_not_be_required"})

    dry_run = evaluate_dry_run_admission(
        command=parse_autopilot_args("dry-run"),
        state=store.status(),
        linear_client=linear,
        work_state_store=_FakeWorkStateStore(),
        repo_policy={
            "kanban": {
                "native_candidate_selection": True,
                "approved_tenants": ["brain-os"],
            }
        },
        kanban_task_provider=lambda: [task],
    )

    assert dry_run["status"] == "would_run"
    assert dry_run["reason"] == "kanban_native_ready_task_selected"
    assert dry_run["source"] == "kanban"
    assert dry_run["selected_issue"] == {
        "identifier": "BO-101",
        "title": "Native queue task",
        "url": None,
        "source": "kanban",
        "task_id": "t_kanban1",
        "public_id": "BO-101",
    }
    assert dry_run["would_create_work_state"] == {
        "work_id": "BO-101",
        "owner": "hermes",
        "executor": "hermes",
        "mode": "kanban",
        "state": "created",
        "parent_id": None,
        "source": "kanban",
        "task_id": "t_kanban1",
    }
    assert dry_run["would_goal_contract"]["command"].startswith("/goal Execute Kanban BO-101")
    assert dry_run["side_effects"] == {
        "work_state_written": False,
        "executor_spawned": False,
        "linear_mutated": False,
        "kanban_task_claimed": False,
    }
    assert linear.queue_calls == []


def test_kanban_native_candidate_selection_rejects_unapproved_tenant_without_linear_fallback(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    task = SimpleNamespace(
        id="t_other",
        public_id="WS-101",
        title="Other tenant task",
        body="",
        status="ready",
        tenant="whystarve",
        review_phase=None,
    )
    linear = _FakeLinearClient({}, queue={"status": "unavailable", "reason": "linear_should_not_be_required"})

    dry_run = evaluate_dry_run_admission(
        command=parse_autopilot_args("dry-run"),
        state=store.status(),
        linear_client=linear,
        work_state_store=_FakeWorkStateStore(),
        repo_policy={
            "kanban_native_candidate_selection": True,
            "approved_kanban_tenants": ["brain-os"],
        },
        kanban_task_provider=lambda: [task],
    )

    assert dry_run["status"] == "blocked"
    assert dry_run["reason"] == "no_eligible_kanban_native_task"
    assert dry_run["candidates"][0]["reason"] == "kanban_tenant_not_approved"
    assert dry_run["selected_issue"] is None
    assert linear.queue_calls == []


def test_kanban_native_candidate_selection_off_pauses_before_queue_lookup(tmp_path):
    store = _state_store(tmp_path)
    calls = []

    dry_run = evaluate_dry_run_admission(
        command=parse_autopilot_args("dry-run"),
        state=store.status(),
        linear_client=_FakeLinearClient({}, queue=[]),
        work_state_store=_FakeWorkStateStore(),
        repo_policy={"kanban_native_candidate_selection": True},
        kanban_task_provider=lambda: calls.append("called") or [],
    )

    assert dry_run["status"] == "paused"
    assert dry_run["reason"] == "controller_disabled_noop"
    assert calls == []


def test_kanban_native_candidate_selection_active_lock_blocks_next_card(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    task = SimpleNamespace(
        id="t_locked",
        public_id="BO-102",
        title="Already locked task",
        body="",
        status="ready",
        tenant="brain-os",
        review_phase=None,
    )
    live = SimpleNamespace(owner="hermes", state="running", work_id="BO-102")

    dry_run = evaluate_dry_run_admission(
        command=parse_autopilot_args("dry-run"),
        state=store.status(),
        linear_client=_FakeLinearClient({}, queue={"status": "unavailable"}),
        work_state_store=_FakeWorkStateStore([live]),
        repo_policy={
            "kanban_native_candidate_selection": True,
            "approved_kanban_tenants": ["brain-os"],
        },
        kanban_task_provider=lambda: [task],
    )

    assert dry_run["status"] == "blocked"
    assert dry_run["candidates"][0]["reason"] == "active_work_state_lock"


def test_kanban_payload_contract_missing_done_when_uses_existing_fail_closed_admission(tmp_path):
    store = _state_store(tmp_path)
    issue = _issue("CH-410", description="## Verification\n- focused tests pass")
    linear = _FakeLinearClient({}, queue=[issue])

    result = handle_autopilot_command(
        "dry-run",
        state_store=store,
        work_state_store=_FakeWorkStateStore(),
        linear_client=linear,
        repo_policy={
            "kanban_payload_dry_run": True,
            "github_repo": "chriskim12/hermes-agent",
            "executor": "kanban-dispatcher",
            "tenant": "hermes",
        },
    )

    dry_run = result.decision["dry_run"]
    assert dry_run["status"] == "paused"
    assert dry_run["reason"] == "controller_disabled_noop"
    store.set_enabled(True, actor="tester")
    result = handle_autopilot_command(
        "dry-run",
        state_store=store,
        work_state_store=_FakeWorkStateStore(),
        linear_client=linear,
        repo_policy={
            "kanban_payload_dry_run": True,
            "github_repo": "chriskim12/hermes-agent",
            "executor": "kanban-dispatcher",
            "tenant": "hermes",
        },
    )
    dry_run = result.decision["dry_run"]
    assert dry_run["status"] == "blocked"
    assert dry_run["reason"] == "no_eligible_execution_ready_issue"
    assert dry_run["candidates"][0]["reason"] == "missing_done_when"


def test_dry_run_multiple_eligible_tasks_uses_deterministic_identifier_order(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    linear = _FakeLinearClient(
        {},
        queue=[
            _issue("CH-410", priority=3),
            _issue("CH-409", priority=3),
            _issue("CH-411", priority=3),
        ],
    )

    result = handle_autopilot_command(
        "dry-run",
        state_store=store,
        work_state_store=_FakeWorkStateStore(),
        linear_client=linear,
    )

    dry_run = result.decision["dry_run"]
    assert dry_run["status"] == "would_run"
    assert dry_run["selected_issue"]["identifier"] == "CH-409"
    assert [candidate["identifier"] for candidate in dry_run["candidates"]] == [
        "CH-409",
        "CH-410",
        "CH-411",
    ]


def test_dry_run_off_returns_noop_without_linear_scan_or_executor(tmp_path):
    store = _state_store(tmp_path)
    linear = _FakeLinearClient({}, queue=[_issue("CH-401")])
    spawner = MagicMock()

    result = handle_autopilot_command(
        "dry-run",
        state_store=store,
        work_state_store=_FakeWorkStateStore(),
        linear_client=linear,
        executor_spawner=spawner,
    )

    assert result.decision["dry_run"] == {
        "status": "paused",
        "reason": "controller_disabled_noop",
        "selected_issue": None,
        "would_create_work_state": None,
        "would_goal_contract": None,
        "candidates": [],
        "groups": {"parents": {}, "projects": {}},
    }
    assert result.decision["admission"] == {
        "status": "paused",
        "reason": "controller_disabled_noop",
        "admission_bypassed": False,
    }
    assert linear.queue_calls == []
    spawner.assert_not_called()


def test_dry_run_active_work_state_lock_blocks_candidate(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    locked = SimpleNamespace(
        work_id="CH-401",
        state="running",
        owner="hermes",
        executor="omx",
        mode="delegated",
    )
    linear = _FakeLinearClient({}, queue=[_issue("CH-401")])

    result = handle_autopilot_command(
        "dry-run",
        state_store=store,
        work_state_store=_FakeWorkStateStore([locked]),
        linear_client=linear,
    )

    dry_run = result.decision["dry_run"]
    assert dry_run["status"] == "blocked"
    assert dry_run["reason"] == "no_eligible_execution_ready_issue"
    assert dry_run["selected_issue"] is None
    assert dry_run["candidates"] == [
        {
            "identifier": "CH-401",
            "eligible": False,
            "reason": "active_work_state_lock",
            "shape": "standalone",
            "parent_id": None,
            "project": "Autopilot",
        }
    ]


def test_autopilot_queue_blocks_next_card_without_prior_pr_cleanup_evidence(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    prior = _autopilot_work_record("CH-401")
    work_state = _FakeWorkStateStore([prior])
    linear = _FakeLinearClient({}, queue=[_issue("CH-402")])

    result = handle_autopilot_command(
        "dry-run",
        state_store=store,
        work_state_store=work_state,
        linear_client=linear,
    )

    dry_run = result.decision["dry_run"]
    assert dry_run["status"] == "blocked"
    assert dry_run["reason"] == "autopilot_closeout_review_gate_blocked"
    assert dry_run["selected_issue"] is None
    assert dry_run["closeout_gate"]["blocked_count"] == 1
    assert dry_run["closeout_gate"]["blocked_cards"][0]["work_id"] == "CH-401"
    assert "pr_url" in dry_run["closeout_gate"]["blocked_cards"][0]["missing"]
    assert "cleanup_proof_pending" in dry_run["closeout_gate"]["blocked_cards"][0]["violations"]
    assert linear.queue_calls == []


def test_autopilot_queue_continuation_allows_valid_prior_pr_cleanup_evidence(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    prior = _autopilot_work_record(
        "CH-401",
        cleanup_proof="task_owned_worktree_removed",
        review_closeout={
            **_valid_pr_closeout_evidence(
                work_id="CH-401",
                remote_branch="yuuka/CH-401-title",
                task_branch="yuuka/CH-401-title",
                pr_head="yuuka/CH-401-title",
                task_worktree_path="/repo/hermes-agent/.worktrees/ch-401-title",
                cleanup_proof="task_owned_worktree_removed",
            ),
            "executor_event": {"status": "executor_finished", "proof": "executor result ingested"},
        },
    )
    work_state = _FakeWorkStateStore([prior])
    linear = _FakeLinearClient({}, queue=[_issue("CH-402")])
    spawner = MagicMock(return_value={"session_id": "exec-402", "proof": "spawned"})

    result = run_autopilot_once(
        state_store=store,
        work_state_store=work_state,
        linear_client=linear,
        executor_spawner=spawner,
        actor="tester",
        owner_session_id="owner-402",
        expected_repo_full_name="chriskim12/hermes-agent",
    )

    assert result["status"] == "started"
    assert result["work_id"] == "CH-402"
    assert result["executor_spawned"] is True
    assert linear.queue_calls == [{"team_key": "CH", "state_name": "Execution Ready", "limit": 100}]
    assert [record.work_id for record in work_state.records] == ["CH-401", "CH-402"]


def test_autopilot_queue_continuation_blocks_valid_prior_pr_without_expected_repo_context(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    prior = _autopilot_work_record(
        "CH-401",
        cleanup_proof="task_owned_worktree_removed",
        review_closeout={
            **_valid_pr_closeout_evidence(
                work_id="CH-401",
                remote_branch="yuuka/CH-401-title",
                task_branch="yuuka/CH-401-title",
                pr_head="yuuka/CH-401-title",
                task_worktree_path="/repo/hermes-agent/.worktrees/ch-401-title",
                cleanup_proof="task_owned_worktree_removed",
                trusted_repo_verified=True,
            ),
            "executor_event": {"status": "executor_finished", "proof": "executor result ingested"},
        },
    )
    work_state = _FakeWorkStateStore([prior])
    linear = _FakeLinearClient({}, queue=[_issue("CH-402")])
    spawner = MagicMock()

    result = run_autopilot_once(
        state_store=store,
        work_state_store=work_state,
        linear_client=linear,
        executor_spawner=spawner,
        actor="tester",
        owner_session_id="owner-402",
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "autopilot_closeout_review_gate_blocked"
    assert result["dry_run"]["closeout_gate"]["blocked_cards"][0]["work_id"] == "CH-401"
    assert "trusted_expected_repo_full_name" in result["dry_run"]["closeout_gate"]["blocked_cards"][0]["missing"]
    assert result["work_state_written"] is False
    assert result["executor_spawned"] is False
    assert linear.queue_calls == []
    spawner.assert_not_called()


def test_targeted_run_autopilot_once_blocks_next_card_without_prior_pr_cleanup_evidence(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    prior = _autopilot_work_record("CH-401")
    work_state = _FakeWorkStateStore([prior])
    linear = _FakeLinearClient({"CH-402": _issue("CH-402")})
    spawner = MagicMock()

    result = run_autopilot_once(
        state_store=store,
        work_state_store=work_state,
        linear_client=linear,
        executor_spawner=spawner,
        actor="tester",
        owner_session_id="owner-402",
        target_id="CH-402",
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "autopilot_closeout_review_gate_blocked"
    assert result["work_state_written"] is False
    assert result["executor_spawned"] is False
    assert result["dry_run"]["selected_issue"] is None
    assert result["dry_run"]["closeout_gate"]["blocked_cards"][0]["work_id"] == "CH-401"
    assert linear.calls == []
    spawner.assert_not_called()


def test_dry_run_unavailable_work_state_fails_closed_before_selection(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    linear = _FakeLinearClient({}, queue=[_issue("CH-401")])

    result = handle_autopilot_command(
        "dry-run",
        state_store=store,
        work_state_store=_BrokenWorkStateStore(),
        linear_client=linear,
    )

    dry_run = result.decision["dry_run"]
    assert dry_run["status"] == "blocked"
    assert dry_run["reason"] == "work_state_unavailable_fail_closed"
    assert dry_run["selected_issue"] is None
    assert dry_run["would_create_work_state"] is None
    assert dry_run["would_goal_contract"] is None
    assert dry_run["work_state_reason"] == "work_state_unavailable:RuntimeError"
    assert result.decision["admission"] == {
        "status": "blocked",
        "reason": "work_state_unavailable_fail_closed",
        "admission_bypassed": False,
    }
    assert linear.queue_calls == []


def test_dry_run_real_corrupt_work_state_file_fails_closed_before_selection(tmp_path):
    from gateway.work_state import WorkStateStore

    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    work_state_path = tmp_path / "gateway_work_state.json"
    work_state_path.write_text("{ not json }", encoding="utf-8")
    linear = _FakeLinearClient({}, queue=[_issue("CH-401")])

    result = handle_autopilot_command(
        "dry-run",
        state_store=store,
        work_state_store=WorkStateStore(work_state_path),
        linear_client=linear,
    )

    dry_run = result.decision["dry_run"]
    assert dry_run["status"] == "blocked"
    assert dry_run["reason"] == "work_state_unavailable_fail_closed"
    assert dry_run["selected_issue"] is None
    assert dry_run["would_create_work_state"] is None
    assert dry_run["would_goal_contract"] is None
    assert dry_run["work_state_reason"] == "work_state_unavailable:JSONDecodeError"
    assert linear.queue_calls == []


def test_dry_run_malformed_work_state_payload_fails_closed_before_selection(tmp_path):
    from gateway.work_state import WorkStateStore

    cases = [
        ({"sessions": [{"work_id": "CH-999"}]}, "work_state_unavailable:invalid_payload_schema"),
        (["bad"], "work_state_unavailable:invalid_record_item_type"),
        ({"records": ["bad"]}, "work_state_unavailable:invalid_record_item_type"),
        ({"records": "bad"}, "work_state_unavailable:invalid_records_type"),
    ]
    for index, (payload, expected_reason) in enumerate(cases):
        store = _state_store(tmp_path / f"state-{index}")
        store.set_enabled(True, actor="tester")
        work_state_path = tmp_path / f"gateway_work_state_{index}.json"
        work_state_path.write_text(json.dumps(payload), encoding="utf-8")
        linear = _FakeLinearClient({}, queue=[_issue("CH-401")])

        result = handle_autopilot_command(
            "dry-run",
            state_store=store,
            work_state_store=WorkStateStore(work_state_path),
            linear_client=linear,
        )

        dry_run = result.decision["dry_run"]
        assert dry_run["status"] == "blocked"
        assert dry_run["reason"] == "work_state_unavailable_fail_closed"
        assert dry_run["selected_issue"] is None
        assert dry_run["work_state_reason"] == expected_reason
        assert linear.queue_calls == []


def test_dry_run_missing_linear_access_fails_closed(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    linear = _FakeLinearClient(
        {},
        queue={"status": "unavailable", "reason": "LINEAR_API_KEY_missing"},
    )

    result = handle_autopilot_command(
        "dry-run",
        state_store=store,
        work_state_store=_FakeWorkStateStore(),
        linear_client=linear,
    )

    assert result.decision["dry_run"] == {
        "status": "blocked",
        "reason": "linear_queue_unavailable_fail_closed",
        "selected_issue": None,
        "would_create_work_state": None,
        "would_goal_contract": None,
        "candidates": [],
        "groups": {"parents": {}, "projects": {}},
        "linear_reason": "LINEAR_API_KEY_missing",
    }
    assert result.decision["admission"] == {
        "status": "blocked",
        "reason": "linear_queue_unavailable_fail_closed",
        "admission_bypassed": False,
    }


def test_dry_run_missing_verification_blocks_candidate(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    linear = _FakeLinearClient(
        {},
        queue=[_issue("CH-401", description="## Done when\n- implementation complete")],
    )

    result = handle_autopilot_command(
        "dry-run",
        state_store=store,
        work_state_store=_FakeWorkStateStore(),
        linear_client=linear,
    )

    dry_run = result.decision["dry_run"]
    assert dry_run["status"] == "blocked"
    assert dry_run["reason"] == "no_eligible_execution_ready_issue"
    assert dry_run["candidates"][0]["reason"] == "missing_verification"
    assert dry_run["selected_issue"] is None


def test_dry_run_parent_terminal_refuses_targeted_parent(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    linear = _FakeLinearClient(
        {
            "CH-173": _issue(
                "CH-173",
                state_name="Done",
                state_type="completed",
                children=[_issue("CH-401")],
            )
        }
    )

    result = handle_autopilot_command(
        "dry-run CH-173",
        state_store=store,
        work_state_store=_FakeWorkStateStore(),
        linear_client=linear,
    )

    dry_run = result.decision["dry_run"]
    assert dry_run["status"] == "blocked"
    assert dry_run["reason"] == "parent_terminal"
    assert dry_run["selected_issue"] is None
    assert result.decision["admission"]["reason"] == "parent_terminal"


def test_targeted_parent_dry_run_selects_at_most_one_eligible_child(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    linear = _FakeLinearClient(
        {
            "CH-173": _issue(
                "CH-173",
                state_name="Backlog",
                state_type="backlog",
                children=[_issue("CH-402"), _issue("CH-401")],
            )
        }
    )

    result = handle_autopilot_command(
        "dry-run CH-173",
        state_store=store,
        work_state_store=_FakeWorkStateStore(),
        linear_client=linear,
    )

    dry_run = result.decision["dry_run"]
    assert dry_run["status"] == "would_run"
    assert dry_run["selected_issue"]["identifier"] == "CH-401"
    assert [candidate["identifier"] for candidate in dry_run["candidates"]] == ["CH-401", "CH-402"]
    assert dry_run["would_create_work_state"]["parent_id"] == "CH-173"


def test_dry_run_forbidden_side_effect_boundary_is_explicit(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    spawner = MagicMock()
    linear = _FakeLinearClient({}, queue=[_issue("CH-401")])

    result = handle_autopilot_command(
        "dry-run",
        state_store=store,
        work_state_store=_FakeWorkStateStore(),
        linear_client=linear,
        executor_spawner=spawner,
    )

    assert result.decision["side_effects"] == {
        "state_written": False,
        "executor_spawned": False,
        "linear_done_mutated": False,
    }
    assert "would_create_work_state" in result.decision["dry_run"]
    assert result.decision["dry_run"]["would_create_work_state"]["work_id"] == "CH-401"
    assert store.status()["enabled"] is True
    spawner.assert_not_called()


def test_run_autopilot_once_materializes_one_task_lock_contract_and_executor(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    work_state = _FakeWorkStateStore()
    linear = _FakeLinearClient({}, queue=[_issue("CH-401")])
    spawner = MagicMock(
        return_value={
            "session_id": "exec-1",
            "proof": "spawned",
            "repo_path": "/repo/hermes-agent",
            "worktree_path": "/repo/hermes-agent/.worktrees/ch-401-title",
            "task_branch": "yuuka/CH-401-title",
        }
    )
    now = datetime(2026, 5, 5, 5, 0, tzinfo=timezone.utc)

    result = run_autopilot_once(
        state_store=store,
        work_state_store=work_state,
        linear_client=linear,
        executor_spawner=spawner,
        actor="tester",
        owner_session_id="owner-1",
        now=now,
    )

    assert result["status"] == "started"
    assert result["selected_issue"]["identifier"] == "CH-401"
    assert result["work_state_written"] is True
    assert result["executor_spawned"] is True
    assert result["linear_done_mutated"] is False
    assert result["work_id"] == "CH-401"
    assert result["owner_session_id"] == "owner-1"
    assert len(work_state.upsert_calls) == 1
    record = work_state.records[0]
    assert record.work_id == "CH-401"
    assert record.owner == "hermes"
    assert record.executor == "hermes"
    assert record.mode == "autopilot"
    assert record.state == "running"
    assert record.executor_session_id == "exec-1"
    assert record.repo_path == "/repo/hermes-agent"
    assert record.worktree_path == "/repo/hermes-agent/.worktrees/ch-401-title"
    assert record.task_branch == "yuuka/CH-401-title"
    assert record.proof == "spawned"
    assert record.close_authority == "autopilot_pr_closeout_gate"
    assert record.cleanup_required is True
    assert record.cleanup_proof == "pending_autopilot_pr_closeout_cleanup"
    assert record.review_closeout["status"] == "pending_review_artifacts"
    assert record.review_closeout["task_worktree_path"] == "/repo/hermes-agent/.worktrees/ch-401-title"
    assert record.review_closeout["task_branch"] == "yuuka/CH-401-title"
    assert "Execute Linear CH-401" in record.objective
    spawner.assert_called_once()
    spawn_kwargs = spawner.call_args.kwargs
    assert spawn_kwargs["work_id"] == "CH-401"
    assert spawn_kwargs["owner_session_id"] == "owner-1"
    assert spawn_kwargs["goal_contract"].startswith("/goal")


def test_run_autopilot_once_does_not_pass_kanban_preview_to_executor(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    work_state = _FakeWorkStateStore()
    linear = _FakeLinearClient({}, queue=[_issue("CH-410")])
    spawner = MagicMock(return_value={"session_id": "exec-410", "proof": "spawned"})

    result = run_autopilot_once(
        state_store=store,
        work_state_store=work_state,
        linear_client=linear,
        executor_spawner=spawner,
        owner_session_id="owner-410",
        repo_policy={
            "kanban_payload_dry_run": True,
            "github_repo": "chriskim12/hermes-agent",
            "executor": "kanban-dispatcher",
            "tenant": "hermes",
        },
    )

    assert result["status"] == "started"
    assert result["dry_run"]["would_kanban_payload"]["task"]["idempotency_key"] == "linear:CH-410"
    spawn_kwargs = spawner.call_args.kwargs
    assert "would_kanban_payload" not in spawn_kwargs["dry_run"]


def test_run_autopilot_once_refuses_when_controller_off_without_queue_scan(tmp_path):
    store = _state_store(tmp_path)
    work_state = _FakeWorkStateStore()
    linear = _FakeLinearClient({}, queue=[_issue("CH-401")])
    spawner = MagicMock()

    result = run_autopilot_once(
        state_store=store,
        work_state_store=work_state,
        linear_client=linear,
        executor_spawner=spawner,
        actor="tester",
        owner_session_id="owner-1",
    )

    assert result["status"] == "paused"
    assert result["reason"] == "controller_disabled_noop"
    assert result["work_state_written"] is False
    assert result["executor_spawned"] is False
    assert result["linear_done_mutated"] is False
    assert linear.queue_calls == []
    spawner.assert_not_called()


def test_run_autopilot_once_rolls_back_lock_when_executor_spawn_fails(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    work_state = _FakeWorkStateStore()
    linear = _FakeLinearClient({}, queue=[_issue("CH-401")])
    spawner = MagicMock(side_effect=RuntimeError("executor unavailable"))

    result = run_autopilot_once(
        state_store=store,
        work_state_store=work_state,
        linear_client=linear,
        executor_spawner=spawner,
        actor="tester",
        owner_session_id="owner-1",
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "executor_spawn_failed"
    assert result["work_state_written"] is True
    assert result["executor_spawned"] is False
    assert result["linear_done_mutated"] is False
    record = work_state.records[0]
    assert record.state == "blocked"
    assert record.close_disposition == "close"
    assert record.usable_outcome == "blocked"
    assert "executor_spawn_failed" in record.blocked_reason


def test_run_autopilot_once_off_after_run_blocks_next_start(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    work_state = _FakeWorkStateStore()
    linear = _FakeLinearClient({}, queue=[_issue("CH-401"), _issue("CH-402")])
    spawner = MagicMock(return_value={"session_id": "exec-1", "proof": "spawned"})

    first = run_autopilot_once(
        state_store=store,
        work_state_store=work_state,
        linear_client=linear,
        executor_spawner=spawner,
        actor="tester",
        owner_session_id="owner-1",
    )
    handle_autopilot_command("OFF", state_store=store, work_state_store=work_state, actor="tester")
    second = run_autopilot_once(
        state_store=store,
        work_state_store=work_state,
        linear_client=linear,
        executor_spawner=spawner,
        actor="tester",
        owner_session_id="owner-2",
    )

    assert first["status"] == "started"
    assert second["status"] == "paused"
    assert second["work_state_written"] is False
    assert spawner.call_count == 1


def test_runtime_command_targeted_one_shot_materializes_lock_goal_and_executor(tmp_path):
    store = _state_store(tmp_path)
    work_state = _FakeWorkStateStore()
    linear = _FakeLinearClient({"CH-401": _issue("CH-401")}, queue=[_issue("CH-401")])
    spawner = MagicMock(return_value={"session_id": "exec-401", "proof": "queued"})

    result = handle_autopilot_runtime_command(
        "CH-401",
        state_store=store,
        work_state_store=work_state,
        linear_client=linear,
        executor_spawner=spawner,
        actor="tester",
        owner_session_id="owner-401",
    )

    assert result.ok is True
    assert result.decision["side_effects"] == {
        "state_written": False,
        "work_state_written": True,
        "executor_spawned": True,
        "linear_done_mutated": False,
    }
    assert result.decision["materialization"]["work_id"] == "CH-401"
    assert work_state.records[0].state == "running"
    assert work_state.records[0].executor_session_id == "exec-401"
    spawner.assert_called_once()
    assert spawner.call_args.kwargs["goal_contract"].startswith("/goal")
    assert "Mode: execution" in result.message


def test_runtime_command_on_records_intent_then_runs_one_queue_candidate(tmp_path):
    store = _state_store(tmp_path)
    work_state = _FakeWorkStateStore()
    linear = _FakeLinearClient({}, queue=[_issue("CH-401"), _issue("CH-402")])
    spawner = MagicMock(return_value={"session_id": "exec-queue", "proof": "queued"})

    result = handle_autopilot_runtime_command(
        "ON",
        state_store=store,
        work_state_store=work_state,
        linear_client=linear,
        executor_spawner=spawner,
        actor="tester",
        owner_session_id="owner-on",
    )

    assert store.status()["enabled"] is True
    assert result.ok is True
    assert result.decision["side_effects"]["state_written"] is True
    assert result.decision["materialization"]["selected_issue"]["identifier"] == "CH-401"
    assert [record.work_id for record in work_state.records] == ["CH-401"]
    assert spawner.call_count == 1


def test_runtime_command_reports_target_candidate_blocker(tmp_path):
    store = _state_store(tmp_path)
    work_state = _FakeWorkStateStore()
    linear = _FakeLinearClient(
        {"CH-401": _issue("CH-401", description="## Verification\n- focused tests pass")},
        queue=[_issue("CH-401", description="## Verification\n- focused tests pass")],
    )
    spawner = MagicMock()

    result = handle_autopilot_runtime_command(
        "CH-401",
        state_store=store,
        work_state_store=work_state,
        linear_client=linear,
        executor_spawner=spawner,
        actor="tester",
        owner_session_id="owner-401",
    )

    assert result.ok is False
    assert "Decision: blocked (no_eligible_execution_ready_issue)." in result.message
    assert "Admission candidate: CH-401 eligible=false reason=missing_done_when" in result.message
    assert "Blocked/no-op reason: no_eligible_execution_ready_issue" in result.message
    assert work_state.records == []
    spawner.assert_not_called()



def test_runtime_command_keeps_status_and_dry_run_read_only(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    work_state = _FakeWorkStateStore()
    linear = _FakeLinearClient({}, queue=[_issue("CH-401")])
    spawner = MagicMock()

    result = handle_autopilot_runtime_command(
        "dry-run",
        state_store=store,
        work_state_store=work_state,
        linear_client=linear,
        executor_spawner=spawner,
        actor="tester",
    )

    assert result.ok is True
    assert result.command.action == ACTION_DRY_RUN
    assert result.decision["side_effects"]["executor_spawned"] is False
    assert work_state.records == []
    spawner.assert_not_called()


def test_work_state_target_classification_includes_read_only_kanban_run_projection():
    record = _autopilot_work_record(
        "CH-411",
        owner="hermes",
        executor="omx",
        mode="delegated",
        owner_session_id="owner-session",
        state="blocked",
        executor_session_id="exec-session",
        tmux_session="omx-CH-411",
        repo_path="/repo/hermes-agent",
        proof="work_state:blocked",
        next_action="Operator review required",
    )

    class _Store:
        def resolve_delegated_signal_candidate(self, *, work_id: str, live_only: bool):
            return {
                "status": "single_match",
                "reason": "eligible",
                "record": record,
                "matches": [record.to_dict()],
            }

    result = classify_work_state_target("CH-411", _Store())

    assert result["resolution_status"] == "single_match"
    projection = result["kanban_run_projection"]
    assert projection["status"] == "mapped"
    assert projection["task_status"] == "blocked"
    assert projection["task_run"]["outcome"] == "blocked"
    assert projection["task_run"]["metadata"]["source"] == "autopilot_work_state_target"


def test_executor_result_ingest_records_controller_event_not_completion_authority(tmp_path):
    now = datetime(2026, 5, 6, 9, 0, tzinfo=timezone.utc)
    record = _autopilot_work_record("CH-405", state="running", cleanup_required=True)
    work_state = _FakeWorkStateStore([record])

    result = ingest_autopilot_executor_result(
        work_state_store=work_state,
        work_id="CH-405",
        owner_session_id="autopilot:CH-405",
        executor_result={
            "status": "succeeded",
            "proof": "executor exited 0 and claims done",
            "repo_full_name": "chriskim12/hermes-agent",
            "commit": "1af1c992b4fe",
        },
        now=now,
    )

    assert result == {
        "status": "ingested",
        "reason": "executor_result_recorded_as_controller_event",
        "work_id": "CH-405",
        "controller_event": "executor_finished",
        "linear_done_mutated": False,
        "next_card_started": False,
    }
    assert work_state.records[0].state == "running"
    assert work_state.records[0].close_disposition is None
    assert work_state.records[0].review_closeout["executor_event"]["status"] == "executor_finished"
    assert work_state.records[0].review_closeout["executor_event"]["linear_done_mutated"] is False
    assert work_state.records[0].review_closeout["executor_event"]["next_card_started"] is False
    assert work_state.records[0].review_closeout["repo_full_name"] == "chriskim12/hermes-agent"
    assert work_state.records[0].review_closeout["commit"] == "1af1c992b4fe"


def test_controller_tick_continues_same_card_when_executor_requests_continuation(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    record = _autopilot_work_record("CH-405", state="running")
    work_state = _FakeWorkStateStore([record])
    linear = _FakeLinearClient({}, queue=[_issue("CH-406")])

    ingest_autopilot_executor_result(
        work_state_store=work_state,
        work_id="CH-405",
        executor_result={
            "status": "needs_continuation",
            "next_action": "Keep working CH-405 closeout gate implementation",
            "proof": "tests still red",
        },
    )
    decision = autopilot_controller_tick(
        state=store.status(),
        work_state_store=work_state,
        linear_client=linear,
        expected_repo_full_name="chriskim12/hermes-agent",
    )

    assert decision["decision"] == "same_card_continuation"
    assert decision["work_id"] == "CH-405"
    assert decision["linear_done_mutated"] is False
    assert decision["next_card_started"] is False
    assert linear.queue_calls == []


def test_controller_tick_finished_executor_requires_closeout_verification_not_next_card(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    record = _autopilot_work_record("CH-405", state="running")
    work_state = _FakeWorkStateStore([record])
    linear = _FakeLinearClient({}, queue=[_issue("CH-406")])

    ingest_autopilot_executor_result(
        work_state_store=work_state,
        work_id="CH-405",
        executor_result={"status": "succeeded", "proof": "executor done, no PR evidence"},
    )
    decision = autopilot_controller_tick(
        state=store.status(),
        work_state_store=work_state,
        linear_client=linear,
        expected_repo_full_name="chriskim12/hermes-agent",
    )

    assert decision["decision"] == "closeout_verification"
    assert decision["reason"] == "autopilot_closeout_review_gate_blocked"
    assert decision["work_id"] == "CH-405"
    assert "pr_url" in decision["closeout_gate"]["missing"]
    assert decision["linear_done_mutated"] is False
    assert decision["next_card_started"] is False
    assert linear.queue_calls == []


def test_controller_tick_selects_next_only_after_verified_pr_closeout(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    prior = _autopilot_work_record(
        "CH-405",
        state="running",
        cleanup_proof="task_owned_worktree_removed",
        review_closeout={"status": "pending", "work_id": "CH-405"},
    )
    work_state = _FakeWorkStateStore([prior])
    linear = _FakeLinearClient({}, queue=[_issue("CH-406")])

    ingest_autopilot_executor_result(
        work_state_store=work_state,
        work_id="CH-405",
        executor_result={
            "status": "succeeded",
            "proof": "executor done with verified PR",
            "review_closeout": _valid_pr_closeout_evidence(
                work_id="CH-405",
                remote_branch="yuuka/CH-405-autopilot-controller-tick",
                task_branch="yuuka/CH-405-autopilot-controller-tick",
                pr_head="yuuka/CH-405-autopilot-controller-tick",
                task_worktree_path="/repo/hermes-agent/.worktrees/CH-405-autopilot-controller-tick",
            ),
        },
    )
    decision = autopilot_controller_tick(
        state=store.status(),
        work_state_store=work_state,
        linear_client=linear,
        expected_repo_full_name="chriskim12/hermes-agent",
        repo_policy={
            "kanban_payload_dry_run": True,
            "github_repo": "chriskim12/hermes-agent",
            "integration_branch": "main",
            "executor": "kanban-dispatcher",
            "tenant": "hermes",
        },
    )

    assert decision["decision"] == "next_card_selection"
    assert decision["selected_issue"]["identifier"] == "CH-406"
    assert decision["would_create_work_state"]["work_id"] == "CH-406"
    assert decision["would_kanban_payload"]["task"]["idempotency_key"] == "linear:CH-406"
    assert decision["linear_done_mutated"] is False
    assert decision["next_card_started"] is False
    assert linear.queue_calls == [{"team_key": "CH", "state_name": "Execution Ready", "limit": 100}]



def test_controller_tick_never_selects_next_without_executor_event_even_with_pr_closeout(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    prior = _autopilot_work_record(
        "CH-405",
        state="completed",
        cleanup_proof="task_owned_worktree_removed",
        review_closeout=_valid_pr_closeout_evidence(
            work_id="CH-405",
            remote_branch="yuuka/CH-405-autopilot-controller-tick",
            task_branch="yuuka/CH-405-autopilot-controller-tick",
            pr_head="yuuka/CH-405-autopilot-controller-tick",
            task_worktree_path="/repo/hermes-agent/.worktrees/CH-405-autopilot-controller-tick",
        ),
    )
    work_state = _FakeWorkStateStore([prior])
    linear = _FakeLinearClient({}, queue=[_issue("CH-406")])

    decision = autopilot_controller_tick(
        state=store.status(),
        work_state_store=work_state,
        linear_client=linear,
        expected_repo_full_name="chriskim12/hermes-agent",
    )

    assert decision["decision"] == "blocked"
    assert decision["reason"] == "executor_result_missing_fail_closed"
    assert decision["work_id"] == "CH-405"
    assert decision["linear_done_mutated"] is False
    assert decision["next_card_started"] is False
    assert linear.queue_calls == []



def test_controller_tick_keeps_active_same_card_awaiting_executor_event(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    prior = _autopilot_work_record("CH-405", state="running")
    work_state = _FakeWorkStateStore([prior])
    linear = _FakeLinearClient({}, queue=[_issue("CH-406")])

    decision = autopilot_controller_tick(
        state=store.status(),
        work_state_store=work_state,
        linear_client=linear,
        expected_repo_full_name="chriskim12/hermes-agent",
    )

    assert decision["decision"] == "same_card_continuation"
    assert decision["reason"] == "awaiting_executor_result_same_card"
    assert decision["work_id"] == "CH-405"
    assert decision["linear_done_mutated"] is False
    assert decision["next_card_started"] is False
    assert linear.queue_calls == []



def test_executor_result_ingest_fails_closed_when_work_state_unavailable():
    class BrokenWorkState:
        def list_records(self):
            raise RuntimeError("store offline")

    result = ingest_autopilot_executor_result(
        work_state_store=BrokenWorkState(),
        work_id="CH-405",
        executor_result={"status": "succeeded", "proof": "executor done"},
    )

    assert result == {
        "status": "blocked",
        "reason": "work_state_unavailable:RuntimeError",
        "work_id": "CH-405",
        "linear_done_mutated": False,
        "next_card_started": False,
    }



def test_executor_result_ingest_rejects_non_mapping_payload_fail_closed():
    record = _autopilot_work_record("CH-405", state="running")
    work_state = _FakeWorkStateStore([record])

    result = ingest_autopilot_executor_result(
        work_state_store=work_state,
        work_id="CH-405",
        executor_result=["not", "a", "mapping"],
    )

    assert result == {
        "status": "blocked",
        "reason": "executor_result_invalid_payload_fail_closed",
        "work_id": "CH-405",
        "linear_done_mutated": False,
        "next_card_started": False,
    }
    assert work_state.update_calls == []



def test_executor_result_ingest_accepts_canonical_executor_finished_event():
    record = _autopilot_work_record("CH-405", state="running")
    work_state = _FakeWorkStateStore([record])

    result = ingest_autopilot_executor_result(
        work_state_store=work_state,
        work_id="CH-405",
        executor_result={"controller_event": "executor_finished", "proof": "executor complete"},
    )

    assert result["status"] == "ingested"
    assert result["controller_event"] == "executor_finished"
    assert work_state.records[0].state == "running"
    assert work_state.records[0].review_closeout["executor_event"]["status"] == "executor_finished"



def test_executor_result_ingest_fails_closed_when_update_record_returns_false():
    class RejectingWorkState(_FakeWorkStateStore):
        def update_record(self, work_id: str, owner_session_id: str, **updates):
            self.update_calls.append(
                {"work_id": work_id, "owner_session_id": owner_session_id, "updates": updates}
            )
            return False

    record = _autopilot_work_record("CH-405", state="running")
    work_state = RejectingWorkState([record])

    result = ingest_autopilot_executor_result(
        work_state_store=work_state,
        work_id="CH-405",
        executor_result={"status": "succeeded", "proof": "executor done"},
    )

    assert result == {
        "status": "blocked",
        "reason": "work_state_update_failed_fail_closed",
        "work_id": "CH-405",
        "controller_event": "executor_finished",
        "linear_done_mutated": False,
        "next_card_started": False,
    }
    assert work_state.records[0].state == "running"



def test_controller_tick_blocks_empty_prior_state_instead_of_selecting_next(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    work_state = _FakeWorkStateStore([])
    linear = _FakeLinearClient({}, queue=[_issue("CH-406")])

    decision = autopilot_controller_tick(
        state=store.status(),
        work_state_store=work_state,
        linear_client=linear,
        expected_repo_full_name="chriskim12/hermes-agent",
    )

    assert decision["decision"] == "blocked"
    assert decision["reason"] == "executor_result_missing_fail_closed"
    assert decision["linear_done_mutated"] is False
    assert decision["next_card_started"] is False
    assert linear.queue_calls == []



def test_controller_tick_blocks_unknown_executor_event_instead_of_selecting_next(tmp_path):
    store = _state_store(tmp_path)
    store.set_enabled(True, actor="tester")
    prior = _autopilot_work_record(
        "CH-405",
        state="completed",
        cleanup_proof="task_owned_worktree_removed",
        review_closeout={
            **_valid_pr_closeout_evidence(
                work_id="CH-405",
                remote_branch="yuuka/CH-405-autopilot-controller-tick",
                task_branch="yuuka/CH-405-autopilot-controller-tick",
                pr_head="yuuka/CH-405-autopilot-controller-tick",
                task_worktree_path="/repo/hermes-agent/.worktrees/CH-405-autopilot-controller-tick",
            ),
            "executor_event": {"status": "nonsense"},
        },
    )
    work_state = _FakeWorkStateStore([prior])
    linear = _FakeLinearClient({}, queue=[_issue("CH-406")])

    decision = autopilot_controller_tick(
        state=store.status(),
        work_state_store=work_state,
        linear_client=linear,
        expected_repo_full_name="chriskim12/hermes-agent",
    )

    assert decision["decision"] == "blocked"
    assert decision["reason"] == "executor_event_unknown_fail_closed"
    assert decision["executor_event"] == "nonsense"
    assert decision["linear_done_mutated"] is False
    assert decision["next_card_started"] is False
    assert linear.queue_calls == []



def test_executor_result_ingest_requires_persistent_update_record():
    class ReadOnlyWorkState:
        def __init__(self, records):
            self.records = records

        def list_records(self):
            return self.records

    record = _autopilot_work_record("CH-405", state="running")
    work_state = ReadOnlyWorkState([record])

    result = ingest_autopilot_executor_result(
        work_state_store=work_state,
        work_id="CH-405",
        executor_result={"status": "succeeded", "proof": "executor done"},
    )

    assert result == {
        "status": "blocked",
        "reason": "work_state_update_missing_fail_closed",
        "work_id": "CH-405",
        "controller_event": "executor_finished",
        "linear_done_mutated": False,
        "next_card_started": False,
    }
    assert "executor_event" not in record.review_closeout



def test_command_registry_exposes_autopilot_to_gateway_surfaces():
    from hermes_cli.commands import (
        is_gateway_known_command,
        resolve_command,
        should_bypass_active_session,
    )

    command = resolve_command("autopilot")

    assert command is not None
    assert command.name == "autopilot"
    assert command.args_hint == "[status|dry-run|ON|OFF|CH-123]"
    assert is_gateway_known_command("autopilot") is True
    assert should_bypass_active_session("autopilot") is True


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    source = _make_source()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock(send=AsyncMock())
    adapter._pending_messages = {}
    adapter.handle_message = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key=build_session_key(source),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_run_generation = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._update_prompt_pending = {}
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = None
    runner._session_db.get_session.return_value = None
    runner._voice_mode = {}
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    runner._background_tasks = set()
    return runner


@pytest.mark.asyncio
async def test_gateway_autopilot_bypasses_running_agent_without_interrupt(monkeypatch):
    calls = []

    def _fake_handle(raw_args, *, actor=None, **kwargs):
        calls.append({"raw_args": raw_args, "actor": actor})
        return SimpleNamespace(message="autopilot handled")

    monkeypatch.setattr("gateway.autopilot.handle_autopilot_runtime_command", _fake_handle)
    runner = _make_runner()
    session_key = build_session_key(_make_source())
    running_agent = SimpleNamespace(interrupt=MagicMock())
    runner._running_agents[session_key] = running_agent
    runner._running_agents_ts[session_key] = 0

    result = await runner._handle_message(_make_event("/autopilot status CH-385"))

    assert result == "autopilot handled"
    assert calls == [{"raw_args": "status CH-385", "actor": "tester"}]
    running_agent.interrupt.assert_not_called()
    assert runner._pending_messages == {}


@pytest.mark.asyncio
async def test_gateway_autopilot_target_dispatches_generated_goal_contract_when_session_idle(monkeypatch):
    fake_work_state = _FakeWorkStateStore()

    class _StoreFactory:
        def __call__(self):
            return fake_work_state

    class _LinearFactory:
        def __call__(self):
            return _FakeLinearClient({"CH-401": _issue("CH-401")}, queue=[_issue("CH-401")])

    monkeypatch.setattr("gateway.work_state.WorkStateStore", _StoreFactory())
    monkeypatch.setattr("gateway.autopilot.EnvLinearIssueClient", _LinearFactory())
    runner = _make_runner()
    goal_sets = []

    class _FakeGoalManager:
        def set(self, goal):
            goal_sets.append(goal)
            return SimpleNamespace(goal=goal)

    runner._get_goal_manager_for_event = lambda _event: (_FakeGoalManager(), runner.session_store.get_or_create_session.return_value)
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter._pending_messages = {}
    event = _make_event("/autopilot CH-401")

    result = await runner._handle_autopilot_command(event)

    assert "Decision: started" in result
    assert "work_state lock: CH-401" in result
    assert fake_work_state.records[0].work_id == "CH-401"
    assert fake_work_state.records[0].state == "running"
    assert fake_work_state.records[0].executor_session_id == "sess-1"
    await asyncio.sleep(0)
    adapter.handle_message.assert_awaited_once()
    dispatched = adapter.handle_message.await_args.args[0]
    assert not dispatched.text.startswith("/goal")
    assert goal_sets == [dispatched.text]
    assert "CH-401" in dispatched.text
    assert dispatched.source == event.source
    assert adapter._pending_messages == {}


def test_cli_autopilot_dispatch_uses_same_controller(monkeypatch):
    import cli as cli_mod
    from cli import HermesCLI

    captured = []
    calls = []

    def _fake_handle(raw_args, *, actor=None):
        calls.append({"raw_args": raw_args, "actor": actor})
        return SimpleNamespace(message="cli autopilot handled")

    monkeypatch.setattr("gateway.autopilot.handle_autopilot_command", _fake_handle)
    monkeypatch.setattr(cli_mod, "_cprint", lambda message, *a, **k: captured.append(str(message)))

    cli = object.__new__(HermesCLI)

    assert cli.process_command("/autopilot dry-run CH-385") is True
    assert calls == [{"raw_args": "dry-run CH-385", "actor": "local-cli"}]
    assert captured == ["cli autopilot handled"]


def _valid_pr_closeout_evidence(**overrides):
    evidence = {
        "commit": "1af1c992b4fe",
        "repo_full_name": "chriskim12/hermes-agent",
        "remote_branch": "feature/CH-309-env-authority-coverage",
        "task_branch": "feature/CH-309-env-authority-coverage",
        "task_worktree_path": "/tmp/hermes-agent/.worktrees/ch-309-env-authority-coverage",
        "branch_pushed": True,
        "pr_created": True,
        "repo_verified": True,
        "pr_verified": True,
        "sha_verified": True,
        "pr_url": "https://github.com/chriskim12/hermes-agent/pull/173",
        "pr_base": "main",
        "pr_head": "feature/CH-309-env-authority-coverage",
        "checks_passed": True,
        "verification_result": "focused tests passed",
        "cleanup_done": True,
        "cleanup_proof": "task_owned_worktree_removed",
        "linear_evidence_comment_id": "comment-123",
    }
    evidence.update(overrides)
    return evidence


def test_autopilot_closeout_gate_blocks_linear_done_without_pr_and_cleanup_evidence():
    gate = evaluate_autopilot_pr_closeout_gate(
        evidence={"commit": "1af1c992b4fe"},
        repo_name="hermes-agent",
        remote_default_branch="main",
    )

    assert gate["allowed"] is False
    assert gate["status"] == "blocked"
    assert gate["linear_done_mutated"] is False
    assert gate["expected_integration_branch"] == "main"
    assert "pr_url" in gate["missing"]
    assert "branch_pushed" in gate["missing"]
    assert "cleanup_done" in gate["missing"]
    assert "cleanup_proof" in gate["missing"]


def test_kanban_closeout_gate_keeps_worker_done_separate_from_review_ready():
    gate = evaluate_kanban_closeout_gate(
        evidence={
            "kanban_task": {"status": "done"},
            "task_run": {"status": "done", "outcome": "completed"},
            "commit": "1af1c992b4fe",
        },
        repo_name="hermes-agent",
        remote_default_branch="main",
        expected_repo_full_name="chriskim12/hermes-agent",
    )

    assert gate["allowed"] is False
    assert gate["status"] == "blocked"
    assert gate["phase"] == "worker_done"
    assert gate["worker_done"] is True
    assert gate["review_ready"] is False
    assert gate["closed"] is False
    assert gate["linear_done_mutated"] is False
    assert gate["kanban_done_project_done"] is False
    assert gate["reason"] == "kanban_worker_done_review_evidence_required"
    assert "pr_url" in gate["missing"]
    assert "cleanup_proof" in gate["missing"]


def test_kanban_closeout_gate_marks_review_ready_only_after_pr_evidence():
    gate = evaluate_kanban_closeout_gate(
        evidence={
            "task_run": {"status": "done", "outcome": "completed"},
            "review_closeout": _valid_pr_closeout_evidence(),
        },
        repo_name="hermes-agent",
        remote_default_branch="main",
        expected_repo_full_name="chriskim12/hermes-agent",
    )

    assert gate["allowed"] is False
    assert gate["status"] == "review_ready"
    assert gate["phase"] == "review_ready"
    assert gate["worker_done"] is True
    assert gate["review_ready"] is True
    assert gate["closed"] is False
    assert gate["linear_done_mutated"] is False
    assert gate["kanban_done_project_done"] is False
    assert gate["reason"] == "kanban_review_ready_final_close_required"
    assert gate["closeout_gate"]["allowed"] is True


def test_kanban_closeout_gate_requires_explicit_final_close_after_review_ready():
    gate = evaluate_kanban_closeout_gate(
        evidence={
            "task_run": {"status": "done", "outcome": "completed"},
            "review_closeout": _valid_pr_closeout_evidence(),
        },
        repo_name="hermes-agent",
        remote_default_branch="main",
        expected_repo_full_name="chriskim12/hermes-agent",
        final_close_approved=True,
    )

    assert gate["allowed"] is True
    assert gate["status"] == "closed"
    assert gate["phase"] == "closed"
    assert gate["worker_done"] is True
    assert gate["review_ready"] is True
    assert gate["closed"] is True
    assert gate["linear_done_mutated"] is False


def test_autopilot_closeout_gate_allows_only_recorded_pr_less_exception():
    blocked = evaluate_autopilot_pr_closeout_gate(
        evidence={"direct_landing_exception": "explicit_direct_landing_approval"},
        repo_name="hermes-agent",
        remote_default_branch="main",
    )

    assert blocked["allowed"] is False
    assert blocked["requires_pr"] is False
    assert "direct_landing_approval_id" in blocked["missing"]
    assert "exception_proof" in blocked["missing"]

    allowed = evaluate_autopilot_pr_closeout_gate(
        evidence={
            "direct_landing_exception": "explicit_direct_landing_approval",
            "commit": "1af1c992b4fe",
            "direct_landing_approval_id": "approval-123",
            "exception_proof": "human approved direct landing in Linear comment",
            "verification_result": "read-only policy exception verified",
            "cleanup_done": True,
            "cleanup_proof": "task_owned_branch_removed",
            "linear_evidence_comment_id": "comment-123",
        },
        repo_name="hermes-agent",
        remote_default_branch="main",
    )

    assert allowed["allowed"] is True
    assert allowed["reason"] == "autopilot_pr_less_closeout_exception_satisfied"
    assert allowed["requires_pr"] is False


def test_autopilot_closeout_gate_allows_verified_pr_and_cleanup_to_integration_branch():
    gate = evaluate_autopilot_pr_closeout_gate(
        evidence=_valid_pr_closeout_evidence(),
        repo_name="hermes-agent",
        remote_default_branch="main",
        expected_repo_full_name="chriskim12/hermes-agent",
    )

    assert gate == {
        "allowed": True,
        "status": "allowed",
        "reason": "autopilot_pr_cleanup_evidence_satisfied",
        "mode": "autopilot",
        "requires_pr": True,
        "expected_integration_branch": "main",
        "linear_done_mutated": False,
        "missing": [],
        "violations": [],
    }


def test_autopilot_closeout_gate_allows_repo_policy_expected_repo():
    gate = evaluate_autopilot_pr_closeout_gate(
        evidence=_valid_pr_closeout_evidence(),
        repo_name="hermes-agent",
        remote_default_branch="main",
        repo_policy={"github_repo": "chriskim12/hermes-agent"},
    )

    assert gate["allowed"] is True
    assert gate["missing"] == []
    assert gate["violations"] == []


def test_autopilot_closeout_gate_uses_dailychingu_develop_not_main():
    assert resolve_autopilot_integration_branch(repo_name="DailyChingu", remote_default_branch="main") == "develop"

    gate = evaluate_autopilot_pr_closeout_gate(
        evidence=_valid_pr_closeout_evidence(pr_base="main"),
        repo_name="DailyChingu",
        remote_default_branch="main",
        expected_repo_full_name="chriskim12/hermes-agent",
    )

    assert gate["allowed"] is False
    assert gate["expected_integration_branch"] == "develop"
    assert "wrong_pr_base:expected=develop:actual=main" in gate["violations"]

    allowed = evaluate_autopilot_pr_closeout_gate(
        evidence=_valid_pr_closeout_evidence(pr_base="develop"),
        repo_name="DailyChingu",
        remote_default_branch="main",
        expected_repo_full_name="chriskim12/hermes-agent",
    )
    assert allowed["allowed"] is True


def test_autopilot_closeout_gate_rejects_pr_head_branch_mismatch_and_bad_url():
    gate = evaluate_autopilot_pr_closeout_gate(
        evidence=_valid_pr_closeout_evidence(
            pr_url="https://example.com/not-a-github-pr",
            pr_head="different-branch",
        ),
        repo_name="hermes-agent",
        remote_default_branch="main",
    )

    assert gate["allowed"] is False
    assert "pr_url_not_github_pull_url" in gate["violations"]
    assert "pr_head_remote_branch_mismatch" in gate["violations"]


def test_autopilot_closeout_gate_fails_closed_without_trusted_expected_repo():
    gate = evaluate_autopilot_pr_closeout_gate(
        evidence=_valid_pr_closeout_evidence(
            repo_full_name="evil/hermes-agent",
            pr_url="https://github.com/evil/hermes-agent/pull/173",
        ),
        repo_name="hermes-agent",
        remote_default_branch="main",
    )

    assert gate["allowed"] is False
    assert "trusted_expected_repo_full_name" in gate["missing"]


def test_autopilot_closeout_gate_rejects_self_certified_trusted_repo_without_expected_repo():
    gate = evaluate_autopilot_pr_closeout_gate(
        evidence=_valid_pr_closeout_evidence(
            repo_full_name="evil/hermes-agent",
            pr_url="https://github.com/evil/hermes-agent/pull/173",
            trusted_repo_verified=True,
        ),
        repo_name="hermes-agent",
        remote_default_branch="main",
    )

    assert gate["allowed"] is False
    assert "trusted_expected_repo_full_name" in gate["missing"]
    assert gate["violations"] == []


def test_autopilot_closeout_gate_dailychingu_develop_cannot_be_overridden_by_policy():
    gate = evaluate_autopilot_pr_closeout_gate(
        evidence=_valid_pr_closeout_evidence(pr_base="main"),
        repo_name="DailyChingu",
        remote_default_branch="main",
        repo_policy={"integration_branch": "main"},
        expected_repo_full_name="chriskim12/hermes-agent",
    )

    assert gate["allowed"] is False
    assert gate["expected_integration_branch"] == "develop"
    assert "wrong_pr_base:expected=develop:actual=main" in gate["violations"]


def test_autopilot_closeout_gate_rejects_pr_url_for_wrong_repo():
    gate = evaluate_autopilot_pr_closeout_gate(
        evidence=_valid_pr_closeout_evidence(
            pr_url="https://github.com/someone-else/hermes-agent/pull/173"
        ),
        repo_name="hermes-agent",
        remote_default_branch="main",
        expected_repo_full_name="chriskim12/hermes-agent",
    )

    assert gate["allowed"] is False
    assert (
        "pr_url_repo_mismatch:expected=chriskim12/hermes-agent:actual=someone-else/hermes-agent"
        in gate["violations"]
    )
