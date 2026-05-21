from datetime import datetime, timezone
from pathlib import Path

from agent.task_card import (
    AcceptanceCriteria,
    CompiledIntent,
    ExecutionPlan,
    TaskCard,
    get_task_cards_dir,
    load_task_card,
    save_task_card,
)


def test_create_populates_core_fields():
    card = TaskCard.create(
        "fix delegation logging",
        session_id="sess-1",
        task_category="code_change",
    )

    assert card.task_id
    assert card.raw_user_request == "fix delegation logging"
    assert card.compiled_intent.real_task == "fix delegation logging"
    assert card.compiled_intent.task_category == "code_change"
    assert card.execution_plan.mode == "self_execute"
    assert card.acceptance_criteria.auto_checkable == []
    assert card.status == "pending"
    assert card.session_id == "sess-1"
    assert card.version == 0
    assert card.created_at
    assert card.updated_at
    datetime.fromisoformat(card.created_at)
    datetime.fromisoformat(card.updated_at)


def test_round_trip_preserves_nested_and_extension_fields():
    card = TaskCard(
        task_id="task-1",
        raw_user_request="build managed agents",
        compiled_intent=CompiledIntent(
            real_task="build managed agents",
            task_category="architecture_change",
            assumptions=["existing registry"],
            must_keep=["compatibility"],
            must_avoid=["breaking change"],
            success_criteria=["registry loads"],
        ),
        execution_plan=ExecutionPlan(
            mode="pipeline",
            agents=["codex", "claude"],
            delegation_reason="split design and implementation",
            require_gate="strategy_spine",
        ),
        acceptance_criteria=AcceptanceCriteria(
            auto_checkable=["tests pass"],
            human_judgment=["design is sane"],
            user_preference_check=["direct and concise"],
        ),
        status="running",
        result_summary="in progress",
        review_result={"quality_score": 88},
        session_id="sess-2",
        routing_basis=["task_category_default"],
        fallback_used="none",
        client="dongqiudi",
        project_topic="managed agents",
        local_project_path="/tmp/project",
        first_output="strategy_spine",
        must_read_local_files=True,
        needs_external_research=False,
    )

    clone = TaskCard.from_dict(card.to_dict())

    assert clone == card
    assert clone.compiled_intent.must_keep == ["compatibility"]
    assert clone.execution_plan.agents == ["codex", "claude"]
    assert clone.review_result == {"quality_score": 88}
    assert clone.first_output == "strategy_spine"


def test_save_increments_version_and_refreshes_timestamp(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.task_card.get_hermes_home", lambda: tmp_path)
    card = TaskCard.create("check persistence", session_id="sess-3")

    first_updated_at = card.updated_at
    first_path = save_task_card(card)
    saved_once = load_task_card(card.task_id)

    assert first_path == get_task_cards_dir() / f"{card.task_id}.json"
    assert saved_once is not None
    assert saved_once.version == 1
    assert saved_once.updated_at != ""
    assert saved_once.updated_at >= first_updated_at

    second_path = save_task_card(saved_once)
    saved_twice = load_task_card(card.task_id)

    assert second_path == first_path
    assert saved_twice is not None
    assert saved_twice.version == 2
    assert saved_twice.updated_at >= saved_once.updated_at


def test_load_missing_task_card_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.task_card.get_hermes_home", lambda: tmp_path)

    assert load_task_card("does-not-exist") is None


def test_from_dict_ignores_unknown_keys():
    card = TaskCard.from_dict(
        {
            "task_id": "task-x",
            "raw_user_request": "hello",
            "compiled_intent": {"real_task": "hello", "task_category": "other"},
            "execution_plan": {"mode": "self_execute"},
            "acceptance_criteria": {},
            "status": "pending",
            "unexpected": "ignored",
            "also_unknown": {"nested": True},
        }
    )

    assert card.task_id == "task-x"
    assert card.raw_user_request == "hello"
    assert card.compiled_intent.real_task == "hello"
    assert card.status == "pending"

