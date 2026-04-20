from pathlib import Path

from model_router import (
    Mode,
    Model,
    Privacy,
    Priority,
    Quota,
    RouterInput,
    TaskType,
    load_config,
    route_model,
)


ROOT = Path(__file__).resolve().parent.parent


def get_config():
    return load_config(ROOT / "router_config.yaml")


def test_coding_execute_high_priority_selects_gpt_and_claude_reviewer():
    decision = route_model(
        RouterInput(
            task_type=TaskType.CODING,
            mode=Mode.EXECUTE,
            priority=Priority.HIGH,
            privacy=Privacy.NORMAL,
            quota=Quota.NORMAL,
            has_code=True,
            has_logs=True,
        ),
        get_config(),
    )

    assert decision.primary_model == Model.GPT
    assert decision.reviewer == Model.CLAUDE
    assert decision.fallback_models == [Model.CLAUDE, Model.DEEPSEEK, Model.OLLAMA]


def test_chat_medium_critical_uses_policy_override_to_claude():
    decision = route_model(
        RouterInput(
            task_type=TaskType.CHAT,
            mode=Mode.DRAFT,
            priority=Priority.MEDIUM,
            privacy=Privacy.NORMAL,
            quota=Quota.CRITICAL,
        ),
        get_config(),
    )

    assert decision.primary_model == Model.CLAUDE
    assert any("policy_override: auto-chat-medium-critical" in item for item in decision.trace)


def test_batch_local_only_forces_ollama():
    decision = route_model(
        RouterInput(
            task_type=TaskType.BATCH,
            mode=Mode.EXECUTE,
            priority=Priority.MEDIUM,
            privacy=Privacy.LOCAL_ONLY,
            quota=Quota.NORMAL,
        ),
        get_config(),
    )

    assert decision.primary_model == Model.OLLAMA


def test_review_mode_for_coding_prefers_claude():
    decision = route_model(
        RouterInput(
            task_type=TaskType.CODING,
            mode=Mode.REVIEW,
            priority=Priority.MEDIUM,
            privacy=Privacy.NORMAL,
            quota=Quota.NORMAL,
        ),
        get_config(),
    )

    assert decision.primary_model == Model.CLAUDE


def test_high_priority_disallows_cheap_primary():
    decision = route_model(
        RouterInput(
            task_type=TaskType.CHAT,
            mode=Mode.DRAFT,
            priority=Priority.HIGH,
            privacy=Privacy.NORMAL,
            quota=Quota.CRITICAL,
        ),
        get_config(),
    )

    assert decision.primary_model == Model.CLAUDE
