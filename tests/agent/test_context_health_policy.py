from agent.context_health_policy import (
    ContextHealthPolicy,
    PromptIntakeDecision,
    TaskBoundaryDecision,
    default_context_health_policy,
    normalize_context_health_policy,
    classify_prompt_for_intake,
    classify_task_boundary,
    resolve_effective_threshold,
)


def test_default_policy_is_disabled_and_runtime_neutral():
    policy = default_context_health_policy()

    assert policy.enabled is False
    assert policy.pre_model_intake.enabled is False
    assert policy.task_boundary.enabled is False
    assert policy.runtime_behavior_enabled is False

    decision = classify_prompt_for_intake("short normal prompt", policy)
    assert decision.action == "pass"
    assert decision.reason == "context_health_disabled"
    assert decision.pre_history_required is False


def test_explicit_enabled_policy_expresses_phase1_controls():
    policy = normalize_context_health_policy(
        {
            "enabled": True,
            "pre_model_intake": {
                "enabled": True,
                "long_prompt_char_threshold": 120,
                "long_prompt_line_threshold": 4,
                "pre_history_required": True,
                "high_risk_keywords": ["credential", "token"],
            },
            "task_boundary": {
                "enabled": True,
                "default_without_clear_continuation": "new_task",
                "ambiguous_action": "hold",
            },
            "thresholds": {
                "max_provider_context_ratio": 0.72,
                "block_provider_call_ratio": 0.85,
                "allow_model_specific_raise": False,
            },
        }
    )

    assert policy.enabled is True
    assert policy.pre_model_intake.enabled is True
    assert policy.pre_model_intake.long_prompt_char_threshold == 120
    assert policy.pre_model_intake.pre_history_required is True
    assert policy.task_boundary.default_without_clear_continuation == "new_task"
    assert policy.task_boundary.ambiguous_action == "hold"
    assert policy.thresholds.max_provider_context_ratio == 0.72
    assert policy.thresholds.allow_model_specific_raise is False


def test_threshold_resolution_preserves_configured_ratio_when_model_raise_disallowed():
    policy = normalize_context_health_policy(
        {
            "enabled": True,
            "thresholds": {
                "max_provider_context_ratio": 0.70,
                "block_provider_call_ratio": 0.85,
                "allow_model_specific_raise": False,
            },
        }
    )

    result = resolve_effective_threshold(
        configured_ratio=0.45,
        model_suggested_ratio=0.85,
        policy=policy,
    )

    assert result.ratio == 0.45
    assert result.clamped_model_suggestion is True
    assert result.reason == "model_raise_disallowed"


def test_threshold_resolution_allows_explicit_raise_only_up_to_policy_max():
    policy = normalize_context_health_policy(
        {
            "enabled": True,
            "thresholds": {
                "max_provider_context_ratio": 0.70,
                "block_provider_call_ratio": 0.85,
                "allow_model_specific_raise": True,
            },
        }
    )

    result = resolve_effective_threshold(
        configured_ratio=0.45,
        model_suggested_ratio=0.85,
        policy=policy,
    )

    assert result.ratio == 0.70
    assert result.clamped_model_suggestion is True
    assert result.reason == "policy_max_provider_context_ratio"


def test_task_boundary_defaults_to_new_task_without_clear_continuation():
    policy = normalize_context_health_policy(
        {
            "enabled": True,
            "task_boundary": {
                "enabled": True,
                "default_without_clear_continuation": "new_task",
                "ambiguous_action": "hold",
            },
        }
    )

    decision = classify_task_boundary(
        user_message="새 리서치 주제로 넘어가자",
        active_task_id="task-a",
        closed_task_ids=["task-a"],
        explicit_continuation_refs=[],
        policy=policy,
    )

    assert decision.action == "new_task"
    assert decision.reason == "no_clear_continuation_evidence"
    assert decision.defaulted is True


def test_task_boundary_holds_ambiguous_relation_when_policy_requires_hold():
    policy = normalize_context_health_policy(
        {
            "enabled": True,
            "task_boundary": {
                "enabled": True,
                "default_without_clear_continuation": "new_task",
                "ambiguous_action": "hold",
            },
        }
    )

    decision = classify_task_boundary(
        user_message="이거 이어서 하면 되는지 새로 봐야 하는지 애매해",
        active_task_id="task-a",
        closed_task_ids=["task-a"],
        explicit_continuation_refs=[],
        ambiguous_relation=True,
        policy=policy,
    )

    assert decision.action == "hold"
    assert decision.reason == "ambiguous_task_relation"
    assert decision.defaulted is False


def test_closed_task_does_not_continue_without_explicit_continuation():
    policy = normalize_context_health_policy(
        {
            "enabled": True,
            "task_boundary": {
                "enabled": True,
                "default_without_clear_continuation": "new_task",
                "ambiguous_action": "hold",
            },
        }
    )

    decision = classify_task_boundary(
        user_message="task-a와 별개로 새 검토를 시작하자",
        active_task_id=None,
        closed_task_ids=["task-a"],
        explicit_continuation_refs=[],
        policy=policy,
    )

    assert decision.action == "new_task"
    assert decision.reason == "no_clear_continuation_evidence"
    assert decision.linked_task_id is None


def test_task_boundary_can_express_explicit_continuation():
    policy = normalize_context_health_policy(
        {
            "enabled": True,
            "task_boundary": {"enabled": True},
        }
    )

    decision = classify_task_boundary(
        user_message="task-a 계속 진행해줘",
        active_task_id=None,
        closed_task_ids=["task-a"],
        explicit_continuation_refs=["task-a"],
        policy=policy,
    )

    assert decision.action == "continue_task"
    assert decision.reason == "explicit_continuation_reference"
    assert decision.linked_task_id == "task-a"


def test_long_prompt_pre_history_intake_policy_is_expressible():
    policy = normalize_context_health_policy(
        {
            "enabled": True,
            "pre_model_intake": {
                "enabled": True,
                "long_prompt_char_threshold": 80,
                "long_prompt_line_threshold": 3,
                "pre_history_required": True,
            },
        }
    )
    prompt = "\n".join(
        [
            "이번 작업은 긴 요구사항이다.",
            "1. WorkSpec으로 분리해야 한다.",
            "2. checklist를 만들어야 한다.",
            "3. task-state를 만들어야 한다.",
        ]
    )

    decision = classify_prompt_for_intake(prompt, policy)

    assert decision.action == "force_md_intake"
    assert decision.reason == "long_prompt"
    assert decision.pre_history_required is True
    assert "line_count" in decision.signals


def test_sensitive_long_prompt_holds_without_safe_raw_storage():
    policy = normalize_context_health_policy(
        {
            "enabled": True,
            "pre_model_intake": {
                "enabled": True,
                "long_prompt_char_threshold": 20,
                "pre_history_required": True,
                "high_risk_keywords": ["token", "password"],
                "sensitive_prompt_action": "hold",
            },
        }
    )

    decision = classify_prompt_for_intake(
        "token value appears in a long prompt that should not enter raw history",
        policy,
    )

    assert decision.action == "hold"
    assert decision.reason == "sensitive_prompt_requires_review"
    assert decision.pre_history_required is True
    assert "high_risk_keyword" in decision.signals


def test_public_decision_types_are_stable_dataclasses():
    assert PromptIntakeDecision(action="pass", reason="x").action == "pass"
    assert TaskBoundaryDecision(action="new_task", reason="x").action == "new_task"
    assert isinstance(default_context_health_policy(), ContextHealthPolicy)
