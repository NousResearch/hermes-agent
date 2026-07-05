from __future__ import annotations


def test_session_stop_with_clean_verification_requires_compact_finalization():
    from hermes_cli.usage_guard import classify_usage_guard_closeout_state

    verdict = classify_usage_guard_closeout_state(
        usage_guard_reason="session_stop: context_estimate >= 260000",
        repo_clean=True,
        tests_run=["pytest tests/unit -q"],
        build_run=True,
        smoke_run=True,
        remaining_work=["final report", "cleanup verification"],
        implementation_commands_needed=False,
        required_model="gpt-5.5",
        fixed_model_policy=True,
    )

    assert verdict["status"] == "compact_finalization_required"
    assert verdict["recommended_action"] == "compact_finalization_prompt"
    assert verdict["required_model"] == "gpt-5.5"
    assert "model switch requires explicit user approval" in verdict["model_policy_note"]
    assert "raw" not in verdict


def test_session_stop_does_not_mask_remaining_implementation_work():
    from hermes_cli.usage_guard import classify_usage_guard_closeout_state

    verdict = classify_usage_guard_closeout_state(
        usage_guard_reason="session_stop",
        repo_clean=True,
        tests_run=["pytest tests/unit -q"],
        remaining_work=["implement retry handler", "run focused tests"],
        implementation_commands_needed=True,
    )

    assert verdict["status"] == "continue_narrow_work"
    assert verdict["recommended_action"] == "narrow_next_step"
    assert "implementation_work_remains" in verdict["reasons"]
