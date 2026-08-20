"""Tests for fail-closed per-turn presentation policy from plugin dispatch hooks."""

from gateway.run import _merge_turn_presentation_policy


def test_final_only_true_wins_across_hook_results():
    policy = {}
    policy = _merge_turn_presentation_policy(
        policy, {"final_only": False, "policy_reason": "looser"}
    )
    policy = _merge_turn_presentation_policy(
        policy, {"final_only": True, "policy_reason": "family-protected-recipient"}
    )

    assert policy == {
        "final_only": True,
        "policy_reason": "family-protected-recipient",
    }


def test_unknown_presentation_keys_are_ignored():
    assert _merge_turn_presentation_policy(
        {}, {"final_only": True, "send_tool_secrets": True}
    ) == {"final_only": True}
