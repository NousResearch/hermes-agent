from propose_config_patch import build_patch_proposal, suggestion_to_override


def test_suggestion_to_override_builds_force_rule():
    suggestion = {
        "type": "segment_low_success",
        "severity": "high",
        "segment": {
            "task_type": "chat",
            "priority": "medium",
            "quota": "critical",
            "primary_model": "deepseek",
        },
        "message": "בעיה בסגמנט הזה",
    }

    override = suggestion_to_override(suggestion)

    assert override is not None
    assert override["when"] == {
        "task_type": "chat",
        "priority": "medium",
        "quota": "critical",
    }
    assert override["force"] == "claude-sonnet-4.6"


def test_build_patch_proposal_adds_generated_overrides():
    config = {
        "router": {"version": "0.3", "default_model": "claude-sonnet-4.6"},
        "policy_overrides": [],
    }

    suggestions = [
        {
            "type": "segment_low_success",
            "severity": "high",
            "segment": {
                "task_type": "chat",
                "priority": "medium",
                "quota": "critical",
                "primary_model": "deepseek",
            },
            "message": "בעיה בסגמנט הזה",
        }
    ]

    result = build_patch_proposal(config, suggestions)

    assert result["generated_count"] == 1
    assert len(result["generated_overrides"]) == 1
    assert result["proposed_config"]["policy_overrides"][0]["force"] == "claude-sonnet-4.6"
