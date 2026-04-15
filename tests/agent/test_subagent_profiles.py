from agent.subagent_profiles import (
    DEFAULT_SUBAGENT_PROFILE,
    apply_subagent_profile_overrides,
    get_review_mesh_specialist_profile,
    get_subagent_profile,
    list_subagent_profiles,
    resolve_subagent_profile,
)


def test_registry_exposes_expected_profiles():
    profiles = {profile.id: profile for profile in list_subagent_profiles()}

    assert "builder" in profiles
    assert "researcher" in profiles
    assert "reviewer" in profiles
    assert "browser_scout" in profiles
    assert "security_reviewer" in profiles
    assert profiles["reviewer"].gstack_skill_hints
    assert "review" in profiles["reviewer"].preferred_tags


def test_unknown_profile_lookup_falls_back_to_default():
    assert get_subagent_profile("definitely-not-real").id == DEFAULT_SUBAGENT_PROFILE.id


def test_profile_resolution_uses_role_hint_and_goal_keywords():
    assert resolve_subagent_profile(role_hint="security").id == "security_reviewer"
    assert resolve_subagent_profile(goal="Research the competing APIs").id == "researcher"
    assert resolve_subagent_profile(goal="Use the browser to inspect login flow", toolsets=["browser"]).id == "browser_scout"


def test_profile_overrides_are_applied_additively():
    overridden = apply_subagent_profile_overrides(
        get_subagent_profile("reviewer"),
        {
            "preferred_skill_names": ["custom-review-skill"],
            "gstack_skill_hints": ["gstack-custom-review"],
            "prompt_preamble": "Focus on bespoke review protocol.",
            "runtime_hints": {"gstack_affinity": "very_high"},
        },
    )

    assert overridden.id == "reviewer"
    assert overridden.preferred_skill_names == ("custom-review-skill",)
    assert overridden.gstack_skill_hints == ("gstack-custom-review",)
    assert overridden.prompt_preamble == "Focus on bespoke review protocol."
    assert overridden.runtime_hints["gstack_affinity"] == "very_high"


def test_gstack_profile_hints_match_imported_gstack_surface():
    assert get_subagent_profile("reviewer").gstack_skill_hints == (
        "gstack-review",
        "gstack-plan-eng-review",
        "gstack-qa",
    )
    assert get_subagent_profile("browser_scout").gstack_skill_hints[:2] == (
        "gstack-browse",
        "gstack-canary",
    )
    assert get_subagent_profile("operator").gstack_skill_hints == (
        "gstack-ship",
        "gstack-land-and-deploy",
        "gstack-document-release",
    )


def test_review_mesh_specialists_map_to_expected_profiles():
    assert get_review_mesh_specialist_profile("testing").id == "reviewer"
    assert get_review_mesh_specialist_profile("security").id == "security_reviewer"
    assert get_review_mesh_specialist_profile("red_team").id == "red_team"
