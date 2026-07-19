from agent.skill_commands import (
    config_default_skill_identifiers,
    merge_preloaded_skill_identifiers,
    normalize_skill_identifiers,
)


def test_normalize_skill_identifiers_accepts_strings_lists_and_newlines():
    assert normalize_skill_identifiers(["alpha,beta", "gamma\nbeta", None]) == [
        "alpha",
        "beta",
        "gamma",
    ]


def test_config_default_skill_identifiers_reads_skills_defaults():
    config = {"skills": {"defaults": ["profile-a", "profile-b,profile-c"]}}

    assert config_default_skill_identifiers(config) == [
        "profile-a",
        "profile-b",
        "profile-c",
    ]


def test_merge_preloaded_skill_identifiers_keeps_defaults_first_and_dedupes():
    assert merge_preloaded_skill_identifiers(
        ["profile-a", "profile-b"],
        ["explicit", "profile-a"],
    ) == ["profile-a", "profile-b", "explicit"]
