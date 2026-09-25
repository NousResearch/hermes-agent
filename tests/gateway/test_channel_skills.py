"""Contract tests for channel skill profile resolution."""

import pytest

from gateway.channel_skills import (
    ChannelSkillProfileCycleError,
    ChannelSkillProfileDepthError,
    ChannelSkillResolver,
)


def test_resolves_legacy_binding_skills_in_declared_order():
    resolver = ChannelSkillResolver({
        "channel_skill_bindings": [
            {"id": "100", "skills": ["base", "specific", "base"]},
        ],
    })

    result = resolver.resolve("100")

    assert result is not None
    assert result.skills == ("base", "specific")
    assert result.profile_chain == ()
    assert result.matched_channel_id == "100"
    assert result.inherited_from_parent is False


def test_exact_channel_binding_wins_over_parent_binding():
    resolver = ChannelSkillResolver({
        "channel_skill_bindings": [
            {"id": "parent", "skills": ["parent-skill"]},
            {"id": "thread", "skills": ["thread-skill"]},
        ],
    })

    result = resolver.resolve("thread", "parent")

    assert result is not None
    assert result.skills == ("thread-skill",)
    assert result.matched_channel_id == "thread"
    assert result.inherited_from_parent is False


def test_thread_inherits_parent_profile_when_no_exact_binding_exists():
    resolver = ChannelSkillResolver({
        "channel_skill_profiles": {
            "vocabulary": {
                "skills": ["vocabulary-skill"],
            },
        },
        "channel_skill_bindings": [
            {"id": "parent", "profile": "vocabulary"},
        ],
    })

    result = resolver.resolve("thread", "parent")

    assert result is not None
    assert result.skills == ("vocabulary-skill",)
    assert result.profile_chain == ("vocabulary",)
    assert result.matched_channel_id == "parent"
    assert result.inherited_from_parent is True


def test_profile_inheritance_expands_parent_before_child_and_supports_base_skill():
    resolver = ChannelSkillResolver({
        "channel_skill_profiles": {
            "base": {
                "skills": ["common"],
            },
            "vocabulary": {
                "extends": "base",
                "base_skill": "formatting",
                "skills": ["vocabulary"],
            },
        },
        "channel_skill_bindings": [
            {"id": "channel", "profile": "vocabulary"},
        ],
    })

    result = resolver.resolve("channel")

    assert result is not None
    assert result.profile_chain == ("base", "vocabulary")
    assert result.skills == ("common", "formatting", "vocabulary")


def test_profile_cycle_is_reported_without_resolving_skills():
    resolver = ChannelSkillResolver({
        "channel_skill_profiles": {
            "a": {"extends": "b", "skills": ["a-skill"]},
            "b": {"extends": "a", "skills": ["b-skill"]},
        },
        "channel_skill_bindings": [
            {"id": "channel", "profile": "a"},
        ],
    })

    with pytest.raises(ChannelSkillProfileCycleError):
        resolver.resolve_profile_chain("a")

    assert resolver.resolve("channel") is None


def test_profile_depth_is_limited():
    profiles = {
        f"p{i}": {"extends": f"p{i + 1}"}
        for i in range(3)
    }
    profiles["p3"] = {"skills": ["leaf"]}
    resolver = ChannelSkillResolver(
        {
            "channel_skill_profiles": profiles,
            "channel_skill_bindings": [{"id": "channel", "profile": "p0"}],
        },
        max_profile_depth=3,
    )

    with pytest.raises(ChannelSkillProfileDepthError):
        resolver.resolve_profile_chain("p0")


def test_unknown_channel_returns_none():
    resolver = ChannelSkillResolver({"channel_skill_bindings": []})

    assert resolver.resolve("unknown") is None
    assert resolver.resolve_skills("unknown") is None


def test_gateway_config_bridge_preserves_profiles_for_discord_extra():
    from gateway.config import Platform
    from gateway.config_loader import bridge_platform_shared_keys

    platforms_data = {}
    bridge_platform_shared_keys(
        {
            "discord": {
                "channel_skill_profiles": {
                    "vocabulary": {"skills": ["discord-vocabulary"]},
                },
                "channel_skill_bindings": [
                    {"id": "channel", "profile": "vocabulary"},
                ],
            },
        },
        None,
        {},
        platforms_data,
        [Platform.DISCORD],
    )

    extra = platforms_data["discord"]["extra"]
    assert extra["channel_skill_profiles"]["vocabulary"]["skills"] == ["discord-vocabulary"]
    assert extra["channel_skill_bindings"][0]["profile"] == "vocabulary"
