"""Tests for first-contact channel skill generation."""

from types import SimpleNamespace

from gateway.channel_skill_generation import (
    ChannelSkillAutoGenerator,
    ChannelSkillRegistry,
)


def _event(channel_id="123", parent_id=None, user_id="user-1", topic=""):
    return SimpleNamespace(
        auto_skill=None,
        text="初回問い合わせ",
        source=SimpleNamespace(
            platform="discord",
            chat_id=channel_id,
            parent_chat_id=parent_id,
            chat_type="channel",
            user_id=user_id,
            chat_topic=topic,
            chat_name="test-channel",
        ),
    )


def test_first_contact_creates_skill_registers_it_and_applies_it(tmp_path):
    registry = ChannelSkillRegistry(tmp_path / "channel-skills.yaml")
    created = []

    def create_skill(name, content):
        created.append((name, content))
        return True

    generator = ChannelSkillAutoGenerator(
        registry=registry,
        create_skill=create_skill,
        generate_skill=lambda **kwargs: "generated skill content",
    )
    event = _event(topic="チャンネルの話題")

    skills = generator.ensure(event, {
        "channel_skill_auto_generate": {
            "enabled": True,
            "allowed_channels": ["*"],
            "allowed_users": ["*"],
        },
    })

    assert skills == ["discord-channel-123"]
    assert event.auto_skill == ["discord-channel-123"]
    assert created[0][0] == "discord-channel-123"
    assert registry.get("123")["status"] == "ready"


def test_fallback_skill_explicitly_allows_common_processing_only(tmp_path):
    registry = ChannelSkillRegistry(tmp_path / "channel-skills.yaml")
    created = []
    generator = ChannelSkillAutoGenerator(
        registry=registry,
        create_skill=lambda name, content: created.append(content) or True,
        generate_skill=lambda **kwargs: "invalid generated content",
    )

    generator.ensure(_event(channel_id="common-only"), {
        "channel_skill_auto_generate": {
            "enabled": True,
            "allowed_channels": ["*"],
            "allowed_users": ["*"],
        },
    })

    assert "共通処理のみで対応可能" in created[0]
    assert "このスキルを編集して反映する" in created[0]


def test_long_generated_description_is_normalized_before_creation(tmp_path):
    registry = ChannelSkillRegistry(tmp_path / "channel-skills.yaml")
    created = []
    long_description = "This description is deliberately longer than the skill index budget and must be normalized."
    generated = f"""---
name: discord-channel-long-description
description: {long_description}
version: 0.1.0
---

# Rules

- Test rule.
"""
    generator = ChannelSkillAutoGenerator(
        registry=registry,
        create_skill=lambda name, content: created.append(content) or True,
        generate_skill=lambda **kwargs: generated,
    )

    result = generator.ensure(_event(channel_id="long-description"), {
        "channel_skill_auto_generate": {
            "enabled": True,
            "allowed_channels": ["*"],
            "allowed_users": ["*"],
        },
    })

    assert result == ["discord-channel-long-description",]
    assert len(created) == 1
    description = next(line for line in created[0].splitlines() if line.startswith("description:"))
    assert len(description.split(":", 1)[1].strip()) <= 60


def test_second_contact_reuses_registry_without_regenerating(tmp_path):
    registry = ChannelSkillRegistry(tmp_path / "channel-skills.yaml")
    calls = []
    generator = ChannelSkillAutoGenerator(
        registry=registry,
        create_skill=lambda name, content: True,
        generate_skill=lambda **kwargs: calls.append(kwargs) or "content",
    )
    config = {
        "channel_skill_auto_generate": {
            "enabled": True,
            "allowed_channels": ["*"],
            "allowed_users": ["*"],
        },
    }

    first = _event()
    second = _event()
    generator.ensure(first, config)
    generator.ensure(second, config)

    assert calls.__len__() == 1
    assert second.auto_skill == ["discord-channel-123"]


def test_generation_is_skipped_when_policy_does_not_allow_channel(tmp_path):
    registry = ChannelSkillRegistry(tmp_path / "channel-skills.yaml")
    calls = []
    generator = ChannelSkillAutoGenerator(
        registry=registry,
        create_skill=lambda name, content: True,
        generate_skill=lambda **kwargs: calls.append(kwargs) or "content",
    )
    event = _event(channel_id="not-allowed")

    result = generator.ensure(event, {
        "channel_skill_auto_generate": {
            "enabled": True,
            "allowed_channels": ["allowed-only"],
            "allowed_users": ["user-1"],
        },
    })

    assert result is None
    assert event.auto_skill is None
    assert calls == []


def test_existing_parent_skills_are_included_before_generated_skill(tmp_path):
    registry = ChannelSkillRegistry(tmp_path / "channel-skills.yaml")
    generator = ChannelSkillAutoGenerator(
        registry=registry,
        create_skill=lambda name, content: True,
        generate_skill=lambda **kwargs: "content",
    )
    event = _event(channel_id="thread", parent_id="parent")

    result = generator.ensure(event, {
        "channel_skill_profiles": {
            "vocabulary": {"skills": ["vocabulary-skill"]},
        },
        "channel_skill_bindings": [
            {"id": "parent", "profile": "vocabulary"},
        ],
        "channel_skill_auto_generate": {
            "enabled": True,
            "allowed_channels": ["*"],
            "allowed_users": ["*"],
        },
    })

    assert result == ["vocabulary-skill", "discord-channel-thread"]
    assert event.auto_skill == result
