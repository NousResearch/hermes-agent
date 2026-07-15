from unittest.mock import patch

from agent.conversation_loop import _expand_inline_skills_for_turn
from agent.skill_commands import (
    expand_inline_skill_mentions,
    extract_skill_mentions,
    extract_user_instruction_from_skill_message,
    scan_skill_commands,
)


def _make_skill(skills_dir, name, body):
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"""---
name: {name}
description: {name} guidance
---

# {name}

{body}
""",
        encoding="utf-8",
    )


def test_extracts_mentions_without_shell_or_literal_forms():
    text = (
        "Use $code-review and $test_driven_development, not $PATH, "
        "\\$escaped, `$inline-code`, or ```\n$fenced-code\n```."
    )

    assert extract_skill_mentions(text) == [
        "code-review",
        "test_driven_development",
    ]


def test_expands_multiple_mentions_in_first_use_order(tmp_path):
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(tmp_path, "code-review", "Inspect the patch carefully.")
        _make_skill(tmp_path, "test-driven-development", "Start with a failing test.")
        scan_skill_commands()
        prompt = (
            "Review this $code-review, use $test_driven_development, "
            "then confirm with $code-review."
        )
        result = expand_inline_skill_mentions(prompt, task_id="turn-1")

    assert result is not None
    message, loaded, missing = result
    assert loaded == ["code-review", "test-driven-development"]
    assert missing == []
    assert "Inspect the patch carefully." in message
    assert "Start with a failing test." in message
    assert extract_user_instruction_from_skill_message(message) == prompt


def test_expands_a_single_inline_mention(tmp_path):
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(tmp_path, "code-review", "Inspect the patch carefully.")
        scan_skill_commands()
        prompt = "Please $code-review this change."
        result = expand_inline_skill_mentions(prompt)

    assert result is not None
    message, loaded, missing = result
    assert loaded == ["code-review"]
    assert missing == []
    assert "Inspect the patch carefully." in message
    assert extract_user_instruction_from_skill_message(message) == prompt


def test_expands_a_skill_whose_name_collides_with_a_slash_command(tmp_path):
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(tmp_path, "help", "Apply the help skill.")
        commands = scan_skill_commands()
        result = expand_inline_skill_mentions("Use $help for this task.")

    assert "/help" not in commands
    assert result is not None
    message, loaded, missing = result
    assert loaded == ["help"]
    assert missing == []
    assert "Apply the help skill." in message


def test_skips_ambiguous_normalized_skill_names(tmp_path):
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(tmp_path, "git-helper", "Hyphenated guidance.")
        _make_skill(tmp_path, "git_helper", "Underscored guidance.")
        scan_skill_commands()
        result = expand_inline_skill_mentions("Use $git-helper for this task.")

    assert result is None


def test_leaves_shell_escapes_and_markdown_code_literal(tmp_path):
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(tmp_path, "home", "Home skill content.")
        _make_skill(tmp_path, "code-review", "Review skill content.")
        scan_skill_commands()
        prompt = (
            "Keep $HOME and \\$code-review literal, then show `$code-review`.\n"
            "```sh\n$code-review\n```"
        )
        result = expand_inline_skill_mentions(prompt)

    assert result is None


def test_lowercase_skill_can_share_a_shell_variable_name(tmp_path):
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(tmp_path, "home", "Home skill content.")
        scan_skill_commands()
        result = expand_inline_skill_mentions("Use $home for this task.")

    assert result is not None
    message, loaded, missing = result
    assert loaded == ["home"]
    assert missing == []
    assert "Home skill content." in message


def test_skips_unknown_and_platform_disabled_skills(tmp_path, monkeypatch):
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        _make_skill(tmp_path, "code-review", "Review skill content.")
        scan_skill_commands()
        monkeypatch.setattr(
            "agent.skill_utils.get_disabled_skill_names",
            lambda platform=None: {"code-review"} if platform == "discord" else set(),
        )
        result = expand_inline_skill_mentions(
            "Use $unknown and $code-review.",
            platform="discord",
        )

    assert result is None


def test_limits_inline_skill_chains_to_five(tmp_path):
    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        for index in range(6):
            _make_skill(tmp_path, f"skill-{index}", f"Guidance {index}.")
        scan_skill_commands()
        result = expand_inline_skill_mentions(
            " ".join(f"$skill-{index}" for index in range(6))
        )

    assert result is not None
    message, loaded, missing = result
    assert loaded == [f"skill-{index}" for index in range(5)]
    assert missing == []
    assert "Guidance 4." in message
    assert "Guidance 5." not in message


def test_turn_expansion_preserves_the_original_message():
    class Agent:
        platform = "telegram"

    with patch(
        "agent.skill_commands.expand_inline_skill_mentions",
        return_value=("expanded", ["code-review"], []),
    ) as expand:
        model_message, persisted = _expand_inline_skills_for_turn(
            Agent(),
            "Please $code-review this.",
            "turn-2",
            None,
        )

    assert model_message == "expanded"
    assert persisted == "Please $code-review this."
    expand.assert_called_once_with(
        "Please $code-review this.",
        task_id="turn-2",
        platform="telegram",
        mention_text="Please $code-review this.",
    )


def test_turn_expansion_keeps_multimodal_parts_and_existing_persistence():
    class Agent:
        platform = "discord"

    user_message = [
        {"type": "text", "text": "Inspect this with $code-review."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
    ]
    with patch(
        "agent.skill_commands.expand_inline_skill_mentions",
        return_value=("expanded", ["code-review"], []),
    ):
        model_message, persisted = _expand_inline_skills_for_turn(
            Agent(),
            user_message,
            "turn-3",
            "Inspect this image with $code-review.",
        )

    assert model_message[0] == {"type": "text", "text": "expanded"}
    assert model_message[1] == user_message[1]
    assert user_message[0]["text"] == "Inspect this with $code-review."
    assert persisted == "Inspect this image with $code-review."


def test_turn_expansion_finds_mentions_in_later_text_parts():
    class Agent:
        platform = "tui"

    user_message = [
        {"type": "text", "text": "Inspect both inputs."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        {"type": "text", "text": "Apply $code-review."},
    ]
    with patch(
        "agent.skill_commands.expand_inline_skill_mentions",
        return_value=("expanded", ["code-review"], []),
    ) as expand:
        model_message, persisted = _expand_inline_skills_for_turn(
            Agent(),
            user_message,
            "turn-4",
            None,
        )

    assert model_message[0] == {"type": "text", "text": "expanded"}
    assert model_message[1:] == user_message[1:]
    assert persisted == user_message
    expand.assert_called_once_with(
        "Inspect both inputs.",
        task_id="turn-4",
        platform="tui",
        mention_text="Inspect both inputs.\nApply $code-review.",
    )


def test_turn_expansion_ignores_mentions_from_synthetic_context():
    class Agent:
        platform = "slack"

    with patch(
        "agent.skill_commands.expand_inline_skill_mentions",
        return_value=None,
    ) as expand:
        model_message, persisted = _expand_inline_skills_for_turn(
            Agent(),
            "[Observed context: $code-review]\nCurrent message: hello",
            "turn-5",
            "hello",
        )

    assert model_message == "[Observed context: $code-review]\nCurrent message: hello"
    assert persisted == "hello"
    expand.assert_not_called()
