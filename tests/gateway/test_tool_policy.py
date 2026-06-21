from __future__ import annotations

from agent.prompt_builder import build_compact_skills_system_prompt
from gateway.tool_policy import (
    select_gateway_toolsets,
    should_use_lightweight_discord_context,
)


def test_discord_casual_chat_uses_lightweight_toolsets(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_LIGHTWEIGHT_DISCORD_TOOLS", "true")

    selected = select_gateway_toolsets(
        platform="discord",
        configured_toolsets={
            "terminal",
            "file",
            "browser",
            "skills",
            "memory",
            "session_search",
            "clarify",
            "chat",
            "channel-notes",
            "budget",
        },
        user_text="what is the 6th highest mountain?",
    )

    assert selected == [
        "budget",
        "channel-notes",
        "chat",
        "clarify",
        "memory",
        "session_search",
        "skills",
    ]


def test_discord_work_intent_keeps_configured_toolsets(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_LIGHTWEIGHT_DISCORD_TOOLS", "true")

    selected = select_gateway_toolsets(
        platform="discord",
        configured_toolsets={"terminal", "file", "skills", "memory"},
        user_text="debug the failing pytest run",
    )

    assert selected == ["file", "memory", "skills", "terminal"]


def test_auto_skill_keeps_configured_toolsets(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_LIGHTWEIGHT_DISCORD_TOOLS", "true")

    selected = select_gateway_toolsets(
        platform="discord",
        configured_toolsets={"terminal", "file", "skills"},
        user_text="hi",
        auto_skill="coding",
    )

    assert selected == ["file", "skills", "terminal"]


def test_non_discord_keeps_configured_toolsets(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_LIGHTWEIGHT_DISCORD_TOOLS", "true")

    selected = select_gateway_toolsets(
        platform="telegram",
        configured_toolsets={"terminal", "skills"},
        user_text="hi",
    )

    assert selected == ["skills", "terminal"]


def test_env_can_disable_lightweight_policy(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_LIGHTWEIGHT_DISCORD_TOOLS", "false")

    selected = select_gateway_toolsets(
        platform="discord",
        configured_toolsets={"terminal", "skills"},
        user_text="hi",
    )

    assert selected == ["skills", "terminal"]


def test_lightweight_context_predicate_matches_casual_discord(monkeypatch) -> None:
    monkeypatch.setenv("HERMES_LIGHTWEIGHT_DISCORD_TOOLS", "true")

    assert should_use_lightweight_discord_context(
        platform="discord",
        user_text="what is the 6th highest mountain?",
    )
    assert not should_use_lightweight_discord_context(
        platform="discord",
        user_text="debug the failing pytest run",
    )
    assert not should_use_lightweight_discord_context(
        platform="discord",
        user_text="hi",
        auto_skill="coding",
    )


def test_compact_skill_prompt_is_not_coding_tool_guidance() -> None:
    prompt = build_compact_skills_system_prompt()

    assert "skills_list" in prompt
    assert "skill_view" in prompt
    assert "ordinary casual chat" in prompt
    assert "use terminal" not in prompt.lower()
    assert "test-driven" not in prompt.lower()
