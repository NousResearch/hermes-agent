from __future__ import annotations

from gateway.tool_policy import select_gateway_toolsets


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

    assert selected == ["budget", "channel-notes", "chat", "clarify", "memory", "session_search", "skills"]


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
