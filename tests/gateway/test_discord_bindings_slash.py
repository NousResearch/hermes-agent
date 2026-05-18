"""Tests for the Discord /bindings interactive skill-binding inspector."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.discord import (
    BindingSkillInfoView,
    DiscordAdapter,
    _discord_binding_summary,
    _format_binding_overview,
)


class FakeTree:
    def __init__(self):
        self.commands = {}

    def command(self, *, name, description):
        def decorator(fn):
            self.commands[name] = fn
            return fn
        return decorator

    def add_command(self, cmd):
        self.commands[cmd.name] = cmd

    def get_commands(self):
        return [SimpleNamespace(name=n) for n in self.commands]


@pytest.fixture
def adapter():
    config = PlatformConfig(
        enabled=True,
        token="***",
        extra={
            "channel_skill_bindings": [
                {
                    "id": "123",
                    "skills": ["discord-project-intake", "github-pr-workflow"],
                    "created_at": "2026-05-14T13:45:00Z",
                }
            ],
            "channel_prompts": {
                "123": "Treat this channel as project intake.",
            },
            "free_response_channels": ["123"],
        },
    )
    adapter = DiscordAdapter(config)
    adapter._client = SimpleNamespace(
        tree=FakeTree(),
        user=SimpleNamespace(id=99999, name="HermesBot"),
    )
    adapter._check_slash_authorization = AsyncMock(return_value=True)
    return adapter


def test_discord_binding_summary_exact_channel_includes_skills_prompt_and_metadata():
    extra = {
        "channel_skill_bindings": [
            {"id": "123", "skills": ["alpha", "beta"], "created_at": "then"}
        ],
        "channel_prompts": {"123": "channel prompt"},
        "free_response_channels": ["123"],
    }

    summary = _discord_binding_summary(extra, "123")

    assert summary["matched_id"] == "123"
    assert summary["skills"] == ["alpha", "beta"]
    assert summary["created_at"] == "then"
    assert summary["channel_prompt"] == "channel prompt"
    assert summary["free_response"] is True


def test_format_binding_overview_renders_buttons_hint_and_bound_since():
    text = _format_binding_overview(
        channel_label="#projekt-start",
        channel_id="123",
        summary={
            "skills": ["discord-project-intake"],
            "created_at": "2026-05-14T13:45:00Z",
            "channel_prompt": "Treat this channel as project intake.",
            "free_response": True,
            "matched_id": "123",
        },
    )

    assert "#projekt-start" in text
    assert "discord-project-intake" in text
    assert "2026-05-14T13:45:00Z" in text
    assert "Free response: ja" in text
    assert "Buttons" in text


@pytest.mark.asyncio
async def test_bindings_slash_sends_overview_with_skill_buttons(adapter):
    adapter._register_slash_commands()
    assert "bindings" in adapter._client.tree.commands

    interaction = SimpleNamespace(
        channel_id=123,
        channel=SimpleNamespace(id=123, name="projekt-start"),
        response=SimpleNamespace(send_message=AsyncMock()),
    )

    await adapter._client.tree.commands["bindings"](interaction)

    adapter._check_slash_authorization.assert_awaited_once_with(interaction, "/bindings")
    kwargs = interaction.response.send_message.await_args.kwargs
    assert "discord-project-intake" in kwargs["content"]
    assert "github-pr-workflow" in kwargs["content"]
    assert kwargs["ephemeral"] is False
    view = kwargs["view"]
    assert isinstance(view, BindingSkillInfoView)
    assert [child.label for child in view.children] == [
        "discord-project-intake",
        "github-pr-workflow",
    ]


@pytest.mark.asyncio
async def test_binding_skill_button_sends_ephemeral_skill_details(monkeypatch):
    view = BindingSkillInfoView(
        skills=["discord-project-intake"],
        channel_id="123",
        binding_created_at="2026-05-14T13:45:00Z",
        skill_info_loader=lambda name: {
            "name": name,
            "description": "Project intake helper",
            "path": "/tmp/SKILL.md",
        },
    )
    interaction = SimpleNamespace(response=SimpleNamespace(send_message=AsyncMock()))

    await view.children[0].callback(interaction)

    kwargs = interaction.response.send_message.await_args.kwargs
    assert kwargs["ephemeral"] is True
    assert "Project intake helper" in kwargs["content"]
    assert "2026-05-14T13:45:00Z" in kwargs["content"]
