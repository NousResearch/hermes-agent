import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, danger=3, green=1, blurple=2, red=3, grey=4, secondary=5)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock()
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    discord_mod.opus = SimpleNamespace(is_loaded=lambda: True)

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

import gateway.platforms.discord as discord_platform  # noqa: E402
from gateway.platforms.discord import DiscordAdapter  # noqa: E402


class FakeForumChannel:
    def __init__(self, channel_id: int, name: str = "forum", topic: str | None = None):
        self.id = channel_id
        self.name = name
        self.topic = topic
        self.guild = SimpleNamespace(name="Neomesh Lab")


class FakeThread:
    def __init__(self, thread_id: int, parent):
        self.id = thread_id
        self.parent = parent
        self.guild = getattr(parent, "guild", None)
        self.name = "thread"
        self.topic = None


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(discord_platform.discord, "ForumChannel", FakeForumChannel, raising=False)
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)
    config = PlatformConfig(
        enabled=True,
        token="fake-token",
        extra={
            "shared_auto_skills": ["neomesh-core"],
            "forum_skill_bindings": [
                {"channel_id": "200", "skill": "backend"},
                {"channel_id": "300", "skills": ["qa", "release-gate"]},
            ],
        },
    )
    return DiscordAdapter(config)


def test_resolve_auto_skills_from_forum_parent_binding(adapter):
    parent = FakeForumChannel(200, topic="Backend work")
    thread = FakeThread(201, parent)

    assert adapter._resolve_auto_skills_for_channel(thread) == ["neomesh-core", "backend"]


def test_resolve_auto_skills_from_direct_channel_binding(adapter):
    channel = SimpleNamespace(id=300, parent=None)

    assert adapter._resolve_auto_skills_for_channel(channel) == ["neomesh-core", "qa", "release-gate"]


def test_resolve_auto_skills_returns_none_without_match(adapter):
    channel = SimpleNamespace(id=999, parent=None)

    assert adapter._resolve_auto_skills_for_channel(channel) is None


def test_resolve_auto_skills_dedupes_and_preserves_order(monkeypatch):
    monkeypatch.setattr(discord_platform.discord, "ForumChannel", FakeForumChannel, raising=False)
    config = PlatformConfig(
        enabled=True,
        token="fake-token",
        extra={
            "shared_auto_skills": ["neomesh-core", "backend"],
            "forum_skill_bindings": [
                {
                    "channel_id": "500",
                    "shared_skills": ["backend", "release-principles"],
                    "skills": ["backend", "qa"],
                }
            ],
        },
    )
    adapter = DiscordAdapter(config)
    channel = FakeForumChannel(500, topic="Release")

    assert adapter._resolve_auto_skills_for_channel(channel) == [
        "neomesh-core",
        "backend",
        "release-principles",
        "qa",
    ]


def test_resolve_auto_skills_supports_binding_specific_shared_skills(monkeypatch):
    monkeypatch.setattr(discord_platform.discord, "ForumChannel", FakeForumChannel, raising=False)
    config = PlatformConfig(
        enabled=True,
        token="fake-token",
        extra={
            "forum_skill_bindings": [
                {
                    "channel_id": "400",
                    "shared_skills": ["neomesh-core"],
                    "skills": ["cto"],
                }
            ]
        },
    )
    adapter = DiscordAdapter(config)
    channel = FakeForumChannel(400, topic="CTO")

    assert adapter._resolve_auto_skills_for_channel(channel) == ["neomesh-core", "cto"]
