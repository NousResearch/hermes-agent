"""Tests for Discord skill slash command registration (issue #5480)."""
from __future__ import annotations

import types
from unittest.mock import MagicMock, patch


def _make_adapter():
    """Return a DiscordAdapter instance with discord.py fully mocked."""
    import sys

    # Build a minimal discord stub
    discord_stub = types.ModuleType("discord")
    discord_stub.Interaction = object
    discord_stub.Intents = MagicMock()
    discord_stub.Message = object
    discord_stub.MessageType = MagicMock()
    discord_stub.DMChannel = object
    discord_stub.Thread = object
    discord_stub.TextChannel = object
    discord_stub.opus = MagicMock()
    discord_stub.opus.is_loaded = MagicMock(return_value=True)
    discord_stub.Color = MagicMock()
    discord_stub.ButtonStyle = MagicMock()
    discord_stub.File = MagicMock()
    discord_stub.FFmpegPCMAudio = MagicMock()
    discord_stub.PCMVolumeTransformer = MagicMock()
    discord_stub.Embed = MagicMock()
    discord_stub.ui = MagicMock()

    # app_commands stub
    app_commands_stub = types.ModuleType("discord.app_commands")
    app_commands_stub.describe = lambda **_: (lambda f: f)
    app_commands_stub.choices = lambda **_: (lambda f: f)
    app_commands_stub.Choice = MagicMock()

    registered_commands: dict = {}

    class FakeCommand:
        def __init__(self, *, name, description, callback):
            self.name = name
            self.description = description
            self.callback = callback

    class FakeTree:
        def __init__(self):
            self._cmds: dict[str, FakeCommand] = {}

        def command(self, *, name, description):
            def decorator(func):
                self._cmds[name] = FakeCommand(name=name, description=description, callback=func)
                return func
            return decorator

        def get_commands(self):
            return list(self._cmds.values())

        def get_command(self, name):
            return self._cmds.get(name)

        def add_command(self, cmd):
            self._cmds[cmd.name] = cmd

        async def sync(self):
            return list(self._cmds.values())

    app_commands_stub.Command = FakeCommand
    app_commands_stub.Group = MagicMock()

    discord_stub.app_commands = app_commands_stub

    ext_stub = types.ModuleType("discord.ext")
    commands_stub = types.ModuleType("discord.ext.commands")

    class FakeBot:
        def __init__(self, **kwargs):
            self.tree = FakeTree()
            self.user = None

        def event(self, func):
            return func

    commands_stub.Bot = FakeBot
    ext_stub.commands = commands_stub
    discord_stub.ext = ext_stub

    http_stub = types.ModuleType("discord.http")
    http_stub.Route = MagicMock()
    discord_stub.http = http_stub

    sys.modules.setdefault("discord", discord_stub)
    sys.modules.setdefault("discord.ext", ext_stub)
    sys.modules.setdefault("discord.ext.commands", commands_stub)
    sys.modules.setdefault("discord.app_commands", app_commands_stub)
    sys.modules.setdefault("discord.http", http_stub)

    from gateway.config import PlatformConfig
    from gateway.platforms.discord import DiscordAdapter

    cfg = PlatformConfig(enabled=True, token="fake-token")
    adapter = DiscordAdapter(cfg)
    # Give adapter a real bot so _register_slash_commands works
    adapter._client = FakeBot()
    return adapter, FakeTree


def test_skills_registered_as_slash_commands():
    """Installed skills should appear in Discord's slash command tree."""
    adapter, FakeTree = _make_adapter()

    fake_skill_cmds = {
        "/gif-search": {
            "name": "gif-search",
            "description": "Search for GIFs",
            "skill_md_path": "/home/sprmn/.hermes/hermes-agent/skills/gif-search/SKILL.md",
            "skill_dir": "/home/sprmn/.hermes/hermes-agent/skills/gif-search",
        },
        "/code-review": {
            "name": "code-review",
            "description": "Review code changes",
            "skill_md_path": "/home/sprmn/.hermes/hermes-agent/skills/code-review/SKILL.md",
            "skill_dir": "/home/sprmn/.hermes/hermes-agent/skills/code-review",
        },
    }

    skills_dir_mock = MagicMock()
    skills_dir_mock.resolve.return_value = MagicMock(
        __str__=lambda self: "/home/user/.hermes/hermes-agent/skills"
    )
    hub_dir = MagicMock()
    hub_dir.__str__ = lambda self: "/home/sprmn/.hermes/hermes-agent/skills/.hub"
    skills_dir_mock.__truediv__ = lambda self, other: hub_dir

    with patch("gateway.platforms.discord.DISCORD_AVAILABLE", True), \
         patch("agent.skill_commands.scan_skill_commands", return_value=fake_skill_cmds), \
         patch("agent.skill_utils.get_disabled_skill_names", return_value=set()), \
         patch("tools.skills_tool.SKILLS_DIR", skills_dir_mock):
        adapter._register_slash_commands()

    tree = adapter._client.tree
    registered = {cmd.name for cmd in tree.get_commands()}

    assert "gif-search" in registered, f"gif-search not registered. Got: {registered}"
    assert "code-review" in registered, f"code-review not registered. Got: {registered}"


def test_skill_cap_at_100_commands(caplog):
    import logging
    adapter,FakeTree=_make_adapter()
    fake_skill_cmds={f"/skill-{i:03d}":{"name":f"skill-{i:03d}","description":f"Skill {i}","skill_md_path":f"/home/sprmn/.hermes/hermes-agent/skills/skill-{i:03d}/SKILL.md","skill_dir":f"/home/sprmn/.hermes/hermes-agent/skills/skill-{i:03d}"} for i in range(150)}
    sm=MagicMock()
    sm.resolve.return_value=MagicMock(__str__=lambda s:"/home/user/.hermes/skills")
    hd=MagicMock()
    hd.__str__=lambda s:"/home/sprmn/.hermes/hermes-agent/skills/.hub"
    sm.__truediv__=lambda s,o:hd
    with patch("gateway.platforms.discord.DISCORD_AVAILABLE",True),patch("agent.skill_commands.scan_skill_commands",return_value=fake_skill_cmds),patch("agent.skill_utils.get_disabled_skill_names",return_value=set()),patch("tools.skills_tool.SKILLS_DIR",sm),caplog.at_level(logging.WARNING,logger="gateway.platforms.discord"):
        adapter._register_slash_commands()
    assert len(adapter._client.tree.get_commands())<=100
    assert any("limit reached" in r.message for r in caplog.records)

def test_disabled_skills_not_registered():
    """Skills disabled for Discord platform should not be registered."""
    adapter, FakeTree = _make_adapter()

    fake_skill_cmds = {
        "/secret-skill": {
            "name": "secret-skill",
            "description": "Should not appear",
            "skill_md_path": "/home/sprmn/.hermes/hermes-agent/skills/secret-skill/SKILL.md",
            "skill_dir": "/home/sprmn/.hermes/hermes-agent/skills/secret-skill",
        },
    }

    skills_dir_mock = MagicMock()
    skills_dir_mock.resolve.return_value = MagicMock(
        __str__=lambda self: "/home/user/.hermes/hermes-agent/skills"
    )
    hub_dir = MagicMock()
    hub_dir.__str__ = lambda self: "/home/sprmn/.hermes/hermes-agent/skills/.hub"
    skills_dir_mock.__truediv__ = lambda self, other: hub_dir

    with patch("gateway.platforms.discord.DISCORD_AVAILABLE", True), \
         patch("agent.skill_commands.scan_skill_commands", return_value=fake_skill_cmds), \
         patch("agent.skill_utils.get_disabled_skill_names", return_value={"secret-skill"}), \
         patch("tools.skills_tool.SKILLS_DIR", skills_dir_mock):
        adapter._register_slash_commands()

    tree = adapter._client.tree
    registered = {cmd.name for cmd in tree.get_commands()}
    assert "secret-skill" not in registered, f"Disabled skill was registered: {registered}"


def test_builtin_name_collision_skipped():
    """A skill whose name collides with a built-in command should be skipped."""
    adapter, FakeTree = _make_adapter()

    # First register built-ins (which include /status)
    fake_skill_cmds = {
        "/status": {
            "name": "status",
            "description": "Skill that collides with built-in",
            "skill_md_path": "/home/sprmn/.hermes/hermes-agent/skills/status/SKILL.md",
            "skill_dir": "/home/sprmn/.hermes/hermes-agent/skills/status",
        },
    }

    skills_dir_mock = MagicMock()
    skills_dir_mock.resolve.return_value = MagicMock(
        __str__=lambda self: "/home/user/.hermes/hermes-agent/skills"
    )
    hub_dir = MagicMock()
    hub_dir.__str__ = lambda self: "/home/sprmn/.hermes/hermes-agent/skills/.hub"
    skills_dir_mock.__truediv__ = lambda self, other: hub_dir

    with patch("gateway.platforms.discord.DISCORD_AVAILABLE", True), \
         patch("agent.skill_commands.scan_skill_commands", return_value=fake_skill_cmds), \
         patch("agent.skill_utils.get_disabled_skill_names", return_value=set()), \
         patch("tools.skills_tool.SKILLS_DIR", skills_dir_mock):
        adapter._register_slash_commands()

    tree = adapter._client.tree
    # The /status built-in should exist; skill's /status should not overwrite it
    status_cmd = tree.get_command("status")
    assert status_cmd is not None
    # Verify it's the built-in (has the built-in description, not the skill's)
    assert "session" in status_cmd.description.lower() or status_cmd.description != "Skill that collides with built-in"
