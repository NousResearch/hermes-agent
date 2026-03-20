"""Import tests for the Discord v2 internal implementation scaffold."""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock


def _ensure_discord_mock():
    """Install a lightweight discord mock when discord.py isn't available."""
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(
        View=object,
        button=lambda *a, **k: (lambda fn: fn),
        Button=object,
    )
    discord_mod.ButtonStyle = SimpleNamespace(
        success=1,
        primary=2,
        danger=3,
        green=1,
        blurple=2,
        red=3,
    )
    discord_mod.Color = SimpleNamespace(
        orange=lambda: 1,
        green=lambda: 2,
        blue=lambda: 3,
        red=lambda: 4,
    )
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    discord_mod.opus = SimpleNamespace(
        is_loaded=lambda: True,
        load_opus=lambda *_args, **_kwargs: None,
    )
    discord_mod.FFmpegPCMAudio = MagicMock
    discord_mod.PCMVolumeTransformer = MagicMock
    discord_mod.http = SimpleNamespace(Route=MagicMock)

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()


def test_discord_impl_package_is_importable():
    package = importlib.import_module("gateway.platforms.discord_impl")
    assert package is not None


def test_discord_impl_submodules_are_importable():
    module_names = (
        "config",
        "intake",
        "delivery",
        "interactions",
        "threads",
        "state",
        "history",
        "permissions",
    )

    for module_name in module_names:
        module = importlib.import_module(f"gateway.platforms.discord_impl.{module_name}")
        assert module is not None


def test_discord_impl_re_exports_nothing():
    package = importlib.import_module("gateway.platforms.discord_impl")
    assert not hasattr(package, "__all__") or not package.__all__


def test_discord_public_import_surface_remains_available():
    from gateway.platforms.discord import (
        DiscordAdapter,
        VoiceReceiver,
        check_discord_requirements,
    )

    assert DiscordAdapter is not None
    assert VoiceReceiver is not None
    assert callable(check_discord_requirements)
