"""Regression tests for Discord category-level channel gates."""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock
import os
import sys


def _ensure_discord_mock():
    """Install a mock discord module when discord.py isn't available."""
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
        secondary=2,
        danger=3,
        green=1,
        grey=2,
        blurple=2,
        red=3,
    )
    discord_mod.Color = SimpleNamespace(
        orange=lambda: 1,
        green=lambda: 2,
        blue=lambda: 3,
        red=lambda: 4,
        purple=lambda: 5,
    )
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from gateway.config import PlatformConfig  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


class FakeCategory:
    id = 500
    name = "Team Ops"


class FakeTextChannel:
    id = 123
    name = "adops"
    category_id = 500
    category = FakeCategory()


class DiscordCategoryChannelGateTests(TestCase):
    def setUp(self):
        self.adapter = object.__new__(DiscordAdapter)
        self.adapter.config = PlatformConfig(enabled=True, token="fake-token", extra={})

    def tearDown(self):
        os.environ.pop("DISCORD_ALLOWED_CATEGORIES", None)
        os.environ.pop("DISCORD_FREE_RESPONSE_CATEGORIES", None)

    def test_channel_keys_do_not_overload_category_ids(self):
        keys = self.adapter._discord_channel_keys_from_channel(FakeTextChannel())

        self.assertIn("123", keys)
        self.assertNotIn("500", keys)
        self.assertNotIn("Team Ops", keys)
        self.assertNotIn("#Team Ops", keys)

    def test_category_keys_include_category_id_and_name(self):
        keys = self.adapter._discord_category_keys_from_channel(FakeTextChannel())

        self.assertEqual({"500", "Team Ops", "#Team Ops"}, keys)

    def test_allowed_categories_uses_explicit_env_var(self):
        os.environ["DISCORD_ALLOWED_CATEGORIES"] = "500,600"

        self.assertTrue(
            self.adapter._discord_category_ids_allowed(
                self.adapter._discord_category_keys_from_channel(FakeTextChannel())
            )
        )

    def test_free_response_categories_uses_explicit_yaml_extra(self):
        self.adapter.config.extra["free_response_categories"] = [500, 600]

        self.assertEqual({"500", "600"}, self.adapter._discord_free_response_categories())
