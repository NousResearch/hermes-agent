import importlib
import sys
from types import SimpleNamespace

import pytest


def _real_discord_available() -> bool:
    try:
        importlib.import_module("discord")
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _real_discord_available(), reason="discord.py not installed")
def test_register_slash_commands_with_real_command_tree(monkeypatch):
    for module_name in (
        "discord",
        "discord.ext",
        "discord.ext.commands",
        "gateway.platforms.discord_impl.native_commands",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    importlib.invalidate_caches()
    discord = importlib.import_module("discord")
    native_commands = importlib.import_module("gateway.platforms.discord_impl.native_commands")

    client = discord.Client(intents=discord.Intents.none())
    tree = discord.app_commands.CommandTree(client)

    native_commands.register_slash_commands(tree, SimpleNamespace())

    assert tree.get_command("status") is not None
    assert tree.get_command("model") is not None
    assert tree.get_command("bash") is not None
