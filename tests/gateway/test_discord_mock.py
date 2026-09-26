"""Behavior contracts for the shared Discord test module loader."""

import importlib
import sys
from types import ModuleType, SimpleNamespace


def _load_helper():
    return importlib.import_module("tests.discord_mock")


def test_existing_discord_mock_is_augmented_without_replacing_it(monkeypatch):
    partial = ModuleType("discord")
    partial.MessageType = SimpleNamespace(default=0, reply=19)
    partial_ext = ModuleType("discord.ext")
    partial_commands = ModuleType("discord.ext.commands")
    partial_opus = ModuleType("discord.opus")
    monkeypatch.setitem(sys.modules, "discord", partial)
    monkeypatch.setitem(sys.modules, "discord.ext", partial_ext)
    monkeypatch.setitem(sys.modules, "discord.ext.commands", partial_commands)
    monkeypatch.setitem(sys.modules, "discord.opus", partial_opus)

    discord_mock = _load_helper()
    loaded = discord_mock.ensure_discord_module()

    assert loaded is partial
    assert sys.modules["discord"] is partial
    assert loaded.ext is partial_ext
    assert loaded.ext.commands is partial_commands
    assert loaded.opus is partial_opus
    assert loaded.opus.is_loaded() is True
    assert loaded.MessageType.default == 0
    assert loaded.app_commands.Command is not None
    assert loaded.ui.Select is not None
    assert loaded.AudioSource().is_opus() is False
    assert issubclass(loaded.Forbidden, Exception)


def test_already_imported_real_discord_module_wins(monkeypatch):
    for name in tuple(sys.modules):
        if name == "discord" or name.startswith("discord."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    real_discord = ModuleType("discord")
    real_discord.__file__ = "/installed/discord/__init__.py"
    monkeypatch.setitem(sys.modules, "discord", real_discord)

    discord_mock = _load_helper()

    assert discord_mock.ensure_discord_module() is real_discord
    assert not hasattr(real_discord, "_hermes_test_mock_configured")
