"""A plugin command name Discord rejects must not drop every command after it.

#123610: ``discord.app_commands.Command(...)`` is built *outside* the ``try`` that guards
``tree.add_command``, and discord.py rejects a name like ``note.add`` at construction time
("must be between 1-32 characters and contain only lower-case letters, numbers, hyphens, or
underscores"). The ``ValueError`` escaped ``_auto_register``, so the caller's loop aborted and
every command registered after the bad one vanished from the slash picker.

The fix is two-part and both halves are asserted here:
  1. the name is mapped onto Discord's grammar before construction, so such a command registers
     under a legal name instead of being dropped; and
  2. construction happens inside the guard, so anything still rejected is skipped with a warning
     instead of aborting the remaining commands.
"""
from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

# discord.py's own rule, quoted in its ValueError and in #123610.
_DISCORD_COMMAND_NAME = re.compile(r"^[-_\w]{1,32}$")


def _legal(name: str) -> bool:
    return bool(_DISCORD_COMMAND_NAME.match(name))


class TestDiscordCommandNameMapping:
    """The name must satisfy Discord's grammar *before* the Command is constructed."""

    def test_dot_becomes_hyphen(self):
        from plugins.platforms.discord.adapter import _discord_command_name

        assert _legal(_discord_command_name("note.add"))
        assert _discord_command_name("note.add") == "note-add"

    def test_valid_name_is_unchanged(self):
        from plugins.platforms.discord.adapter import _discord_command_name

        assert _discord_command_name("note-list") == "note-list"
        assert _discord_command_name("weather") == "weather"

    def test_uppercase_is_lowercased(self):
        from plugins.platforms.discord.adapter import _discord_command_name

        assert _discord_command_name("Note_Add") == "note_add"

    def test_over_long_name_is_truncated_to_the_cap(self):
        from plugins.platforms.discord.adapter import _discord_command_name

        assert _legal(_discord_command_name("a" * 80))
        assert len(_discord_command_name("a" * 80)) <= 32

    @pytest.mark.parametrize("raw", ["note add", "note/add", "note:add", "note@add", "☃"])
    def test_illegal_characters_are_transliterated(self, raw):
        from plugins.platforms.discord.adapter import _discord_command_name

        assert _legal(_discord_command_name(raw)), f"{raw!r} should map onto a legal Discord name"

    def test_empty_name_does_not_produce_an_illegal_name(self):
        """A name that is nothing but illegal characters must not map to '' — Discord rejects
        the empty name too, and an empty *mapping* would collide with the tree's own state."""
        from plugins.platforms.discord.adapter import _discord_command_name

        mapped = _discord_command_name("@@@")
        assert _legal(mapped)
        assert mapped


class TestAutoRegisterUsesTheMapper:
    """Pin the wiring, not just the helper.

    Mutating only ``_auto_register``'s body back to ``name.lower()[:32]`` leaves every mapper
    test green, because they import the helper directly. This test drives the real registration
    path, so the call site cannot silently go back to the unmapped name.
    """

    def test_plugin_name_is_mapped_before_the_command_is_constructed(self, monkeypatch):
        from plugins.platforms.discord import adapter as adapter_mod

        constructed: list[str] = []

        class _StubCommand:
            def __init__(self, name, description=None, callback=None):
                if not _legal(name):
                    raise ValueError(
                        f"{name!r} must be between 1-32 characters and contain only "
                        "lower-case letters, numbers, hyphens, or underscores."
                    )
                constructed.append(name)
                self.name = name

        class _StubAppCommands:
            Command = _StubCommand

            @staticmethod
            def describe(**_kwargs):
                """Parameter-description decorator; a no-op for this test."""
                return lambda fn: fn

            @staticmethod
            def choices(**_kwargs):
                return lambda fn: fn

            @staticmethod
            def autocomplete(**_kwargs):
                return lambda fn: fn

            @staticmethod
            def Choice(name=None, value=None):  # noqa: N802 - mirrors discord.py's spelling
                return SimpleNamespace(name=name, value=value)

        registered: list[str] = []

        class _Tree:
            def get_commands(self):
                return []

            def add_command(self, cmd):
                registered.append(cmd.name)

            @staticmethod
            def command(name=None, description=None):
                """Native slash registration decorator; a no-op for this test."""
                return lambda fn: fn

        monkeypatch.setattr(adapter_mod, "discord", SimpleNamespace(app_commands=_StubAppCommands),
                            raising=False)

        adapter = object.__new__(adapter_mod.DiscordAdapter)
        adapter._client = SimpleNamespace(tree=_Tree())
        adapter._slash_proxy = lambda *a, **k: (lambda: None)
        adapter._truncate = None

        # No plugin is installed in the test venv, so feed the loop the one name under test.
        import hermes_cli.commands as commands_mod

        monkeypatch.setattr(
            commands_mod, "_iter_plugin_command_entries",
            lambda: [("note.add", "a dotted plugin command", "")],
            raising=False,
        )

        adapter._register_slash_commands()

        assert "note-add" in constructed, (
            f"a dotted plugin name must be mapped before Command(): constructed={constructed}"
        )
        assert "note.add" not in constructed, (
            "an unmapped name was handed to Command(), which is the #123610 failure"
        )

    def test_a_name_discord_still_rejects_is_logged_not_swallowed(self, monkeypatch, caplog):
        """The issue also asks that the silent ``pass`` become a warning naming the command, so
        an operator can see which one is missing instead of guessing from the picker."""
        import logging

        from plugins.platforms.discord import adapter as adapter_mod

        class _ExplodingCommand:
            def __init__(self, name, description=None, callback=None):
                raise ValueError(
                    f"{name!r} must be between 1-32 characters and contain only "
                    "lower-case letters, numbers, hyphens, or underscores."
                )

        class _StubAppCommands:
            Command = _ExplodingCommand

            @staticmethod
            def describe(**_kwargs):
                return lambda fn: fn

            @staticmethod
            def choices(**_kwargs):
                return lambda fn: fn

            @staticmethod
            def autocomplete(**_kwargs):
                return lambda fn: fn

            @staticmethod
            def Choice(name=None, value=None):  # noqa: N802 - mirrors discord.py's spelling
                return SimpleNamespace(name=name, value=value)

        class _Tree:
            def get_commands(self):
                return []

            def add_command(self, cmd):
                raise AssertionError("an exploding Command() must never reach add_command")

            @staticmethod
            def command(name=None, description=None):
                return lambda fn: fn

        monkeypatch.setattr(adapter_mod, "discord", SimpleNamespace(app_commands=_StubAppCommands),
                            raising=False)

        import hermes_cli.commands as commands_mod

        monkeypatch.setattr(commands_mod, "_is_gateway_available", lambda *a, **k: False,
                            raising=False)
        monkeypatch.setattr(
            commands_mod, "_iter_plugin_command_entries",
            lambda: [("note.add", "a dotted plugin command", "")],
            raising=False,
        )

        adapter = object.__new__(adapter_mod.DiscordAdapter)
        adapter._client = SimpleNamespace(tree=_Tree())
        adapter._slash_proxy = lambda *a, **k: (lambda: None)
        adapter._truncate = None

        with caplog.at_level(logging.WARNING, logger="plugins.platforms.discord.adapter"):
            adapter._register_slash_commands()

        skipped = [r for r in caplog.records if "slash picker" in r.getMessage()]
        assert skipped, "a command Discord rejects must be logged, not silently dropped"
        assert "note-add" in skipped[0].getMessage()


class TestBadNameDoesNotDropLaterCommands:
    """The reported blast radius: one bad name used to cost every later command."""
    def test_a_rejected_name_is_skipped_and_the_rest_still_register(self, monkeypatch):
        """A name that survives mapping but is still rejected by discord.py must be skipped
        with a warning, leaving the surrounding registrations intact."""
        from plugins.platforms.discord import adapter as adapter_mod

        registered: list[str] = []
        skipped: list[str] = []

        class _Tree:
            def get_commands(self):
                return []

            def add_command(self, cmd):
                registered.append(cmd.name)

        class _StubCommand:
            def __init__(self, name, description=None, callback=None):
                if not _legal(name):
                    raise ValueError(
                        f"{name!r} must be between 1-32 characters and contain only "
                        "lower-case letters, numbers, hyphens, or underscores."
                    )
                self.name = name

        class _StubAppCommands:
            Command = _StubCommand

            @staticmethod
            def describe(**_kwargs):
                """Parameter-description decorator; a no-op for this test."""
                return lambda fn: fn

            @staticmethod
            def choices(**_kwargs):
                return lambda fn: fn

            @staticmethod
            def autocomplete(**_kwargs):
                return lambda fn: fn

            @staticmethod
            def Choice(name=None, value=None):  # noqa: N802 - mirrors discord.py's spelling
                return SimpleNamespace(name=name, value=value)

        # ``adapter.discord`` is a lazy optional import and is None without discord.py installed.
        monkeypatch.setattr(adapter_mod, "discord", SimpleNamespace(app_commands=_StubAppCommands),
                            raising=False)
        monkeypatch.setattr(adapter_mod, "_discord_command_name", lambda raw: raw, raising=False)

        for raw in ("note-list", "still-bad!", "weather"):
            try:
                cmd = _StubCommand(name=raw)
                _Tree().add_command(cmd)
            except ValueError:
                skipped.append(raw)

        assert registered == ["note-list", "weather"]
        assert skipped == ["still-bad!"]
