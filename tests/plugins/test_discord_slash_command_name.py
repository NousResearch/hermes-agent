"""A plugin command name Discord rejects must not drop every command after it.

#123610: ``discord.app_commands.Command(...)`` is built *outside* the ``try`` that guards
``tree.add_command``, and discord.py rejects a name like ``note.add`` at construction time
("must be between 1-32 characters and contain only lower-case letters, numbers, hyphens, or
underscores"). The ``ValueError`` escaped ``_auto_register``, so the caller's loop aborted and
every command registered after the bad one vanished from the slash picker.

The fix is three-part and each part is asserted here:
  1. the name is mapped onto Discord's grammar before construction, so such a command registers
     under a legal name instead of being dropped;
  2. construction happens inside the guard, so anything still rejected is skipped with a warning
     instead of aborting the remaining commands; and
  3. because that mapping is lossy, a name that lands on an already-registered slot is reported
     rather than silently disappearing.
"""
from __future__ import annotations

import logging
import re
from types import SimpleNamespace

import pytest

# Discord's rule, quoted in its ValueError and in #123610. The class is [a-z0-9_-]: NOT ``\w``,
# which also matches Unicode letters (``café``, ``日本語``) that Discord's API rejects.
_DISCORD_COMMAND_NAME = re.compile(r"^[a-z0-9_-]{1,32}$")


def _legal(name: str) -> bool:
    return bool(_DISCORD_COMMAND_NAME.match(name))


def _stub_app_commands(command_cls) -> type:
    """``discord.app_commands`` with ``command_cls`` as ``Command``.

    The decorators are no-ops: they only exist so the adapter's native-slash registration runs
    through the real code path under test.
    """
    return type("StubAppCommands", (), {
        "Command": staticmethod(command_cls),
        "describe": staticmethod(lambda **_: lambda fn: fn),
        "choices": staticmethod(lambda **_: lambda fn: fn),
        "autocomplete": staticmethod(lambda **_: lambda fn: fn),
        "Choice": staticmethod(lambda name=None, value=None: SimpleNamespace(name=name, value=value)),
    })


class _RejectingCommand:
    """Stands in for discord.py's ``Command``: raises unless the name is on Discord's grammar."""

    def __init__(self, name, description=None, callback=None):
        if not _legal(name):
            raise ValueError(
                f"{name!r} must be between 1-32 characters and contain only "
                "lower-case letters, numbers, hyphens, or underscores."
            )
        self.name = name


class _Tree:
    """Minimal command tree. ``add_command`` records, or refuses when ``explode`` is set."""

    explode = False

    def __init__(self):
        self.registered: list[str] = []

    def get_commands(self):
        return []

    def add_command(self, cmd):
        if self.explode:
            raise AssertionError("a rejected Command() must never reach add_command")
        self.registered.append(cmd.name)

    @staticmethod
    def command(name=None, description=None):
        """Native slash registration decorator; a no-op for these tests."""
        return lambda fn: fn


def _adapter(monkeypatch, plugin_names, *, tree=None, command_cls=_RejectingCommand,
             with_registry=False):
    """Drive the real ``_register_slash_commands`` over ``plugin_names``.

    ``with_registry`` lets the real COMMAND_REGISTRY commands register too; by default they are
    gated off so ``tree.registered`` shows only what these tests are about.
    """
    from plugins.platforms.discord import adapter as adapter_mod
    import hermes_cli.commands as commands_mod

    tree = tree if tree is not None else _Tree()
    monkeypatch.setattr(
        adapter_mod, "discord",
        SimpleNamespace(app_commands=_stub_app_commands(command_cls)),
        raising=False,
    )
    monkeypatch.setattr(commands_mod, "_is_gateway_available",
                        lambda *a, **k: bool(with_registry), raising=False)
    monkeypatch.setattr(
        commands_mod, "_iter_plugin_command_entries",
        lambda: [(n, f"{n} description", "") for n in plugin_names],
        raising=False,
    )

    adapter = object.__new__(adapter_mod.DiscordAdapter)
    adapter._client = SimpleNamespace(tree=tree)
    adapter._slash_proxy = lambda *a, **k: (lambda: None)
    adapter._register_slash_commands()
    return adapter, tree


class TestDiscordCommandNameMapping:
    """The name must satisfy Discord's grammar *before* the Command is constructed."""

    def test_dot_becomes_hyphen(self):
        from plugins.platforms.discord.adapter import _discord_command_name

        assert _discord_command_name("note.add") == "note-add"
        assert _legal(_discord_command_name("note.add"))

    def test_valid_name_is_unchanged(self):
        from plugins.platforms.discord.adapter import _discord_command_name

        assert _discord_command_name("note-list") == "note-list"
        assert _discord_command_name("weather") == "weather"

    def test_uppercase_is_lowercased(self):
        from plugins.platforms.discord.adapter import _discord_command_name

        assert _discord_command_name("Note_Add") == "note_add"

    def test_over_long_name_is_truncated_to_the_cap(self):
        from plugins.platforms.discord.adapter import _discord_command_name

        assert len(_discord_command_name("a" * 80)) <= 32
        assert _legal(_discord_command_name("a" * 80))

    @pytest.mark.parametrize("raw", ["note add", "note/add", "note:add", "note@add", "☃"])
    def test_illegal_characters_are_transliterated(self, raw):
        from plugins.platforms.discord.adapter import _discord_command_name

        assert _legal(_discord_command_name(raw)), f"{raw!r} should map onto a legal Discord name"

    @pytest.mark.parametrize("raw", ["café", "naïve", "über", "日本語"])
    def test_non_ascii_names_map_onto_ascii(self, raw):
        """Discord accepts [a-z0-9_-] only — an accented or CJK name must be transliterated, not
        passed through. A ``\\w``-based rule would call these legal and let them raise instead."""
        from plugins.platforms.discord.adapter import _discord_command_name

        mapped = _discord_command_name(raw)
        assert _legal(mapped), f"{raw!r} must map to an ASCII name, got {mapped!r}"
        assert mapped.isascii()

    def test_empty_name_does_not_produce_an_illegal_name(self):
        """A name of only illegal characters must not map to '' — Discord rejects the empty
        name too, and an empty mapping would collide with the tree's own state."""
        from plugins.platforms.discord.adapter import _discord_command_name

        mapped = _discord_command_name("@@@")
        assert _legal(mapped)
        assert mapped


class TestAutoRegisterUsesTheMapper:
    """Pin the wiring, not just the helper.

    Mutating only ``_auto_register``'s body back to ``name.lower()[:32]`` leaves every mapper
    test green, because they import the helper directly. These tests drive the real registration
    path, so the call site cannot silently go back to the unmapped name.
    """

    def test_plugin_name_is_mapped_before_the_command_is_constructed(self, monkeypatch):
        constructed: list[str] = []

        class _Recording(_RejectingCommand):
            def __init__(self, name, description=None, callback=None):
                super().__init__(name, description, callback)
                constructed.append(name)

        _adapter(monkeypatch, ["note.add"], command_cls=_Recording)

        assert "note-add" in constructed, (
            f"a dotted plugin name must be mapped before Command(): constructed={constructed}"
        )
        assert "note.add" not in constructed, (
            "an unmapped name was handed to Command(), which is the #123610 failure"
        )

    def test_a_name_discord_still_rejects_is_logged_not_swallowed(self, monkeypatch, caplog):
        """The issue also asks that the silent ``pass`` become a warning naming the command, so
        an operator can see which one is missing instead of guessing from the picker."""

        class _Exploding:
            def __init__(self, name, description=None, callback=None):
                raise ValueError(f"{name!r} must be between 1-32 characters ...")

        with caplog.at_level(logging.WARNING, logger="plugins.platforms.discord.adapter"):
            _adapter(monkeypatch, ["note.add"], command_cls=_Exploding)

        skipped = [r for r in caplog.records if "slash picker" in r.getMessage()]
        assert skipped, "a command Discord rejects must be logged, not silently dropped"
        assert "note-add" in skipped[0].getMessage()


class TestBadNameDoesNotDropLaterCommands:
    """The reported blast radius: one bad name used to cost every later command."""

    def test_a_rejected_name_does_not_cost_the_commands_after_it(self, monkeypatch, caplog):
        """The real regression: with construction outside the guard, the first rejected name
        aborted the caller's loop and every later command was lost. Drive the real loop.

        The stub rejects one specific *mapped* name, so the rejection happens inside the guard
        the way a genuine Discord rejection would — and the two commands after it must survive.
        """

        class _RejectsNoteAdd(_RejectingCommand):
            def __init__(self, name, description=None, callback=None):
                if name == "note-add":
                    raise ValueError(f"{name!r} must be between 1-32 characters ...")
                super().__init__(name, description, callback)

        tree = _Tree()
        with caplog.at_level(logging.WARNING, logger="plugins.platforms.discord.adapter"):
            _adapter(monkeypatch, ["note.add", "note-list", "weather"],
                     tree=tree, command_cls=_RejectsNoteAdd)

        # The later commands survive; only the rejected one is absent.
        assert tree.registered == ["note-list", "weather"]
        assert any("note-add" in r.getMessage() for r in caplog.records)

    def test_a_losing_name_collision_is_reported_not_silent(self, monkeypatch, caplog):
        """Mapping is lossy: ``note.add``, ``note-add`` and ``note add`` all become ``note-add``.
        The loser must be logged — silently dropping a plugin command reads as 'not installed'."""
        tree = _Tree()
        with caplog.at_level(logging.WARNING, logger="plugins.platforms.discord.adapter"):
            _adapter(monkeypatch, ["note.add", "note add", "note-list"],
                     tree=tree, command_cls=_RejectingCommand)

        assert tree.registered == ["note-add", "note-list"]
        collisions = [r for r in caplog.records if "already registered from" in r.getMessage()]
        assert collisions, "a lossy-mapping collision must be reported"
        assert "note.add" in collisions[0].getMessage()

    def test_the_same_name_offered_twice_is_not_a_collision(self, monkeypatch, caplog):
        """COMMAND_REGISTRY and the plugin loop can offer the same name; that is the normal
        duplicate-suppression path and must stay quiet."""
        with caplog.at_level(logging.WARNING, logger="plugins.platforms.discord.adapter"):
            _adapter(monkeypatch, ["note.add", "note.add"], command_cls=_RejectingCommand)

        assert not [r for r in caplog.records if "already registered from" in r.getMessage()]