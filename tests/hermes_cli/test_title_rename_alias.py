"""The /rename alias must dispatch the same handler as /title on every surface.

Regression for: /title existed but /rename — the spelling every other CLI (git,
docker, tmux, claude, codex) uses — resolved to nothing, so the rename capability
was undiscoverable muscle memory.
"""

from hermes_cli.commands import COMMAND_REGISTRY, resolve_command


def test_rename_resolves_to_title_handler():
    assert resolve_command("/rename") is resolve_command("/title")


def test_rename_spellings_all_resolve():
    # resolve_command takes a bare NAME (the dispatcher does the strip/split), so
    # assert exactly the spellings it is responsible for: case + optional slash.
    for spelling in ("rename", "/rename", "Rename", "/RENAME"):
        assert resolve_command(spelling).name == "title", spelling


def test_rename_is_registered_alias_not_a_second_command():
    """An alias, not a duplicate CommandDef — so help/autocomplete stay single-sourced."""
    titles = [c for c in COMMAND_REGISTRY if c.name == "title"]
    assert len(titles) == 1
    assert "rename" in titles[0].aliases
    assert not [c for c in COMMAND_REGISTRY if c.name == "rename"]
