"""``/rename`` is an alias of ``/title`` (the Claude-Code-style command name)."""

from hermes_cli.commands import resolve_command


def test_rename_is_an_alias_of_title():
    cmd = resolve_command("rename")
    assert cmd is not None
    assert cmd.name == "title"
    assert cmd.args_hint == "[name]"
