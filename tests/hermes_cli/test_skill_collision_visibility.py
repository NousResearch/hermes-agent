"""A skill named like a built-in command is kept out of auto-registration (the 370ebf2d3 guard)
and every listing surface says so where the user looks: the ``/skills`` table, ``/help skills``
and the ``commands.catalog`` palette RPC. A non-colliding skill carries no note."""
from __future__ import annotations

import contextlib
import io

from rich.console import Console

from hermes_constants import get_hermes_home

NOTE = "slash command /handoff unavailable — name taken by built-in; use /skill handoff"


def _write_skill(name: str) -> None:
    skill_dir = get_hermes_home() / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {name} skill.\n---\n# {name}\n", encoding="utf-8")


def test_built_in_name_collision_is_visible_on_every_listing_surface(monkeypatch):
    import cli
    import tools.skills_tool as skills_tool
    from hermes_cli.cli_info_mixin import CLIInfoMixin
    from hermes_cli.skills_hub import do_list
    from tui_gateway import server

    _write_skill("handoff")  # core CommandDef → dropped by scan_skill_commands
    _write_skill("tidy-notes")  # control
    monkeypatch.setattr(skills_tool, "_SKILLS_CACHE", {})
    monkeypatch.setattr(cli, "_skill_commands", None)

    sink = io.StringIO()
    do_list(console=Console(file=sink, force_terminal=False, color_system=None, width=200))
    skills_table = sink.getvalue()
    assert NOTE in skills_table
    assert "tidy-notes" in skills_table and skills_table.count("unavailable") == 1

    class _Cli(CLIInfoMixin):
        config: dict = {}

        def _command_available(self, slash_command):
            return True

    help_out = io.StringIO()
    with contextlib.redirect_stdout(help_out):
        _Cli().show_help("skills")
    assert NOTE in help_out.getvalue()
    assert "/tidy-notes" in help_out.getvalue() and "/handoff" not in help_out.getvalue().replace(NOTE, "")

    catalog = server._methods["commands.catalog"](1, {})["result"]
    assert catalog["warning"] == NOTE
    assert "/tidy-notes" in catalog["skills"] and "/handoff" not in catalog["skills"]

    from hermes_cli.slash_exec import CommandContext, _exec_commands

    gateway_commands = _exec_commands(CommandContext(args="", options={"page_size": 500})).text
    assert f"⚠ {NOTE}" in gateway_commands and "`/tidy-notes`" in gateway_commands


def test_catalog_discovery_failure_warning_outranks_the_collision_note(monkeypatch):
    """A colliding skill must not hide a real discovery failure: the failure stays in ``warning``."""
    import tools.skills_tool as skills_tool
    from tui_gateway import server

    _write_skill("handoff")
    _write_skill("tidy-notes")  # control: skills still list when a loader failed
    monkeypatch.setattr(skills_tool, "_SKILLS_CACHE", {})

    def _broken_cfg():
        raise RuntimeError("config.yaml unreadable")

    monkeypatch.setattr(server, "_load_cfg", _broken_cfg)

    catalog = server._methods["commands.catalog"](1, {})["result"]
    assert catalog["warning"] == "quick_commands discovery unavailable: config.yaml unreadable"
    assert "/tidy-notes" in catalog["skills"]
