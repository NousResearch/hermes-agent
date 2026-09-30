"""A skill named like a built-in command is kept out of auto-registration (the 370ebf2d3 guard)
and every listing surface says so where the user looks: the ``/skills`` table, ``/help skills``
and the ``commands.catalog`` palette RPC. A non-colliding skill carries no note."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest
from rich.console import Console

from hermes_constants import get_hermes_home

NOTE = "slash command /handoff unavailable: a built-in command uses that name"
CLI_NOTE = f"{NOTE}; to load it, start a session with hermes -s handoff"


def _write_skill(name: str) -> None:
    skill_dir = get_hermes_home() / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {name} skill.\n---\n# {name}\n", encoding="utf-8")


def test_built_in_name_collision_is_visible_on_every_listing_surface(monkeypatch):
    import cli
    from tools import skills_tool
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
    assert CLI_NOTE in skills_table
    assert "tidy-notes" in skills_table and skills_table.count("unavailable") == 1

    class _Cli(CLIInfoMixin):
        config: dict = {}

        def _command_available(self, slash_command):
            return True

    help_out = io.StringIO()
    with contextlib.redirect_stdout(help_out):
        _Cli().show_help("skills")
    assert CLI_NOTE in help_out.getvalue()
    assert (
        "/tidy-notes" in help_out.getvalue()
        and "/handoff" not in help_out.getvalue().replace(CLI_NOTE, "")
    )

    catalog = server._methods["commands.catalog"](1, {})["result"]
    assert catalog["warning"] == NOTE
    assert "/tidy-notes" in catalog["skills"] and "/handoff" not in catalog["skills"]

    from hermes_cli.slash_exec import CommandContext, _exec_commands

    gateway_commands = _exec_commands(CommandContext(args="", options={"page_size": 500})).text
    assert f"⚠ {NOTE}" in gateway_commands and "`/tidy-notes`" in gateway_commands


@pytest.mark.parametrize("profile", [None, "work"])
def test_cli_note_advises_a_launch_command_that_loads_the_skill(
    profile, tmp_path, monkeypatch
):
    """The CLI surfaces' advice, run from a fresh shell, preloads the colliding skill from the
    profile that showed the note. The skill exists only in that profile's home."""
    import shlex

    import cli
    import tools.skills_tool as skills_tool
    from agent.skill_commands import (
        build_preloaded_skills_prompt,
        cli_skill_command_collision_note,
    )
    from hermes_cli._parser import build_top_level_parser
    from hermes_cli.main import _scan_profile_flag
    from hermes_cli.profiles import create_profile, resolve_profile_env

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    if profile:
        monkeypatch.setenv(
            "HERMES_HOME", str(create_profile(profile, no_alias=True, no_skills=True))
        )
    _write_skill("handoff")
    note = cli_skill_command_collision_note("handoff")
    assert note is not None

    monkeypatch.setenv("HERMES_HOME", str(root))
    argv = shlex.split(note.partition(" hermes ")[2])
    profile_name, consumed, index = _scan_profile_flag(argv)
    if profile_name:
        monkeypatch.setenv("HERMES_HOME", resolve_profile_env(profile_name))
        del argv[index : index + consumed]
    parser, _subparsers, _chat = build_top_level_parser()
    args = parser.parse_args(argv)

    monkeypatch.setattr(skills_tool, "_SKILLS_CACHE", {})
    _prompt, loaded, missing = build_preloaded_skills_prompt(
        cli._parse_skills_argument(args.skills)
    )
    assert (profile_name, loaded, missing) == (profile, ["handoff"], [])


def test_skills_table_offers_no_launch_for_a_disabled_colliding_skill(monkeypatch):
    """``hermes -s`` refuses a disabled skill, so the table must not suggest it for one."""
    import tools.skills_tool as skills_tool
    from hermes_cli.skills_hub import do_list

    _write_skill("handoff")
    (get_hermes_home() / "config.yaml").write_text(
        "skills:\n  disabled: [handoff]\n", encoding="utf-8"
    )
    monkeypatch.setattr(skills_tool, "_SKILLS_CACHE", {})

    sink = io.StringIO()
    do_list(
        console=Console(file=sink, force_terminal=False, color_system=None, width=200)
    )
    assert NOTE in sink.getvalue() and "hermes -s" not in sink.getvalue()


def test_gated_commands_listing_omits_collision_notes(monkeypatch):
    """A gated non-admin's ``/commands`` hides skill commands, so a colliding skill changes nothing in it."""
    import tools.skills_tool as skills_tool
    from hermes_cli.slash_exec import CommandContext, _exec_commands

    ctx = CommandContext(
        args="", options={"page_size": 500, "allowed_commands": {"help", "whoami"}}
    )
    _write_skill("tidy-notes")
    monkeypatch.setattr(skills_tool, "_SKILLS_CACHE", {})
    without_collision = _exec_commands(ctx).text

    _write_skill("handoff")
    skills_tool._SKILLS_CACHE.clear()
    assert _exec_commands(ctx).text == without_collision


def test_catalog_discovery_failure_warning_outranks_the_collision_note(monkeypatch):
    """A colliding skill must not hide a real discovery failure: the failure stays in ``warning``."""
    from tools import skills_tool
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
