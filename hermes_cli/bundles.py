"""Implementation of the ``hermes bundles`` CLI subcommand."""

from __future__ import annotations

import sys
from typing import List

from rich.console import Console
from rich.table import Table

from hermes_cli.cli_output import line_input

from agent.skill_bundles import (
    _bundles_dir, delete_bundle, get_bundle, list_bundles, reload_bundles, save_bundle, scan_bundles
)


def _console() -> Console:
    return Console()


def _fail(c: Console, message: str) -> None:
    c.print(message)
    sys.exit(1)


def _cmd_list(args) -> None:
    c = _console()
    bundles = list_bundles()
    if not bundles:
        c.print(
            f"[dim]No bundles installed yet. Create one with:\n"
            f"  hermes bundles create <name> --skill skill1 --skill skill2[/]\n"
            f"Bundles directory: [bold]{_bundles_dir()}[/]")
        return

    table = Table(title=f"Skill Bundles ({len(bundles)})", show_lines=False)
    table.add_column("Command", style="bold cyan")
    table.add_column("Name", style="bold")
    table.add_column("Skills", justify="right")
    table.add_column("Description")

    for info in bundles:
        table.add_row(
            f"/{info['slug']}",
            info["name"],
            str(len(info.get("skills", []))),
            info.get("description") or "")
    c.print(table)
    c.print(f"\n[dim]Bundles directory: {_bundles_dir()}[/]")


def _cmd_show(args) -> None:
    c = _console()
    info = get_bundle(args.name)
    if not info:
        _fail(c, f"[bold red]Bundle {args.name!r} not found.[/]")
    c.print(f"[bold cyan]/{info['slug']}[/]  [bold]{info['name']}[/]")
    if info.get("description"):
        c.print(f"  {info['description']}")
    c.print(f"  [dim]File: {info['path']}[/]")
    c.print(f"  [bold]Skills ({len(info['skills'])}):[/]")
    for s in info["skills"]:
        c.print(f"    - {s}")
    if info.get("instruction"):
        c.print(f"  [bold]Instruction:[/]\n    {info['instruction']}")


def _cmd_create(args) -> None:
    c = _console()
    name = args.name
    skills: List[str] = list(args.skill or [])
    if not skills:
        # Interactive prompt for skills if none were passed on the CLI.
        c.print(
            "[dim]No skills passed via --skill. Enter one skill name per line.\n"
            "Submit an empty line to finish.[/]")
        try:
            while True:
                line = line_input("skill> ").strip()
                if not line:
                    break
                skills.append(line)
        except (EOFError, KeyboardInterrupt):
            _fail(c, "\n[yellow]Cancelled.[/]")
    if not skills:
        _fail(c, "[bold red]A bundle must reference at least one skill.[/]")

    try:
        path = save_bundle(
            name, skills, description=args.description or "", instruction=args.instruction or "",
            overwrite=bool(args.force))
    except FileExistsError as exc:
        _fail(c, f"[bold red]{exc}[/]\n[dim]Pass --force to overwrite.[/]")
    except ValueError as exc:
        _fail(c, f"[bold red]{exc}[/]")

    c.print(f"[bold green]Created bundle:[/] {path}")
    info = get_bundle(name)
    if info:
        c.print(
            f"  Invoke with: [bold cyan]/{info['slug']}[/]  "
            f"(loads {len(info['skills'])} skills)")


def _cmd_delete(args) -> None:
    c = _console()
    try:
        path = delete_bundle(args.name)
    except FileNotFoundError as exc:
        _fail(c, f"[bold red]{exc}[/]")
    c.print(f"[bold green]Deleted bundle:[/] {path}")


def _cmd_reload(args) -> None:
    c = _console()
    diff = reload_bundles()
    if diff["added"]:
        c.print(f"[bold green]Added ({len(diff['added'])}):[/]")
        for entry in diff["added"]:
            c.print(f"  + {entry['name']} — {entry.get('description', '')}")
    if diff["removed"]:
        c.print(f"[bold red]Removed ({len(diff['removed'])}):[/]")
        for entry in diff["removed"]:
            c.print(f"  - {entry['name']}")
    if not diff["added"] and not diff["removed"]:
        c.print(f"[dim]No changes. {diff['total']} bundle(s) loaded.[/]")
    else:
        c.print(f"[dim]Total bundles now: {diff['total']}[/]")


def register_cli(subparser) -> None:
    """Build the ``hermes bundles`` argparse tree (called from hermes_cli/main.py, which owns the
    top-level subparser)."""
    subs = subparser.add_subparsers(dest="bundles_action")

    p_list = subs.add_parser("list", help="List installed skill bundles")
    p_list.set_defaults(_bundles_handler=_cmd_list)

    p_show = subs.add_parser("show", help="Show one bundle's contents")
    p_show.add_argument("name", help="Bundle name")
    p_show.set_defaults(_bundles_handler=_cmd_show)

    p_create = subs.add_parser(
        "create",
        help="Create a new skill bundle",
        description=(
            "Create a new bundle. Skills can be passed via --skill (repeat for "
            "multiple) or entered interactively when omitted."))
    p_create.add_argument("name", help="Bundle name (becomes the /slash command)")
    p_create.add_argument(
        "--skill", "-s", action="append", default=[],
        help="Skill name to include (repeat for multiple)")
    p_create.add_argument(
        "--description", "-d", default="",
        help="Human-readable description shown in /help and `hermes bundles list`")
    p_create.add_argument(
        "--instruction", "-i", default="",
        help="Extra guidance prepended to the loaded skill content")
    p_create.add_argument(
        "--force", "-f", action="store_true", help="Overwrite an existing bundle with the same name"
    )
    p_create.set_defaults(_bundles_handler=_cmd_create)

    p_delete = subs.add_parser("delete", help="Delete a skill bundle")
    p_delete.add_argument("name", help="Bundle name")
    p_delete.set_defaults(_bundles_handler=_cmd_delete)

    p_reload = subs.add_parser("reload", help="Re-scan the bundles directory and report changes")
    p_reload.set_defaults(_bundles_handler=_cmd_reload)

    # Ensure a fresh scan when any bundles subcommand runs.
    scan_bundles()


def bundles_command(args) -> None:
    """Dispatch ``hermes bundles <subcommand>`` to the right handler."""
    handler = getattr(args, "_bundles_handler", None) or _cmd_list  # no subcommand → list
    handler(args)
