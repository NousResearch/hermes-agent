"""``hermes plugins update`` with no name: update every installed plugin that has an update.

The list comes from the read-only ``check-updates`` probe (:func:`hermes_cli.plugins_updates.run_checks`);
each plugin is then updated through the single-plugin
path (:func:`~hermes_cli.plugins_cmd_update.update_one`), so catalog consent, the security scan and
capability re-consent still apply per plugin.

Sibling of :mod:`hermes_cli.plugins_cmd` (the facade is imported late, never at module level).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

# Only these classes have an update path `hermes plugins update <name>` can drive: a catalog install
# re-pins to the reviewed sha, a tracked git install pulls. A pip row is a PyPI distribution that
# Hermes never upgrades (#132494), so offering it would promise an update nothing applies.
_UPDATE_KIND = {"catalog": "catalog re-pin", "git": "git pull"}


def _pc():
    from hermes_cli import plugins_cmd
    return plugins_cmd


@dataclass
class UpdatePlan:
    """What ``update`` with no name would do: *movable* rows it updates, *skipped* ``(row, reason)``
    pairs it cannot move, *current* the count already up to date."""

    movable: list[Any] = field(default_factory=list)
    skipped: list[tuple[Any, str]] = field(default_factory=list)
    current: int = 0


def plan_update_all(results: list[Any]) -> UpdatePlan:
    """Split ``run_checks`` rows into the plugins ``update`` can move and the ones it cannot (with why)."""
    plan = UpdatePlan()
    for row in results:
        if row.needs_fixing:
            plan.skipped.append((row, f"needs fixing: {row.needs_fixing}"))
        elif row.update_available is True and row.klass in _UPDATE_KIND:
            plan.movable.append(row)
        elif row.update_available is True:
            plan.skipped.append((row, f"{row.klass} distribution; Hermes does not upgrade it"))
        elif row.update_available is None:
            plan.skipped.append((row, row.reason or "update status unknown"))
        elif row.reason:  # up to date by construction but worth naming (pinned)
            plan.skipped.append((row, row.reason))
        else:
            plan.current += 1
    return plan


def _short(reason: str) -> str:
    first = reason.strip().splitlines()[0] if reason.strip() else ""
    return first if len(first) <= 200 else first[:197] + "..."


def _update_one(row, console, *, interactive: bool) -> tuple[str, str]:
    """Run the single-plugin update for one plan row; return ``(status, detail)``, status being
    updated/current/failed. A refusal is printed like ``update <name>`` prints it, then recorded."""
    from rich.markup import escape
    from hermes_cli.plugins_cmd_catalog import RepinConsentRequired
    from hermes_cli.plugins_cmd_update import update_one
    target = _pc()._user_installed_plugin_dir(row.name)
    if target is None:
        return "failed", "no longer installed"
    try:
        moved = update_one(row.name, target, console, interactive=interactive)
    except RepinConsentRequired as exc:
        console.print(f"[yellow]Update of {escape(row.name)} not applied:[/yellow] {escape(str(exc))}")
        return "failed", _short(f"not applied: {exc}")
    except _pc().PluginOperationError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        return "failed", _short(str(exc))
    if not moved:
        return "current", ""
    return "updated", f"{(row.current or '')[:8]}..{(row.latest or '')[:8]}"


def _print_plan(console, plan: UpdatePlan) -> None:
    from rich.markup import escape
    table = _pc()._table([("#", "dim"), ("Plugin", "bold"), ("Current", ""), ("Latest", ""), ("Update", "")],
                         title="Plugin updates available")
    for i, row in enumerate(plan.movable, 1):
        table.add_row(str(i), row.name, (row.current or "-")[:12], (row.latest or "-")[:12], _UPDATE_KIND[row.klass])
    console.print()
    console.print(table)
    for row, reason in plan.skipped:
        console.print(f"  [dim]skipped {escape(row.name)} ({row.klass}): {escape(reason)}[/dim]")


def cmd_update_all(*, yes: bool = False) -> None:
    """``hermes plugins update [--yes]``: list the plugins with an update and update them all.

    A terminal asks once (``Update all N? [Y/n]``); without one nothing moves unless ``--yes`` is
    given. Each plugin then goes through ``update_one``, interactive only when a terminal is present,
    so per-plugin consent prompts still appear there and stay closed (fail closed) under ``--yes``
    in scripts. One failure does not stop the rest; exit 1 when any update failed.
    """
    from rich.markup import escape
    from hermes_cli.plugins_updates import run_checks

    console = _pc()._console()
    console.print("[dim]Checking installed plugins for updates...[/dim]")
    plan = plan_update_all(run_checks(_pc()._plugins_dir()))
    if not plan.movable:
        for row, reason in plan.skipped:
            console.print(f"  [dim]skipped {escape(row.name)} ({row.klass}): {escape(reason)}[/dim]")
        console.print("[green]✓[/green] No plugin updates to apply.")
        return
    _print_plan(console, plan)
    count = len(plan.movable)
    tty = _pc()._is_tty()
    if not yes:
        if not tty:
            _pc()._fail(console, f"[yellow]Not updating: no terminal to confirm. Re-run with --yes to update "
                                 f"these {count} plugin(s).[/yellow]")
        try:
            answer = input(f"\nUpdate all {count}? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer not in ("", "y", "yes"):
            console.print("Cancelled. Nothing was updated.")
            return

    outcomes = []
    for row in plan.movable:
        console.print(f"\n[bold]── {row.name}[/bold]")
        outcomes.append((row.name, *_update_one(row, console, interactive=tty)))

    console.print("\n[bold]Summary[/bold]")
    marks = {"updated": "[green]✓ updated[/green]", "current": "[dim]= already current[/dim]",
             "failed": "[red]✗ failed[/red]"}
    for name, status, detail in outcomes:
        console.print(f"  {marks[status]}  {escape(name)}" + (f"  [dim]{escape(detail)}[/dim]" if detail else ""),
                      highlight=False)
    if any(status == "failed" for _, status, _ in outcomes):
        sys.exit(1)
