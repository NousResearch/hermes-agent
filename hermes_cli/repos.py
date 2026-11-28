"""``hermes repos`` — one inventory of repo-backed installs.

The read commands (``list`` / ``show``) collect live from disk on every call, so
they can never report a stale state; ``sync`` writes the cache that other
consumers read, and ``audit`` compares the two.

This is the read-only first slice. The lifecycle verbs (``install`` / ``enable``
/ ``disable`` / ``uninstall`` / ``update``) are deliberately NOT here yet — the
rows already carry the exact delegated command under ``manage``, which is what a
follow-up verb would shell out to.
"""

from __future__ import annotations

from typing import Any

from hermes_cli import repos_provenance as prov
from hermes_cli import repos_registry as reg

_SURFACES = (
    prov.SURFACE_PLUGIN,
    prov.SURFACE_SKILL,
    prov.SURFACE_MCP,
    prov.SURFACE_CLONE,
)


def _console():
    from hermes_cli.plugins_cmd import _console as console

    return console()


def _live_rows() -> list[dict[str, Any]]:
    return [install.to_row() for install in prov.collect_installs()]


def _filter_rows(rows: list[dict[str, Any]], args: Any) -> list[dict[str, Any]]:
    surface = getattr(args, "surface", None)
    if surface:
        rows = [row for row in rows if row["surface"] == surface]
    if getattr(args, "enabled", False):
        rows = [row for row in rows if row["status"] == prov.STATUS_ENABLED]
    if getattr(args, "disabled", False):
        rows = [row for row in rows if row["status"] != prov.STATUS_ENABLED]
    return rows


def cmd_list(args: Any | None = None) -> None:
    """List repo-backed installs across every surface."""
    rows = _filter_rows(_live_rows(), args)
    if getattr(args, "json", False):
        print(_dumps(rows))
        return
    if getattr(args, "plain", False):
        for row in rows:
            repo = row["repo"] or "-"
            sha = (row["sha"] or "-")[:10]
            print(
                f"{row['status']:12} {row['surface']:7} {sha:10} {row['name']:28} {repo}"
            )
        return
    if not rows:
        _console().print("[dim]No repo-backed installs found.[/dim]")
        _console().print("[dim]Rebuild the cache:[/dim] hermes repos sync")
        return
    table = _table(
        (
            ("Name", "bold"),
            ("Status", None),
            ("Surface", "dim"),
            ("SHA", "dim"),
            ("Repo", None),
            ("Provenance", "dim"),
        ),
        title="Installed repos",
        show_lines=False,
    )
    for row in rows:
        sha = row["sha"] or "-"
        table.add_row(
            row["name"],
            row["status"],
            row["surface"],
            sha[:10],
            row["repo"] or "-",
            row["provenance"],
        )
    _console().print()
    _console().print(table)
    _console().print()
    _console().print("[dim]Detail:[/dim] hermes repos show <name>")
    _console().print("[dim]Rebuild cache:[/dim] hermes repos sync")


def cmd_show(name: str) -> None:
    """Show every row whose name or id matches *name*."""
    rows = [row for row in _live_rows() if name in (row["name"], row["id"])]
    if not rows:
        _console().print(f"[red]No installed repo matching '{name}'.[/red]")
        _console().print("[dim]List everything:[/dim] hermes repos list")
        return
    console = _console()
    for row in rows:
        console.print()
        console.print(f"[bold]{row['id']}[/bold]")
        console.print(f"[dim]Status:[/dim] {row['status']}")
        console.print(f"[dim]Repo:[/dim] {row['repo'] or '-'}")
        console.print(
            f"[dim]Ref:[/dim] {row['ref'] or '-'}  [dim]SHA:[/dim] {row['sha'] or '-'}"
        )
        console.print(f"[dim]Provenance:[/dim] {row['provenance']}")
        console.print(f"[dim]Path:[/dim] {row['path'] or '-'}")
        if row.get("manage"):
            console.print(
                "[dim]Manage:[/dim] "
                + ", ".join(f"{verb} -> {cmd}" for verb, cmd in row["manage"].items())
            )
    console.print()


def cmd_sync(args: Any | None = None) -> None:
    """Rebuild the registry cache from disk (no network)."""
    rows = _live_rows()
    path = reg.write_registry(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["surface"]] = counts.get(row["surface"], 0) + 1
    summary = ", ".join(
        f"{surface}={counts[surface]}" for surface in _SURFACES if counts.get(surface)
    )
    _console().print(f"[green]Synced {len(rows)} installs[/green] ({summary})")
    _console().print(f"[dim]Registry:[/dim] {path}")
    _console().print("[dim]Drift report:[/dim] hermes repos audit")


def cmd_audit(args: Any | None = None) -> None:
    """Report drift between the cached registry and what is on disk now."""
    cached = reg.read_registry()["installs"]
    live = _live_rows()
    cached_by_id = {row.get("id"): row for row in cached}
    live_ids = {row["id"] for row in live}
    missing = sorted(set(cached_by_id) - live_ids)
    added = sorted(live_ids - set(cached_by_id))
    changed = sorted(
        row["id"]
        for row in live
        if row["id"] in cached_by_id
        and _significant(cached_by_id[row["id"]]) != _significant(row)
    )
    console = _console()
    if not (missing or added or changed):
        console.print("[green]No drift:[/green] the registry matches disk.")
        return
    console.print(
        f"[yellow]Drift detected:[/yellow] "
        f"{len(added)} new, {len(missing)} gone, {len(changed)} changed"
    )
    for row_id in added:
        console.print(f"  [green]+[/green] {row_id}")
    for row_id in missing:
        console.print(f"  [red]-[/red] {row_id}")
    for row_id in changed:
        console.print(f"  [yellow]~[/yellow] {row_id}")
    console.print("[dim]Rebuild:[/dim] hermes repos sync")


def _significant(row: dict[str, Any]) -> tuple:
    """The fields drift cares about — timestamps and paths are noise."""
    return (row.get("repo"), row.get("sha"), row.get("status"), row.get("provenance"))


def _dumps(rows: list[dict[str, Any]]) -> str:
    import json

    return json.dumps(rows, indent=2)


def _table(columns, **kwargs):
    from hermes_cli.plugins_cmd import _table as table

    return table(columns, **kwargs)


def repos_command(args: Any) -> Any:
    """Dispatch a parsed ``hermes repos`` invocation."""
    action = getattr(args, "repos_action", None) or "list"
    if action == "list":
        cmd_list(args)
    elif action == "show":
        cmd_show(args.name)
    elif action == "sync":
        cmd_sync(args)
    elif action == "audit":
        cmd_audit(args)
    else:
        _console().print(f"[red]Unknown repos action: {action}[/red]")
        return 1
    return 0
