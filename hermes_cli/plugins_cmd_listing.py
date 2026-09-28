"""Read-only presentation: ``hermes plugins list``, ``show`` and ``compat``.

Sibling of :mod:`hermes_cli.plugins_cmd` (the facade re-exports the names other modules use and is
imported late here, never at module level).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _pc():
    """The facade, read at call time: tests patch ``plugins_cmd.<name>`` and sibling calls must see it."""
    from hermes_cli import plugins_cmd
    return plugins_cmd


def _filter_plugin_entries(entries: list, args: Any, enabled: set, disabled: set) -> list:
    """Apply ``hermes plugins list`` CLI filters."""
    filtered = entries
    if getattr(args, "no_bundled", False) or getattr(args, "user", False):
        filtered = [entry for entry in filtered if entry[3] != "bundled"]
    if getattr(args, "enabled", False):
        active = _pc()._category_active_names()
        filtered = [entry for entry in filtered if _entry_status(entry, enabled, disabled, active) == "enabled"]
    return filtered


def _entry_status(entry: tuple, enabled: set, disabled: set, active: set) -> str:
    """``_plugin_status`` for one ``_discover_all_plugins`` row."""
    name, _version, _description, source, dir_path, key = entry
    return _pc()._plugin_status(name, enabled, disabled, key, source=source, dir_path=dir_path, active=active)


_STATUS_MARKUP = {"disabled": "[red]disabled[/red]", "enabled": "[green]enabled[/green]"}


def cmd_list(args: Any | None = None) -> None:
    """List all plugins (bundled + user) with enabled/disabled state."""
    console = _pc()._console()
    entries = _pc()._discover_all_plugins()
    if not entries:
        console.print("[dim]No plugins installed.[/dim]")
        console.print("[dim]Install with:[/dim] hermes plugins install owner/repo")
        return

    enabled = _pc()._get_enabled_set()
    disabled = _pc()._get_disabled_set()
    entries = _filter_plugin_entries(entries, args, enabled, disabled)
    from hermes_cli import plugins_cmd_catalog as catalog
    # Source shows catalog provenance (``catalog:<tier>@<sha8>``) or a ``--ref`` pin
    # (``git pinned@<sha8>``) so a team can eyeball that everyone runs the same commit.
    pins = _pc()._read_install_metadata()
    # One kill-list resolution for the whole listing: resolving per row costs a live-catalog
    # fetch per installed plugin when the catalog host is slow or unreachable.
    removed_entries = catalog.resolved_removed_entries()
    active = _pc()._category_active_names()
    rows = [
        (name, _pc()._plugin_status(name, enabled, disabled, key=key, source=source, dir_path=_dir, active=active),
         str(version), description,
         catalog.catalog_annotation(_dir) or _pc()._pin_annotation(name, pins) or source,
         catalog.removed_annotation(name, _dir, removed_entries))
        for name, version, description, source, _dir, key in entries
    ]

    if getattr(args, "json", False):
        keys = ("name", "status", "version", "description", "source", "removed")
        print(json.dumps([dict(zip(keys, row)) for row in rows], indent=2))
        return

    if getattr(args, "plain", False):
        for name, status, version, _description, source, _removed in rows:
            print(f"{status:12} {source:8} {version:8} {name}")
        return

    if not entries:
        console.print("[dim]No plugins matched the selected filters.[/dim]")
        return

    table = _pc()._table(
        (("Name", "bold"), ("Status", None), ("Version", "dim"), ("Description", None), ("Source", "dim")),
        title="Plugins", show_lines=False)
    # provenance class per user-installed dir (bundled entries show '-')
    from hermes_cli.plugins_provenance import plugins_provenance

    prov_classes = {
        p.name: p.klass.value for p in plugins_provenance(_pc()._plugins_dir())
    }

    removed_lines = []
    for name, status_name, version, description, source, removed in rows:
        klass = prov_classes.get(name)
        status = _STATUS_MARKUP.get(status_name, "[yellow]not enabled[/yellow]")
        if removed:
            name = f"[red]{name} ✗[/red]"
            removed_lines.append(f"[red]✗ {name}[/red] was removed from the plugin catalog: {removed}")
        table.add_row(name, status, version, description, source)
        # class line rides the Source column for user plugins
        if source in {"user", "git"}:
            if klass and klass != "git":
                table.add_row(
                    "", "", "", f"[dim]provenance: {klass}[/dim]", ""
                )
    console.print()
    console.print(table)
    for line in removed_lines:
        console.print(line)
    console.print()
    console.print("[dim]Compact view:[/dim] hermes plugins list --plain --no-bundled")
    console.print("[dim]Interactive toggle:[/dim] hermes plugins")
    console.print("[dim]Enable/disable:[/dim] hermes plugins enable/disable <name>")
    console.print("[dim]Plugins are opt-in by default — only 'enabled' plugins load.[/dim]")


def cmd_show(name: str) -> None:
    """Show details for a single plugin, including declared emits/listens."""
    console = _pc()._console()
    match = _pc()._find_plugin_entry(name)
    if match is None:
        console.print(f"[red]Plugin '{name}' not found.[/red]")
        _pc()._fail(console, "[dim]List installed plugins:[/dim] hermes plugins list")

    pname, version, description, source, dir_path, key = match
    manifest = _pc()._read_manifest(Path(dir_path)) if dir_path else {}
    emits = manifest.get("emits") or []
    listens = manifest.get("listens") or []
    status = _pc()._plugin_status(pname, _pc()._get_enabled_set(), _pc()._get_disabled_set(), key=key)
    console.print()
    console.print(f"[bold]{pname}[/bold]" + (f" [dim]v{version}[/dim]" if version else ""))
    if description:
        console.print(description)
    console.print(f"[dim]Status:[/dim] {status}")
    console.print(f"[dim]Source:[/dim] {source}")
    console.print(f"[dim]Key:[/dim] {key}")
    console.print("[dim]Emits:[/dim] " + (", ".join(emits) if emits else "[dim](none)[/dim]"))
    console.print("[dim]Listens:[/dim] " + (", ".join(listens) if listens else "[dim](none)[/dim]"))
    console.print()


def cmd_compat(args: Any | None = None) -> None:
    """``hermes plugins compat`` — which installed plugins import paths scheduled for removal, and where."""
    import sys
    from pathlib import Path
    from hermes_cli.plugin_compat import (
        ALLOW_KEY, COMPAT_REMOVAL, compat_report, removal_in_effect, scan_plugin, summary_lines)
    console = _pc()._console()
    path = getattr(args, "path", None)
    if path:
        hits = scan_plugin(Path(path).expanduser().resolve())
        report = {Path(path).name: hits} if hits else {}
    else:
        report = compat_report(force=True)
    if getattr(args, "json", False):
        print(json.dumps({"removal_date": COMPAT_REMOVAL, "in_effect": removal_in_effect(),
                          "plugins": {k: [h.__dict__ for h in v] for k, v in report.items()}}, indent=2))
        sys.exit(1 if report else 0)
    if not report:
        console.print(f"[green]✓ No enabled plugin imports paths scheduled for removal on {COMPAT_REMOVAL}.[/green]")
        return
    head, tail = summary_lines(report)
    console.print(f"[bold {'red' if removal_in_effect() else 'yellow'}]{head}[/]")
    console.print(f"[dim]{tail}[/dim]")
    for name, hits in sorted(report.items()):
        table = _pc()._table(((f"{name}  ({len(hits)} import{'s' if len(hits) != 1 else ''})", "bold"), ("old path", "yellow"), ("new path", "green")),
                       title=None, show_lines=False)
        for h in hits:
            table.add_row(f"{h.file}:{h.line}", h.old, h.new)
        console.print()
        console.print(table)
    console.print()
    console.print(f"[dim]After {COMPAT_REMOVAL} these plugins are not loaded. Update them, or force-load with "
                  f"plugins.{ALLOW_KEY}: true in config.yaml (the old paths still break once the compat layer is reverted).[/dim]")
    sys.exit(1)
