"""``hermes plugins update`` plus the provenance verbs around it: ``adopt``, ``trust-update-url`` and the
read-only ``check-updates``; the dashboard update path shares the same pull/re-clone core.

Sibling of :mod:`hermes_cli.plugins_cmd` (the facade re-exports the names other modules use and is
imported late here, never at module level).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any


def _pc():
    """The facade, read at call time: tests patch ``plugins_cmd.<name>`` and sibling calls must see it."""
    from hermes_cli import plugins_cmd
    return plugins_cmd


def _pull_plugin_update(target: Path, pinned_msg, not_git_msg, before_pull=None, *, interactive: bool = False) -> str:
    """Shared ``update`` core: refuse pinned checkouts, ``git pull`` (or re-install from the
    recorded source when the tree carries no ``.git`` — subdirectory installs), record the new
    revision. Returns the pull output; raises :class:`PluginOperationError` on any refusal.
    *pinned_msg(install_record)* / *not_git_msg()* build the caller-specific error text."""
    metadata = _pc()._read_install_metadata()
    install_record = metadata.get(target.name, {})
    if install_record.get("pinned") is True:
        raise _pc().PluginOperationError(pinned_msg(install_record))
    # A URL install whose name/repo later landed on the kill list must not keep pulling or
    # re-cloning new code, including subdirectory installs that carry no local .git directory.
    from hermes_cli import plugins_cmd_catalog as catalog
    catalog.refuse_if_installed_removed(target.name, target)
    if not (target / ".git").exists():
        source = install_record.get("source")
        if not isinstance(source, str) or not source:
            raise _pc().PluginOperationError(not_git_msg())
        if before_pull is not None:
            before_pull()
        return _reclone_plugin_update(target, source, install_record.get("revision"))
    if before_pull is not None:
        before_pull()
    from hermes_cli.plugins_transaction import update_plugin

    return update_plugin(target, interactive=interactive)


def _reclone_plugin_update(target: Path, source: str, previous_revision: object) -> str:
    """Update a plugin whose tree is not a git checkout: a subdirectory install ships only
    ``<clone>/<subdir>``, so the ``.git`` stays in the temp clone (#65314). Re-run the install
    from the recorded source (same URL, same subdir) and swap the fresh tree in, carrying the
    user's files across (:class:`~hermes_cli.plugins_cmd_catalog._UserFileCarry`); the metadata
    revision is rewritten by the installer. Returns pull-shaped output for the callers."""
    from hermes_cli.plugins_cmd_catalog import _UserFileCarry
    previous = previous_revision if isinstance(previous_revision, str) else ""
    carry = _UserFileCarry(
        target, _pc()._plugins_dir().parent / "plugins-backup" / f"{target.name}-{previous[:8] or 'old'}")
    new_target, _manifest, _name = _pc()._install_plugin_core(
        source, force=True, before_swap=lambda _manifest, tree: carry(tree))
    revision = str(_pc()._read_install_metadata().get(new_target.name, {}).get("revision") or "")
    if revision and revision == previous:
        return "\n".join(["Already up to date.", *carry.warnings()])
    return "\n".join([f"Re-installed from {source}: {previous[:8]}..{revision[:8]}", *carry.warnings()])


def cmd_update(name: str, *, interactive: bool = True) -> None:
    """Update an installed plugin by pulling latest from its git remote."""
    from rich.markup import escape
    from hermes_cli import plugins_cmd_catalog as catalog
    console = _pc()._console()
    target = _pc()._require_installed_plugin(name, _pc()._plugins_dir(), console)
    sidecar = catalog.catalog_install_record(target)
    if sidecar:  # catalog installs re-pin to the reviewed SHA — never `git pull`
        catalog.cmd_update_catalog(name, target, sidecar, console, interactive=interactive)
        return
    try:
        output = _pull_plugin_update(
            target,
            lambda rec: (
                f"Plugin '{name}' is pinned to {rec.get('revision')}. To move it, run "
                f"`hermes plugins install {escape(str(rec.get('source', '<source>')))} --force "
                "--ref <40-character commit SHA>`."),
            lambda: f"Plugin '{name}' was not installed from git (no .git directory). Cannot update.",
            before_pull=lambda: console.print(f"[dim]Updating {name}...[/dim]"),
            interactive=interactive)
    except _pc().PluginOperationError as exc:
        _pc()._fail(console, f"[red]Error:[/red] {exc}")
    _post_pull_housekeeping(target, console)

    # Update-time re-consent (#64228): if the new version declares
    # capabilities the granted set lacks, surface the diff and require
    # re-consent for the additions. The stored consent hash detects a
    # changed declaration; additions stay ungranted until the user says yes
    # (non-interactive updates leave them ungranted — fail closed).
    updated_manifest = _pc()._read_manifest(target)
    plugin_id = updated_manifest.get("name") or target.name
    declared_caps = _pc()._declared_capabilities_from_manifest(updated_manifest, plugin_id)
    if declared_caps:
        from hermes_cli.plugin_capabilities import declared_set_changed, pending_capabilities
        if pending_capabilities(plugin_id, declared_caps) or declared_set_changed(plugin_id, declared_caps):
            if interactive:
                _pc()._run_capability_consent(console, plugin_id, declared_caps, context="update")
            else:
                console.print(f"[yellow]Plugin {plugin_id} has new capabilities; review them with `hermes plugins capabilities {plugin_id}`.[/yellow]")

    out = output.strip()
    if "Already up to date" in out:
        console.print(f"[green]✓[/green] Plugin [bold]{name}[/bold] is already up to date.")
    else:
        console.print(f"[green]✓[/green] Plugin [bold]{name}[/bold] updated.")
        console.print(f"[dim]{out}[/dim]")


def _post_pull_housekeeping(target: Path, console) -> None:
    """After publication: drop stale bytecode and copy any new example files."""
    # Same stale-bytecode class as the main checkout (#6207/#60242): the pull just changed .py files under
    # this plugin dir, so drop any __pycache__ compiled from the previous revision.
    _clear_plugin_bytecode(target)
    _pc()._copy_example_files(target, console)


def cmd_adopt(name: str) -> None:
    """Adopt a self-cloned plugin dir into the provenance sidecar.

    Reads the dir's git origin URL, validates it, writes the sidecar row
    — from then on a normal git install (check-updates + update). The
    ONLY mutation path for self-cloned dirs (settled: explicit verbs).
    """
    from rich.console import Console

    console = Console()
    plugins_dir = _pc()._plugins_dir()
    target = _pc()._require_installed_plugin(name, plugins_dir, console)

    from hermes_cli.plugins_provenance import ProvenanceClass, plugins_provenance

    prov = next((p for p in plugins_provenance(plugins_dir) if p.name == target.name), None)
    if prov is None:
        console.print(f"[red]Error:[/red] Plugin '{name}' not classifiable.")
        sys.exit(1)
    if prov.klass is not ProvenanceClass.SELF_CLONED:
        console.print(
            f"[red]Error:[/red] Plugin '{name}' is {prov.klass.value}, not "
            "self-cloned — there is nothing to adopt."
        )
        sys.exit(1)
    if not prov.origin_url:
        console.print(
            f"[red]Error:[/red] Plugin '{name}' has no readable git origin "
            "remote. Add one (git remote add origin <url>) and retry."
        )
        sys.exit(1)

    try:
        _pc()._resolve_git_url(prov.origin_url)
    except ValueError as e:
        console.print(f"[red]Error:[/red] The dir's origin url is not installable: {e}")
        sys.exit(1)

    metadata = _pc()._read_install_metadata()
    if target.name in metadata:
        console.print(f"[red]Error:[/red] Plugin '{name}' already has a provenance row.")
        sys.exit(1)

    git_exe = _pc()._resolve_git_executable()
    revision = _pc()._git_head_revision(target, git_exe) if git_exe else ""
    metadata[target.name] = {
        "pinned": False,
        "revision": revision,
        "source": _pc()._canonical_source(prov.origin_url, None),
    }
    _pc()._write_install_metadata(metadata)
    console.print(
        f"[green]✓[/green] Adopted [bold]{name}[/bold] "
        f"(source: {prov.origin_url}, revision: {revision[:12] or 'unknown'}). "
        "It is now a tracked git install."
    )


def cmd_trust_update_url(name: str) -> None:
    """The ONLY path that moves a saved update_url tag.

    A needs-fixing mismatch (manifest update_url vs the saved tag) is
    resolved here: confirms the manifest's url into the sidecar row,
    prints old → new. Refuses when there is nothing to trust.
    """
    from rich.console import Console

    console = Console()
    plugins_dir = _pc()._plugins_dir()
    target = _pc()._require_installed_plugin(name, plugins_dir, console)

    from hermes_cli.plugins_provenance import read_sidecar_rows

    rows = read_sidecar_rows(plugins_dir)
    row = rows.get(target.name)
    if not isinstance(row, dict):
        console.print(
            f"[red]Error:[/red] Plugin '{name}' has no provenance row — "
            "nothing to trust. Reinstall it instead."
        )
        sys.exit(1)

    saved = row.get("update_url") or None
    manifest = _pc()._read_manifest(target)
    claimed = (manifest or {}).get("update_url") or None
    if claimed == saved:
        console.print(
            f"[yellow]Nothing to trust:[/yellow] '{name}' has no "
            "update_url mismatch."
        )
        return
    if claimed is not None:
        from hermes_cli.plugins_updates import https_update_url
        try:
            claimed = https_update_url(claimed)
        except ValueError as exc:
            console.print(f"[red]Error:[/red] Plugin '{name}' {exc}. Not trusted.")
            sys.exit(1)

    row["update_url"] = claimed
    rows[target.name] = row
    _pc()._write_install_metadata(rows)
    console.print(
        f"[green]✓[/green] Trusted [bold]{name}[/bold] update_url:\n"
        f"  old: {saved or '(none)'}\n"
        f"  new: {claimed or '(none)'}"
    )


def cmd_check_updates(args: Any | None = None) -> None:
    """Read-only: is any installed plugin outdated? NEVER mutates."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    plugins_dir = _pc()._plugins_dir()

    from hermes_cli.plugins_updates import run_checks

    results = run_checks(plugins_dir)

    if getattr(args, "json", False):
        print(json.dumps([r.to_json() for r in results], indent=2))
        return

    table = Table(title="Plugin updates", show_lines=False)
    table.add_column("Name", style="bold")
    table.add_column("Class", style="dim")
    table.add_column("Current")
    table.add_column("Latest")
    table.add_column("Status")
    for r in results:
        if r.needs_fixing:
            status = f"[red]needs fixing[/red]\n[dim]{r.needs_fixing}[/dim]"
        elif r.update_available is True:
            status = "[green]update available[/green]"
        elif r.update_available is False:
            status = "[dim]up to date[/dim]"
        else:
            status = f"[yellow]unknown[/yellow]\n[dim]{r.reason}[/dim]"
        table.add_row(
            r.name, r.klass, (r.current or "-")[:12], r.latest or "-", status
        )
    console.print()
    console.print(table)
    console.print()
    console.print("[dim]Check-only. Apply with: hermes plugins update <name>[/dim]")


def dashboard_update_user_plugin(name: str, *, accept_capabilities: bool = False) -> dict[str, Any]:
    """``git pull`` inside ``~/.hermes/plugins/<name>``; catalog installs re-pin instead. A re-pin that
    widens the plugin returns ``{"ok": False, "consent_required": True, "delta": {...}}`` with nothing
    changed — the surface shows the delta and retries with *accept_capabilities*."""
    from hermes_cli import plugins_cmd_catalog as catalog
    target = _pc()._user_installed_plugin_dir(name)
    if target is None:
        return {"ok": False, "error": f"Plugin '{name}' was not found under {_pc()._plugins_dir()}."}
    sidecar = catalog.catalog_install_record(target)
    try:
        if sidecar:
            result = catalog.repin_catalog_plugin(
                target, sidecar, consent_cb=(lambda _delta: True) if accept_capabilities else None)
            warnings = list(result.warnings)
            new_target = target.parent / result.installed_name
            deps = _pc()._python_dependency_summary(new_target, warnings) if result.changed else []
            from hermes_cli.plugins_activation import activate_plugin_now
            activated = activate_plugin_now(result.installed_name) if result.changed else {}
            return {"ok": True, "name": result.installed_name, "sha": result.sha, "unchanged": not result.changed,
                    "python_dependencies": deps, "warnings": warnings, **activated}
        msg = _pull_plugin_update(
            target,
            lambda rec: (
                f"Plugin '{name}' is pinned to {rec.get('revision')}; "
                f"run `hermes plugins install {rec.get('source', '<source>')} --force "
                "--ref <40-character commit SHA>` to move it."),
            lambda: f"Plugin '{name}' is not a git checkout; cannot pull updates.")
    except catalog.RepinConsentRequired as exc:
        return {"ok": False, "consent_required": True, "error": str(exc), "name": exc.name, "sha": exc.sha,
                "delta": exc.delta, "delta_lines": catalog.surface_delta_lines(exc.delta)}
    except _pc().PluginOperationError as exc:
        return {"ok": False, "error": str(exc)}
    _post_pull_housekeeping(target, _pc()._console())
    return {"ok": True, "name": name, "output": msg, "unchanged": "Already up to date" in msg}


def _clear_plugin_bytecode(target: Path) -> int:
    """Remove ``__pycache__`` dirs under a just-updated plugin checkout. Plugin dirs sit outside
    the repo, so the launch-time bytecode sweep never covers them and stale bytecode after a pull
    can ImportError in the next process. Never raises.

    See #60242, #6207.
    """
    removed = 0
    try:
        for cache_dir in target.rglob("__pycache__"):
            if cache_dir.is_dir():
                shutil.rmtree(cache_dir, ignore_errors=True)
                removed += 0 if cache_dir.exists() else 1
    except OSError:
        pass
    return removed
