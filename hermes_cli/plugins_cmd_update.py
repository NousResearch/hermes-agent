"""``hermes plugins update`` plus the provenance verbs around it: ``adopt``, ``trust-update-url`` and the
read-only ``check-updates``. :func:`update_installed_plugin` is the one update path every surface calls.

Sibling of :mod:`hermes_cli.plugins_cmd` (the facade re-exports the names other modules use and is
imported late here, never at module level).
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _pc():
    """The facade, read at call time: tests patch ``plugins_cmd.<name>`` and sibling calls must see it."""
    from hermes_cli import plugins_cmd
    return plugins_cmd


def _pull_plugin_update(target: Path, pinned_msg, not_git_msg, before_pull=None, *, interactive: bool = False,
                        force: bool = False, scan_decision_cb=None) -> str:
    """Shared ``update`` core: refuse pinned checkouts, ``git pull`` (or re-install from the
    recorded source when the tree carries no ``.git`` — subdirectory installs), record the new
    revision. Returns the pull output; raises :class:`PluginOperationError` on any refusal.
    *pinned_msg(install_record)* / *not_git_msg()* build the caller-specific error text.
    *force* / *scan_decision_cb* answer the scan findings on the updated tree."""
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
        return _reclone_plugin_update(target, source, install_record.get("revision"),
                                      force=force, scan_decision_cb=scan_decision_cb)
    if before_pull is not None:
        before_pull()
    from hermes_cli.plugins_transaction import update_plugin

    return update_plugin(target, interactive=interactive, force=force, scan_decision_cb=scan_decision_cb)


def _reclone_plugin_update(target: Path, source: str, previous_revision: object, *,
                           force: bool = False, scan_decision_cb=None) -> str:
    """Update a subdirectory install by re-cloning its source and atomically replacing the tree.

    These installs carry no local ``.git``, so the staged replacement also carries user-owned
    config/data from *target* before publication instead of silently deleting it (#122006).
    ``replace=True`` overwrites the tree; *force* stays the user's answer to the scan findings.
    """
    from hermes_cli.plugins_cmd_catalog import _carry_user_files

    new_target, _manifest, _name = _pc()._install_plugin_core(
        source,
        force=force,
        replace=True,
        scan_decision_cb=scan_decision_cb,
        confirm_dangerous=True,
        before_swap=lambda _manifest, tree: _carry_user_files(target, tree, None),
    )
    revision = str(_pc()._read_install_metadata().get(new_target.name, {}).get("revision") or "")
    previous = previous_revision if isinstance(previous_revision, str) else ""
    if revision and revision == previous:
        return "Already up to date."
    return f"Re-installed from {source}: {previous[:8]}..{revision[:8]}"


@dataclass
class UpdateOutcome:
    """What :func:`update_installed_plugin` did: *name*/*target* are the installed name and dir after the
    update (a catalog re-pin can rename), *sha* the revision now installed."""

    name: str
    target: Path
    changed: bool
    sha: str
    output: str
    warnings: list[str] = field(default_factory=list)
    catalog: bool = False


def update_installed_plugin(target: Path, *, force: bool = False, interactive: bool = False,
                            consent_cb=None, scan_decision_cb=None) -> UpdateOutcome:
    """The one update path for every surface (CLI, TUI/Desktop RPC, dashboard REST, gateway auto-apply).

    Catalog installs re-pin to the reviewed catalog sha (never ``git pull``); custom installs pull, or
    re-clone a subdirectory install. *force* is the user's yes to what the update shows: a re-pin that
    widens the plugin and the security scan findings on the updated tree, any verdict included.
    Without it, *consent_cb(delta)* / *scan_decision_cb(scan_result)* may ask; otherwise the update
    raises :class:`~hermes_cli.plugins_cmd_catalog.RepinConsentRequired` or
    :class:`~hermes_cli.plugins_cmd.PluginScanBlocked` with nothing changed. *interactive*: a terminal
    user can consent to newly declared Python dependencies.
    """
    from hermes_cli import plugins_cmd_catalog as catalog
    name = target.name
    sidecar = catalog.catalog_install_record(target)
    if sidecar:
        result = catalog.repin_catalog_plugin(target, sidecar, interactive=interactive, consent_cb=consent_cb,
                                              force=force, scan_decision_cb=scan_decision_cb)
        output = (f"Updated to catalog pin {result.sha[:8]}" if result.changed
                  else f"Already at catalog pin {result.sha[:8]}")
        return UpdateOutcome(result.installed_name, target.parent / result.installed_name, result.changed,
                             result.sha, output, list(result.warnings), catalog=True)
    before = str(_pc()._read_install_metadata().get(name, {}).get("revision") or "")
    output = _pull_plugin_update(
        target,
        lambda rec: (
            f"Plugin '{name}' is pinned to {rec.get('revision')}. To move it, run "
            f"`hermes plugins install {rec.get('source', '<source>')} --force "
            "--ref <40-character commit SHA>`."),
        lambda: f"Plugin '{name}' was not installed from git (no .git directory). Cannot update.",
        interactive=interactive, force=force, scan_decision_cb=scan_decision_cb)
    sha = str(_pc()._read_install_metadata().get(name, {}).get("revision") or "")
    return UpdateOutcome(name, target, sha != before, sha, output.strip())


def scan_finding_lines(scan_result, limit: int = 20) -> list[str]:
    """Scan findings worst first, one line each, for a surface that shows them before a retry with consent."""
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    findings = sorted(scan_result.findings if scan_result is not None else (),
                      key=lambda f: order.get(f.severity, 4))
    lines = [f"{f.severity.upper()} {f.category} {f.file}:{f.line} {f.description}" for f in findings[:limit]]
    if len(findings) > limit:
        lines.append(f"...and {len(findings) - limit} more")
    return lines


def update_consent_payload(name: str, exc: Exception) -> dict[str, Any] | None:
    """The ``consent_required`` answer for an update that needs the user's yes, else None. Nothing
    changed on disk; the surface shows ``delta_lines`` and retries with ``force``."""
    from hermes_cli import plugins_cmd_catalog as catalog
    if isinstance(exc, catalog.RepinConsentRequired):
        return {"ok": False, "consent_required": True, "name": exc.name, "sha": exc.sha, "delta": exc.delta,
                "delta_lines": catalog.surface_delta_lines(exc.delta) + exc.findings, "error": str(exc)}
    if isinstance(exc, _pc().PluginScanBlocked):
        result = exc.scan_result
        verdict = getattr(result, "verdict", "")
        count = len(result.findings) if result is not None else 0
        return {"ok": False, "consent_required": True, "name": name, "delta_lines": scan_finding_lines(result),
                "error": f"The update of '{name}' has {count} security scan finding(s) (verdict: {verdict}). "
                         "Review them and confirm to continue."}
    return None


def cmd_update(name: str, *, interactive: bool = True, force: bool = False) -> None:
    """``hermes plugins update <name> [--force]``: a terminal shows what the update brings and asks;
    ``--force`` accepts it; without either the update is refused with nothing changed."""
    from rich.markup import escape
    console = _pc()._console()
    target = _pc()._require_installed_plugin(name, _pc()._plugins_dir(), console)
    ask = interactive and _pc()._is_tty()
    refused = "[yellow]Not applied. Re-run in a terminal to review it, or pass --force to accept it.[/yellow]"

    def _confirm_widening(delta: dict[str, list[str]]) -> bool:
        from hermes_cli.plugins_cmd_catalog import surface_delta_lines
        console.print(f"\n  [yellow]The new pin of [bold]{name}[/bold] adds:[/yellow]")
        for line in surface_delta_lines(delta):
            console.print(f"    {line}")
        if not ask:
            console.print(f"  {refused}")
            return False
        return _pc()._ask_yes("  Apply this update? [y/N]: ")

    def _confirm_findings(scan_result) -> bool:
        console.print(f"\n[yellow]⚠ Security scan flagged the update (verdict: {scan_result.verdict}):[/yellow]")
        for line in scan_finding_lines(scan_result):
            console.print(f"    {escape(line)}")
        if not ask:
            console.print(refused)
            return False
        return _pc()._ask_yes("  Update anyway? Only continue if you trust the source. [y/N]: ")

    from hermes_cli.plugins_cmd_catalog import RepinConsentRequired
    console.print(f"[dim]Updating {name}...[/dim]")
    try:
        outcome = update_installed_plugin(target, force=force, interactive=interactive,
                                          consent_cb=_confirm_widening, scan_decision_cb=_confirm_findings)
    except (RepinConsentRequired, _pc().PluginScanBlocked):
        # The findings or the widening were already shown above, with the reason it was not applied.
        _pc()._fail(console, f"[yellow]Update of {name} not applied.[/yellow]")
    except _pc().PluginOperationError as exc:
        _pc()._fail(console, f"[red]Error:[/red] {exc}")
    if outcome.changed:
        _post_pull_housekeeping(outcome.target, console)

    # Update-time re-consent (#64228): if the new version declares
    # capabilities the granted set lacks, surface the diff and require
    # re-consent for the additions. The stored consent hash detects a
    # changed declaration; additions stay ungranted until the user says yes
    # (non-interactive updates leave them ungranted — fail closed).
    if outcome.changed or not outcome.catalog:
        updated_manifest = _pc()._read_manifest(outcome.target)
        plugin_id = updated_manifest.get("name") or outcome.name
        declared_caps = _pc()._declared_capabilities_from_manifest(updated_manifest, plugin_id)
        if declared_caps:
            from hermes_cli.plugin_capabilities import declared_set_changed, pending_capabilities
            if pending_capabilities(plugin_id, declared_caps) or declared_set_changed(plugin_id, declared_caps):
                if interactive:
                    _pc()._run_capability_consent(console, plugin_id, declared_caps, context="update")
                else:
                    console.print(f"[yellow]Plugin {plugin_id} has new capabilities; review them with "
                                  f"`hermes plugins capabilities {plugin_id}`.[/yellow]")

    for warning in outcome.warnings:
        console.print(f"[yellow]⚠ {warning}[/yellow]")
    if not outcome.changed:
        console.print(f"[green]✓[/green] Plugin [bold]{outcome.name}[/bold] is already up to date.")
    else:
        console.print(f"[green]✓[/green] Plugin [bold]{outcome.name}[/bold] updated.")
        console.print(f"[dim]{outcome.output}[/dim]")


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


def dashboard_update_user_plugin(name: str, *, accept_capabilities: bool = False,
                                 force: bool = False) -> dict[str, Any]:
    """Non-interactive :func:`update_installed_plugin` for the dashboard and the TUI/Desktop RPC.

    An update that needs the user's yes (a widened catalog pin, or security scan findings on the
    updated tree) returns :func:`update_consent_payload` with nothing changed; the surface shows
    ``delta_lines`` and retries with *force* (*accept_capabilities* is the older name for it).
    """
    target = _pc()._user_installed_plugin_dir(name)
    if target is None:
        return {"ok": False, "error": f"Plugin '{name}' was not found under {_pc()._plugins_dir()}."}
    from hermes_cli.plugins_cmd_catalog import RepinConsentRequired
    try:
        outcome = update_installed_plugin(target, force=force or accept_capabilities)
    except (RepinConsentRequired, _pc().PluginOperationError) as exc:
        return update_consent_payload(target.name, exc) or {"ok": False, "error": str(exc)}
    warnings = list(outcome.warnings)
    if not outcome.changed:
        return {"ok": True, "name": outcome.name, "sha": outcome.sha, "unchanged": True,
                "python_dependencies": [], "warnings": warnings}
    _post_pull_housekeeping(outcome.target, _pc()._console())
    deps = _pc()._python_dependency_summary(outcome.target, warnings)
    from hermes_cli.plugins_activation import activate_plugin_now
    return {"ok": True, "name": outcome.name, "sha": outcome.sha, "unchanged": False,
            "python_dependencies": deps, "warnings": warnings, **activate_plugin_now(outcome.name)}


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
