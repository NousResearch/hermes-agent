"""``hermes plugins`` catalog surface: resolution, provenance sidecar, search/browse/info/validate,
catalog-aware update, plus the dashboard/TUI-facing catalog payload helpers.

Sibling of :mod:`hermes_cli.plugins_cmd` (the installer core, enable/disable state and console helpers
live there and are imported late — this module is imported BY ``plugins_cmd``).
"""

from __future__ import annotations

import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_cli.plugin_catalog import (
    PluginCatalogEntry, entry_capability_summary, filter_entries, find_removed, get_live_catalog_entry,
    load_catalog_live, load_removed_list, _NAME_RE,
)

logger = logging.getLogger(__name__)

CATALOG_SIDECAR = ".hermes-catalog.json"


# ── Resolution / provenance ──────────────────────────────────────────────────

def looks_like_catalog_name(identifier: str) -> bool:
    """Bare ``[a-z0-9_-]`` token — not a URL, ``owner/repo`` or path."""
    from hermes_cli.plugins_cmd import _URL_SCHEMES
    return bool(identifier) and "/" not in identifier and "\\" not in identifier \
        and not identifier.startswith(_URL_SCHEMES) and bool(_NAME_RE.match(identifier))


def raise_if_removed(*candidates: str) -> None:
    """``PluginOperationError`` when any candidate (name or repo URL) is on the kill list."""
    from hermes_cli.plugins_cmd import PluginOperationError
    for candidate in candidates:
        removed = find_removed(candidate)
        if removed is not None:
            detail = removed.reason or "no reason recorded"
            if removed.date:
                detail += f" (removed {removed.date})"
            raise PluginOperationError(
                f"Plugin '{removed.name}' was removed from the Hermes plugin catalog and is blocked from "
                f"installation: {detail}")


def resolve_catalog_name(identifier: str, console) -> PluginCatalogEntry:
    """Bare name → live catalog entry, or exit 1 with a pointer to ``search``."""
    from hermes_cli.plugins_cmd import _fail
    entry = get_live_catalog_entry(identifier)
    if entry is None:
        _fail(console, (
            f"[red]Error:[/red] '{identifier}' is not in the Hermes plugin catalog and is not a Git URL or "
            "owner/repo shorthand. Browse entries with `hermes plugins search`."))
        raise SystemExit(1)  # _fail exits; keeps type-checkers honest
    return entry


def write_catalog_sidecar(target: Path, entry: PluginCatalogEntry) -> None:
    """``.hermes-catalog.json`` inside the install dir — how ``update``/``list``/dashboards know the plugin
    came from the catalog and at which pin."""
    sidecar = {
        "catalog_name": entry.name, "repo": entry.repo, "sha": entry.sha, "tier": entry.tier,
        "installed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    }
    try:
        (target / CATALOG_SIDECAR).write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to write catalog sidecar in %s: %s", target, exc)


def read_catalog_sidecar(plugin_dir) -> Optional[dict]:
    """Parsed sidecar, or ``None`` (absent/corrupt = a non-catalog install)."""
    path = Path(plugin_dir) / CATALOG_SIDECAR if plugin_dir else None
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) and data.get("catalog_name") else None


def catalog_annotation(dir_path) -> Optional[str]:
    """``catalog:<tier>@<sha8>`` for a catalog install (``list`` Source column), else ``None``."""
    sidecar = read_catalog_sidecar(dir_path)
    if not sidecar:
        return None
    return f"catalog:{sidecar.get('tier') or 'community'}@{str(sidecar.get('sha') or '')[:8]}"


def removed_annotation(name: str, dir_path) -> Optional[str]:
    """Kill-list reason when an INSTALLED plugin matches by name, catalog name or repo, else ``None``."""
    sidecar = read_catalog_sidecar(dir_path) or {}
    for candidate in (name, sidecar.get("catalog_name"), sidecar.get("repo")):
        removed = find_removed(str(candidate)) if candidate else None
        if removed is not None:
            return removed.reason or "no reason recorded"
    return None


# ── Catalog-aware install / update ───────────────────────────────────────────

def install_catalog_entry(entry: PluginCatalogEntry, *, force: bool, ref: Optional[str] = None,
                          allow_removed: bool = False, scan_decision_cb=None) -> tuple:
    """``_install_plugin_core`` at the catalog pin (an explicit *ref* wins) + provenance sidecar.
    Returns the core's ``(target, manifest, installed_name)``."""
    from hermes_cli.plugins_cmd import _install_plugin_core
    if not allow_removed:
        raise_if_removed(entry.name, entry.repo)
    target, manifest, installed_name = _install_plugin_core(
        entry.install_identifier, force=force, ref=ref or entry.sha, scan_decision_cb=scan_decision_cb)
    write_catalog_sidecar(target, entry)
    return target, manifest, installed_name


def repin_catalog_plugin(target: Path, sidecar: dict) -> tuple[str, bool]:
    """Re-pin a catalog install to the current catalog SHA (never ``git pull``). Returns
    ``(new_sha, changed)``; raises ``PluginOperationError`` when the entry left the catalog."""
    from hermes_cli.plugins_cmd import PluginOperationError, _get_enabled_set, _save_enabled_set
    catalog_name = str(sidecar["catalog_name"])
    entry = get_live_catalog_entry(catalog_name)
    if entry is None:
        raise PluginOperationError(
            f"Plugin '{catalog_name}' is no longer in the catalog — it may have been removed. "
            "See `hermes plugins info` and the removed blocklist.")
    if str(sidecar.get("sha") or "").strip().lower() == entry.sha:
        return entry.sha, False
    was_enabled = _get_enabled_set()  # the force reinstall must not flip activation state
    install_catalog_entry(entry, force=True)
    _save_enabled_set(was_enabled)
    return entry.sha, True


def cmd_update_catalog(name: str, target: Path, sidecar: dict, console) -> None:
    from hermes_cli.plugins_cmd import PluginOperationError, _fail
    console.print(f"[dim]Checking catalog pin for {name}...[/dim]")
    try:
        sha, changed = repin_catalog_plugin(target, sidecar)
    except PluginOperationError as exc:
        _fail(console, f"[red]Error:[/red] {exc}")
        raise SystemExit(1)
    verb = "updated to" if changed else "is already at catalog pin"
    console.print(f"[green]✓[/green] Plugin [bold]{name}[/bold] {verb} {sha[:8]}.")


# ── search / browse / info / validate ────────────────────────────────────────

def _capability_counts(entry: PluginCatalogEntry) -> str:
    caps = entry.capabilities
    parts = [f"{len(items)} {label}{'s' if len(items) != 1 and label != 'middleware' else ''}"
             for items, label in ((caps.provides_tools, "tool"), (caps.provides_hooks, "hook"),
                                  (caps.provides_middleware, "middleware")) if items]
    if caps.requires_env:
        parts.append(f"{len(caps.requires_env)} env")
    return ", ".join(parts) or "—"


def _render_entries(entries: List[PluginCatalogEntry], console) -> None:
    from hermes_cli.plugins_cmd import _table
    table = _table(((("Name", "bold")), ("Tier", None), ("Description", None), ("Pinned", "dim"),
                    ("Capabilities", "dim")), title="Hermes Plugin Catalog (curated)")
    for e in entries:
        tier = "[cyan]official[/cyan]" if e.tier == "official" else "[magenta]community[/magenta]"
        desc = e.description if len(e.description) <= 60 else e.description[:57] + "..."
        table.add_row(e.name, tier, desc, e.sha[:8], _capability_counts(e))
    console.print()
    console.print(table)
    console.print()
    console.print("[dim]Details:[/dim] hermes plugins info <name>    [dim]Install:[/dim] hermes plugins install <name>")


def cmd_search(term: str = "", *, json_output: bool = False) -> None:
    """Search the curated catalog (name/description/declared tools); empty term = browse everything."""
    from hermes_cli.plugins_cmd import _console
    matches = filter_entries(load_catalog_live(), term)
    if json_output:
        print(json.dumps({"query": term, "results": [e.to_dict() for e in matches]}, indent=2))
        return
    console = _console()
    if not matches:
        console.print(f"[yellow]No catalog entries matched '{term}'[/yellow]" if term
                      else "[dim]No catalog entries available.[/dim]")
        return
    _render_entries(matches, console)


def cmd_info(name: str) -> None:
    """Full catalog entry for *name*; falls back to installed-plugin details for non-catalog names."""
    from hermes_cli.plugins_cmd import _console, cmd_show
    entry = get_live_catalog_entry(name)
    if entry is None:
        cmd_show(name)
        return
    console = _console()
    caps = entry.capabilities
    console.print()
    console.print(f"[bold]{entry.name}[/bold] [cyan]\\[{entry.tier}][/cyan]")
    if entry.description:
        console.print(entry.description)
    console.print()
    rows = [("Repo", entry.repo), ("Subdir", entry.subdir), ("Pinned SHA", entry.sha),
            ("Maintainer", entry.maintainer), ("Requires", f"hermes {entry.requires_hermes}" if entry.requires_hermes else ""),
            ("Platforms", ", ".join(entry.platforms)), ("Docs", entry.docs_url)]
    for label, value in rows:
        if value:
            console.print(f"[dim]{label + ':':<12}[/dim] {value}")
    console.print()
    for label, items in (("Tools", caps.provides_tools), ("Hooks", caps.provides_hooks),
                         ("Middleware", caps.provides_middleware), ("Env vars", caps.requires_env)):
        console.print(f"[dim]{label + ':':<12}[/dim] {', '.join(items) or '(none)'}")
    console.print()
    removed = find_removed(entry.name) or find_removed(entry.repo)
    if removed is not None:
        console.print(f"[red bold]✗ REMOVED from catalog: {removed.reason or 'no reason recorded'}"
                      f"{f' ({removed.date})' if removed.date else ''}[/red bold]")
        console.print()
    console.print(f"[dim]Install:[/dim]     hermes plugins install {entry.name}")
    console.print()


def cmd_validate(path: str, as_json: bool = False) -> None:
    """Catalog-admission validation of a plugin directory (the CI gate); exits 0/1."""
    from hermes_cli.plugin_validate import validate_plugin_dir
    from hermes_cli.plugins_cmd import _console
    report = validate_plugin_dir(Path(path))
    if as_json:
        print(json.dumps(report.to_dict(), indent=2))
        sys.exit(report.exit_code)
    console = _console()
    console.print()
    for check_name, ok, detail in report.checks:
        console.print(f"{'[green]✓[/green]' if ok else '[red]✗[/red]'} {check_name}"
                      + (f" [dim]— {detail}[/dim]" if detail else ""))
    for warning in report.warnings:
        console.print(f"[yellow]⚠ {warning}[/yellow]")
    console.print()
    console.print("[green bold]Validation passed.[/green bold]" if report.ok else "[red bold]Validation failed.[/red bold]")
    sys.exit(report.exit_code)


# ── Dashboard / TUI payloads ─────────────────────────────────────────────────

def installed_catalog_state(installed: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Catalog entries merged with local state for the dashboard. *installed* maps every alias (name
    and registry key) of a discovered plugin to ``{"dir", "runtime_status"}``. A catalog name rarely
    equals the manifest name (``hermes-plugin-x`` vs ``x``), so installs are matched through the
    sidecar's ``catalog_name`` first and by name only as a fallback."""
    by_catalog_name: Dict[str, Dict[str, Any]] = {}
    for local in installed.values():
        sidecar = read_catalog_sidecar(local["dir"])
        if sidecar:
            by_catalog_name[str(sidecar["catalog_name"])] = {**local, "sidecar": sidecar}
    entries = []
    for entry in load_catalog_live():
        local = by_catalog_name.get(entry.name) or installed.get(entry.name)
        sidecar = local.get("sidecar") if local else None
        installed_sha = str(sidecar["sha"]) if sidecar and sidecar.get("sha") else None
        entries.append({
            **entry.to_dict(), "sha_short": entry.sha[:7],
            "capability_summary": entry_capability_summary(entry),
            "installed": local is not None, "installed_sha": installed_sha,
            "update_available": bool(installed_sha) and installed_sha != entry.sha,
            "runtime_status": local["runtime_status"] if local else None,
        })
    return {
        "entries": entries,
        "removed": [{"name": r.name, "repo": r.repo, "reason": r.reason, "date": r.date} for r in load_removed_list()],
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def catalog_row_fields(dir_path, pins: Dict[str, str]) -> Dict[str, Any]:
    """Provenance fields for one installed-plugin row (TUI/desktop ``plugins.manage list``): catalog
    name/tier/installed SHA and, when *pins* has the entry, the current pin + ``update_available``."""
    sidecar = read_catalog_sidecar(dir_path)
    if not sidecar:
        return {}
    installed_sha = str(sidecar.get("sha") or "").lower()
    row: Dict[str, Any] = {
        "catalog_name": sidecar["catalog_name"], "catalog_tier": str(sidecar.get("tier") or "community"),
        "installed_sha": installed_sha}
    pin = pins.get(str(sidecar["catalog_name"]))
    if pin:
        row["catalog_sha"] = pin
        row["update_available"] = bool(installed_sha) and installed_sha != pin
    return row


def catalog_pins() -> Dict[str, str]:
    """``{catalog_name: pinned_sha}`` from the live catalog; empty on failure (best effort)."""
    try:
        return {e.name: e.sha for e in load_catalog_live()}
    except Exception:
        return {}
