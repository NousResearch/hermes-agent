"""``hermes plugins`` catalog surface: resolution, provenance sidecar, search/browse/info/validate,
catalog-aware update, plus the dashboard/TUI-facing catalog payload helpers.

Sibling of :mod:`hermes_cli.plugins_cmd` (the installer core, enable/disable state and console helpers
live there and are imported late — this module is imported BY ``plugins_cmd``).
"""

from __future__ import annotations

import datetime
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

from hermes_cli.plugin_catalog import (
    PluginCatalogEntry, RemovedEntry, cached_removed_entries, entry_capability_summary, filter_entries,
    find_removed, get_live_catalog_entry, load_catalog_live, match_removed, resolved_removed_entries,
    _NAME_RE, _normalize_repo,
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


def write_catalog_sidecar(target: Path, entry: PluginCatalogEntry, sha: Optional[str] = None) -> None:
    """Human/Desktop-readable ``.hermes-catalog.json`` inside the install dir. It is a CONVENIENCE COPY:
    the authoritative provenance is the ``catalog`` block on the ``.install-metadata.json`` record (see
    :func:`read_catalog_sidecar`), because anything inside the tree is under the repo's control."""
    sidecar = {
        "catalog_name": entry.name, "repo": entry.repo, "sha": sha or entry.sha, "tier": entry.tier,
        "installed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    }
    try:
        (target / CATALOG_SIDECAR).write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to write catalog sidecar in %s: %s", target, exc)


def _install_record(plugin_dir: Path) -> Optional[dict]:
    """The installer-owned ``.install-metadata.json`` record for a dir under the plugins dir, else ``None``."""
    from hermes_cli.plugins_cmd import PluginOperationError, _plugins_dir, _read_install_metadata
    if plugin_dir.parent != _plugins_dir():
        return None
    try:
        record = _read_install_metadata().get(plugin_dir.name)
    except PluginOperationError:
        return None
    return record if isinstance(record, dict) else None


def _adopt_legacy_sidecar(plugin_dir: Path, record: dict) -> Optional[dict]:
    """Installs made before provenance moved out of the tree carry only the in-tree file. Trust it once —
    only when the installer record agrees (pinned at that sha, cloned from that catalog entry's repo) —
    and copy it onto the record so later reads never consult the tree again."""
    from hermes_cli.plugins_cmd import _update_install_record
    path = plugin_dir / CATALOG_SIDECAR
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except Exception:
        return None
    if not isinstance(data, dict) or not data.get("catalog_name"):
        return None
    sha = str(data.get("sha") or "").lower()
    entry = get_live_catalog_entry(str(data["catalog_name"]))
    if entry is None or record.get("pinned") is not True or record.get("revision") != sha:
        return None
    source = str(record.get("source") or "").split("#", 1)[0]
    if _normalize_repo(source) != _normalize_repo(entry.repo):
        return None
    block = {"name": entry.name, "repo": entry.repo, "tier": str(data.get("tier") or entry.tier), "pin": sha, "sha": sha}
    _update_install_record(plugin_dir.name, lambda current: {**current, "catalog": block} if current else None)
    return block


def read_catalog_sidecar(plugin_dir) -> Optional[dict]:
    """Catalog provenance of an installed plugin (``catalog_name``/``repo``/``sha``/``tier``/``pin``), or
    ``None`` for a non-catalog install. Read from the installer-owned metadata record, never from the
    tree: a URL-installed repo that ships its own ``.hermes-catalog.json`` must not render as a reviewed
    catalog install nor mark the real entry installed."""
    if not plugin_dir:
        return None
    plugin_dir = Path(plugin_dir)
    record = _install_record(plugin_dir)
    if record is None:
        return None
    block = record.get("catalog")
    if not isinstance(block, dict):
        block = _adopt_legacy_sidecar(plugin_dir, record)
    if not block or not block.get("name"):
        return None
    return {"catalog_name": block["name"], "repo": block.get("repo", ""), "sha": block.get("sha", ""),
            "tier": block.get("tier") or "community", "pin": block.get("pin", "")}


def at_catalog_pin(sidecar: dict, entry_sha: str) -> bool:
    """The install satisfies the catalog pin *entry_sha*: HEAD is that commit, or the pin is an
    annotated tag whose commit was checked out (``sha`` records the peeled commit, ``pin`` the tag
    object the installer verified). Neither matches after a re-pin or for an off-pin ``--ref`` install."""
    return bool(entry_sha) and entry_sha in (
        str(sidecar.get("sha") or "").lower(), str(sidecar.get("pin") or "").lower())


def catalog_annotation(dir_path) -> Optional[str]:
    """``catalog:<tier>@<sha8>`` for a catalog install (``list`` Source column), else ``None``."""
    sidecar = read_catalog_sidecar(dir_path)
    if not sidecar:
        return None
    return f"catalog:{sidecar.get('tier') or 'community'}@{str(sidecar.get('sha') or '')[:8]}"


def removed_annotation(name: str, dir_path, removed_entries: List[RemovedEntry]) -> Optional[str]:
    """Kill-list reason when an INSTALLED plugin matches by name, catalog name or repo, else ``None``.

    ``removed_entries`` is required: callers annotating many rows (``plugins list``, the dashboard hub)
    resolve the kill list once with :func:`plugin_catalog.resolved_removed_entries` and pass it in.
    Resolving per row cost one live-catalog fetch — one network timeout, offline — per plugin.
    """
    sidecar = read_catalog_sidecar(dir_path) or {}
    for candidate in (name, sidecar.get("catalog_name"), sidecar.get("repo")):
        removed = match_removed(str(candidate), removed_entries) if candidate else None
        if removed is not None:
            return removed.reason or "no reason recorded"
    return None


# ── Catalog-aware install / update ───────────────────────────────────────────

_PLATFORM_ALIASES = {"windows": "win32", "macos": "darwin"}


def normalized_platforms(platforms: List[str]) -> set[str]:
    """Return catalog platform names in host OS-family vocabulary."""
    return {_PLATFORM_ALIASES.get(value.lower(), value.lower()) for value in platforms}


def _refuse_unsupported_catalog_platform(entry: PluginCatalogEntry) -> None:
    if not entry.platforms:
        return
    from hermes_cli.plugins_cmd import PluginOperationError
    from hermes_platform.host.facts import os_family

    current = os_family()
    if current not in normalized_platforms(entry.platforms):
        raise PluginOperationError(
            f"Plugin '{entry.name}' is unavailable on {current}; supported platforms: "
            f"{', '.join(entry.platforms)}."
        )


def install_catalog_entry(entry: PluginCatalogEntry, *, force: bool, ref: Optional[str] = None,
                          allow_removed: bool = False, scan_decision_cb=None, python_deps: bool = True,
                          before_swap=None) -> tuple:
    """``_install_plugin_core`` at the catalog pin (an explicit *ref* wins) + provenance recorded on the
    install-metadata record at the sha ACTUALLY checked out (a ``--ref`` install is not at the reviewed
    pin, so ``update_available`` must say so). Returns the core's ``(target, manifest, installed_name)``."""
    from hermes_cli.plugins_cmd import _install_plugin_core, pinned_revision
    if not allow_removed:
        raise_if_removed(entry.name, entry.repo)
    _refuse_unsupported_catalog_platform(entry)
    target, manifest, installed_name = _install_plugin_core(
        entry.install_identifier, force=force, ref=ref or entry.sha, scan_decision_cb=scan_decision_cb,
        reviewed_pin=entry.sha, python_deps=python_deps, allow_removed=allow_removed, before_swap=before_swap,
        catalog={"name": entry.name, "repo": entry.repo, "tier": entry.tier, "pin": entry.sha})
    write_catalog_sidecar(target, entry, sha=pinned_revision(target.name))
    return target, manifest, installed_name


def installed_plugin_removal(name: str, plugin_dir) -> Optional[RemovedEntry]:
    """Kill-list verdict for an INSTALLED plugin (manifest name, dir name, catalog name or recorded
    source), or ``None``. A record carrying ``allow_removed`` (the user bypassed the list at install) is
    honoured; the check is offline (in-tree list + cached live copy) so load time never blocks on the
    catalog host."""
    plugin_dir = Path(plugin_dir) if plugin_dir else None
    record = (_install_record(plugin_dir) if plugin_dir else None) or {}
    if record.get("allow_removed") is True:
        return None
    block = record.get("catalog") if isinstance(record.get("catalog"), dict) else {}
    source = str(record.get("source") or "").split("#", 1)[0]
    candidates = [name, source, block.get("name"), block.get("repo"), plugin_dir.name if plugin_dir else None]
    entries = cached_removed_entries()
    for candidate in candidates:
        removed = match_removed(str(candidate), entries) if candidate else None
        if removed is not None:
            return removed
    return None


def refuse_if_installed_removed(name: str, plugin_dir) -> None:
    """``PluginOperationError`` form of :func:`installed_plugin_removal` for ``update``/``enable``, which
    otherwise keep pulling and activating code the catalog recalled."""
    from hermes_cli.plugins_cmd import PluginOperationError
    removed = installed_plugin_removal(name, plugin_dir)
    if removed is not None:
        raise PluginOperationError(
            f"Plugin '{name}' was removed from the Hermes plugin catalog: "
            f"{removed.reason or 'no reason recorded'}. Remove it with `hermes plugins remove {name}`, "
            "or reinstall with `hermes plugins install <source> --force --allow-removed` if you trust it.")


_PRESERVE_SKIP = ("__pycache__",)


def _local_changes(target: Path) -> tuple[list[str], list[str]]:
    """``(untracked_or_ignored, modified_tracked)`` relative paths in a git checkout; empty for a
    non-git tree (subdir installs carry no ``.git``, so nothing can be told apart from the clone)."""
    from hermes_cli.plugins_cmd import _resolve_git_executable, _run_plugin_git
    git_exe = _resolve_git_executable()
    if not git_exe or not (target / ".git").exists():
        return [], []
    status = _run_plugin_git(git_exe, target, "status", "--porcelain", "--ignored", "-z", "--untracked-files=all",
                             "--ignored=matching", timeout=30)
    if status.returncode != 0:
        return [], []
    local, modified = [], []
    for item in status.stdout.split("\0"):
        if len(item) < 4:
            continue
        code, rel = item[:2], item[3:]
        if any(part in _PRESERVE_SKIP or part.endswith(".pyc") for part in Path(rel).parts):
            continue
        (local if code in ("??", "!!") else modified).append(rel)
    return local, modified


def _stash_local_files(target: Path, rels: list[str], stash: Path) -> None:
    for rel in rels:
        src = target / rel
        if src.is_file():
            dst = stash / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


class RepinResult(NamedTuple):
    sha: str
    changed: bool
    installed_name: str
    warnings: list[str]


# Surfaces a re-pin can widen without the user seeing a diff: each is a list of identifiers the
# new manifest adds (``desktop`` = a Desktop half appeared). Compared as sets — removals are not consent events.
_SURFACE_LABELS = {"capabilities": "host capabilities", "tools": "tools", "hooks": "hooks",
                   "python_dependencies": "Python dependencies", "desktop": "Desktop UI half"}


def plugin_surface(manifest: dict, tree: Path) -> Dict[str, set]:
    """What an installed tree exposes: declared host capabilities, tools, hooks, Python deps, Desktop half."""
    from hermes_cli.plugins_cmd import _declared_capabilities_from_manifest
    manifest = manifest or {}

    def _list(key: str, *alts: str) -> set:
        for k in (key, *alts):
            raw = manifest.get(k)
            if isinstance(raw, list):
                return {str(x) for x in raw if isinstance(x, (str, int, float))}
        return set()

    return {
        "capabilities": set(_declared_capabilities_from_manifest(manifest, str(manifest.get("name") or "?"))),
        "tools": _list("provides_tools"), "hooks": _list("provides_hooks", "hooks"),
        "python_dependencies": _list("python_dependencies"),
        "desktop": {"desktop/plugin.js"} if (tree / "desktop" / "plugin.js").is_file() else set(),
    }


def surface_delta(old: Dict[str, set], new: Dict[str, set]) -> Dict[str, List[str]]:
    """``{surface: [added...]}`` for every surface the new tree widens; empty when nothing widened."""
    return {k: sorted(new.get(k, set()) - old.get(k, set())) for k in _SURFACE_LABELS
            if new.get(k, set()) - old.get(k, set())}


def surface_delta_lines(delta: Dict[str, List[str]]) -> List[str]:
    return [f"{_SURFACE_LABELS[k]}: {', '.join(v)}" for k, v in delta.items()]


class RepinConsentRequired(Exception):
    """The new pin widens the plugin's surface and no consent was given; nothing was changed on disk.
    ``delta`` is :func:`surface_delta`'s mapping — surfaces hand it to the user and retry with consent."""

    def __init__(self, name: str, sha: str, delta: Dict[str, List[str]]):
        self.name, self.sha, self.delta = name, sha, delta
        super().__init__(
            f"Updating '{name}' to {sha[:8]} adds {'; '.join(surface_delta_lines(delta))}. Confirm to continue.")


def repin_catalog_plugin(target: Path, sidecar: dict, *, consent_cb=None) -> RepinResult:
    """Re-pin a catalog install to the current catalog SHA (never ``git pull``). The force reinstall
    replaces the whole tree, so the user's untracked/ignored files (installer-created ``config.yaml``,
    data) are carried into the new tree and edits to TRACKED files are saved under
    ``<HERMES_HOME>/plugins-backup/<name>-<sha8>/`` with a warning. A manifest rename between pins moves the
    enabled flag and removes the stale dir. Raises ``PluginOperationError`` when the entry left the catalog.

    A new pin that WIDENS the plugin (new tools, hooks, Python deps, host capabilities or a Desktop half
    vs the installed manifest) is a new grant, not a refresh: ``consent_cb(delta) -> bool`` decides
    before anything moves; absent or declined → :class:`RepinConsentRequired`, tree untouched."""
    from hermes_cli.plugins_cmd import (
        PluginOperationError, _get_enabled_set, _plugins_dir, _read_manifest, _remove_plugin_core,
        _save_enabled_set)
    catalog_name = str(sidecar["catalog_name"])
    entry = get_live_catalog_entry(catalog_name)
    if entry is None:
        raise PluginOperationError(
            f"Plugin '{catalog_name}' is no longer in the catalog — it may have been removed. "
            "See `hermes plugins info` and the removed blocklist.")
    if at_catalog_pin(sidecar, entry.sha):
        return RepinResult(entry.sha, False, target.name, [])
    was_enabled = _get_enabled_set()  # the force reinstall must not flip activation state
    local, modified = _local_changes(target)
    old_sha8 = str(sidecar.get("sha") or "old")[:8]
    installed_surface = plugin_surface(_read_manifest(target), target)

    def _consent_gate(manifest: dict, tree: Path) -> None:
        delta = surface_delta(installed_surface, plugin_surface(manifest, tree))
        if delta and not (consent_cb is not None and consent_cb(delta)):
            raise RepinConsentRequired(catalog_name, entry.sha, delta)

    with tempfile.TemporaryDirectory(prefix=".repin-", dir=_plugins_dir()) as tmp:
        stash = Path(tmp) / "local"
        _stash_local_files(target, local, stash)
        # Outside the plugins dir: the discovery scanners recurse into every subdirectory there.
        backup = _plugins_dir().parent / "plugins-backup" / f"{target.name}-{old_sha8}"
        _stash_local_files(target, modified, backup)
        new_target, _manifest, installed_name = install_catalog_entry(entry, force=True, before_swap=_consent_gate)
        if stash.exists():
            shutil.copytree(stash, new_target, dirs_exist_ok=True)
    warnings: list[str] = []
    if modified:
        warnings.append(f"Local edits to {len(modified)} tracked file(s) were not carried over; copies are under "
                        f"{backup} (the previous version's files, re-apply by hand).")
    if new_target != target and target.exists():
        _remove_plugin_core(target)
        if target.name in was_enabled:
            was_enabled = (was_enabled - {target.name}) | {installed_name}
        warnings.append(f"Plugin renamed itself from '{target.name}' to '{installed_name}'; the old directory was removed.")
    _save_enabled_set(was_enabled)
    return RepinResult(entry.sha, True, installed_name, warnings)


def cmd_update_catalog(name: str, target: Path, sidecar: dict, console) -> None:
    from hermes_cli.plugins_cmd import (
        PluginOperationError, _ask_yes, _declared_capabilities_from_manifest, _fail, _is_tty, _read_manifest,
        _run_capability_consent)
    console.print(f"[dim]Checking catalog pin for {name}...[/dim]")

    def _confirm_widening(delta: Dict[str, List[str]]) -> bool:
        console.print(f"\n  [yellow]The new pin of [bold]{name}[/bold] adds:[/yellow]")
        for line in surface_delta_lines(delta):
            console.print(f"    {line}")
        if not _is_tty():
            console.print("  [yellow]Non-interactive session: update NOT applied (fail closed). "
                          "Re-run `hermes plugins update` in a terminal to review and confirm.[/yellow]")
            return False
        return _ask_yes("  Apply this update? [y/N]: ")

    try:
        result = repin_catalog_plugin(target, sidecar, consent_cb=_confirm_widening)
    except RepinConsentRequired as exc:
        _fail(console, f"[yellow]Update of {name} not applied:[/yellow] {exc}")
        raise SystemExit(1)
    except PluginOperationError as exc:
        _fail(console, f"[red]Error:[/red] {exc}")
        raise SystemExit(1)
    verb = "updated to" if result.changed else "is already at catalog pin"
    console.print(f"[green]✓[/green] Plugin [bold]{result.installed_name}[/bold] {verb} {result.sha[:8]}.")
    for warning in result.warnings:
        console.print(f"[yellow]⚠ {warning}[/yellow]")
    if result.changed:
        from hermes_cli.plugins_cmd import _install_python_dependencies
        new_target = target.parent / result.installed_name
        _install_python_dependencies(new_target, console)
        # Same grant path as the git-pull ``update``: host capabilities the new pin declares stay
        # ungranted (fail closed) until the user says yes here.
        declared = _declared_capabilities_from_manifest(_read_manifest(new_target), result.installed_name)
        if declared:
            from hermes_cli.plugin_capabilities import declared_set_changed, pending_capabilities
            if pending_capabilities(result.installed_name, declared) or declared_set_changed(result.installed_name, declared):
                _run_capability_consent(console, result.installed_name, declared, context="update")


# ── search / browse / info / validate ────────────────────────────────────────

def _capability_counts(entry: PluginCatalogEntry) -> str:
    caps = entry.capabilities
    parts = [f"{len(items)} {label}{'s' if len(items) != 1 and label != 'middleware' else ''}"
             for items, label in ((caps.provides_tools, "tool"), (caps.provides_hooks, "hook"),
                                  (caps.provides_middleware, "middleware")) if items]
    if caps.requires_env:
        parts.append(f"{len(caps.requires_env)} env")
    return ", ".join(parts) or "—"


def pin_label(entry: PluginCatalogEntry) -> str:
    """``1.4.0 @ abcd1234`` when the entry carries a version label, else the short sha."""
    return f"{entry.version} @ {entry.sha[:8]}" if entry.version else entry.sha[:8]


def _render_entries(entries: List[PluginCatalogEntry], console) -> None:
    from hermes_cli.plugins_cmd import _table
    table = _table(((("Name", "bold")), ("Category", None), ("Tier", None), ("Description", None),
                    ("Pinned", "dim"), ("Capabilities", "dim")), title="Hermes Plugin Catalog (curated)")
    for e in sorted(entries, key=lambda e: (e.category, e.tier != "official", e.name)):
        tier = "[cyan]official[/cyan]" if e.tier == "official" else "[magenta]community[/magenta]"
        desc = e.description if len(e.description) <= 60 else e.description[:57] + "..."
        table.add_row(e.name, e.category, tier, desc, pin_label(e), _capability_counts(e))
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
    rows = [("Repo", entry.repo), ("Subdir", entry.subdir), ("Version", entry.version), ("Pinned SHA", entry.sha),
            ("Image", entry.image),
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


def cmd_validate(path: str, as_json: bool = False, install_deps: bool = False) -> None:
    """Catalog-admission validation of a plugin directory (the CI gate); exits 0/1. *install_deps*
    installs the declared Python deps first so the capability probe imports what an install would."""
    from hermes_cli.plugin_validate import validate_plugin_dir
    from hermes_cli.plugins_cmd import _console
    if install_deps:
        from hermes_cli.plugin_python_deps import install_for_plugin_dir
        outcome = install_for_plugin_dir(Path(path))
        if outcome.status in ("failed", "invalid"):
            print(outcome.message, file=sys.stderr)
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
            "update_available": bool(installed_sha) and not at_catalog_pin(sidecar or {}, entry.sha),
            "runtime_status": local["runtime_status"] if local else None,
        })
    return {
        "entries": entries,
        "removed": [{"name": r.name, "repo": r.repo, "reason": r.reason, "date": r.date} for r in resolved_removed_entries()],
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def catalog_row_fields(dir_path, pins: Dict[str, str], versions: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Provenance fields for one installed-plugin row (TUI/desktop ``plugins.manage list``): catalog
    name/tier/installed SHA and, when *pins* has the entry, the current pin (+ its version label from
    *versions*) and ``update_available``."""
    versions = versions or {}
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
        row["catalog_version"] = versions.get(str(sidecar["catalog_name"])) or None
        row["update_available"] = bool(installed_sha) and not at_catalog_pin(sidecar, pin)
    return row


def catalog_pins() -> Dict[str, str]:
    """``{catalog_name: pinned_sha}`` from the live catalog; empty on failure (best effort)."""
    try:
        return {e.name: e.sha for e in load_catalog_live()}
    except Exception:
        return {}


def catalog_versions() -> Dict[str, str]:
    """``{catalog_name: version_label}`` for entries that carry one; empty on failure (best effort)."""
    try:
        return {e.name: e.version for e in load_catalog_live() if e.version}
    except Exception:
        return {}
