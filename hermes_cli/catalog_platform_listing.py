"""Messaging platforms the curated plugin catalog offers that this home does not provide yet.

``hermes gateway setup``, ``hermes setup gateway``, the dashboard Channels page and the Desktop
Messaging page list the same rows and install a pick through the same path: the catalog install
``hermes plugins install <name>`` uses (pinned commit, kill list, security scan), then enable, so
the platform is configurable the moment the install returns. Only Nous-built entries are offered
here (``official`` / ``verified`` tiers and the Hermes Labs repo); community platforms stay in the
plugin browser, where their unreviewed status is the first thing shown. A Hermes Labs entry needs
an explicit confirmation: it is built by Nous Research but not supported.

Callers bind the owning profile (``_profile_scope`` / ``-p``) before calling; every read and the
install resolve ``get_hermes_home()``. A setup surface outlives the installs it lists: a dashboard
or Desktop backend that loaded plugins before the gateway's background default install finished,
or before another terminal's ``hermes plugins install``, must see that plugin on the next read, so
every read goes through :func:`refresh_platform_plugins` instead of the process-lifetime discovery.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any, Collection, Literal, Optional

LABS_REPO = "github.com/nousresearch/hermes-labs-plugins"
Badge = Literal["official", "verified", "labs"]
_TIER_BADGES: dict[str, Badge] = {"official": "official", "verified": "verified"}
# Why a row cannot install, keyed by the reason code the surfaces translate.
LAZY_OFF = "lazy_installs_off"
BLOCKED_SENTENCES = {
    LAZY_OFF: "security.allow_lazy_installs is off, so Hermes does not install plugins from setup.",
}


@dataclass(frozen=True)
class CatalogPlatform:
    """One not-installed catalog platform row. ``blocked`` is empty when a pick can install it, else
    a :data:`BLOCKED_SENTENCES` key; ``install_command`` is the one command that installs it into
    this home either way."""

    name: str
    title: str
    badge: Badge
    summary: str
    disclosure: str
    blocked: str
    install_command: str

    @property
    def needs_confirm(self) -> bool:
        return self.badge == "labs"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "needs_confirm": self.needs_confirm,
                "blocked_reason": BLOCKED_SENTENCES.get(self.blocked, "")}


def badge_for(entry: Any) -> Optional[Badge]:
    """The row badge for a catalog entry, or None when this list does not offer it."""
    from hermes_cli.plugin_catalog import _normalize_repo
    if _normalize_repo(entry.repo) == LABS_REPO:
        return "labs"
    return _TIER_BADGES.get(entry.tier)


# Hermes home key -> the plugin-tree signature the loaded registry was refreshed against.
_refreshed_at: dict[str, tuple] = {}
_refresh_lock = threading.Lock()


def _plugin_tree_signature() -> tuple:
    """What decides which platform plugins the bound home loads: every user platform manifest
    (added, removed, replaced, edited) and the ``plugins.enabled`` / ``plugins.disabled`` lists."""
    from hermes_cli.config import platform_manifest_stamp
    from hermes_cli.plugins_cmd import _get_disabled_set, _get_enabled_set
    return platform_manifest_stamp(), tuple(sorted(_get_enabled_set())), tuple(sorted(_get_disabled_set()))


def refresh_platform_plugins() -> None:
    """Load the bound home's plugins so the platform registry matches its plugin tree now. Discovery
    is cached for the process; a tree that moved since this process last looked (or a registry some
    other caller loaded before this module first looked) is rediscovered once, then reads are a
    stat-cheap signature compare."""
    from hermes_cli.plugins import discover_plugins, get_plugin_manager
    from hermes_constants import hermes_home_key
    key = hermes_home_key()
    with _refresh_lock:
        signature = _plugin_tree_signature()
        if _refreshed_at.get(key) == signature:
            discover_plugins()
            return
        discover_plugins(force=bool(get_plugin_manager()._discovered))
        _refreshed_at[key] = signature


def _mtime_ns(path: Path) -> Optional[int]:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def platform_list_signature(home: Path) -> str:
    """Digest of what a channels page lists for ``home``: platform connect/disconnect/health
    (``gateway_state.json``), a platform plugin installed, removed or replaced (another terminal, the
    background default install, a Plugins-page install), and config.yaml (a plugin enabled, a platform
    toggled elsewhere). Desktop gets it pushed as ``platforms.changed``; the dashboard polls it, so
    both refresh on the same moves. Stat-cheap: no plugin discovery, no catalog read."""
    from hermes_cli.config import platform_manifest_stamp
    parts = (_mtime_ns(home / "gateway_state.json"), _mtime_ns(home / "config.yaml"),
             platform_manifest_stamp(home))
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:16]


def installed_catalog_names() -> frozenset[str]:
    """Catalog entries installed into the bound home and enabled, read from the install records
    ``hermes plugins list`` reads, not from what this process loaded: an enabled plugin this process
    cannot import yet (its dependencies arrived in a newer environment) is installed all the same."""
    from hermes_cli.plugins_cmd import (
        PluginOperationError, _get_enabled_set, _plugins_dir, _read_install_metadata, _read_manifest)
    from hermes_cli.plugins_cmd_catalog import catalog_install_record
    try:
        keys = list(_read_install_metadata())
    except PluginOperationError:
        return frozenset()
    enabled = _get_enabled_set()
    names = set()
    for key in keys:
        target = _plugins_dir() / key
        record = catalog_install_record(target) if target.is_dir() else None
        if record and {key, target.name, _read_manifest(target).get("name")} & enabled:
            names.add(record["catalog_name"])
    return frozenset(names)


def provided_platforms() -> frozenset[str]:
    """Platform ids and owning plugin names the bound home loads now (built-in or plugin)."""
    from gateway.platform_registry import platform_registry
    refresh_platform_plugins()
    return frozenset(name for entry in platform_registry.all_entries()
                     for name in (entry.name, entry.plugin_name) if name)


def _runs_here(entry: Any) -> bool:
    from hermes_cli.plugins_cmd_catalog import catalog_hermes_error, normalized_platforms
    from hermes_platform.host.facts import os_family
    if entry.platforms and os_family() not in normalized_platforms(entry.platforms):
        return False
    return catalog_hermes_error(entry) is None


def available_catalog_platforms(provided: Optional[Collection[str]] = None) -> list[CatalogPlatform]:
    """Catalog ``category: platform`` entries this list offers that nothing loaded provides, sorted by
    title. A removed (kill-listed) entry, one for another OS or one this Hermes is too old for is left
    out: its install would be refused, and so is an entry installed and enabled here that this process
    has not loaded (:func:`installed_catalog_names`). Never raises; an unreadable catalog lists nothing."""
    from hermes_cli.default_plugins import fetch_allowed
    from hermes_cli.memory_provider_migration import _install_command
    from hermes_cli.plugin_catalog import load_catalog_live, match_removed, resolved_removed_entries
    from hermes_cli.plugin_catalog_presence import disclosure, first_sentence
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    have = (provided_platforms() if provided is None else frozenset(provided)) | installed_catalog_names()
    entries = [e for e in load_catalog_live() if e.category == "platform" and e.name not in have]
    offered: list[tuple[Any, Badge]] = []
    for entry in entries:
        badge = badge_for(entry)
        if badge is not None:
            offered.append((entry, badge))
    if not offered:
        return []
    removed = resolved_removed_entries()
    blocked = "" if fetch_allowed(home) else LAZY_OFF
    rows = [CatalogPlatform(name=e.name, title=e.title or e.name, badge=badge,
                            summary=first_sentence(e.description), disclosure=disclosure(e.description),
                            blocked=blocked, install_command=_install_command(e.name, home))
            for e, badge in offered
            if not (match_removed(e.name, removed) or match_removed(e.repo, removed)) and _runs_here(e)]
    return sorted(rows, key=lambda row: row.title.lower())


def platforms_of(plugin_name: str) -> list[str]:
    """Platform ids *plugin_name* registered in this process (after its install loaded it)."""
    from gateway.platform_registry import platform_registry
    return [entry.name for entry in platform_registry.all_entries() if entry.plugin_name == plugin_name]


def install_catalog_platform(name: str, *, confirmed: bool = False) -> dict[str, Any]:
    """Install and enable catalog platform *name* into the bound home. ``{"ok": True, "plugin_name",
    "platforms": [ids], "warnings"}`` on success; else ``{"ok": False, "error", "install_command"}``,
    plus ``"confirm_required": True`` when *name* is a Hermes Labs entry and *confirmed* is False.
    Only a row :func:`available_catalog_platforms` offers installs: a community or unknown name is
    refused here, never installed past the list's tier rule."""
    from hermes_cli.left_core_migration import _first_cause
    from hermes_cli.plugins_cmd_install import dashboard_install_plugin

    if (present := _already_installed(name)) is not None:
        return present
    row = next((r for r in available_catalog_platforms() if r.name == name), None)
    if row is None:
        return {"ok": False, "error": f"'{name}' is not a catalog platform this home can add.", "install_command": ""}
    if row.blocked:
        return {"ok": False, "error": BLOCKED_SENTENCES[row.blocked], "install_command": row.install_command}
    if row.needs_confirm and not confirmed:
        return {"ok": False, "confirm_required": True, "error": "Hermes Labs plugins need an explicit confirmation.",
                "install_command": row.install_command}
    result = dashboard_install_plugin("", force=False, enable=True, catalog_name=name)
    if not result.get("ok"):
        # Another installer (the gateway's default install) may have landed it while this one ran.
        return _already_installed(name) or {"ok": False, "error": _first_cause(str(result.get("error") or "")),
                                            "install_command": row.install_command}
    plugin_name = str(result.get("plugin_name") or name)
    refresh_platform_plugins()
    return {"ok": True, "plugin_name": plugin_name, "platforms": platforms_of(plugin_name),
            "warnings": list(result.get("warnings") or [])}


def _already_installed(name: str) -> Optional[dict[str, Any]]:
    """The install result for catalog entry *name* when it is already installed and enabled here (a
    page listed it before the background default install finished), else None. ``platforms`` is
    empty when this process cannot load the plugin yet; its card appears after a restart."""
    from hermes_cli.plugins_cmd import _catalog_installed_dir, _read_manifest
    refresh_platform_plugins()
    if name not in installed_catalog_names():
        return None
    target = _catalog_installed_dir(name)
    plugin_name = str((_read_manifest(target).get("name") if target else None) or name)
    platforms = platforms_of(plugin_name)
    warnings = [] if platforms else [f"{plugin_name} is installed; restart Hermes to load it in this window."]
    return {"ok": True, "plugin_name": plugin_name, "platforms": platforms, "warnings": warnings,
            "already_installed": True}
