"""Gateway start: catalog platform plugins whose credentials a profile already has.

A catalog entry of ``category: platform`` names the env its adapter needs (``capabilities.requires_env``).
When every one of them is set for a served home and the plugin is not there, the gateway does what the
user's setup implies, under the same consent rules as every other unattended install:

* A default entry (``default: true``, official/verified tier: :mod:`hermes_cli.default_plugins`) is
  installed now, through that module's own pass (its record, failure stamps and leftover handling), so
  the platform starts in this same gateway start instead of after the detached default install.
* Any other entry gets one actionable line naming ``hermes plugins install <name>``: Hermes installs
  third-party code only when the user asks.

Never for a plugin either installer already placed once (a removal sticks), one ``plugins.disabled``
names, a row :mod:`hermes_cli.left_core_migration` owns, or where ``security.allow_lazy_installs`` is off
(that home gets the line instead). Once per process per home and settings fingerprint, like the
left-core migration, so credentials added to a served profile after boot are picked up by its rescan.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_attempted: dict[str, str] = {}  # home key -> settings fingerprint last tried


def _configured(entry, home: Path) -> bool:
    """Every env name *entry* requires resolves for *home* (dotenv layers, its secret scope, the process
    env for the launch home): read as :func:`left_core_migration._home_env` reads a platform credential."""
    from hermes_cli.left_core_migration import _home_env
    names = entry.capabilities.requires_env
    return bool(names) and all(_home_env(home, name, process_env=True) for name in names)


def _served_env(home: Path) -> set[str]:
    """Env names a platform plugin *home* already has (bundled or installed, not disabled) requires: read
    from the manifests, never by loading a plugin, so it holds before *home*'s first discovery."""
    from hermes_cli.left_core_migration import _plugins_listed
    from hermes_cli.plugins_discovery import collect_directory_manifests
    from agent.runtime_scope import bind_profile_scope
    from hermes_cli.plugins_manifest import manifest_key
    with bind_profile_scope(home, hydrate_secrets=False):
        manifests = collect_directory_manifests()
    served: set[str] = set()
    for manifest in manifests:
        if manifest.kind != "platform" or _plugins_listed(home, (manifest.name, manifest_key(manifest)))[1]:
            continue
        for item in manifest.requires_env:
            name = item.get("name") if isinstance(item, dict) else item
            if isinstance(name, str):
                served.add(name)
    return served


def _catalog_entries(*, allow_network: bool) -> list:
    """Live catalog when unattended network fetches are allowed; otherwise the fresh on-disk cache
    or in-tree catalog with zero network I/O."""
    from hermes_cli.plugin_catalog import load_catalog, load_catalog_cached, load_catalog_live
    if allow_network:
        return load_catalog_live()
    cached = load_catalog_cached()
    return cached if cached is not None else load_catalog()


def pending_entries(home: Path, *, _catalog: list | None = None) -> list:
    """Platform entries *home* has the credentials for and has never had installed. Read-only.

    An entry whose required env a platform plugin *home* already has also requires is not pending: those
    credentials are served (``hermes-tag`` overrides Feishu with the same ``FEISHU_*``), so naming it would
    tell a working Feishu user on every start that a plugin they do not need is missing."""
    from hermes_cli.default_plugins import _installed_dir, _recorded, fetch_allowed
    from hermes_cli.left_core_migration import LEFT_CORE, _plugins_listed
    owned = {feature.plugin for feature in LEFT_CORE}
    done = _recorded(home)
    catalog = _catalog if _catalog is not None else _catalog_entries(allow_network=fetch_allowed(home))
    candidates = [entry for entry in catalog
                  if entry.category == "platform" and entry.name not in owned and entry.name not in done
                  and _configured(entry, home) and _installed_dir(home, entry) is None
                  and not _plugins_listed(home, (entry.name,))[1]]
    served = _served_env(home) if candidates else set()
    return [entry for entry in candidates if not set(entry.capabilities.requires_env) <= served]


def recover_in(home: Path, *, report: Callable[[str], None]) -> list[str]:
    """Install or point to every :func:`pending_entries` entry for *home* (the active, bound home). Returns
    the installed plugin names; never raises. Blocking: call it off the event loop."""
    from hermes_cli.left_core_migration import settings_fingerprint
    from hermes_constants import hermes_home_key
    key = hermes_home_key(home)
    if _attempted.get(key) == settings_fingerprint(home):
        return []
    installed: list[str] = []
    seen: set[str] = set()
    for _ in range(3):  # again while the settings changed during the pass (left_core_migration.recover_at_startup)
        before = settings_fingerprint(home)
        installed += _recover(home, report=report, seen=seen)
        _attempted[key] = settings_fingerprint(home)
        if _attempted[key] == before:
            break
    return installed


def _recover(home: Path, *, report: Callable[[str], None], seen: set[str]) -> list[str]:
    """One pass over the pending entries not in *seen* (which it extends): a re-pass never repeats one."""
    from hermes_cli.default_plugins import _filter_defaults, fetch_allowed, install_defaults, installer_lock
    from hermes_cli.left_core_migration import startup_lock_wait
    from hermes_cli.memory_provider_migration import _install_command
    try:
        allowed = fetch_allowed(home)
        offline_catalog = None if allowed else _catalog_entries(allow_network=False)
        entries = [entry for entry in pending_entries(home, _catalog=offline_catalog) if entry.name not in seen]
        seen.update(entry.name for entry in entries)
        if not entries:
            return []
        if allowed:
            from hermes_cli.default_plugins import default_entries
            defaults = {entry.name for entry in default_entries()}  # one catalog snapshot for the pass
        else:
            defaults = {entry.name for entry in _filter_defaults(offline_catalog or [])}
        unattended = [entry for entry in entries if entry.name in defaults] if allowed else []
        for entry in entries:
            if entry in unattended:
                continue
            env = ", ".join(entry.capabilities.requires_env)
            why = ("security.allow_lazy_installs is off, so Hermes did not fetch it" if entry.name in defaults
                   else f"Hermes installs {entry.tier}-tier plugins only when you ask")
            label = f" for {entry.title}" if entry.title else ""
            report(f"⚠ This profile sets {env}, but the '{entry.name}' platform plugin{label} is not installed, so "
                   f"the gateway does not serve it ({why}). Run `{_install_command(entry.name, home)}`.")
        if not unattended:
            return []
        try:
            with installer_lock(startup_lock_wait(gateway_start=True)):
                outcome = install_defaults(home, say=lambda message: report(message.strip()), entries=unattended)
        except TimeoutError:
            report(f"⚠ Another Hermes process is still installing plugins, so "
                   f"{', '.join(repr(e.name) for e in unattended)} was not installed now; Hermes retries at the "
                   "next gateway start.")
            return []
        return list(outcome["installed"])
    except Exception as exc:  # health: allow BLE001 -- never take a gateway start down
        logger.warning("catalog platform install failed for %s: %s", home, exc, exc_info=True)
        return []
