"""Catalog plugins every home gets: entries marked ``default: true`` (official or verified tier).

Each home gets each default plugin installed at its catalog pin and enabled exactly once. The home
records it under ``_default_plugins_installed`` in its config.yaml (the left-core migrator records its
one-time install the same way), so a user who removes or disables a default plugin keeps that choice
through every later update. A default the home already has (installed by hand before it became a
default, or carried by a profile clone) is recorded without reinstalling.

Callers: the end of every source install and ``hermes update`` (``update_cmd_maint``'s maintenance
tail) for every profile home; the boot bootstrap, for installs that never run that tail (bundled
Desktop, Docker, Nix, Termux), once per installed revision and on later starts while a fetch is owed;
and profile creation, for the new home. Nothing is fetched where ``security.allow_lazy_installs`` is
off. A start that left a plugin uninstalled backs the next start off (an hour, doubling to a week)
before any network, so an offline machine pays the round trip once, not on every start.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_INSTALLED_KEY = "_default_plugins_installed"
_BACKOFF = "default-plugins"  # memory_provider_migration failure-stamp name, in the default home


def default_entries() -> list:
    """Catalog entries marked default that this OS and this Hermes can run (live catalog, else in-tree)."""
    from hermes_cli.plugin_catalog import load_catalog_live
    from hermes_cli.plugins_cmd_catalog import catalog_hermes_error, normalized_platforms
    from hermes_platform.host.facts import os_family
    host = os_family()
    return [entry for entry in load_catalog_live()
            if entry.default and (not entry.platforms or host in normalized_platforms(entry.platforms))
            and catalog_hermes_error(entry) is None]


def _recorded(home: Path) -> set[str]:
    from hermes_cli.left_core_migration import _read_config
    done = _read_config(home).get(_INSTALLED_KEY)
    return set(map(str, done)) if isinstance(done, list) else set()


def _record(home: Path, names: set[str]) -> None:
    from hermes_cli.config import atomic_config_write, read_user_config_raw
    from hermes_cli.config_backend import config_exists
    from hermes_cli.left_core_migration import _stamp_created
    path = home / "config.yaml"
    created = not config_exists(path)
    config = read_user_config_raw(path)
    done = config.get(_INSTALLED_KEY)
    config[_INSTALLED_KEY] = sorted({*(map(str, done) if isinstance(done, list) else ()), *names})
    if created:
        _stamp_created(home, config)
    atomic_config_write(path, config)


def _present(home: Path, entry) -> bool:
    """*home* already has *entry* installed (enabled or not), by catalog provenance or by its name."""
    from hermes_cli.plugins_provenance import read_sidecar_rows
    plugins = home / "plugins"
    if (plugins / entry.name).is_dir():
        return True
    for name, row in read_sidecar_rows(plugins).items():
        block = row.get("catalog") if isinstance(row, dict) else None
        if isinstance(block, dict) and block.get("name") == entry.name and (plugins / name).is_dir():
            return True
    return False


def _fetch_allowed(home: Path) -> bool:
    from agent.runtime_scope import bind_profile_scope
    from pm.install import lazy_installs_allowed
    with bind_profile_scope(home, hydrate_secrets=False):
        return lazy_installs_allowed()


def install_defaults(home: Path, *, say: Callable[[str], None] = logger.info,
                     entries: Optional[list] = None) -> dict:
    """Install and enable every default plugin *home* never had. ``installed`` lists what this call
    installed, ``owed`` what is still to fetch (a failed install). One plugin's failure never raises."""
    from hermes_cli.left_core_migration import _first_cause, _install_into
    from hermes_cli.memory_provider_migration import _home_label
    result: dict = {"installed": [], "owed": []}
    if not _fetch_allowed(home):
        return {**result, "skipped": "lazy-installs-disabled"}
    done = _recorded(home)
    pending = [entry for entry in (default_entries() if entries is None else entries) if entry.name not in done]
    have = {entry.name for entry in pending if _present(home, entry)}
    install = _install_into(home, consent=True)  # a reviewed default: its dependencies come with it
    for entry in pending:
        if entry.name in have:
            continue
        try:
            outcome = install(entry.name)
        except Exception as exc:  # health: allow BLE001 -- network, uv or kill list: reported per plugin, never raised
            outcome = {"ok": False, "error": str(exc)}
        if outcome.get("ok"):
            have.add(entry.name)
            result["installed"].append(entry.name)
            say(f"  ✓ Installed default plugin '{entry.name}' for {_home_label(home)}")
            continue
        result["owed"].append(entry.name)
        say(f"  ⚠ Default plugin '{entry.name}' could not be installed for {_home_label(home)}: "
            f"{_first_cause(str(outcome.get('error') or ''))}. Hermes retries at a later start; "
            f"`hermes plugins install {entry.name}` installs it now.")
    if have:
        _record(home, have)
    return result


def install_defaults_all_homes(*, say: Callable[[str], None] = print) -> dict:
    """Every profile home sharing this install (the default home plus each live profile). The catalog is
    read once; one home's failure never stops the next."""
    from pm.plugins_state import dependency_homes
    summary: dict = {"installed": [], "owed": []}
    homes = [home for home in dependency_homes() if _fetch_allowed(home)]
    entries = default_entries() if homes else []  # no catalog read (or fetch) where nothing may install
    for home in homes if entries else ():
        try:
            outcome = install_defaults(home, say=say, entries=entries)
        except Exception:  # health: allow BLE001 -- one home (unreadable config.yaml) never stops the rest; retried next run
            logger.warning("default plugins skipped for %s", home, exc_info=True)
            outcome = {"installed": [], "owed": [str(home)]}
        summary["installed"] += outcome["installed"]
        summary["owed"] += outcome["owed"]
    return summary


def step_install_default_plugins() -> dict:
    """Boot-bootstrap step. ``owed`` makes boot retry it on later starts (``boot_bootstrap._owed_steps``);
    a start inside the failure backoff skips before any catalog read or network."""
    from hermes_cli.memory_provider_migration import _failed_recently, _note_failure, _note_success
    from pm.environments import dependency_home_root
    root = Path(dependency_home_root())
    if _failed_recently(root, _BACKOFF):
        return {"ok": True, "skipped": "backoff", "owed": True}
    outcome = install_defaults_all_homes(say=lambda message: print(message.strip(), file=sys.stderr))
    if outcome["owed"]:
        _note_failure(root, _BACKOFF)
        return {"ok": True, "installed": outcome["installed"], "owed": True}
    _note_success(root, _BACKOFF)
    return {"ok": True, "installed": outcome["installed"]}
