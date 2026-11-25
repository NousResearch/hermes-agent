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


def pending_entries(home: Path) -> list:
    """Platform entries *home* has the credentials for and has never had installed. Read-only."""
    from hermes_cli.default_plugins import _installed_dir, _recorded
    from hermes_cli.left_core_migration import LEFT_CORE, _plugins_listed
    from hermes_cli.plugin_catalog import load_catalog_live
    owned = {feature.plugin for feature in LEFT_CORE}
    done = _recorded(home)
    return [entry for entry in load_catalog_live()
            if entry.category == "platform" and entry.name not in owned and entry.name not in done
            and _configured(entry, home) and _installed_dir(home, entry) is None
            and not _plugins_listed(home, (entry.name,))[1]]


def _installs_unattended(entry) -> bool:
    from hermes_cli.default_plugins import default_entries
    return any(default.name == entry.name for default in default_entries())


def _label(entry) -> str:
    return entry.title or entry.name


def recover_in(home: Path, *, report: Callable[[str], None]) -> list[str]:
    """Install or point to every :func:`pending_entries` entry for *home* (the active, bound home). Returns
    the installed plugin names; never raises. Blocking: call it off the event loop."""
    from hermes_cli.left_core_migration import settings_fingerprint
    from hermes_constants import hermes_home_key
    key = hermes_home_key(home)
    if _attempted.get(key) == settings_fingerprint(home):
        return []
    try:
        return _recover(home, report=report)
    finally:  # after the attempt: its own config.yaml writes (the install record) are no change by the user
        _attempted[key] = settings_fingerprint(home)


def _recover(home: Path, *, report: Callable[[str], None]) -> list[str]:
    from hermes_cli.default_plugins import fetch_allowed, install_defaults, installer_lock
    from hermes_cli.left_core_migration import _STARTUP_LOCK_WAIT
    from hermes_cli.memory_provider_migration import _install_command
    try:
        entries = pending_entries(home)
        unattended = [entry for entry in entries if _installs_unattended(entry)] if fetch_allowed(home) else []
        for entry in entries:
            if entry in unattended:
                continue
            env = ", ".join(entry.capabilities.requires_env)
            why = ("security.allow_lazy_installs is off, so Hermes did not fetch it"
                   if not fetch_allowed(home) and _installs_unattended(entry)
                   else f"Hermes installs {entry.tier}-tier plugins only when you ask")
            report(f"⚠ This profile sets {env}, but the '{entry.name}' platform plugin for {_label(entry)} is not "
                   f"installed, so the gateway does not serve it ({why}). Run `{_install_command(entry.name, home)}`.")
        if not unattended:
            return []
        try:
            with installer_lock(_STARTUP_LOCK_WAIT):
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
