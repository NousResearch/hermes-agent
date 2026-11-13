"""Catalog plugins every home gets: entries marked ``default: true`` (official or verified tier).

Each home gets each default plugin installed at its catalog pin and enabled exactly once. The home
records it under ``_default_plugins_installed`` in its config.yaml (the left-core migrator records its
one-time install the same way), so a user who removes or disables a default plugin keeps that choice
through every later update. A default the home already has, enabled or disabled by the user, is
recorded without reinstalling; one left on disk but never enabled (a refused enable leaves the tree)
is enabled on the next attempt.

When it runs: at the end of every source install and ``hermes update`` (``update_cmd_maint``'s
maintenance tail, in the foreground, for every profile home). Installs that never run that tail
(bundled Desktop, Docker, Nix, Termux) and a newly created profile start a detached child instead
(:func:`spawn_install`), so neither a start nor a create waits on git and dependency installs: the
boot bootstrap starts it once per installed revision and on later starts until a child reports every
default settled. Nothing is fetched where ``security.allow_lazy_installs`` is off, and a failed
attempt backs the next one off (an hour, doubling to a week).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_INSTALLED_KEY = "_default_plugins_installed"
_BACKOFF = "default-plugins"  # memory_provider_migration failure-stamp name, in the default home


def default_entries() -> list:
    """Catalog entries marked default that this OS and this Hermes can run (live catalog, else in-tree).
    The tier is checked here too: the live/in-tree merge can carry ``default`` onto a community entry."""
    from hermes_cli.plugin_catalog import DEFAULT_TIERS, load_catalog_live
    from hermes_cli.plugins_cmd import PluginOperationError
    from hermes_cli.plugins_cmd_catalog import refuse_incompatible_catalog_entry

    def runs_here(entry) -> bool:
        try:
            refuse_incompatible_catalog_entry(entry)  # the install gate's own platform/version verdict
        except PluginOperationError:
            return False
        return True
    return [entry for entry in load_catalog_live() if entry.default and entry.tier in DEFAULT_TIERS and runs_here(entry)]


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


def _installed_dir(home: Path, entry) -> Optional[Path]:
    """*home*'s install of *entry*: the dir its install record ties to the catalog name, else its name."""
    from hermes_cli.plugins_provenance import read_sidecar_rows
    plugins = home / "plugins"
    for name, row in read_sidecar_rows(plugins).items():
        block = row.get("catalog") if isinstance(row, dict) else None
        if isinstance(block, dict) and block.get("name") == entry.name and (plugins / name).is_dir():
            return plugins / name
    return plugins / entry.name if (plugins / entry.name).is_dir() else None


def _settled(home: Path, entry) -> bool:
    """*home* already has *entry*: a tree the user put there (enabled or not, it is theirs), or one an
    earlier automatic attempt left that someone has since enabled or disabled. A tree a refused enable
    left behind (stamped by that attempt, in neither list) is retried."""
    from hermes_cli.left_core_migration import _plugins_listed
    from hermes_cli.memory_provider_migration import _failure_stamp
    target = _installed_dir(home, entry)
    if target is None:
        return False
    if not _failure_stamp(home, _attempt(entry)).exists():
        return True
    names = {entry.name, target.name}
    try:
        from hermes_yaml import safe_load
        names.add(str((safe_load((target / "plugin.yaml").read_text(encoding="utf-8-sig")) or {}).get("name") or ""))
    except (OSError, ValueError, AttributeError):  # no or malformed manifest: the dir names count
        pass
    enabled, disabled = _plugins_listed(home, tuple(names - {""}))
    return enabled or disabled


def _attempt(entry) -> str:
    """Failure-stamp name of this home's automatic install of *entry*."""
    return f"default-{entry.name}"


def fetch_allowed(home: Path) -> bool:
    """``security.allow_lazy_installs`` for *home* (managed overlay applied), read without binding the
    profile: a whole-profile bind at boot would freeze the launch environment before boot sets it up."""
    if os.environ.get("HERMES_DISABLE_LAZY_INSTALLS", "").strip().lower() in ("1", "true", "yes"):
        return False
    from gateway.config_loader import read_yaml_layers
    try:
        security = read_yaml_layers(home).get("security")
    except Exception:  # health: allow BLE001 -- a malformed config never grants a fetch (as pm.install decides)
        logger.debug("default plugins: unreadable config for %s", home, exc_info=True)
        return False
    return (security if isinstance(security, dict) else {}).get("allow_lazy_installs", True) is True


def install_defaults(home: Path, *, say: Callable[[str], None] = logger.info,
                     entries: Optional[list] = None) -> dict:
    """Install and enable every default plugin *home* never had. ``installed`` lists what this call
    installed, ``owed`` what is still to fetch (a failed install). One plugin's failure never raises."""
    from hermes_cli.left_core_migration import _first_cause, _install_into
    from hermes_cli.memory_provider_migration import _home_label, _note_failure, _note_success
    result: dict = {"installed": [], "owed": []}
    if not fetch_allowed(home):
        return {**result, "skipped": "lazy-installs-disabled"}
    done = _recorded(home)
    pending = [entry for entry in (default_entries() if entries is None else entries) if entry.name not in done]
    have = {entry.name for entry in pending if _settled(home, entry)}
    install = _install_into(home, consent=True)  # a reviewed default: its dependencies come with it
    for entry in pending:
        if entry.name in have:
            continue
        try:
            outcome = install(entry.name)  # reuses, and enables, a tree a refused enable left behind
        except Exception as exc:  # health: allow BLE001 -- network, uv or kill list: reported per plugin, never raised
            outcome = {"ok": False, "error": str(exc)}
        if outcome.get("ok"):
            _note_success(home, _attempt(entry))
            have.add(entry.name)
            result["installed"].append(entry.name)
            say(f"  ✓ Installed default plugin '{entry.name}' for {_home_label(home)}")
            continue
        result["owed"].append(entry.name)
        _note_failure(home, _attempt(entry))  # marks a tree it left as ours to retry, not the user's
        say(f"  ⚠ Default plugin '{entry.name}' could not be installed for {_home_label(home)}: "
            f"{_first_cause(str(outcome.get('error') or ''))}. Hermes retries at a later start; "
            f"`hermes plugins install {entry.name}` installs it now.")
    if have:
        _record(home, have)
    return result


def install_defaults_all_homes(*, say: Callable[[str], None] = print, wait: float = 900.0) -> dict:
    """Every profile home sharing this install (the default home plus each live profile). The catalog is
    read once; one home's failure never stops the next. Waits up to *wait* seconds for another installer
    to finish (``TimeoutError`` past that)."""
    from hermes_cli.plugin_catalog import fetch_live_catalog
    from pm.plugins_state import dependency_homes
    summary: dict = {"installed": [], "owed": []}
    homes = [home for home in dependency_homes() if fetch_allowed(home)]
    entries = default_entries() if homes else []  # no catalog read (or fetch) where nothing may install
    if homes and fetch_live_catalog() is None:
        # Offline: the in-tree defaults still install, but one the live catalog added since this release
        # is unknown, so the pass is owed until the live catalog is read.
        summary["owed"].append("live catalog")
    if not entries:
        return summary
    with _machine_lock(wait):  # never two installers in one home (an update tail beside a detached child)
        for home in homes:
            try:
                outcome = install_defaults(home, say=say, entries=entries)
            except Exception:  # health: allow BLE001 -- one home (unreadable config.yaml) never stops the rest; retried next run
                logger.warning("default plugins skipped for %s", home, exc_info=True)
                outcome = {"installed": [], "owed": [str(home)]}
            summary["installed"] += outcome["installed"]
            summary["owed"] += outcome["owed"]
    return summary


def _machine_lock(wait: float):
    """The OS-held lock every installer of the default set holds (the kernel frees it when its holder dies).
    Raises ``TimeoutError`` when another holder keeps it past *wait* seconds."""
    from hermes_cli.auth import _file_lock
    from pm.environments import dependency_home_root
    return _file_lock(_state(Path(dependency_home_root()), "lock"), threading.local(), wait,
                      "another default plugin install is running")


# ── detached child: first start, profile create ──────────────────────────────

def _state(root: Path, name: str) -> Path:
    return root / "cache" / f"default-plugins.{name}"


def _identity() -> str:
    from hermes_cli.boot_bootstrap import current_install_identity
    from pm.paths import install_root
    return current_install_identity(install_root()) or ""


def spawn_install(home: Optional[Path] = None) -> bool:
    """Start the detached child that installs the defaults for every home (*home*, when given, must allow
    fetching), unless the last attempt failed recently. A running child sees the request when it finishes
    and scans again. Never waits."""
    from hermes_cli._launchers import runtime_command
    from hermes_cli._subprocess_compat import windows_detach_popen_kwargs
    from hermes_cli.memory_provider_migration import _failed_recently
    from pm.environments import dependency_home_root
    from pm.paths import install_root
    root = Path(dependency_home_root())
    if (home is not None and not fetch_allowed(home)) or _failed_recently(root, _BACKOFF):
        return False
    rerun = _state(root, "rerun")
    rerun.parent.mkdir(parents=True, exist_ok=True)
    rerun.touch()
    argv = runtime_command(install_root(), module="hermes_cli.default_plugins")
    kwargs = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
    # detached-spawn: ok — git and dependency installs must neither delay a boot or a create nor die with it; one installer at a time (OS lock)
    subprocess.Popen(argv, **kwargs, **windows_detach_popen_kwargs())
    return True


def _run_child() -> int:
    """The detached child. Scans until no request arrived while it worked, then records the outcome the boot
    step reads; a child that finds another installer running leaves the request to it."""
    import time
    from pm.environments import dependency_home_root
    root = Path(dependency_home_root())
    (root / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s default-plugins %(message)s",
                        filename=str(root / "logs" / "default-plugins.log"))
    try:
        while True:
            started = time.time()
            outcome = install_and_record(say=logger.info, wait=1.0)
            rerun = _state(root, "rerun")
            if not (rerun.exists() and rerun.stat().st_mtime > started):
                break
    except TimeoutError:
        return 0
    return 1 if outcome["owed"] else 0


def install_and_record(*, say: Callable[[str], None] = print, wait: float = 900.0) -> dict:
    """:func:`install_defaults_all_homes`, then the outcome the boot step reads: a failure begins the
    backoff, a clean pass marks this revision settled (so its first boot spawns nothing)."""
    from hermes_cli.memory_provider_migration import _note_failure, _note_success
    from pm.environments import dependency_home_root
    root = Path(dependency_home_root())
    outcome = install_defaults_all_homes(say=say, wait=wait)
    if outcome["owed"]:
        _note_failure(root, _BACKOFF)
        # A later pass (a profile created after this revision settled) can fail too: owed again.
        _state(root, "settled").unlink(missing_ok=True)
        return outcome
    _note_success(root, _BACKOFF)
    _state(root, "settled").parent.mkdir(parents=True, exist_ok=True)
    _state(root, "settled").write_text(_identity(), encoding="utf-8")
    return outcome


def step_install_default_plugins() -> dict:
    """Boot-bootstrap step, never blocking the start: owed (later starts re-check, without network) until a
    child reports every default settled for this revision; an owed start outside the backoff starts one."""
    from pm.environments import dependency_home_root
    from pm.plugins_state import dependency_homes
    try:
        if not any(fetch_allowed(home) for home in dependency_homes()):
            return {"ok": True, "skipped": "lazy-installs-disabled"}
        settled = _state(Path(dependency_home_root()), "settled")
        if settled.is_file() and settled.read_text(encoding="utf-8-sig").strip() == _identity():
            return {"ok": True}
        return {"ok": True, "spawned": spawn_install(), "owed": True}
    except (OSError, ValueError) as exc:  # unreadable profiles dir, failed spawn: still owed, retried next boot
        return {"ok": False, "error": str(exc), "owed": True}


if __name__ == "__main__":
    sys.exit(_run_child())
