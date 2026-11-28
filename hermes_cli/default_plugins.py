"""Catalog plugins every home gets: entries marked ``default: true`` (official or verified tier).

Each home gets each default plugin installed at its catalog pin and enabled exactly once. The home
records it under ``_default_plugins_installed`` in its config.yaml (the left-core migrator records its
one-time install the same way), so a user who removes or disables a default plugin keeps that choice
through every later update. A default the home already has, enabled or disabled by the user, is
recorded without reinstalling; one left on disk but never enabled (a refused enable leaves the tree)
is enabled on the next attempt.

When it runs:
- At the end of every source install and ``hermes update`` (``update_cmd_maint``'s maintenance tail,
  in the foreground, for every profile home).
- As a bounded foreground preflight (:func:`ensure_defaults` / ``python -m hermes_cli.default_plugins
  ensure``) before the first backend/gateway start in Docker, Desktop, and ``hermes profile create``
  (the install runs in a detached child that survives a timeout so a deadline never cuts an install
  short).
- On CLI starts (including Nix/Termux and already-maintained revisions), the boot-bootstrap step
  (:func:`step_install_default_plugins`) checks local state only (zero network) and spawns the detached
  child (:func:`spawn_install`) whenever the revision, live catalog cache, or default set needs a pass.

Nothing is fetched where ``security.allow_lazy_installs`` is off, and a failed attempt backs the next
one off (an hour, doubling to a week).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from hermes_cli.default_plugins_ensure import (
    DEFAULT_ENSURE_TIMEOUT, ENSURE_EXIT_OK, ENSURE_EXIT_OWED, ENSURE_EXIT_SKIPPED, ENSURE_EXIT_TIMEOUT,
    PROFILE_CREATE_ENSURE_TIMEOUT, _CHILD_LOCK_WAIT, _RESULT_MAX_AGE, _check_ensure_fast_path,
    _cleanup_stale_results, _ensure_child_work, _ensure_payload, _install_subset_and_record,
    _launch_ensure_child, _pending_names, _read_child_result, _record_settled_only, _result_path,
    _run_child, _write_child_result, ensure_defaults, main,
)

logger = logging.getLogger(__name__)

_INSTALLED_KEY = "_default_plugins_installed"
_BACKOFF = "default-plugins"  # memory_provider_migration failure-stamp name, in the default home


def _is_remote_config_backend(home: Optional[Path] = None) -> bool:
    """Whether the configured config backend is remote, checked without network I/O."""
    from hermes_cli.env_loader import selected_config_backend_name
    return selected_config_backend_name(home) == "remote"


def _live_catalog_unreachable() -> bool:
    """True when the live catalog fetch failed (empty/None OR fell back to an expired stale cache)."""
    from hermes_cli.plugin_catalog import fetch_live_catalog, live_fetch_recently_failed
    return not (fetch_live_catalog() or {}).get("entries") or live_fetch_recently_failed()


def _runs_here(entry) -> bool:
    from hermes_cli.plugins_cmd import PluginOperationError
    from hermes_cli.plugins_cmd_catalog import refuse_incompatible_catalog_entry
    try:
        refuse_incompatible_catalog_entry(entry)  # the install gate's own platform/version verdict
    except PluginOperationError:
        return False
    return True


def _filter_defaults(entries: list) -> list:
    from hermes_cli.plugin_catalog import DEFAULT_TIERS
    return [entry for entry in entries if entry.default and entry.tier in DEFAULT_TIERS and _runs_here(entry)]


def default_entries() -> list:
    """Catalog entries marked default that this OS and this Hermes can run (live catalog, else in-tree).
    The tier is checked here too: the live/in-tree merge can carry ``default`` onto a community entry."""
    from hermes_cli.plugin_catalog import load_catalog_live
    return _filter_defaults(load_catalog_live())


_UNPATCHED_DEFAULT_ENTRIES = default_entries


def _cached_default_entries() -> Optional[list]:
    """Default entries readable without network I/O, or ``None`` when the live cache is missing/expired.

    Honours a monkeypatched :func:`default_entries` in tests; otherwise reads ``cache/plugin-catalog.json``
    via :func:`hermes_cli.plugin_catalog.load_catalog_cached` so startup never blocks on HTTP.
    """
    if default_entries is not _UNPATCHED_DEFAULT_ENTRIES:
        return default_entries()
    from hermes_cli.plugin_catalog import load_catalog_cached
    merged = load_catalog_cached()
    return None if merged is None else _filter_defaults(merged)


def _recorded(home: Path, *, own_only: bool = False) -> set[str]:
    """Plugins *home* already got automatically: this installer's marker, plus (unless *own_only*)
    the left-core migrator's, so a plugin either one installed and the user removed stays removed."""
    from hermes_cli.left_core_migration import _INSTALLED_KEY as _LEFT_CORE_KEY, _read_config
    config = _read_config(home)
    keys = (_INSTALLED_KEY,) if own_only else (_INSTALLED_KEY, _LEFT_CORE_KEY)
    return {str(n) for key in keys if isinstance(config.get(key), list) for n in config[key]}


def _all_recorded(homes: list[Path], entries: list) -> bool:
    """True when every home in *homes* has recorded every entry in *entries*."""
    if not entries:
        return True
    for home in homes:
        done = _recorded(home)
        if any(entry.name not in done for entry in entries):
            return False
    return True


def _record(home: Path, names: set[str]) -> None:
    """Set only the marker key: a one-key write re-reads the file as it writes, so another process's
    edit to the rest of *home*'s config (a running gateway, the user) is never overwritten."""
    from hermes_cli.config import atomic_config_write, read_user_config_raw
    from hermes_cli.config_backend import config_exists, write_config_key
    from hermes_cli.left_core_migration import _stamp_created
    path = home / "config.yaml"
    done = _recorded(home, own_only=True)
    marker = sorted(done | names)
    if config_exists(path):
        write_config_key(path, _INSTALLED_KEY, marker)
        return
    config = read_user_config_raw(path)
    config[_INSTALLED_KEY] = marker
    _stamp_created(home, config)
    atomic_config_write(path, config)


def _installed_dir(home: Path, entry) -> Optional[Path]:
    """*home*'s install of *entry*: the dir its install record ties to the catalog name, in either record
    shape (``catalog.name``, or the older top-level ``catalog_name``), else the dir of that name. Read by
    path, not by binding *home* (the installer's lookup reads the bound home only, and a bind at boot would
    freeze the launch environment)."""
    from hermes_cli.plugins_provenance import read_sidecar_rows
    plugins = home / "plugins"
    for name, row in read_sidecar_rows(plugins).items():
        block = row.get("catalog") if isinstance(row, dict) else None
        named = block.get("name") if isinstance(block, dict) else (row.get("catalog_name") if isinstance(row, dict) else None)
        if named == entry.name and (plugins / name).is_dir():
            return plugins / name
    return plugins / entry.name if (plugins / entry.name).is_dir() else None


def _settled(home: Path, entry) -> bool:
    """*home* already has *entry*: a plugin the user explicitly disabled, a tree the user put there
    (enabled or not, it is theirs), or one an earlier automatic attempt left (this installer's failed
    attempt, or the left-core migrator's refused enable) that someone has since enabled or disabled.
    Explicit ``plugins.disabled`` is checked before any stale ``installing`` marker so a crashed
    automatic attempt never deletes or re-enables a plugin the user disabled."""
    from hermes_cli.left_core_migration import _placed as _left_core_placed
    from hermes_cli.left_core_migration import _plugins_listed, legacy_plugin_keys
    from hermes_cli.memory_provider_migration import _failure_stamp
    from hermes_cli.plugin_identity import plugin_identity, tree_identity
    target = _installed_dir(home, entry)
    # Judged by the names the loader gives the tree (its core-era keys too), else the catalog name's.
    identity = plugin_identity(entry.name) if target is None else tree_identity(home / "plugins", target,
                                                                                 default=entry.name)
    enabled, disabled = _plugins_listed(home, identity)
    if disabled:
        return True
    if _installing(home, entry).exists():
        return False
    if target is None:
        return _placed(home, entry).exists()  # the tree a failed attempt left is gone: the user removed it
    if not _failure_stamp(home, _attempt(entry)).exists() and not _left_core_placed(home, entry.name).exists():
        return True  # neither installer left it unfinished: the user's tree
    # An unfinished tree is kept when the user chose it on or off by its core-era key too (the loader honours
    # both, ``plugins_discovery.gate_manifest``): the retry would discard it.
    return enabled or any(_plugins_listed(home, tuple(sorted(legacy_plugin_keys(entry.name)))))


def _attempt(entry) -> str:
    """Failure-stamp name of this home's automatic install of *entry*."""
    return f"default-{entry.name}"


def _placed(home: Path, entry) -> Path:
    """Written when a failed automatic attempt left *entry*'s tree on disk (a refused enable)."""
    return home / "cache" / f"default-plugins.placed-{entry.name}"


def _installing(home: Path, entry) -> Path:
    """Present while an automatic install of *entry* runs: written before the install starts, removed once
    it reported (ok, or a handled failure that stamped it). One left behind marks an interrupted install
    (the tree may be on disk without its dependencies admitted), which the next pass redoes."""
    return home / "cache" / f"default-plugins.installing-{entry.name}"


def left_unfinished(home: Path, name: str) -> bool:
    """An automatic install of *name* in *home* did not finish: its ``installing`` marker outlived it (killed),
    or its ``placed`` marker says the tree is on disk but the enable was refused. Either tree is owed."""
    cache = home / "cache"
    return (cache / f"default-plugins.installing-{name}").exists() or (cache / f"default-plugins.placed-{name}").exists()


def _phase_timer() -> tuple[Callable[[object], None], Callable[[], str]]:
    """An ``on_step`` hook and a formatter for one install's phases: fetch+place runs until the Python
    packages phase (or, without Python dependencies, the load phase: enabling is folded into placing),
    enable+deps until the load phase, load until the install returned."""
    started = time.monotonic()
    marks: dict[str, float] = {}

    def on_step(phase: object) -> None:
        marks.setdefault(str(getattr(phase, "value", phase)), time.monotonic())

    def report() -> str:
        end = time.monotonic()
        load = marks.get("loading_tools", end)
        deps = min(marks.get("python_packages", load), load)
        return (f"fetch+place {deps - started:.1f}s, enable+deps {load - deps:.1f}s, "
                f"load {end - load:.1f}s, total {end - started:.1f}s")
    return on_step, report


def _invoke_install(install: Callable, name: str, *, discard: Optional[Path],
                    on_step: Callable[[object], None]) -> dict:
    """Call *install*, passing ``on_step`` when the callable accepts it (keeps older test doubles working)."""
    params = inspect.signature(install).parameters
    if "on_step" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return install(name, discard=discard, on_step=on_step)
    return install(name, discard=discard)


def fetch_policy(home: Path) -> str:
    """``"allowed"``, ``"denied"`` (``security.allow_lazy_installs`` off, or the env opt-out) or
    ``"unreadable"`` for *home* (managed overlay applied). Nothing is fetched unless allowed, but an
    unreadable config keeps the home owed, so a repaired file is picked up. Read without binding the
    profile: a whole-profile bind at boot would freeze the launch environment before boot sets it up."""
    if os.environ.get("HERMES_DISABLE_LAZY_INSTALLS", "").strip().lower() in ("1", "true", "yes"):
        return "denied"
    from gateway.config_loader import read_yaml_layers
    try:
        security = read_yaml_layers(home).get("security")
    except Exception:  # health: allow BLE001 -- a malformed or unreadable config never grants a fetch
        logger.debug("default plugins: unreadable config for %s", home, exc_info=True)
        return "unreadable"
    allowed = (security if isinstance(security, dict) else {}).get("allow_lazy_installs", True) is True
    return "allowed" if allowed else "denied"


def fetch_allowed(home: Path) -> bool:
    return fetch_policy(home) == "allowed"


def install_defaults(home: Path, *, say: Callable[[str], None] = logger.info,
                     entries: Optional[list] = None) -> dict:
    """Install and enable every default plugin *home* never had. ``installed`` lists what this call
    installed, ``owed`` what is still to fetch (a failed install), ``raced`` the owed ones whose enable only
    lost to other writers of the plugin selection (only when there are any). One plugin's failure never raises."""
    from hermes_cli.left_core_migration import _first_cause, _install_into
    from hermes_cli.memory_provider_migration import _home_label, _note_failure, _note_success, kept_off_sentence
    result: dict = {"installed": [], "owed": []}
    policy = fetch_policy(home)
    if policy == "unreadable":
        return {**result, "owed": [str(home)]}
    if policy == "denied":
        return {**result, "skipped": "lazy-installs-disabled"}
    done = _recorded(home)
    pending = [entry for entry in (default_entries() if entries is None else entries) if entry.name not in done]
    have = {entry.name for entry in pending if _settled(home, entry)}
    install = _install_into(home, consent=True)  # a reviewed default: its dependencies come with it
    for entry in pending:
        if entry.name in have:
            _installing(home, entry).unlink(missing_ok=True)
            continue
        on_step, timing = _phase_timer()
        try:
            # A tree still on disk here is one an earlier automatic attempt left and nobody chose since
            # (_settled): discarded first, so a re-pin cannot wedge the retry at "already exists". Its
            # placed marker goes with it, since a tree missing because WE removed it is not the user's
            # removal. The install runs at this home's catalog pin, never an explicit ref (a ref off that
            # pin is recorded as a user's ``--ref`` hold that unattended updates never move), and
            # unforced, so it never overwrites another plugin already at the destination.
            # The installing marker goes down first and comes off only once the install reported, so
            # a kill between placing the tree and admitting its dependencies is redone next pass.
            leftover = _installed_dir(home, entry)
            marker = _installing(home, entry)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
            if leftover is not None:
                _placed(home, entry).unlink(missing_ok=True)
            outcome = _invoke_install(install, entry.name, discard=leftover, on_step=on_step)
        except Exception as exc:  # health: allow BLE001 -- network, uv or kill list: reported per plugin, never raised
            outcome = {"ok": False, "error": str(exc)}
        say(f"  · timing {entry.name}: {timing()} ({'ok' if outcome.get('ok') else 'failed'})")
        if outcome.get("ok"):
            _note_success(home, _attempt(entry))
            _placed(home, entry).unlink(missing_ok=True)
            _installing(home, entry).unlink(missing_ok=True)
            have.add(entry.name)
            result["installed"].append(entry.name)
            say(f"  ⚠ Installed default plugin '{entry.name}' for {_home_label(home)}, but "
                f"{kept_off_sentence(entry.name, home)}" if outcome.get("kept_disabled") else
                f"  ✓ Installed default plugin '{entry.name}' for {_home_label(home)}")
            continue
        result["owed"].append(entry.name)
        if outcome.get("raced"):
            result.setdefault("raced", []).append(entry.name)
        _note_failure(home, _attempt(entry))  # marks a tree it left as ours to retry, not the user's
        if _installed_dir(home, entry) is not None:
            _placed(home, entry).parent.mkdir(parents=True, exist_ok=True)
            _placed(home, entry).touch()
        _installing(home, entry).unlink(missing_ok=True)  # a handled failure: the stamp above now governs
        say(f"  ⚠ Default plugin '{entry.name}' could not be installed for {_home_label(home)}: "
            f"{_first_cause(str(outcome.get('error') or ''))}. Hermes retries at a later start; "
            f"`hermes plugins install {entry.name}` installs it now.")
    if have:
        _record(home, have)
    return result


def _install_defaults_for_homes(
    target_homes: list[Path],
    *,
    say: Callable[[str], None] = print,
    wait: float = 900.0,
    lock: Optional[Callable] = None,
    entries: Optional[list] = None,
    _entries_out: Optional[list] = None,
    _allowed_out: Optional[list] = None,
) -> dict:
    """Install defaults for *target_homes*. Catalog is read once; one home's failure never stops the next."""
    summary: dict = {"installed": [], "owed": []}
    policies = {home: fetch_policy(home) for home in target_homes}
    summary["owed"] += [str(home) for home, policy in policies.items() if policy == "unreadable"]
    homes = [home for home, policy in policies.items() if policy == "allowed"]
    if _allowed_out is not None:
        _allowed_out[:] = homes
    resolved_entries = (default_entries() if entries is None else list(entries)) if homes else []
    if _entries_out is not None:
        _entries_out[:] = resolved_entries
    if homes and _live_catalog_unreachable():  # an expired offline cache has none or just failed HTTP
        # Offline: the in-tree defaults still install, but one the live catalog added since this release
        # is unknown, so the pass is owed until the live catalog is read.
        summary["owed"].append("live catalog")
    if not resolved_entries:
        return summary
    acquire_lock = lock if lock is not None else (lambda: installer_lock(wait))
    asked = time.monotonic()
    with acquire_lock():  # never two installers in one home (an update tail beside a detached child)
        locked = time.monotonic()
        for home in homes:
            try:
                outcome = install_defaults(home, say=say, entries=resolved_entries)
            except Exception:  # health: allow BLE001 -- one home (unreadable config.yaml) never stops the rest; retried next run
                logger.warning("default plugins skipped for %s", home, exc_info=True)
                outcome = {"installed": [], "owed": [str(home)]}
            summary["installed"] += outcome["installed"]
            summary["owed"] += outcome["owed"]
            if outcome.get("raced"):
                summary.setdefault("raced", []).extend(outcome["raced"])
    if summary["installed"] or summary["owed"]:  # a pass that installed nothing stays quiet (update tail)
        say(f"default plugins pass: lock wait {locked - asked:.1f}s, installs {time.monotonic() - locked:.1f}s, "
            f"{len(homes)} home(s), installed {summary['installed'] or 'none'}, owed {summary['owed'] or 'none'}")
    return summary


def install_defaults_all_homes(*, say: Callable[[str], None] = print, wait: float = 900.0) -> dict:
    """Every profile home sharing this install (the default home plus each live profile). The catalog is
    read once; one home's failure never stops the next. Waits up to *wait* seconds for another installer
    to finish (``TimeoutError`` past that)."""
    from pm.plugins_state import dependency_homes
    return _install_defaults_for_homes(list(dependency_homes()), say=say, wait=wait)


def installer_lock(wait: float):
    """The OS-held lock every unattended plugin installer holds: this default set, the left-core migrator
    and the gateway-start catalog platform install (the kernel frees it when its holder dies). Raises
    ``TimeoutError`` when another holder keeps it past *wait* seconds. Not reentrant: hold it around one
    pass, never around a call that takes it again."""
    from hermes_cli.auth import _file_lock
    from pm.environments import dependency_home_root
    return _file_lock(_state(Path(dependency_home_root()), "lock"), threading.local(), wait,
                      "another default plugin install is running")


# ── settled marker (identity + default-set hash) ─────────────────────────────

def _pass_in_flight(root: Path) -> bool:
    """Another live process holds :func:`installer_lock` (a holder stamps its pid into the lock file
    while it holds it). Read, never acquired: a boot must not wait on it."""
    from hermes_cli._subprocess_compat import pid_exists_stdlib
    try:
        pid = int(_state(root, "lock").read_text(encoding="utf-8-sig").split()[0])
    except (OSError, ValueError, IndexError):
        return False
    return pid > 0 and pid != os.getpid() and pid_exists_stdlib(pid)


def _state(root: Path, name: str) -> Path:
    return root / "cache" / f"default-plugins.{name}"


def _identity() -> str:
    from hermes_cli.boot_bootstrap import current_install_identity
    from pm.paths import install_root
    return current_install_identity(install_root()) or ""


def _defaults_hash(entries: list) -> str:
    """Stable digest of the sorted default plugin names and pinned SHAs."""
    spec = "\n".join(sorted(f"{e.name}:{getattr(e, 'sha', '')}" for e in entries))
    return hashlib.sha256(spec.encode("utf-8")).hexdigest()


def _live_cache_signature() -> Optional[str]:
    """Fast freshness + file-stat signature of ``cache/plugin-catalog.json`` when present and within TTL."""
    from hermes_cli.plugin_catalog import live_cache_signature
    return live_cache_signature()


def _read_settled_record(path: Path) -> dict:
    """Read ``cache/default-plugins.settled``; legacy plain-text identity files return ``defaults_hash=None``."""
    empty = {"identity": "", "defaults_hash": None, "defaults": None, "live_cache_sig": None}
    try:
        raw = path.read_text(encoding="utf-8-sig").strip() if path.is_file() else ""
    except OSError:
        return empty
    if not raw:
        return empty
    if not raw.startswith("{"):
        return {**empty, "identity": raw}
    try:
        data = json.loads(raw)
    except ValueError:
        return empty
    if not isinstance(data, dict):
        return empty
    dhash, live_sig, defaults = data.get("defaults_hash"), data.get("live_cache_sig"), data.get("defaults")
    return {
        "identity": data["identity"] if isinstance(data.get("identity"), str) else "",
        "defaults_hash": dhash if isinstance(dhash, str) and dhash else None,
        "defaults": [str(n) for n in defaults] if isinstance(defaults, list) else None,
        "live_cache_sig": live_sig if isinstance(live_sig, str) and live_sig else None,
    }


def _read_settled(path: Path) -> tuple[str, Optional[str]]:
    """Read ``(identity, defaults_hash)`` from ``cache/default-plugins.settled``."""
    rec = _read_settled_record(path)
    return rec["identity"], rec["defaults_hash"]


def _write_settled(path: Path, identity: str, defaults_hash: str, *,
                   default_names: Optional[list[str]] = None, live_cache_sig: Optional[str] = None) -> None:
    from utils import atomic_write_text
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {"identity": identity, "defaults_hash": defaults_hash}
    if default_names is not None:
        payload["defaults"] = sorted(default_names)
    if live_cache_sig is not None:
        payload["live_cache_sig"] = live_cache_sig
    atomic_write_text(path, json.dumps(payload), tmp_prefix=f"{path.name}.tmp-")


def _is_settled_for(root: Path, entries: list) -> bool:
    saved_id, saved_hash = _read_settled(_state(root, "settled"))
    return saved_hash is not None and saved_id == _identity() and saved_hash == _defaults_hash(entries)


def _fast_settled_hit(root: Path, allowed_homes: list[Path]) -> bool:
    """True when the settled marker proves the install identity, fresh live catalog cache, and every
    allowed home's recorded defaults are unchanged — avoiding parsing all in-tree catalog YAML files."""
    if default_entries is not _UNPATCHED_DEFAULT_ENTRIES:
        return False
    rec = _read_settled_record(_state(root, "settled"))
    if rec["defaults_hash"] is None or rec["live_cache_sig"] is None or rec["defaults"] is None:
        return False
    live_sig = _live_cache_signature()
    if live_sig is None or live_sig != rec["live_cache_sig"] or rec["identity"] != _identity():
        return False
    expected = rec["defaults"]
    return not expected or all(all(name in _recorded(home) for name in expected) for home in allowed_homes)


# ── detached child & boot-bootstrap step ─────────────────────────────────────

def _child_argv(args: tuple[str, ...] = ()) -> list[str]:
    """The installer child's command: this installation's runtime running this module."""
    from hermes_cli._launchers import runtime_command
    from pm.paths import install_root
    return runtime_command(install_root(), args, module="hermes_cli.default_plugins")


def _popen_detached(argv: list[str]) -> subprocess.Popen:
    """Start *argv* in its own session (Windows: own process group, out of the parent's job), with no
    stdio shared with this process, so neither killing this process nor its parent (Desktop kills only
    its direct child) stops it, and no pipe of the caller is held open past this process."""
    from hermes_cli._subprocess_compat import (
        IS_WINDOWS,
        windows_detach_flags_without_breakaway,
        windows_detach_popen_kwargs,
    )
    kwargs = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
    try:
        # detached-spawn: ok — git and dependency installs must neither delay a boot or a create nor die with it; one installer at a time (OS lock)
        return subprocess.Popen(argv, **kwargs, **windows_detach_popen_kwargs())
    except OSError:
        if not IS_WINDOWS:
            raise
        # A job object without BREAKAWAY_OK refuses CREATE_BREAKAWAY_FROM_JOB; retry without it.
        return subprocess.Popen(argv, **kwargs, creationflags=windows_detach_flags_without_breakaway())


def spawn_install(home: Optional[Path] = None) -> bool:
    """Start the detached child that installs the defaults for every home, unless the last attempt failed
    recently or an install pass is running now (it reports for this revision; a boot only asks that one
    runs). *home* (a profile just created; it must allow fetching) is a rescan request: a running pass
    scans again when it finishes cleanly. Never waits."""
    from hermes_cli.memory_provider_migration import _failed_recently
    from pm.environments import dependency_home_root
    root = Path(dependency_home_root())
    if home is not None and fetch_policy(home) == "denied":
        return False
    _state(root, "settled").unlink(missing_ok=True)  # unsettled until a pass installs it, so every boot re-checks
    if _failed_recently(root, _BACKOFF):
        return False
    if home is not None:
        rerun = _state(root, "rerun")
        rerun.parent.mkdir(parents=True, exist_ok=True)
        rerun.touch()
    if _pass_in_flight(root):
        return False
    _popen_detached(_child_argv())
    return True


def install_and_record(*, say: Callable[[str], None] = print, wait: float = 900.0) -> dict:
    """The one coordinator (update tail, foreground ensure, and detached child): installs defaults across
    every home again until no rescan was requested meanwhile (a profile created during the pass), then
    records the outcome: a failure begins the backoff, a clean pass marks this revision + default set
    settled. Both are written before the installer lock is released, so no boot in between starts another
    attempt; a failed pass is never repeated here (the backoff decides when the next one runs)."""
    from contextlib import nullcontext
    from hermes_cli.memory_provider_migration import _note_failure, _note_success
    from pm.environments import dependency_home_root
    from pm.plugins_state import dependency_homes
    root = Path(dependency_home_root())
    rerun = _state(root, "rerun")
    used_entries: list = []
    allowed_homes: list = []
    installed: list[str] = []
    with installer_lock(wait):
        while True:
            started = time.time()
            outcome = _install_defaults_for_homes(
                list(dependency_homes()),
                say=say,
                lock=nullcontext,
                _entries_out=used_entries,
                _allowed_out=allowed_homes,
            )
            # A rescan pass (requested while this one ran) usually finds nothing left: keep what the earlier
            # passes installed, so the reported outcome is this run's, not only its last pass's.
            installed += [name for name in outcome.get("installed", []) if name not in installed]
            # An enable that only lost to other selection writers is owed without a backoff: the next start retries.
            failed = set(outcome["owed"]) - set(outcome.get("raced", ()))
            if failed or not (rerun.exists() and rerun.stat().st_mtime >= started):
                break
        outcome = {**outcome, "installed": installed}
        if outcome["owed"]:
            if failed:
                _note_failure(root, _BACKOFF)
            # A later pass (a profile created after this revision settled) can fail too: owed again.
            _state(root, "settled").unlink(missing_ok=True)
            return outcome
        _note_success(root, _BACKOFF)
        if allowed_homes:
            _write_settled(
                _state(root, "settled"),
                _identity(),
                _defaults_hash(used_entries),
                default_names=[e.name for e in used_entries],
                live_cache_sig=_live_cache_signature(),
            )
        return outcome


def step_install_default_plugins() -> dict:
    """Boot-bootstrap step, never blocking startup on the network: owed until a pass reports every default
    settled for this revision AND recorded in every allowed home for the cached catalog's default set;
    when local files cannot prove settled (e.g. cache missing/expired), spawns the detached child."""
    from pm.environments import dependency_home_root
    from pm.plugins_state import dependency_homes
    try:
        if _is_remote_config_backend():
            return {"ok": True, "skipped": "remote-config-backend"}
        homes = list(dependency_homes())
        if any(_is_remote_config_backend(home) for home in homes):
            return {"ok": True, "skipped": "remote-config-backend"}
        policies = {home: fetch_policy(home) for home in homes}
        if homes and all(policy == "denied" for policy in policies.values()):
            return {"ok": True, "skipped": "lazy-installs-disabled"}
        root = Path(dependency_home_root())
        allowed_homes = [home for home, policy in policies.items() if policy == "allowed"]
        if all(policy != "unreadable" for policy in policies.values()):
            if _fast_settled_hit(root, allowed_homes):
                return {"ok": True}
            saved_id, saved_hash = _read_settled(_state(root, "settled"))
            if saved_hash is not None and saved_id == _identity():
                cached_entries = _cached_default_entries()
                if (
                    cached_entries is not None
                    and saved_hash == _defaults_hash(cached_entries)
                    and _all_recorded(allowed_homes, cached_entries)
                ):
                    return {"ok": True}
        return {"ok": True, "spawned": spawn_install(), "owed": True}
    except (OSError, ValueError) as exc:  # unreadable profiles dir, failed spawn: still owed, retried next boot
        return {"ok": False, "error": str(exc), "owed": True}


__all__ = [
    "DEFAULT_ENSURE_TIMEOUT", "ENSURE_EXIT_OK", "ENSURE_EXIT_OWED", "ENSURE_EXIT_SKIPPED",
    "ENSURE_EXIT_TIMEOUT", "PROFILE_CREATE_ENSURE_TIMEOUT", "default_entries", "ensure_defaults",
    "fetch_allowed", "fetch_policy", "install_and_record", "install_defaults",
    "install_defaults_all_homes", "installer_lock", "main", "spawn_install", "step_install_default_plugins",
]


if __name__ == "__main__":
    sys.exit(main())
