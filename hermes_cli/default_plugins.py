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

import argparse
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

logger = logging.getLogger(__name__)

_INSTALLED_KEY = "_default_plugins_installed"
_BACKOFF = "default-plugins"  # memory_provider_migration failure-stamp name, in the default home

ENSURE_EXIT_OK = 0
ENSURE_EXIT_OWED = 2
ENSURE_EXIT_TIMEOUT = 3
ENSURE_EXIT_SKIPPED = 4
DEFAULT_ENSURE_TIMEOUT = 300.0
PROFILE_CREATE_ENSURE_TIMEOUT = 120.0
_CHILD_LOCK_WAIT = 900.0  # an ensure child waits out another installer instead of exiting with nothing done


def _runs_here(entry) -> bool:
    from hermes_cli.plugins_cmd import PluginOperationError
    from hermes_cli.plugins_cmd_catalog import refuse_incompatible_catalog_entry
    try:
        refuse_incompatible_catalog_entry(entry)  # the install gate's own platform/version verdict
    except PluginOperationError:
        return False
    return True


def default_entries() -> list:
    """Catalog entries marked default that this OS and this Hermes can run (live catalog, else in-tree).
    The tier is checked here too: the live/in-tree merge can carry ``default`` onto a community entry."""
    from hermes_cli.plugin_catalog import DEFAULT_TIERS, load_catalog_live
    return [entry for entry in load_catalog_live() if entry.default and entry.tier in DEFAULT_TIERS and _runs_here(entry)]


_UNPATCHED_DEFAULT_ENTRIES = default_entries


def _entries_from_catalog_doc(data: dict) -> Optional[list]:
    """Resolve default entries from an already-loaded live-catalog JSON dict without network I/O."""
    from hermes_cli import plugin_catalog as pc
    raw_entries = data.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        return None
    entries = [e for i, raw in enumerate(raw_entries)
               if (e := pc.entry_from_mapping(raw, f"{pc.LIVE_CATALOG_URL}#{i}")) is not None]
    if not entries:
        return None
    in_tree = {e.name: e for e in pc.load_catalog()}
    live_t, tree_t = pc._live_generated_time(data), pc.in_tree_catalog_time()
    tree_is_newer = (tree_t > live_t) if (live_t is not None and tree_t is not None) else None
    raw_by_name = {str(raw.get("name")): raw for raw in raw_entries if isinstance(raw, dict)}
    merged = [
        in_tree[e.name] if e.name in in_tree and pc._prefer_in_tree_entry(in_tree[e.name], e, tree_is_newer)
        else pc._with_curated_fields(e, in_tree.get(e.name), raw_by_name.get(e.name) or {})
        for e in entries
    ]
    return [entry for entry in merged if entry.default and entry.tier in pc.DEFAULT_TIERS and _runs_here(entry)]


def _cached_default_entries() -> Optional[list]:
    """Default entries readable without network I/O, or ``None`` when the live cache is missing/expired.

    Honours a monkeypatched :func:`default_entries` in tests; otherwise reads ``cache/plugin-catalog.json``
    only while it is within ``LIVE_CATALOG_TTL_SECONDS`` so CLI startup never blocks on HTTP.
    """
    if default_entries is not _UNPATCHED_DEFAULT_ENTRIES:
        return default_entries()
    from hermes_cli.plugin_catalog import LIVE_CATALOG_TTL_SECONDS, _live_cache_path
    cache = _live_cache_path()
    try:
        st = cache.stat()
        if time.time() - st.st_mtime >= LIVE_CATALOG_TTL_SECONDS:
            return None
        data = json.loads(cache.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return _entries_from_catalog_doc(data)


def _recorded(home: Path, *, own_only: bool = False) -> set[str]:
    """Plugins *home* already got automatically: this installer's marker, plus (unless *own_only*)
    the left-core migrator's, so a plugin either one installed and the user removed stays removed."""
    from hermes_cli.left_core_migration import _INSTALLED_KEY as _LEFT_CORE_KEY, _read_config
    config = _read_config(home)
    keys = (_INSTALLED_KEY,) if own_only else (_INSTALLED_KEY, _LEFT_CORE_KEY)
    return {str(n) for key in keys if isinstance(config.get(key), list) for n in config[key]}


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
    """*home* already has *entry*: a tree the user put there (enabled or not, it is theirs), or one an
    earlier automatic attempt left that someone has since enabled or disabled. A tree a refused enable
    left behind (stamped by that attempt, in neither list) is retried, and so is anything an install that
    never finished left (its ``installing`` marker survives a kill, crash or power loss)."""
    from hermes_cli.left_core_migration import _plugins_listed
    from hermes_cli.memory_provider_migration import _failure_stamp
    if _installing(home, entry).exists():
        return False
    target = _installed_dir(home, entry)
    if target is None:
        return _placed(home, entry).exists()  # the tree a failed attempt left is gone: the user removed it
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


def _placed(home: Path, entry) -> Path:
    """Written when a failed automatic attempt left *entry*'s tree on disk (a refused enable)."""
    return home / "cache" / f"default-plugins.placed-{entry.name}"


def _installing(home: Path, entry) -> Path:
    """Present while an automatic install of *entry* runs: written before the install starts, removed once
    it reported (ok, or a handled failure that stamped it). One left behind marks an interrupted install
    (the tree may be on disk without its dependencies admitted), which the next pass redoes."""
    return home / "cache" / f"default-plugins.installing-{entry.name}"


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
    from hermes_cli.memory_provider_migration import _home_label, _note_failure, _note_success
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
            say(f"  ✓ Installed default plugin '{entry.name}' for {_home_label(home)}")
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
    from hermes_cli.plugin_catalog import fetch_live_catalog
    summary: dict = {"installed": [], "owed": []}
    policies = {home: fetch_policy(home) for home in target_homes}
    summary["owed"] += [str(home) for home, policy in policies.items() if policy == "unreadable"]
    homes = [home for home, policy in policies.items() if policy == "allowed"]
    if _allowed_out is not None:
        _allowed_out[:] = homes
    resolved_entries = (default_entries() if entries is None else list(entries)) if homes else []
    if _entries_out is not None:
        _entries_out[:] = resolved_entries
    if homes and not (fetch_live_catalog() or {}).get("entries"):  # an expired offline cache has none
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
    from hermes_cli.plugin_catalog import LIVE_CATALOG_TTL_SECONDS, _live_cache_path
    cache = _live_cache_path()
    try:
        st = cache.stat()
    except OSError:
        return None
    if time.time() - st.st_mtime >= LIVE_CATALOG_TTL_SECONDS:
        return None
    return f"{st.st_mtime_ns}:{st.st_size}"


def _read_settled_record(path: Path) -> dict:
    """Read the settled marker record from ``cache/default-plugins.settled``.

    Legacy settled files written by older code store only a plain identity string; those return
    ``defaults_hash=None`` so callers treat them as unsettled-for-hash safely (one extra pass, no crash).
    """
    empty = {"identity": "", "defaults_hash": None, "defaults": None, "live_cache_sig": None}
    if not path.is_file():
        return empty
    try:
        raw = path.read_text(encoding="utf-8-sig").strip()
    except OSError:
        return empty
    if not raw:
        return empty
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
        except ValueError:
            return empty
        if isinstance(data, dict):
            identity = data.get("identity")
            dhash = data.get("defaults_hash")
            defaults = data.get("defaults")
            live_sig = data.get("live_cache_sig")
            return {
                "identity": identity if isinstance(identity, str) else "",
                "defaults_hash": dhash if isinstance(dhash, str) and dhash else None,
                "defaults": [str(n) for n in defaults] if isinstance(defaults, list) else None,
                "live_cache_sig": live_sig if isinstance(live_sig, str) and live_sig else None,
            }
        return empty
    # Legacy plain-text identity file (pre-hash format).
    return {**empty, "identity": raw}


def _read_settled(path: Path) -> tuple[str, Optional[str]]:
    """Read ``(identity, defaults_hash)`` from ``cache/default-plugins.settled``."""
    rec = _read_settled_record(path)
    return rec["identity"], rec["defaults_hash"]


def _write_settled(
    path: Path,
    identity: str,
    defaults_hash: str,
    *,
    default_names: Optional[list[str]] = None,
    live_cache_sig: Optional[str] = None,
) -> None:
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
    if saved_hash is None:
        return False
    return saved_id == _identity() and saved_hash == _defaults_hash(entries)


def _fast_settled_hit(root: Path, allowed_homes: list[Path]) -> bool:
    """True when the settled marker proves the install identity, fresh live catalog cache, and every
    allowed home's recorded defaults are unchanged — avoiding parsing all in-tree catalog YAML files."""
    if default_entries is not _UNPATCHED_DEFAULT_ENTRIES:
        return False
    rec = _read_settled_record(_state(root, "settled"))
    if rec["defaults_hash"] is None or rec["live_cache_sig"] is None or rec["defaults"] is None:
        return False
    live_sig = _live_cache_signature()
    if live_sig is None or live_sig != rec["live_cache_sig"]:
        return False
    if rec["identity"] != _identity():
        return False
    expected = rec["defaults"]
    if expected:
        for home in allowed_homes:
            done = _recorded(home)
            if any(name not in done for name in expected):
                return False
    return True


# ── detached child: first start, profile create, ensure handoff ──────────────

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


def _launch_ensure_child(run_id: str, home: Optional[Path]) -> subprocess.Popen:
    """The detached child an ``ensure`` waits on: it installs (for *home* only, when given, else every
    home), then writes its outcome under *run_id* to ``cache/default-plugins.result.json``."""
    args: list[str] = ["_child", "--run-id", run_id]
    if home is not None:
        args += ["--home", str(home)]
    return _popen_detached(_child_argv(tuple(args)))


def _write_child_result(root: Path, run_id: str, outcome: dict) -> None:
    from utils import atomic_write_text
    path = _state(root, "result.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"run_id": run_id, "installed": [str(x) for x in outcome.get("installed") or ()],
               "owed": [str(x) for x in outcome.get("owed") or ()]}
    atomic_write_text(path, json.dumps(payload), tmp_prefix=f"{path.name}.tmp-")


def _read_child_result(root: Path, run_id: str) -> Optional[dict]:
    """The result *run_id*'s child wrote, else None (none yet, unreadable, or another run's)."""
    try:
        data = json.loads(_state(root, "result.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("run_id") != run_id:
        return None
    return {key: [str(x) for x in data.get(key) or ()] for key in ("installed", "owed")}


def _ensure_child_work(run_id: str, home: Optional[Path], *, say: Callable[[str], None]) -> int:
    """The ensure child's pass (every home, or *home* only), its outcome written under *run_id* whatever
    happens short of the process dying (that leaves no result: the waiting ensure reports it owed)."""
    from hermes_cli.memory_provider_migration import _note_failure
    from pm.environments import dependency_home_root
    root = Path(dependency_home_root())
    outcome: dict = {"installed": [], "owed": []}
    try:
        if home is None:
            outcome = install_and_record(say=say, wait=_CHILD_LOCK_WAIT)
        else:
            outcome = _install_subset_and_record([home], default_entries(), root, say=say, wait=_CHILD_LOCK_WAIT)
    except Exception as exc:  # health: allow BLE001 -- reported to the waiting ensure as owed, and backed off
        logger.exception("default plugin install failed")
        _note_failure(root, _BACKOFF)
        outcome = {"installed": [], "owed": [f"error: {exc}"]}
    _write_child_result(root, run_id, outcome)  # not on a BaseException: no result reads as owed
    return 1 if outcome["owed"] else 0


def _run_child(*, run_id: str = "", home: Optional[Path] = None) -> int:
    """The detached child; one that finds another installer running leaves its request to that one. Started
    by an ``ensure`` (with *run_id*), it waits for the lock and reports to that ensure."""
    from pm.environments import dependency_home_root
    root = Path(dependency_home_root())
    (root / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s default-plugins %(message)s",
                        filename=str(root / "logs" / "default-plugins.log"))
    if run_id:
        return _ensure_child_work(run_id, home, say=logger.info)
    try:
        return 1 if install_and_record(say=logger.info, wait=1.0)["owed"] else 0
    except TimeoutError:
        return 0
    except Exception:  # health: allow BLE001 -- a detached child has no stderr: log the cause and back off
        from hermes_cli.memory_provider_migration import _note_failure
        logger.exception("default plugin install failed")
        _note_failure(root, _BACKOFF)
        return 1


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
    settled for this revision AND the current cached catalog's default set; when local files cannot prove
    settled (e.g. cache missing or expired), spawns the detached child immediately without HTTP."""
    from pm.environments import dependency_home_root
    from pm.plugins_state import dependency_homes
    try:
        homes = list(dependency_homes())
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
                if cached_entries is not None and saved_hash == _defaults_hash(cached_entries):
                    return {"ok": True}
        return {"ok": True, "spawned": spawn_install(), "owed": True}
    except (OSError, ValueError) as exc:  # unreadable profiles dir, failed spawn: still owed, retried next boot
        return {"ok": False, "error": str(exc), "owed": True}


# ── foreground ensure (Docker, Desktop, profile create) ──────────────────────

def _pending_names(homes: list[Path], entries: list) -> list[str]:
    pending: list[str] = []
    seen: set[str] = set()
    for home in homes:
        done = _recorded(home)
        for entry in entries:
            if entry.name not in done and not _settled(home, entry):
                if entry.name not in seen:
                    seen.add(entry.name)
                    pending.append(entry.name)
    return pending


def _ensure_payload(
    state: str,
    pending_before: list[str],
    installed: list[str],
    owed: list[str],
    started_monotonic: float,
) -> dict:
    return {
        "state": state,
        "pending_before": list(pending_before),
        "installed": list(installed),
        "owed": list(owed),
        "elapsed_ms": max(0, round((time.monotonic() - started_monotonic) * 1000)),
    }


def _install_subset_and_record(
    target_homes: list[Path],
    entries: list,
    root: Path,
    *,
    say: Callable[[str], None],
    wait: float,
) -> dict:
    from hermes_cli.memory_provider_migration import _note_failure
    outcome = _install_defaults_for_homes(target_homes, say=say, wait=wait, entries=entries)
    if outcome["owed"]:
        failed = set(outcome["owed"]) - set(outcome.get("raced", ()))
        if failed:
            _note_failure(root, _BACKOFF)
        _state(root, "settled").unlink(missing_ok=True)
    return outcome


def _check_ensure_fast_path(
    homes: Optional[list[Path]],
    allowed_homes: list[Path],
    unreadable_homes: list[Path],
    root: Path,
    entries: list,
    pending_before: list[str],
    t0: float,
) -> Optional[tuple[int, dict]]:
    """Return a settled/owed response immediately when no install or record write is needed."""
    from hermes_cli.plugin_catalog import fetch_live_catalog
    if pending_before or unreadable_homes:
        return None
    if homes is None:
        if _is_settled_for(root, entries):
            return ENSURE_EXIT_OK, _ensure_payload("nothing-pending", [], [], [], t0)
        return None
    if allowed_homes and not (fetch_live_catalog() or {}).get("entries"):
        _state(root, "settled").unlink(missing_ok=True)
        return ENSURE_EXIT_OWED, _ensure_payload("owed", [], [], ["live catalog"], t0)
    needs_record = any(entry.name not in _recorded(home) for home in allowed_homes for entry in entries)
    if not needs_record:
        return ENSURE_EXIT_OK, _ensure_payload("nothing-pending", [], [], [], t0)
    return None


def ensure_defaults(
    *,
    timeout: float = DEFAULT_ENSURE_TIMEOUT,
    homes: Optional[list[Path]] = None,
    say: Optional[Callable[[str], None]] = None,
) -> tuple[int, dict]:
    """Ensure every default catalog plugin is installed before a service starts (bounded foreground pass).

    The install itself runs in one detached child (:func:`_launch_ensure_child`) that this call waits on
    until the deadline; the child survives this process, so a timeout never cuts an install short.

    Returns ``(exit_code, payload)`` where ``exit_code`` is:
    - ``0`` (``nothing-pending`` or ``installed``)
    - ``2`` (``owed`` — offline / failed / unreadable home; retried next start)
    - ``3`` (``timeout`` — hard deadline hit; the detached installer child keeps going and finishes)
    - ``4`` (``skipped`` — fetch policy denied for all target homes)
    """
    from hermes_cli.plugin_catalog import fetch_live_catalog
    from pm.environments import dependency_home_root
    from pm.plugins_state import dependency_homes

    t0 = time.monotonic()
    budget = max(0.0, float(timeout))
    deadline = t0 + budget
    emit: Callable[[str], None] = say if say is not None else (lambda msg: print(msg, file=sys.stderr))

    target_homes = list(dependency_homes()) if homes is None else [Path(h) for h in homes]
    policies = {home: fetch_policy(home) for home in target_homes}
    if target_homes and all(policy == "denied" for policy in policies.values()):
        return ENSURE_EXIT_SKIPPED, _ensure_payload("skipped", [], [], [], t0)

    allowed_homes = [home for home, policy in policies.items() if policy == "allowed"]
    unreadable_homes = [home for home, policy in policies.items() if policy == "unreadable"]
    root = Path(dependency_home_root())

    # Zero-parse fast path: install identity, fresh live catalog cache, and recorded defaults unchanged.
    if homes is None and not unreadable_homes and _fast_settled_hit(root, allowed_homes):
        return ENSURE_EXIT_OK, _ensure_payload("nothing-pending", [], [], [], t0)

    entries = default_entries() if allowed_homes else []
    pending_before = _pending_names(allowed_homes, entries)

    fast = _check_ensure_fast_path(homes, allowed_homes, unreadable_homes, root, entries, pending_before, t0)
    if fast is not None:
        return fast

    # Offline: the live catalog read just failed, so every git/uv install would fail too, each after its
    # own network timeout. Fail open at once; the next start retries (same backoff and owed state).
    if pending_before and allowed_homes and not (fetch_live_catalog() or {}).get("entries"):
        from hermes_cli.memory_provider_migration import _note_failure
        _state(root, "settled").unlink(missing_ok=True)
        _note_failure(root, _BACKOFF)
        emit("  ⚠ Default plugins not installed: the plugin catalog is unreachable (offline?). "
             "Hermes retries at a later start.")
        return ENSURE_EXIT_OWED, _ensure_payload("owed", pending_before, [],
                                                 ["live catalog", *pending_before], t0)

    if not pending_before:
        # Nothing to fetch: only records (defaults already on disk) and the settled marker are written, each
        # atomically, so this short pass runs here, synchronously (no thread, no child interpreter start).
        return _record_only_pass(None if homes is None else target_homes, entries, root, emit, deadline, t0)

    # The install runs in a detached child, never in a thread of this process: at the deadline this
    # returns at once and the child carries on (a thread would die with this process, possibly between
    # placing a tree and admitting its dependencies, while still holding the machine lock).
    import uuid
    run_id = uuid.uuid4().hex
    log_tail = _LogTail(root / "logs" / "default-plugins.log")
    try:
        child = _launch_ensure_child(run_id, target_homes[0] if homes is not None and len(target_homes) == 1 else None)
    except OSError as exc:
        emit(f"default plugins ensure failed: could not start the installer: {exc}")
        return ENSURE_EXIT_OWED, _ensure_payload("owed", pending_before, [], list(pending_before) or [str(exc)], t0)

    result = _wait_for_child(child, root, run_id, deadline, log_tail, emit)
    if result == "timeout":
        emit(f"default plugins: still installing after {budget:.0f}s; continuing in the background "
             f"(pid {child.pid}, log {root / 'logs' / 'default-plugins.log'})")
        return ENSURE_EXIT_TIMEOUT, _ensure_payload("timeout", pending_before, [], list(pending_before), t0)
    if result is None:
        emit(f"default plugins ensure failed: the installer exited ({child.returncode}) without a result")
        return ENSURE_EXIT_OWED, _ensure_payload("owed", pending_before, [], list(pending_before)
                                                 or [f"installer exit {child.returncode}"], t0)
    return _outcome_exit(pending_before, result, t0)


def _outcome_exit(pending_before: list[str], outcome: dict, t0: float) -> tuple[int, dict]:
    installed, owed = list(outcome.get("installed") or []), list(outcome.get("owed") or [])
    if owed:
        return ENSURE_EXIT_OWED, _ensure_payload("owed", pending_before, installed, owed, t0)
    if installed:
        return ENSURE_EXIT_OK, _ensure_payload("installed", pending_before, installed, [], t0)
    return ENSURE_EXIT_OK, _ensure_payload("nothing-pending", pending_before, [], [], t0)


def _record_only_pass(subset: Optional[list[Path]], entries: list, root: Path, emit: Callable[[str], None],
                      deadline: float, t0: float) -> tuple[int, dict]:
    """The in-process pass when nothing is pending: every home (*subset* None) or *subset* only."""
    wait = max(0.0, deadline - time.monotonic())
    try:
        if subset is None:
            outcome = install_and_record(say=emit, wait=wait)
        else:
            outcome = _install_subset_and_record(subset, entries, root, say=emit, wait=wait)
    except TimeoutError:  # another installer holds the lock; its pass covers this one
        return ENSURE_EXIT_TIMEOUT, _ensure_payload("timeout", [], [], [], t0)
    except Exception as exc:  # health: allow BLE001 -- reported as owed, retried next start
        emit(f"default plugins ensure failed: {exc}")
        return ENSURE_EXIT_OWED, _ensure_payload("owed", [], [], [str(exc)], t0)
    return _outcome_exit([], outcome, t0)


def _wait_for_child(child, root: Path, run_id: str, deadline: float, log_tail: _LogTail,
                    emit: Callable[[str], None]):
    """*run_id*'s result once the child wrote it, None when it exited without one, or ``"timeout"`` at
    the deadline (the child is left running). Relays the child's log lines meanwhile."""
    while True:
        log_tail.pump(emit)
        result = _read_child_result(root, run_id)
        if result is not None:
            break
        if child.poll() is not None:
            result = _read_child_result(root, run_id)  # written just before it exited
            break
        if time.monotonic() >= deadline:
            return "timeout"
        time.sleep(0.05)
    log_tail.pump(emit)
    return result


class _LogTail:
    """Relays the lines the installer child appends to its log (per-plugin timing included) to the
    ensure's own output while it waits. Starts at the log's current end."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self.offset = path.stat().st_size
        except OSError:
            self.offset = 0
        self.partial = b""

    def pump(self, emit: Callable[[str], None]) -> None:
        try:
            with self.path.open("rb") as fh:
                fh.seek(self.offset)
                chunk = fh.read()
        except OSError:
            return
        if not chunk:
            return
        self.offset += len(chunk)
        *lines, self.partial = (self.partial + chunk).split(b"\n")
        for line in lines:
            text = line.decode("utf-8", "replace").rstrip("\r")
            if text.strip():
                emit(text)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for ``python -m hermes_cli.default_plugins``.

    Bare invocation (no args) preserves the detached-child behavior (:func:`_run_child`).
    ``ensure [--timeout SECONDS] [--json]`` runs the bounded foreground preflight.
    ``_child --run-id ID [--home PATH]`` is the internal detached installer spawned by ``ensure``.
    """
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        return _run_child()

    parser = argparse.ArgumentParser(prog="python -m hermes_cli.default_plugins")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ensure_p = subparsers.add_parser("ensure", help="Install pending default catalog plugins in the foreground")
    ensure_p.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_ENSURE_TIMEOUT,
        help="Seconds to wait for the detached installer child before returning 3 (default: 300)",
    )
    ensure_p.add_argument(
        "--json",
        action="store_true",
        help="Emit a single JSON status line on stdout (human/log output goes to stderr)",
    )
    child_p = subparsers.add_parser("_child", help=argparse.SUPPRESS)
    child_p.add_argument("--run-id", required=True)
    child_p.add_argument("--home", default="")
    parsed = parser.parse_args(args_list)
    if parsed.command == "_child":
        target = parsed.home.strip()
        return _run_child(run_id=parsed.run_id.strip(), home=Path(target) if target else None)
    if parsed.command == "ensure":
        real_stdout = sys.stdout
        if parsed.json:
            sys.stdout = sys.stderr
        try:
            code, payload = ensure_defaults(
                timeout=parsed.timeout,
                say=lambda msg: print(msg, file=sys.stderr),
            )
            if parsed.json:
                real_stdout.write(json.dumps(payload) + "\n")
                real_stdout.flush()
            else:
                print(
                    f"default-plugins ensure: state={payload['state']} "
                    f"installed={len(payload['installed'])} owed={len(payload['owed'])} "
                    f"elapsed_ms={payload['elapsed_ms']}",
                    file=sys.stderr,
                )
            return code
        finally:
            sys.stdout = real_stdout
    return 0


if __name__ == "__main__":
    sys.exit(main())
