"""Plumbing shared by the kanban notifier and dispatcher loops.

Thread offload, board enumeration, live-config coercers and the singleton
dispatcher lock live here so the notifier, dispatcher and mixin modules read
them from one place.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from contextvars import Context
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

# Keep the logger name run.py used so extracted log records are unchanged.
logger = logging.getLogger("gateway.run")


def _run_in_fresh_context(func: Callable[..., Any], /, *args: Any) -> Any:
    """Run *func* in an empty ``Context`` so request-local ContextVars stay behind.

    ``asyncio.to_thread`` copies the caller's context; a lingering
    ``delegate_task`` child marker would make ``write_txn`` false-trip for
    these process-owned writers. An empty Context keeps the DB guard intact
    for real children without exempting dispatcher writes.
    """
    return Context().run(func, *args)


async def _to_thread_process_service(func: Callable[..., Any], /, *args: Any) -> Any:
    """Offload blocking process-service work without inheriting request ContextVars."""
    return await asyncio.to_thread(_run_in_fresh_context, func, *args)


def _list_boards(kb: Any) -> list:
    """Enumerate live boards; fall back to the default board when listing fails."""
    try:
        return kb.list_boards(include_archived=False)
    except Exception:
        return [kb.read_board_metadata(kb.DEFAULT_BOARD)]


def _board_slugs(kb: Any) -> list:
    return [b.get("slug") or kb.DEFAULT_BOARD for b in _list_boards(kb)]


def _positive_int_setting(kanban_cfg: dict, key: str) -> Optional[int]:
    """Parse an optional ``kanban.<key>`` int cap; None when unset or invalid (< 1 is invalid)."""
    raw = kanban_cfg.get(key)
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning("kanban dispatcher: invalid kanban.%s=%r; ignoring", key, raw)
        return None
    if value < 1:
        logger.warning("kanban dispatcher: kanban.%s=%r is below 1; ignoring", key, raw)
        return None
    logger.info("kanban dispatcher: %s=%d", key, value)
    return value


def _resolve_auto_decompose_settings(load_config: Callable[[], Any]) -> "tuple[bool, int]":
    """Live (enabled, per_tick) auto-decompose settings, re-read every dispatcher tick.

    Fails safe: a config read error returns ``(False, 3)`` rather than
    re-enabling a feature the user turned off. ``per_tick`` is clamped to ``>= 1``.

    Read fresh from config on every dispatcher tick (#49638) so that flipping ``kanban.auto_decompose:
    false`` to STOP runaway fan-out takes effect on the next tick instead of requiring a gateway restart.
    Auto-decompose is a safety toggle — a user who sees it create and launch tasks they didn't intend
    reaches for this flag to halt it, and a stale boot-captured value silently ignoring that change is the
    bug reported in #49638.
    """
    try:
        cfg = load_config()
    except Exception:
        return False, 3
    kcfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    try:
        per_tick = int(kcfg.get("auto_decompose_per_tick", 3) or 3)
    except (TypeError, ValueError):
        per_tick = 3
    return bool(kcfg.get("auto_decompose", True)), max(per_tick, 1)


def _gc_retention_days() -> int:
    """``kanban.done_sub_retention_days`` (default 30; 0 disables), re-read per sweep; fails safe to 30."""
    try:
        from hermes_cli.config import load_config

        return int(((load_config() or {}).get("kanban") or {}).get("done_sub_retention_days", 30))
    except Exception:
        return 30


def _kanban_dispatch_allowed() -> bool:
    """False while the global emergency stop (`hermes pause`) is engaged.

    Checked every tick before spawning, so a pause applies on the next tick;
    in-flight workers are never touched. Fails open if estop is unimportable.
    """
    try:
        from agent.estop import check_paused
    except ImportError:
        return True
    return not check_paused("kanban", logger)


def _acquire_singleton_lock(lock_path) -> "tuple[Optional[object], str]":
    """Take the exclusive, non-blocking advisory lock for the sole dispatcher.

    Only one gateway machine-wide may run the embedded dispatcher: concurrent
    dispatchers double reclaim frequency and claim events, and with
    ``wal_autocheckpoint=0`` concurrent manual checkpoints can corrupt index
    pages. ``dispatch_in_gateway`` is the primary control; this is the backstop.

    Returns ``(handle, "held")`` (release via :func:`_release_singleton_lock`),
    ``(None, "contended")`` when another process holds it (caller must NOT
    dispatch), or ``(None, "unavailable")`` when locking cannot be performed
    (caller falls back to config control).
    """
    try:
        from gateway.status import _try_acquire_file_lock  # deferred; same package
    except ImportError:
        return None, "unavailable"
    try:
        Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
        handle = open(str(lock_path), "a+", encoding="utf-8")  # windows-footgun: ok (append-mode lock handle, write not read)
    except OSError:
        return None, "unavailable"
    if not _try_acquire_file_lock(handle):
        handle.close()
        return None, "contended"
    return handle, "held"


def _release_singleton_lock(handle) -> None:
    """Release a lock acquired via :func:`_acquire_singleton_lock`."""
    if handle is None:
        return
    with contextlib.suppress(Exception):
        from gateway.status import _release_file_lock

        _release_file_lock(handle)
    with contextlib.suppress(Exception):
        handle.close()


# --- Dispatcher owner record -------------------------------------------------
#
# The dispatcher lock is STORE-scoped: whichever gateway wins
# ``<store>/kanban/.dispatcher.lock`` serves that store's dispatcher for the life
# of the process. The lock cannot be *probed* from another process without
# stealing it — ``_kanban_dispatcher_boot`` reads a contended acquire as a
# permanent disable — so presence is announced instead: the lock holder writes
# this record beside the lock, and every consumer asks the record rather than
# inferring from a gateway PID file. A PID file is scoped to the gateway's own
# ``HERMES_HOME`` and says nothing about which store that gateway serves, which
# is what made a profile-scoped probe disagree with a healthy root gateway.

_DISPATCHER_OWNER_FILENAME = ".dispatcher.owner.json"
_DISPATCHER_OWNER_VERSION = 1


def _resolve_store_home(store_home: Optional[Path] = None) -> Optional[Path]:
    """The kanban store's home (``HERMES_KANBAN_HOME`` else the default root)."""
    if store_home is not None:
        return Path(store_home).expanduser()
    try:
        from hermes_cli import kanban_db as _kb  # deferred: kanban_db imports are heavy

        return _kb.kanban_home()
    except Exception:
        return None


def _dispatcher_owner_path(store_home: Optional[Path] = None) -> Optional[Path]:
    home = _resolve_store_home(store_home)
    return None if home is None else home / "kanban" / _DISPATCHER_OWNER_FILENAME


def _same_store_home(left: Path | str, right: Path | str) -> bool:
    try:
        a = os.path.normcase(str(Path(left).expanduser().resolve(strict=False)))
        b = os.path.normcase(str(Path(right).expanduser().resolve(strict=False)))
        return a == b
    except Exception:
        return False


def _owner_process_home() -> str:
    """The lock holder's OWN launch home — the identity its gateway process carries."""
    try:
        from gateway.status import _get_process_hermes_home  # deferred; same package

        return str(_get_process_hermes_home())
    except Exception:
        try:
            from hermes_constants import get_process_hermes_home

            return str(get_process_hermes_home())
        except Exception:
            return ""


def write_dispatcher_owner(
    store_home: Optional[Path] = None, *, pid: Optional[int] = None,
    home: Optional[str] = None, tree: Optional[str] = None,
) -> Optional[Path]:
    """Announce THIS process as ``store_home``'s dispatcher; return the record path.

    Lock holders only: a process that did not win ``<store>/kanban/.dispatcher.lock``
    owns no store and must not claim one. Best-effort by design — a dispatcher that
    cannot write its record still dispatches, so failures are logged and swallowed.
    Written atomically (tmp + ``os.replace``) so a reader never sees a torn record.
    """
    path = _dispatcher_owner_path(store_home)
    if path is None:
        return None
    try:
        record = {
            "version": _DISPATCHER_OWNER_VERSION,
            "pid": os.getpid() if pid is None else int(pid),
            "home": _owner_process_home() if home is None else str(home),
            "store": str(path.parent.parent.resolve(strict=False)),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "tree": str(tree) if tree else None,
        }
    except Exception:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        tmp.write_text(json.dumps(record), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("kanban dispatcher: could not write owner record %s: %s", path, exc)
        return None
    logger.debug("kanban dispatcher: owner record written at %s (pid=%s)", path, record["pid"])
    return path


def _live_gateway_for_home(pid: int, home: str) -> bool:
    """True when ``pid`` is alive AND is a hermes gateway serving ``home``.

    Reuses the gateway identity primitives so a dead or recycled PID never counts.
    An unreadable command line (Windows/EACCES) cannot be *disproven*; the record
    then stands on its own, mirroring ``gateway.status._record_matches_live_gateway_pid``.
    """
    try:
        from gateway.status import (
            _command_line_belongs_to_profile,
            _host_gateway_serves_home,
            _pid_exists,
            _read_process_cmdline,
            looks_like_gateway_runtime_command_line,
        )
    except Exception:
        return False
    try:
        if not _pid_exists(pid):
            return False
        home_path = Path(home)
        if _host_gateway_serves_home(pid, home_path):
            return True
        cmdline = _read_process_cmdline(pid)
        if not cmdline:
            return True
        if not looks_like_gateway_runtime_command_line(cmdline):
            return False
        return _command_line_belongs_to_profile(cmdline, home_path)
    except Exception:
        return False


def read_dispatcher_owner(store_home: Optional[Path] = None) -> Optional[dict]:
    """The live dispatcher owner of ``store_home``, or None.

    None means "no dispatcher will pick up this store's cards": the record is absent,
    unreadable, a version mismatch, names a DIFFERENT store, or its PID is dead or no
    longer a gateway for the home it claims (a recycled PID must never count). The
    record's ``home`` is deliberately NOT compared to the store — a gateway under a
    normal ``HERMES_HOME`` legitimately serves a custom ``HERMES_KANBAN_HOME``.

    Never acquires the lock: the record is written by the lock holder, so presence is
    proven without contending for it (a probe that took the lock would disable the next
    gateway boot for the rest of that process's life).
    """
    path = _dispatcher_owner_path(store_home)
    if path is None:
        return None
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.debug("kanban dispatcher: owner record unreadable at %s: %s", path, exc)
        return None
    try:
        record = json.loads(raw)
    except ValueError:
        logger.debug("kanban dispatcher: owner record is not JSON at %s", path)
        return None
    if not isinstance(record, dict):
        return None
    if record.get("version") != _DISPATCHER_OWNER_VERSION:
        logger.debug("kanban dispatcher: owner record version %r unsupported", record.get("version"))
        return None
    recorded_store = record.get("store")
    if not isinstance(recorded_store, str) or not _same_store_home(recorded_store, path.parent.parent):
        return None
    pid = record.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    home = record.get("home")
    if not isinstance(home, str) or not home.strip():
        return None
    if not _live_gateway_for_home(pid, home):
        logger.debug("kanban dispatcher: owner pid %s is not a live gateway for %s", pid, home)
        return None
    return record


def dispatcher_active(store_home: Optional[Path] = None) -> bool:
    """True when a live dispatcher owns ``store_home`` (see :func:`read_dispatcher_owner`)."""
    return read_dispatcher_owner(store_home) is not None


def clear_dispatcher_owner(store_home: Optional[Path] = None, *, pid: Optional[int] = None) -> None:
    """Remove ``store_home``'s owner record IFF ``pid`` (default: this process) owns it.

    Owner-scoped so a starting dispatcher can never delete the record of one that is
    still running; a stale record whose PID has since died is rejected by the reader,
    so a crashed holder needs no cleanup to stay honest.
    """
    path = _dispatcher_owner_path(store_home)
    if path is None:
        return
    owner_pid = os.getpid() if pid is None else int(pid)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return
    if not isinstance(record, dict) or record.get("pid") != owner_pid:
        return
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)
