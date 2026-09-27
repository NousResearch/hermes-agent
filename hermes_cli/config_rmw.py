"""Locked read-modify-write for ``config.yaml`` (#96571, #66752).

Two gaps let one config write silently undo another:

* Writers read, mutated and replaced ``config.yaml`` with no lock shared across writers or processes
  (``_CONFIG_LOCK`` is per process, and ``save_config_value`` never took it), so a write landing
  between another writer's read and its replace was overwritten. :func:`config_write_lock` is held
  from the fresh read through the atomic replace, by every writer, in every process.
* Whole-document saves (``save_config(cfg)``, the TUI's ``_save_cfg(cfg)``) wrote back EVERY value
  of a dict loaded earlier, reverting keys other writers had changed since. A dict served by a
  tracked loader (``load_config``, ``read_raw_config``, ``read_user_config_raw``, the TUI's
  ``_load_cfg_raw``) remembers the snapshot it was served from; :func:`rebase_onto_disk` applies
  only the paths the caller changed relative to that snapshot onto a fresh disk read. A key the
  caller never touched is never reverted, and a value that came from the default merge is never
  mistaken for a change (the false conflict of #62232), because the snapshot carries the same
  default.

Dicts no loader served (literals, ``dict(cfg)`` copies) are saved whole, as before, under the lock.
"""

from __future__ import annotations

import copy
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Tuple

from hermes_cli.config_read_errors import FailedConfigRead
from hermes_constants import mkdir_under_hermes_home

logger = logging.getLogger(__name__)

# A writer blocked this long on another process's lock fails loudly instead of hanging a CLI or RPC.
_LOCK_TIMEOUT_SECONDS = 30.0
# Lock files this thread already holds, so nested writers re-enter instead of deadlocking on their
# own lock. Only touched under _CONFIG_LOCK (an RLock), so only by the thread that holds it.
_held_depth: Dict[str, int] = {}
_UNLOCKABLE_WARNED: set = set()

# id(served dict) -> (served dict, snapshot it was served from, real config path, form). An entry
# lives exactly as long as a caller holds its dict: entries only this registry still references are
# pruned on every insert. There is deliberately no count cap. Evicting a dict its caller still holds
# would silently turn that caller's next save back into the whole-document write that reverts
# concurrent updates; a plain dict cannot be weakly referenced, so the refcount prune is how the
# registry lets go, and its size is bounded by what callers hold anyway.
_tracked: "Dict[int, Tuple[dict, Any, str, str]]" = {}
_tracked_lock = threading.Lock()

_DELETED = object()
_AUDIT_MAX_PATHS = 20
# Frames that are the write machinery, not the surface that asked for the write (audit log).
_MACHINERY_MODULES = frozenset({__name__, "utils", "contextlib"})
_MACHINERY_FUNCS = frozenset({"atomic_config_write", "save_config", "_write_user_config"})


def _registry_only_refcount() -> int:
    """``sys.getrefcount`` of a dict referenced only from a registry tuple (measured, not assumed)."""
    entry = ({},)
    return sys.getrefcount(entry[0])


_ORPHAN_REFCOUNT = _registry_only_refcount()


def _lock_path(config_path: Path) -> Path:
    # Beside the config in its (resolved) home, not beside a symlink target: a config.yaml linked
    # into a dotfiles repo must not grow a lock file there.
    config_path = Path(config_path)
    return Path(os.path.realpath(config_path.parent)) / f".{config_path.name}.lock"


@contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    """Exclusive cross-process lock on *lock_path* (the ``hermes_cli/backup.py`` pattern). A lock the
    filesystem cannot provide degrades to the in-process lock with a warning, never a failed write."""
    try:
        handle = lock_path.open("a+b")
    except PermissionError:
        if os.name == "nt":
            raise
        handle = lock_path.open("rb")  # created by another user (a `sudo hermes` run); flock needs no write
    locked = False
    try:
        if os.name == "nt":
            import msvcrt
            if lock_path.stat().st_size == 0:
                with suppress(OSError):  # another process won the race to seed the byte
                    handle.write(b" ")
                    handle.flush()

            def _lock_op(flag: int) -> None:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), flag, 1)
            lock_flag, unlock_flag, busy = msvcrt.LK_NBLCK, msvcrt.LK_UNLCK, OSError
        else:
            import fcntl

            def _lock_op(flag: int) -> None:
                fcntl.flock(handle.fileno(), flag)
            lock_flag, unlock_flag, busy = fcntl.LOCK_EX | fcntl.LOCK_NB, fcntl.LOCK_UN, BlockingIOError
        deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
        delay = 0.001
        while not locked:
            try:
                _lock_op(lock_flag)
                locked = True
            except busy:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Another Hermes process has held {lock_path} for {_LOCK_TIMEOUT_SECONDS:.0f}s, "
                        "so this config change was not saved. Try again.") from None
                time.sleep(delay)
                delay = min(delay * 2, 0.05)
            except OSError as exc:  # ENOLCK / EOPNOTSUPP: this filesystem has no flock
                if str(lock_path) not in _UNLOCKABLE_WARNED:
                    _UNLOCKABLE_WARNED.add(str(lock_path))
                    logger.warning("Cannot lock %s (%s): config.yaml writes from other Hermes processes "
                                   "are not serialized on this filesystem.", lock_path, exc)
                break
        try:
            yield
        finally:
            if locked:
                with suppress(OSError):
                    _lock_op(unlock_flag)
    finally:
        handle.close()


@contextmanager
def config_write_lock(config_path: Path) -> Iterator[None]:
    """Hold THE config.yaml write lock: ``_CONFIG_LOCK`` for this process plus a lock file beside
    *config_path* for every other process. Reentrant within a thread. Hold it from the read a write
    is based on through the atomic replace."""
    from hermes_cli.config import _CONFIG_LOCK

    lock_path = _lock_path(Path(config_path))
    key = str(lock_path)
    with _CONFIG_LOCK:
        if _held_depth.get(key):
            _held_depth[key] += 1
            try:
                yield
            finally:
                _held_depth[key] -= 1
            return
        mkdir_under_hermes_home(lock_path.parent)
        with _file_lock(lock_path):
            _held_depth[key] = 1
            try:
                yield
            finally:
                del _held_depth[key]


def track_served(served: Any, snapshot: Any, config_path: Path, form: str) -> None:
    """Remember that the loader serving *form* (``"effective"`` or ``"raw"``) handed out *served*,
    a private copy of *snapshot*. *snapshot* must never be mutated afterwards."""
    if not isinstance(served, dict) or isinstance(snapshot, FailedConfigRead):
        return
    real = os.path.realpath(config_path)
    with _tracked_lock:
        for key in [k for k, entry in _tracked.items() if sys.getrefcount(entry[0]) <= _ORPHAN_REFCOUNT]:
            del _tracked[key]
        _tracked[id(served)] = (served, snapshot, real, form)


def _tracked_entry(data: Any, config_path: Path) -> Optional[Tuple[dict, Any, str, str]]:
    with _tracked_lock:
        entry = _tracked.get(id(data))
    if entry is None or entry[0] is not data or entry[2] != os.path.realpath(config_path):
        return None
    return entry


def _same(a: Any, b: Any) -> bool:
    """Equal as YAML values: ``True`` and ``1`` differ (``True == 1`` in Python)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(map(_same, a, b))
    return a == b and isinstance(a, bool) is isinstance(b, bool)


def _changes(base: dict, current: dict, prefix: Tuple = ()) -> list:
    """``(path, value)`` per leaf *current* sets differently from *base*; ``_DELETED`` marks a removal.
    Mappings on both sides, and new non-empty ones, are descended so every path names a leaf."""
    out = []
    for key, value in current.items():
        path = (*prefix, key)
        old = base.get(key, _DELETED)
        if isinstance(value, dict) and value and (old is _DELETED or isinstance(old, dict)):
            out.extend(_changes({} if old is _DELETED else old, value, path))
        elif old is _DELETED or not _same(old, value):
            out.append((path, value))
    out.extend(((*prefix, key), _DELETED) for key in base if key not in current)
    return out


def changed_paths(before: dict, after: dict) -> list:
    """Key paths (tuples) whose value differs between two config mappings."""
    return [path for path, _ in _changes(before, after)]


def _apply(target: dict, changes: Iterable[Tuple[Tuple, Any]]) -> None:
    for path, value in changes:
        *parents, leaf = path
        node = target
        for key in parents:
            child = node.get(key)
            if not isinstance(child, dict):
                if value is _DELETED:
                    break  # nothing to remove under a missing/scalar parent
                child = node[key] = {}
            node = child
        else:
            if value is _DELETED:
                node.pop(leaf, None)
            else:
                node[leaf] = copy.deepcopy(value)


def rebase_onto_disk(data: Any, config_path: Path, read_fresh: Mapping[str, Callable[[], dict]]) -> Optional[dict]:
    """What a whole-document save of *data* should write: a fresh read (``read_fresh[form]()``, taken
    under :func:`config_write_lock`) with only the paths the caller changed since *data* was served
    applied on top. ``None`` when *data* is not a tracked dict of a form the caller can re-read."""
    entry = _tracked_entry(data, config_path)
    if entry is None or entry[3] not in read_fresh:
        return None
    fresh = read_fresh[entry[3]]()
    _apply(fresh, _changes(entry[1], data))
    return fresh


def remember_saved(data: Any, config_path: Path) -> None:
    """After a successful save, a tracked *data*'s next save diffs against what it holds now."""
    entry = _tracked_entry(data, config_path)
    if entry is not None:
        track_served(data, copy.deepcopy(data), config_path, entry[3])


def _calling_surface(depth: int = 3) -> str:
    frame = sys._getframe(1)
    names: list = []
    while frame is not None and len(names) < depth:
        module, func = frame.f_globals.get("__name__", "?"), frame.f_code.co_name
        if module not in _MACHINERY_MODULES and not (module == "hermes_cli.config" and func in _MACHINERY_FUNCS):
            names.append(f"{module}:{func}")
        frame = frame.f_back
    return " < ".join(names) or "?"


def log_config_write(config_path: Path, paths: Iterable[Tuple]) -> None:
    """One INFO line per config.yaml write: the key paths it changed and who called. Never values."""
    names = list(dict.fromkeys(".".join(map(str, p)) for p in paths))
    shown = ", ".join(names[:_AUDIT_MAX_PATHS])
    if len(names) > _AUDIT_MAX_PATHS:
        shown += f" (+{len(names) - _AUDIT_MAX_PATHS} more)"
    logger.info("config write %s: %s [via %s]", config_path, shown or "no key changes", _calling_surface())
