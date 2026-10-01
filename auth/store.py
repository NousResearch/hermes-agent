"""Authentication-store paths, locking and private atomic persistence."""

from __future__ import annotations
import json
import logging
import os
import shutil
import threading
import time
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from hermes_constants import get_hermes_home, secure_parent_dir
from utils import atomic_json_write, file_signature
from auth.store_migrations import _migrate_stale_nous_portal_url

logger = logging.getLogger(__name__)
try:
    import fcntl
except Exception:
    fcntl = None
try:
    import msvcrt
except Exception:
    msvcrt = None

AUTH_STORE_VERSION = 1


AUTH_LOCK_TIMEOUT_SECONDS = 15.0


def _auth_file_path() -> Path:
    path = get_hermes_home() / "auth.json"
    # Seat belt: under pytest, refuse to touch the real user's auth store (tests that forgot to
    # monkeypatch HERMES_HOME or escaped the hermetic conftest). In production: one dict lookup.
    if os.environ.get("PYTEST_CURRENT_TEST") and _same_path(
        path, Path.home() / ".hermes" / "auth.json"
    ):
        raise RuntimeError(
            f"Refusing to touch real user auth store during test run: {path}. "
            "Set HERMES_HOME to a tmp_path in your test fixture, or run "
            "via scripts/run_tests.sh for hermetic CI-parity env."
        )
    return path


def _global_auth_file_path() -> Optional[Path]:
    """Global-root auth.json in profile mode; None when profile and global root are the same dir.

    Read-only fallback path, so no pytest seat belt here (it lives on ``_auth_file_path()``)."""
    try:
        from hermes_constants import get_default_hermes_root

        global_root = get_default_hermes_root()
    except Exception:
        return None
    return (
        None
        if _same_path(get_hermes_home(), global_root)
        else global_root / "auth.json"
    )


def _load_global_auth_store() -> Dict[str, Any]:
    """Load the global-root auth store (read-only fallback, mtime-memoised); ``{}`` when absent or
    unreadable — a malformed global store must never break profile reads."""
    global _global_auth_store_cache
    global_path = _global_auth_file_path()
    if global_path is None or not global_path.exists():
        _global_auth_store_cache = None
        return {}
    try:
        cache_key: Optional[Tuple[str, Tuple[int, int, int, int]]] = (
            str(global_path.resolve(strict=False)),
            file_signature(global_path.stat()),
        )
    except Exception:
        cache_key = None
    cached = _global_auth_store_cache
    if cache_key is not None and cached is not None and cached[:2] == cache_key:
        return cached[2]
    if os.environ.get("PYTEST_CURRENT_TEST") and os.environ.get("HOME"):
        real_root = Path(os.environ["HOME"]) / ".hermes" / "auth.json"
        try:
            if os.path.normcase(os.path.abspath(global_path)) == os.path.normcase(
                os.path.abspath(real_root)
            ):
                _global_auth_store_cache = None
                return {}
        except Exception:
            pass
    try:
        store = _load_auth_store(global_path)
    except Exception:
        _global_auth_store_cache = None
        return {}
    if cache_key is not None:
        _global_auth_store_cache = (*cache_key, store)
    return store


_auth_target_lock_holders: Dict[str, threading.local] = {}


_auth_target_lock_holders_guard = threading.Lock()


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve(strict=False) == right.resolve(strict=False)
    except Exception:
        return left == right


def _is_same_auth_store(left: Path, right: Path) -> bool:
    """True when two auth paths name ONE store rather than two copies.
    ``_same_path`` resolves symlinks and ``..``; ``samefile`` adds hardlinks and bind-mounts
    (same inode under two resolved names). Used by the forked-grant heal: a shared store has
    no "other side" to consolidate.

    See #101356.
    """
    if _same_path(left, right):
        return True
    try:
        return left.samefile(right)
    except OSError:
        return False


def _resolved_key(path: Path) -> str:
    """Canonical string for *path* (resolved when possible) used as a cache / lock-holder key."""
    try:
        return str(path.resolve(strict=False))
    except Exception:
        return str(path)


def _auth_lock_holder_for(target_path: Path) -> threading.local:
    """Return a reentrancy tracker keyed to one canonical auth-store path."""
    with _auth_target_lock_holders_guard:
        return _auth_target_lock_holders.setdefault(
            _resolved_key(target_path), threading.local()
        )


def _kernel_lock(lock_file: Any, acquire: bool) -> None:
    """Non-blocking exclusive flock (fcntl) or 1-byte msvcrt lock at offset 0; ``acquire=False`` releases."""
    if fcntl:
        fcntl.flock(
            lock_file.fileno(),
            (fcntl.LOCK_EX | fcntl.LOCK_NB) if acquire else fcntl.LOCK_UN,
        )
    else:
        lock_file.seek(0)
        msvcrt.locking(
            lock_file.fileno(), msvcrt.LK_NBLCK if acquire else msvcrt.LK_UNLCK, 1
        )


@contextmanager
def _file_lock(
    lock_path: Path,
    holder: threading.local,
    timeout_seconds: float,
    timeout_message: str,
):
    """Cross-process advisory flock helper, reentrant per-thread via ``holder.depth``.

    Falls back to a depth-only guard when neither ``fcntl`` nor ``msvcrt`` is available. Callers
    supply their own ``threading.local`` so independent locks (profile store vs global root vs the
    shared Nous store) track reentrancy separately."""
    if getattr(holder, "depth", 0) > 0:
        holder.depth += 1
        try:
            yield
        finally:
            holder.depth -= 1
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock_file = None
        if fcntl is not None or msvcrt is not None:
            # msvcrt.locking needs a non-empty file with the pointer at 0. This convenience write can
            # race another holder's byte-range lock and raise PermissionError (reproduced with 20
            # concurrent processes on Windows); losing the race just means the file already has
            # content, so swallow it.
            if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
                try:
                    lock_path.write_text(" ", encoding="utf-8")
                except (OSError, PermissionError):
                    pass
            lock_file = stack.enter_context(
                lock_path.open("r+" if msvcrt else "a+", encoding="utf-8")
            )
            deadline = time.monotonic() + max(1.0, timeout_seconds)
            while True:
                try:
                    _kernel_lock(lock_file, True)
                    break
                except (BlockingIOError, OSError, PermissionError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(timeout_message)
                    time.sleep(0.05)

        holder.depth = 1
        try:
            yield
        finally:
            holder.depth = 0
            if lock_file is not None:
                try:
                    _kernel_lock(lock_file, False)
                except (OSError, IOError):
                    pass


@contextmanager
def _auth_store_lock(
    timeout_seconds: float = AUTH_LOCK_TIMEOUT_SECONDS,
    *,
    target_path: Optional[Path] = None,
):
    """Cross-process advisory lock for one auth.json read/write transaction.

    ``target_path`` is required for profile-to-global write-throughs: each path has its own
    reentrancy tracker and kernel lock. Lock ordering invariant: ``_auth_store_lock`` FIRST (outer),
    ``_nous_shared_store_lock`` SECOND (inner), else deadlock against a concurrent shared import."""
    auth_path = target_path if target_path is not None else _auth_file_path()
    with _file_lock(
        auth_path.with_suffix(".lock"),
        _auth_lock_holder_for(auth_path),
        timeout_seconds,
        "Timed out waiting for auth store lock",
    ):
        yield


def _empty_auth_store() -> Dict[str, Any]:
    return {"version": AUTH_STORE_VERSION, "providers": {}}


def _load_auth_store(auth_file: Optional[Path] = None) -> Dict[str, Any]:
    auth_file = auth_file or _auth_file_path()
    if not auth_file.exists():
        return _empty_auth_store()
    try:
        raw = json.loads(auth_file.read_text(encoding="utf-8-sig"))
    except OSError:
        # Exists but unreadable (EMFILE, EACCES, EIO, stalled mount): contents are not bad, and this
        # module read-modify-writes everywhere, so an empty store here is one _save_auth_store()
        # away from erasing every credential. Fail loudly.
        logger.warning(
            "auth: could not read %s, leaving the store on disk untouched "
            "rather than degrading to an empty one",
            auth_file,
            exc_info=True,
        )
        raise
    except Exception as exc:
        # Genuine corruption: unparseable JSON or non-UTF-8 bytes. Preserve a copy, but never
        # advertise a backup that was not written.
        corrupt_path = auth_file.with_suffix(".json.corrupt")
        try:
            shutil.copy2(auth_file, corrupt_path)
            preserved = True
        except Exception:
            preserved = False
            logger.debug(
                "auth: could not preserve a copy of the corrupt store at %s",
                corrupt_path,
                exc_info=True,
            )
        logger.warning(
            "auth: failed to parse %s (%s), starting with empty store. %s %s",
            auth_file,
            exc,
            "Corrupt file preserved at"
            if preserved
            else "A copy could NOT be preserved at",
            corrupt_path,
        )
        return _empty_auth_store()

    if isinstance(raw, dict) and (
        isinstance(raw.get("providers"), dict)
        or isinstance(raw.get("credential_pool"), dict)
    ):
        raw.setdefault("providers", {})
        if isinstance(raw.get("providers"), dict):
            _migrate_stale_nous_portal_url(raw["providers"])
        return raw

    if isinstance(raw, dict) and isinstance(
        raw.get("systems"), dict
    ):  # legacy "systems" format
        systems = raw["systems"]
        providers = {"nous": systems["nous_portal"]} if "nous_portal" in systems else {}
        return {
            **_empty_auth_store(),
            "providers": providers,
            "active_provider": "nous" if providers else None,
        }
    return _empty_auth_store()


def _save_private_json(
    target: Path, data: Any, *, fsync_dir: bool = False, **dump_kwargs: Any
) -> None:
    """0600 credential JSON under a 0700 parent (``secure_parent_dir`` refuses ``/``, top-level dirs
    and the install tree). ``atomic_json_write`` creates the temp file 0600 before any byte lands."""
    from hermes_constants import mkdir_under_hermes_home

    mkdir_under_hermes_home(target.parent)
    secure_parent_dir(target)
    atomic_json_write(target, data, mode=0o600, fsync_dir=fsync_dir, **dump_kwargs)


def _save_auth_store(
    auth_store: Dict[str, Any], target_path: Optional[Path] = None
) -> Path:
    """Atomically persist *auth_store* (0o600, parent tightened to 0o700) to the active store, or to
    an explicit *target_path* (e.g. the global-root write-through for rotating xAI OAuth grants)."""
    auth_file = target_path if target_path is not None else _auth_file_path()
    auth_store["version"] = AUTH_STORE_VERSION
    auth_store["updated_at"] = datetime.now(timezone.utc).isoformat()
    _save_private_json(auth_file, auth_store, fsync_dir=True)
    if target_path is not None:
        # A write-through to the global root must not be masked by the mtime memo: on coarse-mtime
        # filesystems a read-after-write in the same tick would keep serving the pre-write store.
        global _global_auth_store_cache
        _global_auth_store_cache = None
    return auth_file


def _store_section(auth_store: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Return ``auth_store[key]`` as a dict, replacing a missing/non-dict value in place."""
    section = auth_store.get(key)
    if not isinstance(section, dict):
        section = auth_store[key] = {}
    return section


_global_auth_store_cache: Optional[Tuple[str, int, Dict[str, Any]]] = None
