"""Cooperative cross-process compare-and-set for profile UI metadata."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
import threading
import time
from typing import Optional


# The stable sidecar survives atomic replacement of profile.yaml. Never unlink it:
# replacing the lock inode would let independent writers enter separate locks.
_PROFILE_UI_META_LOCK_TIMEOUT_S = 10.0
_PROFILE_UI_META_LOCK_POLL_S = 0.05
_ui_meta_thread_lock = threading.Lock()
try:
    import fcntl as _uimeta_fcntl
except ImportError:  # pragma: no cover - Windows
    _uimeta_fcntl = None
try:
    import msvcrt as _uimeta_msvcrt
except ImportError:  # pragma: no cover - POSIX
    _uimeta_msvcrt = None


def _clean_ui_meta_revisions(raw) -> dict:
    """Normalise a ``_ui_meta_revisions`` map: str keys, non-bool ints clamped at 0."""
    if not isinstance(raw, dict):
        return {}
    return {str(k): max(0, int(v)) for k, v in raw.items()
            if isinstance(v, int) and not isinstance(v, bool)}


@contextlib.contextmanager
def _profile_ui_meta_file_lock(profile_dir: Path):
    """Bounded OS advisory lock: flock on POSIX, byte zero on Windows.

    A process exit releases its handle's lock. Platforms without either backend
    fail closed. Windows acquisition and release must address the SAME byte;
    append-mode file position is not a stable lock offset.
    """
    import errno

    if _uimeta_fcntl is None and _uimeta_msvcrt is None:
        raise NotImplementedError("ui_meta requires OS advisory locking")
    lock_path = Path(profile_dir) / "profile.yaml.uimeta.lock"
    deadline = time.monotonic() + _PROFILE_UI_META_LOCK_TIMEOUT_S
    with open(lock_path, "a+b") as handle:
        if _uimeta_fcntl is None:
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
        while True:
            try:
                if _uimeta_fcntl is not None:
                    _uimeta_fcntl.flock(handle.fileno(), _uimeta_fcntl.LOCK_EX | _uimeta_fcntl.LOCK_NB)
                else:
                    handle.seek(0)
                    _uimeta_msvcrt.locking(handle.fileno(), _uimeta_msvcrt.LK_NBLCK, 1)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out acquiring ui_meta lock") from exc
                time.sleep(_PROFILE_UI_META_LOCK_POLL_S)
        try:
            yield
        finally:
            if _uimeta_fcntl is not None:
                _uimeta_fcntl.flock(handle.fileno(), _uimeta_fcntl.LOCK_UN)
            else:
                handle.seek(0)
                _uimeta_msvcrt.locking(handle.fileno(), _uimeta_msvcrt.LK_UNLCK, 1)


def apply_ui_meta_cas(
    profile_dir: Path, ui_meta: dict, *, expected_revisions: Optional[dict] = None,
) -> dict:
    """Cooperatively serialize metadata read/check/merge/write across processes.

    Merge keys (None deletes); any expected-revision mismatch rejects the whole
    write. Revisions persist across deletion. The size limit preserves the
    gateway's historical JSON string-length boundary, not a UTF-8 byte limit.
    Only writers using this helper participate; unrelated whole-profile writes
    are not made transactional by this lock. Lock refusal raises to callers.
    The inherited permissive YAML reader treats malformed/unreadable/nonmapping
    documents as empty; accepted writes can replace such documents.
    """
    if not isinstance(ui_meta, dict):
        raise ValueError("ui_meta must be an object")
    if expected_revisions is not None and not isinstance(expected_revisions, dict):
        raise ValueError("expected_revisions must be an object")
    if profile_dir is None or not Path(profile_dir).is_dir():
        raise FileNotFoundError(f"profile directory does not exist: {profile_dir}")
    payload = json.dumps(ui_meta)
    if len(payload) > 65536:
        return {"applied": False, "revisions": {}, "oversize": True}
    profile_dir = Path(profile_dir)
    # The module lock serializes threads; the sidecar lock fences other processes.
    with _ui_meta_thread_lock, _profile_ui_meta_file_lock(profile_dir):
        from hermes_cli.profiles import _load_yaml_dict

        existing = _load_yaml_dict(profile_dir / "profile.yaml")
        existing = existing if isinstance(existing, dict) else {}
        revisions = _clean_ui_meta_revisions(existing.get("_ui_meta_revisions"))
        conflicts = {}
        for key in ui_meta if expected_revisions is not None else ():
            wanted, actual = expected_revisions.get(key), revisions.get(key, 0)
            if not isinstance(wanted, int) or isinstance(wanted, bool) or wanted < 0 or wanted != actual:
                conflicts[key] = {"expected": wanted, "actual": actual}
        if conflicts:
            return {"applied": False, "revisions": {k: revisions.get(k, 0) for k in ui_meta},
                    "conflicts": conflicts}
        current = existing.get("ui_meta")
        current = current if isinstance(current, dict) else {}
        for key, value in ui_meta.items():
            if value is None:
                current.pop(key, None)
            else:
                current[key] = value
            revisions[key] = revisions.get(key, 0) + 1
        if current:
            existing["ui_meta"] = current
        else:
            existing.pop("ui_meta", None)
        existing["_ui_meta_revisions"] = revisions
        from utils import atomic_yaml_write
        atomic_yaml_write(profile_dir / "profile.yaml", existing, sort_keys=False)
        return {"applied": True, "revisions": {k: revisions[k] for k in ui_meta}}
