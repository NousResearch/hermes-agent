"""Shared cache substrate for external secret-source backends.

Two-layer fetch cache (in-process + on-disk); the disk half writes atomically
with ``0600`` permissions and honours a TTL, so that logic is audited in exactly
one place. Each backend supplies only its cache-key shape and a serializer.
The disk layer is strictly best-effort: a miss just triggers a refetch, because
a cache problem must never block Hermes startup.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Generic, Optional, TypeVar

__all__ = [
    "CachedFetch",
    "DiskCache",
    "SecretCache",
    "atomic_write_json",
    "entry_from_payload",
    "fingerprint",
    "resolve_cache_home",
]


def fingerprint(material: str) -> str:
    """SHA-256 prefix used as a cache key — never logged, never displayed."""
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


@dataclass
class CachedFetch:
    """A set of fetched secret values plus when they were fetched."""

    secrets: Dict[str, str]
    fetched_at: float

    def is_fresh(self, ttl_seconds: float) -> bool:
        return ttl_seconds > 0 and (time.time() - self.fetched_at) < ttl_seconds


def resolve_cache_home(home_path: Optional[Path] = None) -> Path:
    """``home_path`` as resolved by ``load_hermes_dotenv()``, else ``$HERMES_HOME``/``~/.hermes``."""
    if home_path is None:
        from hermes_constants import get_hermes_home

        home_path = get_hermes_home()
    return home_path


def entry_from_payload(payload: object) -> Optional[CachedFetch]:
    """``{"secrets": {...}, "fetched_at": n}`` → :class:`CachedFetch`, or None if malformed.

    Only str→str pairs survive (JSON permits other types; env vars need strings).
    """
    if not isinstance(payload, dict):
        return None
    secrets, fetched_at = payload.get("secrets"), payload.get("fetched_at")
    if not isinstance(secrets, dict) or not isinstance(fetched_at, (int, float)):
        return None
    typed = {k: v for k, v in secrets.items() if isinstance(k, str) and isinstance(v, str)}
    return CachedFetch(secrets=typed, fetched_at=float(fetched_at))


def atomic_write_json(path: Path, payload: dict, *, tmp_prefix: str) -> None:
    """Write ``payload`` to ``path`` via mkstemp → chmod 0600 → os.replace.

    The containing dir is forced to ``0700`` (``mkdir``'s mode is umask-subject,
    so the chmod is the reliable form). Raises ``OSError`` on failure; callers
    decide whether that is best-effort.
    """
    cache_dir = path.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(cache_dir, 0o700)
    except OSError:
        pass
    # tempfile honours os.umask, so chmod 0600 explicitly before the rename.
    fd, tmp = tempfile.mkstemp(prefix=tmp_prefix, suffix=".tmp", dir=str(cache_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


K = TypeVar("K")


class DiskCache(Generic[K]):
    """Best-effort, profile-aware on-disk cache for fetched secret values.

    One JSON object per backend at ``<hermes_home>/cache/<basename>``::

        {"key": "<serialized cache key>", "secrets": {...}, "fetched_at": 1.0}

    The file holds only secret *values*, never raw auth material — backends
    fingerprint tokens/sessions before they reach ``key_serializer``. Both
    ``read`` and ``write`` short-circuit when ``ttl_seconds <= 0``, so a TTL of
    zero disables both layers symmetrically: an opted-out user never gets
    secret values written to disk at all.
    """

    def __init__(self, basename: str, *, key_serializer: Callable[[K], str]) -> None:
        self._basename = basename
        self._key_serializer = key_serializer
        # Per-backend temp prefix so concurrent writers in one dir never collide.
        self._tmp_prefix = f".{basename.split('.', 1)[0]}_"

    def path(self, home_path: Optional[Path] = None) -> Path:
        return resolve_cache_home(home_path) / "cache" / self._basename

    def read(self, key: K, ttl_seconds: float, home_path: Optional[Path] = None) -> Optional[CachedFetch]:
        """Fresh cached entry for ``key``, or None (I/O error, mismatch, stale)."""
        if ttl_seconds <= 0:
            return None
        try:
            with open(self.path(home_path), "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or payload.get("key") != self._key_serializer(key):
            return None
        entry = entry_from_payload(payload)
        return entry if entry is not None and entry.is_fresh(ttl_seconds) else None

    def write(self, key: K, entry: CachedFetch, ttl_seconds: float, home_path: Optional[Path] = None) -> None:
        """Persist ``entry`` atomically at mode 0600; no-op when ``ttl_seconds <= 0`` or on I/O error."""
        if ttl_seconds <= 0:
            return
        payload = {"key": self._key_serializer(key), "secrets": entry.secrets, "fetched_at": entry.fetched_at}
        try:
            atomic_write_json(self.path(home_path), payload, tmp_prefix=self._tmp_prefix)
        except OSError:
            pass  # best-effort — a disk-cache miss next invocation is fine

    def clear(self, home_path: Optional[Path] = None) -> None:
        """Delete the on-disk cache file if present (idempotent)."""
        try:
            self.path(home_path).unlink()
        except (FileNotFoundError, OSError):
            pass


class SecretCache(Generic[K]):
    """Two-layer cache: in-process dict (L1) over a :class:`DiskCache` (L2).

    L1 saves repeated fetches WITHIN one process (CLI startup, gateway
    hot-reload); L2 saves them ACROSS back-to-back short-lived processes.
    """

    def __init__(self, basename: str, *, key_serializer: Callable[[K], str]) -> None:
        self.memory: Dict[K, CachedFetch] = {}
        self.disk: DiskCache[K] = DiskCache(basename, key_serializer=key_serializer)

    def lookup(self, key: K, ttl_seconds: float, home_path: Optional[Path] = None,
               read_disk: Optional[Callable[[], Optional[CachedFetch]]] = None) -> Optional[CachedFetch]:
        """Fresh entry from L1, else from L2 (promoted into L1), else None.

        ``read_disk`` swaps in an alternative L2 reader (e.g. an encrypted file).
        """
        cached = self.memory.get(key)
        if cached and cached.is_fresh(ttl_seconds):
            return cached
        disk_cached = read_disk() if read_disk else self.disk.read(key, ttl_seconds, home_path)
        if disk_cached is not None:
            self.memory[key] = disk_cached
        return disk_cached

    def store(self, key: K, entry: CachedFetch, ttl_seconds: float, home_path: Optional[Path] = None) -> None:
        self.memory[key] = entry
        self.disk.write(key, entry, ttl_seconds, home_path)

    def clear(self, home_path: Optional[Path] = None) -> None:
        self.memory.clear()
        self.disk.clear(home_path)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'FetchResult': ('agent.secret_sources.base', 'FetchResult'),
    'is_valid_env_name': ('agent.secret_sources.base', 'is_valid_env_name'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
