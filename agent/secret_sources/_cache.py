"""Shared substrate for external secret-source backends.

Every backend (Bitwarden, 1Password, …) needs the same handful of
security-sensitive primitives:

  * a uniform result object (:class:`FetchResult`),
  * environment-variable name validation (:func:`is_valid_env_name`),
  * a two-layer fetch cache whose disk half writes atomically with ``0600``
    permissions and honours a TTL (:class:`DiskCache`, :class:`CachedFetch`).

These used to live inline inside ``bitwarden.py``.  Pulling them here means
the atomic-write / ``0600`` / TTL logic is audited and fixed in exactly one
place instead of drifting across copy-pasted per-backend modules — each
backend supplies only its own cache-key shape and a serializer for it.

Nothing in this module ever raises out to the caller's hot path: the disk
layer is strictly best-effort (a miss just triggers a refetch), because a
cache problem must never block Hermes startup.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Generic, List, Optional, TypeVar

__all__ = [
    "FetchResult",
    "CachedFetch",
    "DiskCache",
    "is_valid_env_name",
    "resolve_cache_home",
]


# ---------------------------------------------------------------------------
# Result object — the contract every ``apply_*`` entry point returns
# ---------------------------------------------------------------------------


@dataclass
class FetchResult:
    """Outcome of a single secret-source pull.

    ``error`` is set only for *fatal* conditions (nothing was fetched);
    non-fatal issues go into ``warnings`` so the values that did resolve
    are still applied.  ``ok`` is the convenience inverse of ``error``.
    """

    secrets: Dict[str, str] = field(default_factory=dict)
    applied: List[str] = field(default_factory=list)   # set into os.environ
    skipped: List[str] = field(default_factory=list)   # already set / protected
    warnings: List[str] = field(default_factory=list)  # non-fatal issues
    error: Optional[str] = None                        # fatal: nothing fetched
    binary_path: Optional[Path] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


@dataclass
class CachedFetch:
    """A set of fetched secret values plus when they were fetched."""

    secrets: Dict[str, str]
    fetched_at: float

    def is_fresh(self, ttl_seconds: float) -> bool:
        if ttl_seconds <= 0:
            return False
        return (time.time() - self.fetched_at) < ttl_seconds


# ---------------------------------------------------------------------------
# Env-var name validation
# ---------------------------------------------------------------------------


def is_valid_env_name(name: str) -> bool:
    """Return True if ``name`` is a usable POSIX environment-variable name.

    Must be non-empty, start with a letter or underscore, and contain only
    alphanumerics and underscores.  Backends use this to drop secret names
    that couldn't be exported (e.g. ``"has spaces"`` or ``"1LEADING_DIGIT"``).
    """
    if not name:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(c.isalnum() or c == "_" for c in name)


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------


def resolve_cache_home(home_path: Optional[Path] = None) -> Path:
    """Resolve the Hermes home used for cache paths.

    ``home_path`` is whatever ``load_hermes_dotenv()`` already resolved;
    falling back to ``$HERMES_HOME`` / ``~/.hermes`` keeps direct callers
    (and tests that don't thread a home through) working.
    """
    if home_path is None:
        home_path = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    return home_path


K = TypeVar("K")


class DiskCache(Generic[K]):
    """Best-effort, profile-aware on-disk cache for fetched secret values.

    One JSON object per backend lives at ``<hermes_home>/cache/<basename>``::

        {"key": "<serialized cache key>", "secrets": {...}, "fetched_at": 1.0}

    The file holds only secret *values* keyed by the serialized cache key —
    never raw auth material.  Backends are responsible for fingerprinting
    tokens/sessions *before* they reach ``key_serializer`` so the token can't
    land in the key.

    Writes are atomic (``mkstemp`` → ``chmod 0600`` → ``os.replace``) and the
    containing ``cache/`` directory is forced to ``0700`` — ``mkdir``'s mode is
    umask-subject, so the chmod is the reliable form.  Both ``read`` and
    ``write`` short-circuit when ``ttl_seconds <= 0``, so setting the TTL to
    zero disables *both* cache layers symmetrically: a user opting out never
    gets secret values written to disk at all.
    """

    def __init__(self, basename: str, *, key_serializer: Callable[[K], str]) -> None:
        self._basename = basename
        self._key_serializer = key_serializer
        # Temp-file prefix derived from the basename so concurrent writers for
        # different backends in the same dir don't collide on the staging name.
        stem = basename.split(".", 1)[0]
        self._tmp_prefix = f".{stem}_"

    def path(self, home_path: Optional[Path] = None) -> Path:
        return resolve_cache_home(home_path) / "cache" / self._basename

    def read(
        self,
        key: K,
        ttl_seconds: float,
        home_path: Optional[Path] = None,
    ) -> Optional[CachedFetch]:
        """Return a fresh cached entry for ``key``, or None.

        Best-effort: any I/O or parse error, a key mismatch, or a stale entry
        all return None so the caller re-fetches.
        """
        if ttl_seconds <= 0:
            return None
        path = self.path(home_path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("key") != self._key_serializer(key):
            return None
        secrets = payload.get("secrets")
        fetched_at = payload.get("fetched_at")
        if not isinstance(secrets, dict) or not isinstance(fetched_at, (int, float)):
            return None
        # JSON permits non-string values; env vars need strings, so coerce by
        # dropping anything that isn't a str→str pair.
        typed: Dict[str, str] = {
            k: v for k, v in secrets.items() if isinstance(k, str) and isinstance(v, str)
        }
        entry = CachedFetch(secrets=typed, fetched_at=float(fetched_at))
        if not entry.is_fresh(ttl_seconds):
            return None
        return entry

    def write(
        self,
        key: K,
        entry: CachedFetch,
        ttl_seconds: float,
        home_path: Optional[Path] = None,
    ) -> None:
        """Persist ``entry`` for ``key`` atomically at mode ``0600``.

        No-op when ``ttl_seconds <= 0`` (so caching is genuinely off) or on any
        I/O error — the next invocation just re-fetches.
        """
        if ttl_seconds <= 0:
            return
        path = self.path(home_path)
        try:
            cache_dir = path.parent
            cache_dir.mkdir(parents=True, exist_ok=True)
            # mkdir's mode is umask-subject; chmod the dir to 0700 so cache
            # metadata isn't exposed if HERMES_HOME is ever made traversable.
            try:
                os.chmod(cache_dir, 0o700)
            except OSError:
                pass
            payload = {
                "key": self._key_serializer(key),
                "secrets": entry.secrets,
                "fetched_at": entry.fetched_at,
            }
            # Write to a sibling temp file and atomic-rename.  tempfile honours
            # os.umask, so we explicitly chmod 0600 before the rename.
            fd, tmp = tempfile.mkstemp(
                prefix=self._tmp_prefix, suffix=".tmp", dir=str(cache_dir)
            )
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
        except OSError:
            pass  # best-effort — a disk-cache miss next invocation is fine

    def clear(self, home_path: Optional[Path] = None) -> None:
        """Delete the on-disk cache file if present (idempotent)."""
        try:
            self.path(home_path).unlink()
        except (FileNotFoundError, OSError):
            pass
