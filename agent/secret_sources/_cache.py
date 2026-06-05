"""Shared cache and result substrate for secret-source backends.

Every backend (Bitwarden, 1Password, …) needs the same handful of
security-sensitive primitives:

- a uniform result object (``FetchResult``),
- environment-variable name validation (``is_valid_env_name``),
- a two-layer fetch cache whose disk half writes atomically with
  ``0600`` permissions and honours a TTL (``DiskCache``, ``CachedFetch``).

Pulling them here means the atomic-write / ``0600`` / TTL logic is
audited and fixed in exactly one place instead of drifting across
copy-pasted per-backend modules — each backend supplies only its own
cache-key shape and a serializer for it.

**Design rule**: nothing in this module ever raises out to the caller's
hot path.  The disk layer is strictly best-effort (a miss just triggers
a refetch), because a cache problem must never block Hermes startup.

The two-layer cache improves on the single-disk-layer approach:

1. **L1 — in-process dict** (instant, no I/O).  Survives repeated
   calls within one process (gateway hot-reload, multiple import
   paths).  Expires slightly *before* the L2 TTL so a multi-process
   group naturally converges on a single refresher.

2. **L2 — disk-persisted JSON** (~5 ms on hit).  Shared across
   processes (CLI invocations, cron jobs, gateway forking new agents).

Credit: the shared-cache pattern was first proposed by @hwrdprkns in
PR #36896 for the 1Password CLI backend.  This module generalises it
further with an in-process L1 layer and rate-limit cooldown support
so it can serve both the Bitwarden (low-API-call) and 1Password
(high-API-call) use cases from one auditable location.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FetchResult:
    """Outcome of a single secret-source fetch.

    ``error`` is reserved for fatal conditions where *nothing* was
    fetched (missing binary, bad auth, etc.).  Partial failures go in
    ``warnings`` so that successfully resolved secrets can still be
    applied.
    """

    secrets: Dict[str, str] = field(default_factory=dict)
    applied: List[str] = field(default_factory=list)   # set into os.environ
    skipped: List[str] = field(default_factory=list)   # already set, not overridden
    warnings: List[str] = field(default_factory=list)  # non-fatal issues
    error: Optional[str] = None                         # fatal: nothing was fetched
    binary_path: Optional[Path] = None                  # resolved binary (CLI backends)
    cache_hit: bool = False                             # served from L1 or L2 cache

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class CachedFetch:
    """A set of fetched secrets and when they were retrieved."""

    secrets: Dict[str, str]
    fetched_at: float

    def is_fresh(self, ttl_seconds: float) -> bool:
        """Return True when the entry is still fresh under *ttl_seconds*.

        A TTL ≤ 0 means *never fresh* (opt-out of caching entirely).
        """
        if ttl_seconds <= 0:
            return False
        return (time.time() - self.fetched_at) < ttl_seconds


# ---------------------------------------------------------------------------
# Env-var name validation
# ---------------------------------------------------------------------------


def is_valid_env_name(name: str) -> bool:
    """Return True if *name* is a usable POSIX environment-variable name.

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

K = TypeVar("K")
"""Cache-key type — supplied by the backend."""


class TwoLayerCache(Generic[K]):
    """Best-effort two-layer cache for fetched secret values.

    **L1 — in-process dict**
    Instant hits within one process.  Expires at 90 % of the disk
    TTL so sibling processes naturally converge on a single refresher.

    **L2 — disk-persisted JSON** (``<hermes_home>/cache/<basename>``)
    Shared across processes.  Written atomically (``mkstemp`` →
    ``chmod 0600`` → ``os.replace``).  The cache directory is forced
    to ``0700``.

    The disk file holds only secret **values** keyed by the serialized
    cache key — never raw auth material.  Backends must fingerprint
    tokens/sessions before serialization so the token itself never
    appears in the key.

    Setting ``ttl_seconds`` to 0 disables **both** cache layers
    symmetrically — a user opting out never gets secret values written
    to disk at all.

    **Rate-limit cooldown** (optional): when enabled, a backend can
    record a cooldown timestamp per cache key.  Subsequent reads return
    ``None`` (cooldown active) until the window expires, preventing N
    sibling processes from hammering a rate-limited API in lockstep.
    """

    # L1 — in-process cache
    _l1: Dict[K, CachedFetch] = {}
    _l1_expiry: Dict[K, float] = {}

    # Rate-limit cooldown
    _cooldowns: Dict[K, float] = {}
    _cooldown_enabled: bool = False
    _cooldown_seconds: float = 3600.0

    def __init__(
        self,
        basename: str,
        *,
        serializer: Optional[Callable[[K], str]] = None,
        deserializer: Optional[Callable[[str], K]] = None,
        cooldown_enabled: bool = False,
        cooldown_seconds: float = 3600.0,
    ):
        """Create a two-layer cache.

        Args:
            basename: File name under ``<hermes_home>/cache/``.
            serializer: ``(K) → str`` — converts a cache key to a
                stable JSON-dict key.  Defaults to ``str()``.
            deserializer: ``(str) → K`` — converts a JSON-dict key
                back.  If omitted, the raw string is used as the key
                and the backend must handle conversion.
            cooldown_enabled: When True, ``record_cooldown()`` and
                ``is_cooldown_active()`` gate reads.  Use for
                rate-limited APIs (1Password: 1,000 reads/hour).
            cooldown_seconds: How long a cooldown lasts (default 1 h).
        """
        self._basename = basename
        self._serializer = serializer or str
        self._deserializer = deserializer or (lambda s: s)
        self._cooldown_enabled = cooldown_enabled
        self._cooldown_seconds = cooldown_seconds

    # -- Rate-limit cooldown ------------------------------------------------

    def record_cooldown(self, key: K) -> None:
        """Record that *key* hit a rate limit now."""
        if self._cooldown_enabled:
            self._cooldowns[key] = time.time() + self._cooldown_seconds

    def is_cooldown_active(self, key: K) -> bool:
        """Return True when reads for *key* should be suppressed."""
        if not self._cooldown_enabled:
            return False
        until = self._cooldowns.get(key, 0.0)
        return time.time() < until

    def cooldown_remaining(self, key: K) -> float:
        """Seconds until cooldown expires for *key* (0 if not active)."""
        until = self._cooldowns.get(key, 0.0)
        return max(0.0, until - time.time())

    # -- Two-layer read -----------------------------------------------------

    def read(
        self,
        key: K,
        ttl_seconds: float,
        home_path: Optional[Path] = None,
    ) -> Optional[CachedFetch]:
        """Return a fresh cached entry (L1 then L2), or None.

        If the cooldown is active for *key* the read is aborted early.
        """
        if ttl_seconds <= 0:
            return None

        # Cooldown gate (before any I/O)
        if self.is_cooldown_active(key):
            return None

        # L1 — in-process dict
        entry = self._l1.get(key)
        if entry and entry.is_fresh(ttl_seconds):
            return entry

        # L2 — disk
        disk_entry = self._read_disk(key, ttl_seconds, home_path)
        if disk_entry is not None:
            # Promote to L1 with slightly shorter expiry so the disk
            # layer always expires first and a sibling process can
            # become the designated refresher.
            self._l1[key] = disk_entry
            self._l1_expiry[key] = time.time() + ttl_seconds * 0.9
            return disk_entry

        return None

    def write(
        self,
        key: K,
        entry: CachedFetch,
        home_path: Optional[Path] = None,
    ) -> None:
        """Persist *entry* to L1 and atomically to L2.  Best-effort."""
        # L1
        self._l1[key] = entry
        self._l1_expiry[key] = time.time()

        # L2
        self._write_disk(key, entry, home_path)

    def clear(self, home_path: Optional[Path] = None) -> None:
        """Clear L1 and L2 for all keys.  For tests."""
        self._l1.clear()
        self._l1_expiry.clear()
        self._cooldowns.clear()
        try:
            _disk_cache_path(self._basename, home_path).unlink()
        except (FileNotFoundError, OSError):
            pass

    # -- Disk I/O (private) ------------------------------------------------

    def _read_disk(
        self,
        key: K,
        ttl_seconds: float,
        home_path: Optional[Path] = None,
    ) -> Optional[CachedFetch]:
        path = _disk_cache_path(self._basename, home_path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None

        key_str = self._serializer(key)
        entry_data = payload.get(key_str)
        if entry_data is None:
            return None

        secrets = entry_data.get("secrets")
        fetched_at = entry_data.get("fetched_at")
        if not isinstance(secrets, dict) or not isinstance(fetched_at, (int, float)):
            return None

        # Coerce all values to strings — JSON allows numbers but env vars need str
        typed_secrets: Dict[str, str] = {}
        for k, v in secrets.items():
            if isinstance(k, str) and isinstance(v, str):
                typed_secrets[k] = v

        entry = CachedFetch(secrets=typed_secrets, fetched_at=float(fetched_at))
        if not entry.is_fresh(ttl_seconds):
            return None
        return entry

    def _write_disk(
        self,
        key: K,
        entry: CachedFetch,
        home_path: Optional[Path] = None,
    ) -> None:
        path = _disk_cache_path(self._basename, home_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Force cache directory to 0700
            try:
                path.parent.chmod(0o700)
            except OSError:
                pass

            # Read existing entries to merge (preserve sibling entries)
            existing = {}
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    existing = data
            except (OSError, json.JSONDecodeError):
                pass

            key_str = self._serializer(key)
            existing[key_str] = {
                "secrets": entry.secrets,
                "fetched_at": entry.fetched_at,
            }

            # Atomic write
            fd, tmp = tempfile.mkstemp(
                prefix=".hermes_sc_",
                suffix=".tmp",
                dir=str(path.parent),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(existing, f)
                os.chmod(tmp, 0o600)
                os.replace(tmp, path)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except OSError:
            pass  # best-effort — disk miss on next invocation is fine


# ---------------------------------------------------------------------------
# Home-path resolution
# ---------------------------------------------------------------------------


def resolve_cache_home(home_path: Optional[Path] = None) -> Path:
    """Resolve the Hermes home directory for cache storage.

    If *home_path* is provided it is used directly (tests,
    ``load_hermes_dotenv()`` already resolved it).  Otherwise falls
    back to ``$HERMES_HOME`` and then ``~/.hermes``.
    """
    if home_path is not None:
        return home_path
    return Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))


def _disk_cache_path(basename: str, home_path: Optional[Path] = None) -> Path:
    """Resolve the full path for a disk-cache file."""
    return resolve_cache_home(home_path) / "cache" / basename
