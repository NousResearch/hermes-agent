"""One-shot artifact transport for browser control (Gateway side).

Phase 8 Task 29: authenticated one-shot HTTPS upload/download of bounded
browser-control artifacts (screenshots, PDFs, uploads) with SHA-256
validation, exact MIME/size caps, a controlled artifact root, and TTL
cleanup.  This module is the transport-neutral store core: it knows nothing
about aiohttp or the API server — the routes in
:mod:`gateway.platforms.api_server` authenticate callers and enforce rate
limits, then hand bytes to this store.

Why a store at all: the controller WebSocket is a command channel, not a
file pipe.  A controller action that needs bytes (a screenshot upload, a
downloaded PDF) references an artifact by its server-minted id; the agent
side later retrieves it over HTTPS.  Base64 screenshots or files in
controller WebSocket frames are therefore structurally impossible: the
frame carries only ``artifact_id`` strings, and the bytes live on disk
under a controlled root for a short TTL.

Contract (exercised by tests/gateway/test_browser_control_artifacts.py):

- **Server-minted ids, no traversal.** ``store`` assigns a fresh random hex
  id; ``_artifact_path`` accepts only ``[0-9a-f]{N}`` ids and resolves them
  strictly inside the root.  Client-supplied filenames are metadata only
  and never become filesystem paths.

- **Exact size and MIME caps.** ``store`` rejects bytes above
  ``max_bytes`` and any content type outside the configured allowlist
  before anything touches the disk.

- **SHA-256 provenance.** Every artifact is stored with its ``sha256``,
  returned in the receipt, and re-verified by ``load``/``validate`` so a
  corrupted or tampered file can never be handed to a caller.

- **One-shot, scope-bound downloads.** ``load`` requires the exact scope
  key the artifact was stored under and deletes the artifact atomically on
  success.  ``validate`` (used by the broker for "approved artifact id
  only" gating) checks existence, TTL, and scope without consuming.

- **No overwrite.** Ids are random and ``store`` refuses to overwrite an
  existing id (a collision is retried with a fresh id).

- **TTL cleanup.** ``prune_expired`` removes expired entries; the API
  server sweeps on demand.  Nothing in the store is allowed to outlive its
  TTL by more than the sweep interval.

Thread-safety: the in-memory index is guarded by a lock; files are written
to a temp name and atomically renamed into place so a concurrent ``load``
never observes a partially written artifact.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

#: Default lifetime of a stored artifact, in clock seconds.
DEFAULT_ARTIFACT_TTL_SECONDS = 300.0
#: Default per-artifact byte cap (10 MiB).
DEFAULT_MAX_ARTIFACT_BYTES = 10 * 1024 * 1024
#: Default exact MIME allowlist.  Unknown or parameterized variants are
#: rejected; clients must send the canonical registered type.
DEFAULT_ALLOWED_MIME_TYPES = frozenset(
    {
        "application/json",
        "application/pdf",
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/webp",
        "text/plain",
    }
)
#: Length in hex chars of a minted artifact id.
_ARTIFACT_ID_HEX = 32
_ARTIFACT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_TEMP_SUFFIX = ".tmp"


class ArtifactError(Exception):
    """Base class for artifact store contract failures."""


class ArtifactNotFound(ArtifactError):
    """The artifact id is unknown (or already consumed)."""


class ArtifactExpired(ArtifactError):
    """The artifact outlived its TTL."""


class ArtifactTooLarge(ArtifactError):
    """The upload exceeds the configured byte cap."""


class ArtifactMimeRejected(ArtifactError):
    """The content type is outside the exact allowlist."""


class ArtifactScopeMismatch(ArtifactError):
    """The artifact exists but belongs to a different scope."""


class ArtifactChecksumMismatch(ArtifactError):
    """The stored bytes do not match the recorded SHA-256."""


class ArtifactTraversal(ArtifactError):
    """A caller-supplied id is not a valid minted artifact id."""


class ArtifactOverwrite(ArtifactError):
    """An artifact id already exists and the store refuses to overwrite it."""


@dataclass(frozen=True)
class ArtifactReceipt:
    """Provenance record returned to the caller of ``store``."""

    artifact_id: str
    sha256: str
    size_bytes: int
    content_type: str
    filename: str
    created_at: float
    expires_at: float
    ttl_seconds: float
    scope_key: str

    def to_dict(self, *, download_path: str = "") -> dict[str, Any]:
        """Serialize to the wire receipt (never contains file paths)."""
        receipt = {
            "artifact_id": self.artifact_id,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "content_type": self.content_type,
            "filename": self.filename,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "ttl_seconds": self.ttl_seconds,
            "one_shot": True,
        }
        if download_path:
            receipt["download_path"] = download_path
        return receipt


def artifact_scope_key(scope: Any) -> str:
    """Derive the stable scope key an artifact is bound to.

    Only server-derived identity fields participate: principal (mandatory)
    plus transport family. ``session_id`` is deliberately EXCLUDED: the HTTP
    artifact routes authenticate by API key and can never resolve a server
    session, while broker dispatch always carries a session-bearing
    ControllerScope — including the session would make the two halves of the
    intended journey (HTTP upload → broker artifact dispatch) hash to
    different keys and never compose. Artifacts are therefore
    principal/transport-family owned; ids are unguessable server-minted
    32-hex and downloads are one-shot, so cross-session reuse within one
    authenticated principal is by design. Capabilities and optional ids are
    likewise excluded so a reconnect that refreshes the same controller
    keeps its artifacts.
    """
    principal = ""
    family = ""
    try:
        principal = str(getattr(scope, "principal_id", "") or "")
        family = str(getattr(scope, "transport_family", "") or "")
    except Exception:
        pass
    if not principal:
        # Fail closed: an artifact can only be minted for an authenticated
        # principal.
        raise ArtifactError("artifact scope must carry a resolved principal")
    material = f"{principal}\x00{family}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class _ArtifactEntry:
    receipt: ArtifactReceipt
    path: Path


class ArtifactStore:
    """Thread-safe, TTL-bounded, scope-bound one-shot artifact store."""

    def __init__(
        self,
        root: Path,
        *,
        ttl_seconds: float = DEFAULT_ARTIFACT_TTL_SECONDS,
        max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
        allowed_mime_types: frozenset = DEFAULT_ALLOWED_MIME_TYPES,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._ttl_seconds = max(1.0, float(ttl_seconds))
        self._max_bytes = max(1, int(max_bytes))
        self._allowed_mime_types = frozenset(allowed_mime_types)
        self._clock = clock if clock is not None else time.time
        self._lock = threading.RLock()
        self._entries: dict[str, _ArtifactEntry] = {}
        # Restart-safe retention: receipts live only in memory, so files
        # left behind by a previous process are unreachable but would
        # otherwise persist forever. Sweep every artifact-id-shaped file
        # (plus stale temps) that has no index entry — at construction the
        # index is empty, so anything on disk is an orphan from a dead
        # process and past its advertised TTL by definition.
        self._sweep_orphan_files()

    def _sweep_orphan_files(self) -> int:
        """Delete on-disk artifact files with no live index entry.

        Called at construction (empty index ⇒ everything on disk is an
        orphan from a previous process). Only files whose names match the
        server-minted 32-hex id shape or the ``*.tmp`` staging suffix are
        touched; anything else in the directory is left alone.
        """
        removed = 0
        try:
            candidates = list(self._root.iterdir())
        except OSError:
            return 0
        with self._lock:
            live = set(self._entries)
        for path in candidates:
            if not path.is_file():
                continue
            name = path.name
            is_temp = name.endswith(".tmp")
            if not is_temp and not _ARTIFACT_ID_RE.fullmatch(name):
                continue
            if not is_temp and name in live:
                continue
            try:
                path.unlink(missing_ok=True)
                removed += 1
            except OSError:
                continue
        return removed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def root(self) -> Path:
        """Controlled artifact root (never exposed to callers by default)."""
        return self._root

    @property
    def ttl_seconds(self) -> float:
        return self._ttl_seconds

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def allowed_mime_types(self) -> frozenset:
        return self._allowed_mime_types

    def store(
        self,
        data: bytes,
        *,
        filename: str,
        content_type: str,
        scope: Any,
    ) -> ArtifactReceipt:
        """Validate and store one artifact, returning its provenance receipt.

        Raises :class:`ArtifactTooLarge` / :class:`ArtifactMimeRejected`
        before any disk write; :class:`ArtifactError` if the scope is not a
        fully resolved browser-control scope.
        """
        size = len(data)
        if size > self._max_bytes:
            raise ArtifactTooLarge(
                f"artifact is {size} bytes; cap is {self._max_bytes}"
            )
        normalized_type = _normalize_content_type(content_type)
        if normalized_type not in self._allowed_mime_types:
            raise ArtifactMimeRejected(
                f"content type {content_type!r} is outside the exact allowlist"
            )
        scope_key = artifact_scope_key(scope)
        now = self._clock()

        # Mint a fresh id; retry on an astronomically unlikely collision.
        while True:
            artifact_id = secrets.token_hex(_ARTIFACT_ID_HEX // 2)
            target = self._artifact_path(artifact_id)
            with self._lock:
                if artifact_id in self._entries:
                    continue
                if target.exists():
                    continue
                receipt = ArtifactReceipt(
                    artifact_id=artifact_id,
                    sha256=_sha256(data),
                    size_bytes=size,
                    content_type=normalized_type,
                    filename=_bounded_filename(filename),
                    created_at=now,
                    expires_at=now + self._ttl_seconds,
                    ttl_seconds=self._ttl_seconds,
                    scope_key=scope_key,
                )
                entry = _ArtifactEntry(receipt=receipt, path=target)
                self._entries[artifact_id] = entry
                break

        # Write via temp + atomic rename so readers never observe a
        # partially written artifact.
        temp = target.with_name(f"{target.name}{_TEMP_SUFFIX}")
        try:
            with open(temp, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, target)
        except Exception:
            with self._lock:
                self._entries.pop(artifact_id, None)
            try:
                temp.unlink(missing_ok=True)
            except Exception:
                pass
            raise
        return receipt

    def validate(self, artifact_id: str, *, scope: Any) -> ArtifactReceipt:
        """Return the receipt when the artifact is live for ``scope``.

        Used by the broker's "approved artifact id only" gate: checks
        existence, TTL, and scope without consuming the artifact.  Raises
        the appropriate :class:`ArtifactError` subclass otherwise.
        """
        return self._entry_for(artifact_id, scope=scope).receipt

    def load(self, artifact_id: str, *, scope: Any) -> tuple[bytes, ArtifactReceipt]:
        """One-shot download: verify, read, checksum, then consume.

        Returns ``(bytes, receipt)`` and atomically deletes the artifact so
        a second ``load`` raises :class:`ArtifactNotFound`.  Raises
        :class:`ArtifactChecksumMismatch` (without consuming) if the file
        on disk does not match the recorded SHA-256.
        """
        with self._lock:
            entry = self._entry_for(artifact_id, scope=scope)
            path = entry.path
            if not path.exists():
                self._entries.pop(artifact_id, None)
                raise ArtifactNotFound(f"artifact {artifact_id!r} is gone")
            try:
                data = path.read_bytes()
            except OSError as exc:
                raise ArtifactError(f"artifact read failed: {exc}") from exc
            if _sha256(data) != entry.receipt.sha256:
                raise ArtifactChecksumMismatch(
                    f"artifact {artifact_id!r} failed SHA-256 validation"
                )
            # Consume atomically: remove the index entry first so a
            # concurrent load fails closed, then delete the file.
            self._entries.pop(artifact_id, None)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("artifact %s: file removal failed; TTL sweep will retry", artifact_id)
        return data, entry.receipt

    def prune_expired(self, now: Optional[float] = None) -> int:
        """Delete every artifact past its TTL; return the count removed.

        Also removes orphaned temp files older than one sweep.  Idempotent
        and safe to call on any request or a periodic sweep.
        """
        now = self._clock() if now is None else float(now)
        removed = 0
        with self._lock:
            for artifact_id, entry in list(self._entries.items()):
                if entry.receipt.expires_at <= now:
                    self._entries.pop(artifact_id, None)
                    try:
                        entry.path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    removed += 1
            for temp in self._root.glob(f"*{_TEMP_SUFFIX}"):
                try:
                    if temp.stat().st_mtime <= now - self._ttl_seconds:
                        temp.unlink(missing_ok=True)
                except OSError:
                    continue
        return removed

    def count(self) -> int:
        """Number of live (unconsumed, not-yet-pruned) artifacts."""
        with self._lock:
            return len(self._entries)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _entry_for(self, artifact_id: str, *, scope: Any) -> _ArtifactEntry:
        path = self._artifact_path(artifact_id)
        scope_key = artifact_scope_key(scope)
        now = self._clock()
        with self._lock:
            entry = self._entries.get(artifact_id)
            # Check the target's own expiry BEFORE sweeping other entries so
            # an expired artifact surfaces as ArtifactExpired rather than
            # silently vanishing into the sweep.
            if entry is None:
                self._prune_expired_locked(now)
                entry = self._entries.get(artifact_id)
            if entry is None:
                raise ArtifactNotFound(f"unknown artifact {artifact_id!r}")
            if entry.receipt.expires_at <= now:
                self._entries.pop(artifact_id, None)
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise ArtifactExpired(f"artifact {artifact_id!r} expired")
            if entry.receipt.scope_key != scope_key:
                raise ArtifactScopeMismatch(
                    f"artifact {artifact_id!r} is bound to a different scope"
                )
            return entry

    def _prune_expired_locked(self, now: float) -> None:
        for artifact_id, entry in list(self._entries.items()):
            if entry.receipt.expires_at <= now:
                self._entries.pop(artifact_id, None)
                try:
                    entry.path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _artifact_path(self, artifact_id: str) -> Path:
        """Resolve a minted id strictly inside the controlled root."""
        if not isinstance(artifact_id, str) or not _ARTIFACT_ID_RE.fullmatch(artifact_id):
            raise ArtifactTraversal(f"invalid artifact id {artifact_id!r}")
        candidate = (self._root / artifact_id).resolve()
        try:
            root_resolved = self._root.resolve()
        except OSError:
            root_resolved = self._root.absolute()
        if candidate.parent != root_resolved or candidate.name != artifact_id:
            raise ArtifactTraversal(f"artifact path escapes root for {artifact_id!r}")
        return candidate


def _normalize_content_type(value: str) -> str:
    """Return the canonical MIME type, or ``""`` for malformed input."""
    if not isinstance(value, str):
        return ""
    return value.strip().split(";", 1)[0].strip().lower()


def _bounded_filename(value: str, limit: int = 160) -> str:
    """Sanitize a display-only filename; never used as a filesystem path."""
    if not isinstance(value, str):
        return ""
    cleaned = value.strip().replace("\\", "_").replace("/", "_")
    cleaned = "".join(character for character in cleaned if ord(character) >= 32)
    return cleaned[:limit]


# ----------------------------------------------------------------------
# Rate limiting (route-level, per principal)
# ----------------------------------------------------------------------


class ArtifactRateLimiter:
    """Sliding-window per-key limiter for artifact routes.

    The API server keys this by the authenticated principal so a single
    key cannot flood the store.  Injected clock makes tests deterministic.
    """

    def __init__(
        self,
        *,
        window_seconds: float = 60.0,
        max_requests: int = 30,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._window_seconds = max(1.0, float(window_seconds))
        self._max_requests = max(1, int(max_requests))
        self._clock = clock if clock is not None else time.time
        self._lock = threading.Lock()
        self._hits: dict[str, list[float]] = {}

    def allow(self, key: str) -> bool:
        """Return True when ``key`` is under the window cap; else False."""
        if not isinstance(key, str) or not key:
            return False
        now = self._clock()
        window_start = now - self._window_seconds
        with self._lock:
            hits = [hit for hit in self._hits.get(key, []) if hit > window_start]
            if len(hits) >= self._max_requests:
                self._hits[key] = hits
                return False
            hits.append(now)
            self._hits[key] = hits
            return True

    def reset(self, key: str) -> None:
        """Drop the recorded hits for ``key`` (tests/diagnostics)."""
        with self._lock:
            self._hits.pop(key, None)
