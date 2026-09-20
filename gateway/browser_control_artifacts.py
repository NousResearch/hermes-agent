"""One-shot artifact transport for browser control (Gateway side); :mod:`gateway.platforms.api_server`
authenticates and rate-limits, then hands bytes here. Controller frames carry only server-minted
``[0-9a-f]{32}`` ids (client filenames are metadata, never paths); bytes live under a controlled root
for a short TTL. Size/MIME caps apply before any write; SHA-256 is re-verified on read; ``load`` needs
the exact scope key and consumes atomically. The index is lock-guarded and files are temp-written then
renamed so readers never see partials. The index is in memory unless a store opts in to a durable one
(``index_path``), which only long-lived scopes such as run artifacts want."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT_TTL_SECONDS = 300.0
DEFAULT_MAX_ARTIFACT_BYTES = 10 * 1024 * 1024
#: Exact allowlist — parameterized/unknown variants are rejected.
DEFAULT_ALLOWED_MIME_TYPES = frozenset({
    "application/json", "application/pdf", "image/gif", "image/jpeg", "image/png", "image/webp", "text/plain",
})
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
    one_shot: bool = True

    def to_dict(self, *, download_path: str = "") -> dict[str, Any]:
        """Serialize to the wire receipt (never contains file paths)."""
        return {
            "artifact_id": self.artifact_id, "sha256": self.sha256, "size_bytes": self.size_bytes,
            "content_type": self.content_type, "filename": self.filename, "created_at": self.created_at,
            "expires_at": self.expires_at, "ttl_seconds": self.ttl_seconds, "one_shot": self.one_shot,
            **({"download_path": download_path} if download_path else {}),
        }


def artifact_scope_key(scope: Any) -> str:
    """Derive the stable scope key an artifact is bound to.

    Only principal (mandatory) + transport family participate. ``session_id`` is deliberately EXCLUDED: HTTP
    artifact routes authenticate by API key and can't resolve a session while broker dispatch always carries
    one, so hashing it would make upload and dispatch never compose (ids are unguessable and downloads
    one-shot). Capabilities/optional ids are excluded so a reconnect keeps its artifacts."""
    principal = family = ""
    try:
        principal = str(getattr(scope, "principal_id", "") or "")
        family = str(getattr(scope, "transport_family", "") or "")
    except Exception:
        pass
    if not principal:
        # Fail closed: only an authenticated principal may mint artifacts.
        raise ArtifactError("artifact scope must carry a resolved principal")
    return hashlib.sha256(f"{principal}\x00{family}".encode("utf-8")).hexdigest()


@dataclass
class _ArtifactEntry:
    receipt: ArtifactReceipt
    path: Path


class _ReceiptIndex:
    """Durable receipt index for an :class:`ArtifactStore` that opts in.

    Bytes outlive the process but an in-memory index does not, so a restart silently orphans every
    live artifact: durable run status keeps advertising them while the download 404s. Only receipt
    metadata is stored -- never a filesystem path, so a tampered index cannot redirect a read out of
    the controlled root; paths are always re-derived from the current root on restore."""

    _COLUMNS = ("artifact_id", "sha256", "size_bytes", "content_type", "filename",
                "created_at", "expires_at", "ttl_seconds", "scope_key", "one_shot")
    _SCHEMA = """CREATE TABLE IF NOT EXISTS artifact_receipts (
        artifact_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL,
        content_type TEXT NOT NULL, filename TEXT NOT NULL, created_at REAL NOT NULL,
        expires_at REAL NOT NULL, ttl_seconds REAL NOT NULL, scope_key TEXT NOT NULL,
        one_shot INTEGER NOT NULL DEFAULT 0)"""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._durable = True
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._path), check_same_thread=False, timeout=30)
        except Exception as exc:
            # A read-only or full HERMES_HOME must not take the gateway down with it; artifacts
            # still work for the life of this process, they just will not survive a restart.
            logger.warning("artifact index unavailable (%s); receipts will not survive a restart",
                           type(exc).__name__)
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._durable = False
        with contextlib.suppress(Exception):
            from hermes_state_wal import apply_wal_with_fallback
            apply_wal_with_fallback(self._conn, db_label=self._path.name)
        self._conn.execute(self._SCHEMA)
        self._conn.commit()
        self._tighten_permissions()

    @property
    def durable(self) -> bool:
        """Whether receipts written here survive this process."""
        return self._durable

    def _tighten_permissions(self) -> None:
        """Scope keys are hashes rather than secrets, but the index still describes a principal's
        artifacts; keep it owner-only like every other Hermes state DB."""
        for suffix in ("", "-wal", "-shm") if self._durable else ():
            candidate = Path(f"{self._path}{suffix}")
            with contextlib.suppress(OSError):
                if candidate.exists():
                    candidate.chmod(0o600)

    def put(self, receipt: ArtifactReceipt) -> None:
        placeholders = ",".join("?" * len(self._COLUMNS))
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO artifact_receipts ({','.join(self._COLUMNS)}) VALUES ({placeholders})",
                (receipt.artifact_id, receipt.sha256, receipt.size_bytes, receipt.content_type,
                 receipt.filename, receipt.created_at, receipt.expires_at, receipt.ttl_seconds,
                 receipt.scope_key, int(receipt.one_shot)))
            self._conn.commit()

    def delete(self, artifact_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM artifact_receipts WHERE artifact_id=?", (artifact_id,))
            self._conn.commit()

    def rows(self) -> list:
        with self._lock:
            return list(self._conn.execute(f"SELECT {','.join(self._COLUMNS)} FROM artifact_receipts"))

    def close(self) -> None:
        with self._lock, contextlib.suppress(Exception):
            self._conn.close()


class ArtifactStore:
    """Thread-safe, TTL-bounded, scope-bound one-shot artifact store."""
    def __init__(self, root: Path, *, ttl_seconds: float = DEFAULT_ARTIFACT_TTL_SECONDS, max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
                 allowed_mime_types: frozenset = DEFAULT_ALLOWED_MIME_TYPES, clock: Optional[Callable[[], float]] = None,
                 one_shot: bool = True, index_path: Optional[Path] = None) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._ttl_seconds = max(1.0, float(ttl_seconds))
        self._max_bytes = max(1, int(max_bytes))
        self._allowed_mime_types = frozenset(allowed_mime_types)
        self._clock = clock if clock is not None else time.time
        self._one_shot = bool(one_shot)
        self._lock = threading.RLock()
        self._entries: dict[str, _ArtifactEntry] = {}
        # Without a durable index receipts live only in memory, so files left by a previous
        # process are unreachable orphans by definition. With one, restore first so the sweep
        # below only removes what is genuinely unreachable.
        self._index = _ReceiptIndex(Path(index_path)) if index_path is not None else None
        if self._index is not None:
            self._restore_from_index()
        self._sweep_orphan_files()

    def _sweep_orphan_files(self) -> int:
        """Delete on-disk files with no live index entry; only minted-id-shaped and ``*.tmp`` names are
        touched. Returns the number removed."""
        removed = 0
        try:
            candidates = list(self._root.iterdir())
        except OSError:
            return 0
        with self._lock:
            live = set(self._entries)
        for path in candidates:
            orphan = path.name.endswith(_TEMP_SUFFIX) or (_ARTIFACT_ID_RE.fullmatch(path.name) and path.name not in live)
            if path.is_file() and orphan:
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)
                    removed += 1
        return removed

    def _restore_from_index(self) -> int:
        """Rebuild live entries from the durable index; return how many came back.

        The index is state written by an earlier process, so nothing in it is trusted: ids are
        re-matched against the minted shape, the path is re-derived from the current root rather
        than read from the row, and the content type is re-checked against the live allowlist (which
        may have tightened since). Anything expired, unreadable, file-less or malformed is dropped
        and its row deleted, so a restart can only ever shrink what is reachable."""
        now = self._clock()
        restored: dict[str, _ArtifactEntry] = {}
        stale: list[str] = []
        try:
            rows = self._index.rows()
        except Exception:
            logger.warning("artifact index unreadable; starting empty")
            return 0
        for row in rows:
            artifact_id = row[0] if isinstance(row[0], str) else ""
            if not _ARTIFACT_ID_RE.fullmatch(artifact_id):
                # No safe path to derive, so drop the row without touching the filesystem.
                stale.append(row[0])
                continue
            try:
                receipt = ArtifactReceipt(
                    artifact_id=artifact_id, sha256=str(row[1]), size_bytes=int(row[2]),
                    content_type=_normalize_content_type(row[3]), filename=_bounded_filename(row[4]),
                    created_at=float(row[5]), expires_at=float(row[6]), ttl_seconds=float(row[7]),
                    scope_key=str(row[8]), one_shot=bool(row[9]))
                path = self._artifact_path(artifact_id)
            except (ArtifactError, TypeError, ValueError):
                stale.append(artifact_id)
                continue
            if (receipt.expires_at <= now or not receipt.scope_key
                    or receipt.content_type not in self._allowed_mime_types or not path.is_file()):
                stale.append(artifact_id)
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)
                continue
            restored[artifact_id] = _ArtifactEntry(receipt=receipt, path=path)
        with self._lock:
            self._entries.update(restored)
        for artifact_id in stale:
            self._forget_durable(artifact_id)
        return len(restored)

    def _forget_durable(self, artifact_id: str) -> None:
        """Drop a receipt from the durable index, if this store keeps one."""
        if self._index is None:
            return
        try:
            self._index.delete(artifact_id)
        except Exception:
            logger.warning("artifact %s: index delete failed; a stale row may linger", artifact_id)

    def close(self) -> None:
        """Release the durable index handle. In-memory state is unaffected."""
        if self._index is not None:
            self._index.close()

    @property
    def root(self) -> Path:
        """Controlled artifact root (never exposed to callers by default)."""
        return self._root

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def allowed_mime_types(self) -> frozenset:
        return self._allowed_mime_types

    def store(self, data: bytes, *, filename: str, content_type: str, scope: Any) -> ArtifactReceipt:
        """Validate and store one artifact, returning its receipt; size/MIME rejections fire before any disk write."""
        size = len(data)
        if size > self._max_bytes:
            raise ArtifactTooLarge(f"artifact is {size} bytes; cap is {self._max_bytes}")
        normalized_type = _normalize_content_type(content_type)
        if normalized_type not in self._allowed_mime_types:
            raise ArtifactMimeRejected(f"content type {content_type!r} is outside the exact allowlist")
        scope_key = artifact_scope_key(scope)
        now = self._clock()
        # Mint a fresh id; retry on an astronomically unlikely collision.
        while True:
            artifact_id = secrets.token_hex(16)
            target = self._artifact_path(artifact_id)
            with self._lock:
                if artifact_id not in self._entries and not target.exists():
                    receipt = ArtifactReceipt(
                        artifact_id=artifact_id, sha256=hashlib.sha256(data).hexdigest(), size_bytes=size,
                        content_type=normalized_type, filename=_bounded_filename(filename), created_at=now,
                        expires_at=now + self._ttl_seconds, ttl_seconds=self._ttl_seconds, scope_key=scope_key,
                        one_shot=self._one_shot,
                    )
                    self._entries[artifact_id] = _ArtifactEntry(receipt=receipt, path=target)
                    break
        # Temp + atomic rename so readers never observe a partial artifact.
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
            with contextlib.suppress(Exception):
                temp.unlink(missing_ok=True)
            raise
        if self._index is not None:
            # After the rename, never before: the index must not advertise a file that is not there.
            try:
                self._index.put(receipt)
            except Exception:
                # Durability is best effort; the artifact is live in this process either way.
                logger.warning("artifact %s: index write failed; it will not survive a restart", artifact_id)
        return receipt

    def validate(self, artifact_id: str, *, scope: Any) -> ArtifactReceipt:
        """Receipt when the artifact is live for ``scope`` (existence, TTL, scope), without consuming."""
        return self._entry_for(artifact_id, scope=scope).receipt

    def load(self, artifact_id: str, *, scope: Any) -> tuple[bytes, ArtifactReceipt]:
        """Verify and read bytes; one-shot stores consume only after checksum validation."""
        with self._lock:
            entry = self._entry_for(artifact_id, scope=scope)
            if not entry.path.exists():
                self._entries.pop(artifact_id, None)
                raise ArtifactNotFound(f"artifact {artifact_id!r} is gone")
            try:
                data = entry.path.read_bytes()
            except OSError as exc:
                raise ArtifactError(f"artifact read failed: {exc}") from exc
            if hashlib.sha256(data).hexdigest() != entry.receipt.sha256:
                raise ArtifactChecksumMismatch(f"artifact {artifact_id!r} failed SHA-256 validation")
            if self._one_shot:
                # Drop the index entry first so a concurrent load fails closed.
                self._entries.pop(artifact_id, None)
                self._forget_durable(artifact_id)
        if self._one_shot:
            try:
                entry.path.unlink(missing_ok=True)
            except OSError:
                logger.warning("artifact %s: file removal failed; TTL sweep will retry", artifact_id)
        return data, entry.receipt

    def prune_expired(self, now: Optional[float] = None) -> int:
        """Delete every artifact past its TTL (and stale temp files); return the count removed."""
        now = self._clock() if now is None else float(now)
        with self._lock:
            removed = self._prune_expired_locked(now)
            for temp in self._root.glob(f"*{_TEMP_SUFFIX}"):
                with contextlib.suppress(OSError):
                    if temp.stat().st_mtime <= now - self._ttl_seconds:
                        temp.unlink(missing_ok=True)
        return removed

    def count(self) -> int:
        """Number of live (unconsumed, not-yet-pruned) artifacts."""
        with self._lock:
            return len(self._entries)

    def _discard_locked(self, artifact_id: str, path: Path) -> None:
        """Drop the index entry and best-effort unlink its file."""
        self._entries.pop(artifact_id, None)
        self._forget_durable(artifact_id)
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)

    def _entry_for(self, artifact_id: str, *, scope: Any) -> _ArtifactEntry:
        path = self._artifact_path(artifact_id)
        scope_key = artifact_scope_key(scope)
        now = self._clock()
        with self._lock:
            entry = self._entries.get(artifact_id)
            # Check the target's own expiry BEFORE sweeping so an expired
            # artifact surfaces as ArtifactExpired, not ArtifactNotFound.
            if entry is None:
                self._prune_expired_locked(now)
                entry = self._entries.get(artifact_id)
            if entry is None:
                raise ArtifactNotFound(f"unknown artifact {artifact_id!r}")
            if entry.receipt.expires_at <= now:
                self._discard_locked(artifact_id, path)
                raise ArtifactExpired(f"artifact {artifact_id!r} expired")
            if entry.receipt.scope_key != scope_key:
                raise ArtifactScopeMismatch(f"artifact {artifact_id!r} is bound to a different scope")
            return entry

    def _prune_expired_locked(self, now: float) -> int:
        expired = [(aid, e.path) for aid, e in self._entries.items() if e.receipt.expires_at <= now]
        for artifact_id, path in expired:
            self._discard_locked(artifact_id, path)
        return len(expired)

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
    return value.strip().split(";", 1)[0].strip().lower() if isinstance(value, str) else ""


def _bounded_filename(value: str, limit: int = 160) -> str:
    """Sanitize a display-only filename; never used as a filesystem path."""
    cleaned = value.strip().replace("\\", "_").replace("/", "_") if isinstance(value, str) else ""
    return "".join(character for character in cleaned if ord(character) >= 32)[:limit]


class ArtifactRateLimiter:
    """Sliding-window per-key limiter; the API server keys it by principal."""
    def __init__(self, *, window_seconds: float = 60.0, max_requests: int = 30, clock: Optional[Callable[[], float]] = None) -> None:
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
        with self._lock:
            hits = [hit for hit in self._hits.get(key, []) if hit > now - self._window_seconds]
            allowed = len(hits) < self._max_requests
            if allowed:
                hits.append(now)
            self._hits[key] = hits
            return allowed

    def reset(self, key: str) -> None:
        """Drop the recorded hits for ``key`` (tests/diagnostics)."""
        with self._lock:
            self._hits.pop(key, None)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

class ArtifactOverwrite(ArtifactError):
    """An artifact id already exists and the store refuses to overwrite it."""
# ---- END PLUGIN-COMPAT ----
