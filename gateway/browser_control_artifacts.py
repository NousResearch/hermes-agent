"""One-shot artifact transport for browser control (Gateway side); :mod:`gateway.platforms.api_server`
authenticates and rate-limits, then hands bytes here. Controller frames carry only server-minted
``[0-9a-f]{32}`` ids (client filenames are metadata, never paths); bytes live under a controlled root
for a short TTL. Size/MIME caps apply before any write; SHA-256 is re-verified on read; ``load`` needs
the exact scope key and consumes atomically. The index is lock-guarded and files are temp-written then
renamed so readers never see partials."""

from __future__ import annotations

import contextlib
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

    def to_dict(self, *, download_path: str = "") -> dict[str, Any]:
        """Serialize to the wire receipt (never contains file paths)."""
        return {
            "artifact_id": self.artifact_id, "sha256": self.sha256, "size_bytes": self.size_bytes,
            "content_type": self.content_type, "filename": self.filename, "created_at": self.created_at,
            "expires_at": self.expires_at, "ttl_seconds": self.ttl_seconds, "one_shot": True,
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


class ArtifactStore:
    """Thread-safe, TTL-bounded, scope-bound one-shot artifact store."""
    def __init__(self, root: Path, *, ttl_seconds: float = DEFAULT_ARTIFACT_TTL_SECONDS, max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
                 allowed_mime_types: frozenset = DEFAULT_ALLOWED_MIME_TYPES, clock: Optional[Callable[[], float]] = None) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._ttl_seconds = max(1.0, float(ttl_seconds))
        self._max_bytes = max(1, int(max_bytes))
        self._allowed_mime_types = frozenset(allowed_mime_types)
        self._clock = clock if clock is not None else time.time
        self._lock = threading.RLock()
        self._entries: dict[str, _ArtifactEntry] = {}
        # Receipts live only in memory, so files left by a previous process
        # are unreachable orphans past their TTL by definition — sweep them.
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
        return receipt

    def validate(self, artifact_id: str, *, scope: Any) -> ArtifactReceipt:
        """Receipt when the artifact is live for ``scope`` (existence, TTL, scope), without consuming."""
        return self._entry_for(artifact_id, scope=scope).receipt

    def load(self, artifact_id: str, *, scope: Any) -> tuple[bytes, ArtifactReceipt]:
        """One-shot download: verify, read, checksum, then consume (a checksum mismatch does not consume)."""
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
            # Drop the index entry first so a concurrent load fails closed.
            self._entries.pop(artifact_id, None)
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
