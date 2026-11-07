"""Data contracts shared by every skill backend (local, plugin, MCP).

Three version axes are deliberately separate and must never be collapsed into one another:

* ``snapshot_revision`` — the host's publication batch for one origin's catalog.
* ``content_revision`` — a fingerprint of one entry's manifest (uri/digest/size set).
* ``connection_generation`` — which MCP transport session produced a result.

None of them is a ``ttlMs`` (freshness only) and none of them is proof that the
server is trustworthy. ``skill://`` URIs are names, not capabilities: holding one
proves nothing about the resource behind it.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

BACKEND_LOCAL = "local"
BACKEND_PLUGIN = "plugin"
BACKEND_MCP = "mcp"


class SkillErrorCode(str, Enum):
    """Typed failures every surface maps to its own affordance."""

    UNSUPPORTED_EXTENSION = "unsupported_extension"
    VERSION_UNSUPPORTED = "version_unsupported"
    PROTOCOL_INVALID = "protocol_invalid"
    PROTOCOL_MISMATCH = "protocol_mismatch"
    NOT_PROVIDED = "not_provided"
    AMBIGUOUS_IDENTITY = "ambiguous_identity"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    VERIFICATION_FAILED = "verification_failed"
    VERSION_CHANGED = "version_changed"
    INCOMPLETE_CATALOG = "incomplete_catalog"
    LIMIT_EXCEEDED = "limit_exceeded"
    DYNAMIC_DISABLED = "dynamic_disabled"
    MUTATION_UNSUPPORTED = "mutation_unsupported"


class SkillError(Exception):
    """A typed skill failure. ``message`` is user-facing; ``code`` drives behaviour."""

    def __init__(self, code: SkillErrorCode, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def as_payload(self) -> dict[str, Any]:
        """The ``skill_view`` JSON error body for this failure."""
        return {"success": False, "error": self.message, "error_code": self.code.value, **self.details}


@dataclass(frozen=True)
class SkillIdentity:
    """A skill's routing identity: which backend, which origin inside it, which locator.

    ``name`` is a display label and never a key — two servers may both publish
    ``refunds``, and one server may publish it twice under different URIs.
    """

    backend_id: str
    origin_id: str
    locator: str

    def as_key(self) -> str:
        """Stable, collision-free string form (``backend\\x1forigin\\x1flocator``).

        NUL separators because every component is user/server-supplied text that
        may itself contain any printable punctuation, including ``:`` and ``/``.
        """
        return f"{self.backend_id}\x1f{self.origin_id}\x1f{self.locator}"


@dataclass(frozen=True)
class OriginInfo:
    """Where an entry came from, in a form the model may be shown."""

    backend_id: str
    origin_id: str
    label: str
    detail: str = ""


@dataclass(frozen=True)
class BackendFeatures:
    """What a backend can do with an entry — capabilities, never permissions."""

    attachments: bool = False
    directory: bool = False
    editable: bool = False
    materializable: bool = False
    executable: bool = False
    remote: bool = False


@dataclass(frozen=True)
class ManifestFile:
    """One file inside a skill, as declared by the server."""

    uri: str
    digest: str
    size: int
    relative_path: str


@dataclass(frozen=True)
class ResourceManifest:
    """An entry's file list. ``dynamic`` manifests carry no content binding and are
    refused for activation by default (per spec a host MAY refuse them)."""

    files: tuple[ManifestFile, ...]
    dynamic: bool = False

    def file_count(self) -> int:
        return len(self.files)

    def total_size(self) -> int:
        return sum(f.size for f in self.files)

    def find(self, uri: str) -> Optional[ManifestFile]:
        return next((f for f in self.files if f.uri == uri), None)

    def fingerprint(self) -> str:
        """Canonical sha256 over the uri/digest set — the ``content_revision``.

        Sorting makes the fingerprint insensitive to listing order; extra host
        fields (size) are excluded so a cosmetic re-serialization is not a new
        version, while a changed digest always is.
        """
        canonical = json.dumps(sorted((f.uri, f.digest) for f in self.files), ensure_ascii=False)
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SkillEntry:
    """One skill as published by a backend: metadata plus its resource manifest.

    ``frontmatter`` is the raw mapping, unknown keys preserved — validation and
    display must not drop fields the server considered meaningful.
    """

    identity: SkillIdentity
    name: str
    description: str
    frontmatter: Mapping[str, Any] = field(default_factory=dict)
    manifest: Optional[ResourceManifest] = None
    content_revision: Optional[str] = None
    features: BackendFeatures = field(default_factory=BackendFeatures)
    provenance: Optional[OriginInfo] = None

    def main_uri(self) -> Optional[str]:
        """The SKILL.md resource URI (the manifest's own main file)."""
        return self.identity.locator if self.identity.backend_id == BACKEND_MCP else None


class BackendState(str, Enum):
    """Transport reachability of one backend instance, independent of catalog freshness."""

    READY = "ready"
    CONNECTING = "connecting"
    SUSPECT = "suspect"
    DISCONNECTED = "disconnected"
    DISABLED = "disabled"


class SkillsSupport(str, Enum):
    """Whether the skills channel of an instance is usable. Orthogonal to BackendState:
    an instance with ``UNSUPPORTED`` skills still serves its ordinary MCP tools."""

    UNKNOWN = "unknown"
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    VERSION_UNSUPPORTED = "version_unsupported"
    PROTOCOL_INVALID = "protocol_invalid"
    PROTOCOL_MISMATCH = "protocol_mismatch"


class CatalogState(str, Enum):
    ABSENT = "absent"
    LOADING = "loading"
    FRESH = "fresh"
    STALE = "stale"
    REFRESH_FAILED = "refresh_failed"


@dataclass(frozen=True)
class CatalogSnapshot:
    """One origin's published catalog. Immutable: a refresh publishes a new one."""

    origin_id: str
    scope_id: str
    entries: Mapping[str, SkillEntry]
    snapshot_revision: int = 0
    connection_generation: Optional[int] = None
    discovered_at: float = field(default_factory=time.monotonic)
    fresh_until: Optional[float] = None
    availability: BackendState = BackendState.READY
    support: SkillsSupport = SkillsSupport.UNKNOWN
    refresh_status: str = ""
    pagination_finished: bool = False
    diagnostics: str = ""

    def entry(self, identity: SkillIdentity) -> Optional[SkillEntry]:
        return self.entries.get(identity.as_key())

    def is_fresh(self, now: Optional[float] = None) -> bool:
        if self.fresh_until is None:
            return True
        return (now if now is not None else time.monotonic()) < self.fresh_until

    def usable(self) -> bool:
        """Whether this snapshot may serve routing/read right now."""
        return self.availability is BackendState.READY and self.support is SkillsSupport.SUPPORTED


@dataclass(frozen=True)
class VerificationResult:
    """What was checked on raw bytes before they were allowed into context."""

    size_ok: bool
    digest_ok: bool
    frontmatter_ok: bool
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.size_ok and self.digest_ok and self.frontmatter_ok and not self.errors


@dataclass(frozen=True)
class VerifiedRead:
    """Content that passed verification, with the provenance needed to attribute it.

    ``skill_dir`` is None for every remote read: a host directory would be a lie —
    remote content has a ``skill_root_uri``, not a path on this machine.
    """

    identity: SkillIdentity
    resource_locator: str
    raw_bytes: bytes
    text: Optional[str]
    mime_type: Optional[str] = None
    content_revision: Optional[str] = None
    provenance: Optional[OriginInfo] = None
    verification: Optional[VerificationResult] = None
    skill_root_uri: Optional[str] = None
    display_name: str = ""
    linked_files: Mapping[str, Sequence[str]] = field(default_factory=dict)

    def with_display(self, name: str, linked: Mapping[str, Sequence[str]]) -> VerifiedRead:
        return replace(self, display_name=name, linked_files=linked)
