"""Media claim-check spool (Tier-2, design §6).

A ``media_ref`` is an opaque token, **never a path on the wire**.  The front
mints refs for inbound media; the worker resolves each ref to local bytes and
materializes them into its own cache, feeding the existing media path with zero
downstream change.  Outbound, the worker mints refs the front resolves + uploads.

MVP resolver = a shared spool directory both processes can read.  The wire
contract (``MediaRef.to_wire``) carries only metadata, so the end-state
HTTP-fetch resolver is an additive swap with no schema change.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

VALID_KINDS = frozenset({"image", "voice", "video", "document"})


def sign_ref(run_id: str, ref: str, secret: str) -> str:
    """HMAC binding a ref to its run, so a token can't be replayed across runs."""
    return hmac.new(secret.encode(), f"{run_id}:{ref}".encode(), hashlib.sha256).hexdigest()


def verify_ref(run_id: str, ref: str, token: str, secret: str) -> bool:
    return hmac.compare_digest(sign_ref(run_id, ref, secret), token or "")


def confine_to_safe_root(path: str, safe_root: Path | str) -> Path:
    """Resolve *path* under *safe_root*, rejecting any escape (harvested from #18510)."""
    root = Path(safe_root).resolve()
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path {path!r} escapes safe_root {safe_root!r}")
    return resolved


def default_spool_root() -> Path:
    """Shared spool dir both front and worker can read.

    Anchored to the default hermes root (not a per-profile home) so the
    transport-owner front and a worker that resolved a different HERMES_HOME
    still meet on the same filesystem path.  Override with HERMES_MEDIA_SPOOL.
    """
    override = os.getenv("HERMES_MEDIA_SPOOL")
    if override:
        return Path(override)
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root() / "cache" / "gateway_media_spool"


@dataclass
class MediaRef:
    ref: str
    filename: str
    mime: str
    kind: str
    size: int
    as_document: bool = False
    is_voice: bool = False

    def to_wire(self) -> dict:
        """Serialize for the request/SSE body — metadata only, never a path."""
        return {
            "ref": self.ref,
            "filename": self.filename,
            "mime": self.mime,
            "kind": self.kind,
            "size": self.size,
            "as_document": self.as_document,
            "is_voice": self.is_voice,
        }

    @classmethod
    def from_wire(cls, data: dict) -> "MediaRef":
        return cls(
            ref=str(data["ref"]),
            filename=str(data.get("filename", "")),
            mime=str(data.get("mime", "application/octet-stream")),
            kind=str(data.get("kind", "document")),
            size=int(data.get("size", 0)),
            as_document=bool(data.get("as_document", False)),
            is_voice=bool(data.get("is_voice", False)),
        )


class MediaSpool:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._filenames: dict[str, str] = {}

    def _path(self, ref: str) -> Path:
        # Refs come off the wire — confine to the spool root, never trust them.
        return confine_to_safe_root(ref, self.root)

    def mint(self, data: bytes, *, filename: str, mime: str, kind: str, **flags) -> MediaRef:
        ref = secrets.token_hex(24)
        self._path(ref).write_bytes(data)
        self._filenames[ref] = filename
        return MediaRef(
            ref=ref, filename=filename, mime=mime, kind=kind, size=len(data),
            as_document=bool(flags.get("as_document", False)),
            is_voice=bool(flags.get("is_voice", False)),
        )

    def resolve(self, ref: str) -> bytes:
        path = self._path(ref)
        if not path.is_file():
            raise KeyError(f"unknown media ref {ref!r}")
        return path.read_bytes()

    def materialize(self, ref: str, dest_dir: Path, *, filename: str | None = None) -> Path:
        """Write the ref's bytes into *dest_dir*, preserving the original suffix."""
        data = self.resolve(ref)
        suffix = Path(filename or self._filenames.get(ref, "") or ref).suffix
        dest = Path(dest_dir) / f"{ref}{suffix}"
        dest.write_bytes(data)
        return dest

    def unlink(self, ref: str) -> None:
        self._filenames.pop(ref, None)
        try:
            os.unlink(self._path(ref))
        except FileNotFoundError:
            pass


_VIDEO_EXTS = frozenset({".mp4", ".mov", ".webm", ".mkv", ".avi"})
_AUDIO_EXTS = frozenset({".ogg", ".opus", ".mp3", ".wav", ".m4a", ".flac", ".aac"})
_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".svg"})


def kind_for(path: str, is_voice: bool) -> str:
    ext = Path(path).suffix.lower()
    if is_voice:
        return "voice"
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _VIDEO_EXTS:
        return "video"
    if ext in _AUDIO_EXTS:
        return "voice"
    return "document"


def mint_outbound(spool: MediaSpool, media_files: list[tuple[str, bool]]) -> list[dict]:
    """Mint refs for worker-produced files (path, is_voice) → wire dicts."""
    import mimetypes

    refs = []
    for path, is_voice in media_files:
        p = Path(path)
        kind = kind_for(path, is_voice)
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        ref = spool.mint(p.read_bytes(), filename=p.name, mime=mime, kind=kind, is_voice=is_voice)
        refs.append(ref.to_wire())
    return refs


def materialize_inbound(spool: MediaSpool, refs: list[dict], dest_dir: Path) -> list[tuple[Path, MediaRef]]:
    """Resolve inbound wire refs to local files in *dest_dir* (worker side).

    Uses only the wire metadata, so it works cross-process from a spool that
    never minted these refs itself.
    """
    out: list[tuple[Path, MediaRef]] = []
    for data in refs:
        mref = MediaRef.from_wire(data)
        out.append((spool.materialize(mref.ref, dest_dir, filename=mref.filename), mref))
    return out
