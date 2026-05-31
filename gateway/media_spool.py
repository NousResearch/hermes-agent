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

import os
import secrets
from dataclasses import dataclass
from pathlib import Path

VALID_KINDS = frozenset({"image", "voice", "video", "document"})


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
        return self.root / ref

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
