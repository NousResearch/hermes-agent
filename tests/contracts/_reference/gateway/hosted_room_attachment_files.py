"""The attachment blob files of the in-memory reference: one file map per placement, beside that placement's hosted-room tables.

The reference keeps blob bytes in a :class:`BlobFiles` map (file name to bytes) under the ``"hosted_room_attachment_files"`` key of the same namespace that holds the placement's tables (``hosted_room_tables``), outside the copy-on-write table copy: a file written inside a write that then refuses stays behind, and the orphan sweep reclaims it. The file steps: the verified read and the slice read with their size fence, the atomic write, the unlink after the commit and the orphan sweep.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from persistence.gateway.hosted_rooms import AttachmentIntegrityError, HostedPlacement

from .hosted_room_attachment_rules import BLOB_ID_RE, MAX_ATTACHMENT_BYTES
from .hosted_room_tables import read_tables, shared_state_path

__all__ = ["BlobFiles", "blob_files", "read_blob_slice", "read_verified_blob", "sweep_orphan_blobs", "unlink_blobs", "write_blob_file"]

_FILES_KEY = "hosted_room_attachment_files"


class BlobFiles(dict[str, bytes]):
    """One placement's blob directory: file name to bytes. ``before_read``, when set, runs at the start of every blob read: the reference's seam for a write that commits while the bytes are read."""

    before_read: Callable[[], None] | None = None


def blob_files(store: Any, placement: HostedPlacement) -> BlobFiles:
    """The blob directory of ``placement``: the bound profile's for ``'profile'``, the install's root ``shared-state.db`` directory for ``'shared'``."""
    if placement == "profile":
        parent: dict[str, Any] = store.scope_store(store.scope)
    elif placement == "shared":
        parent = store.backend.file_store(shared_state_path(Path(store.scope.profile_dir)))
    else:
        raise ValueError(f"unknown hosted-room placement {placement!r}")
    files: BlobFiles = parent.setdefault(_FILES_KEY, BlobFiles())
    return files


# A blob's file name: its id, which must be a valid blob id.
def _blob_name(blob_id: str) -> str:
    if BLOB_ID_RE.fullmatch(blob_id) is None:
        raise AttachmentIntegrityError("stored blob id is invalid")
    return blob_id


def write_blob_file(files: BlobFiles, blob_id: str, data: bytes) -> None:
    """File step ``write_blob_file``: the atomic write lands the whole file under its blob name."""
    files[_blob_name(blob_id)] = bytes(data)


# A blob file's bytes, refused when the file is missing or its size changed.
def _open_blob(files: BlobFiles, blob_id: str, size: int) -> bytes:
    name = _blob_name(blob_id)
    if files.before_read is not None:
        files.before_read()
    data = files.get(name)
    if data is None:
        raise AttachmentIntegrityError("canonical attachment blob is unavailable")
    if len(data) != size:
        raise AttachmentIntegrityError("canonical attachment blob size changed")
    return data


def read_verified_blob(files: BlobFiles, *, blob_id: str, size: int, sha256: str) -> bytes:
    """File step ``read_verified_blob``: size fence, then the SHA-256 check."""
    data = _open_blob(files, blob_id, size)[:MAX_ATTACHMENT_BYTES + 1]
    if len(data) != size or hashlib.sha256(data).hexdigest() != sha256:
        raise AttachmentIntegrityError("canonical attachment blob failed SHA-256 validation")
    return data


def read_blob_slice(files: BlobFiles, *, blob_id: str, size: int, offset: int, length: int) -> bytes:
    """File step ``read_blob_slice``: size fence, no hash."""
    data = _open_blob(files, blob_id, size)[offset:offset + length]
    if len(data) != min(length, size - offset):
        raise AttachmentIntegrityError("canonical attachment blob size changed")
    return data


def unlink_blobs(files: BlobFiles, blob_ids: list[str]) -> None:
    """The unlinks after a prune commits; a missing file is fine."""
    for blob_id in blob_ids:
        files.pop(_blob_name(blob_id), None)


def sweep_orphan_blobs(store: Any, placement: HostedPlacement) -> None:
    """Staging (``.tmp-``), quarantine (``.prune-``) and unreferenced ``blob_*`` files are reclaimed against the committed blob rows."""
    files = blob_files(store, placement)
    live = set(read_tables(store, placement)["hosted_room_attachment_blobs"])
    for name in list(files):
        if name.startswith((".tmp-", ".prune-")) or (BLOB_ID_RE.fullmatch(name) is not None and name not in live):
            del files[name]
