"""Pure filesystem helpers behind the ``GET/PUT /api/soul`` endpoints.

This module deliberately has no aiohttp, gateway, or session-DB imports:
everything the HTTP layer needs is a pure function over the profile home
directory, so the persona-mutation semantics (hash gate, backup
rotation, atomic replace, byte cap) are unit-testable without a running
gateway.

Design: docs/specs/2026-11-09-miri-soul-direct-editor-design.md (miri-app
repo). The mutating contract:

- writes are validated (non-empty, <= CAP_BYTES UTF-8 bytes);
- a caller-supplied ``base_sha256`` implements optimistic concurrency:
  mismatch -> ``SoulChangedSinceRead``, file vanished -> ``SoulMissing``
  (refusing to silently recreate a file the caller believed it read);
- the previous file is copied to ``SOUL.md.bak-<UTC stamp>.md`` before
  replacement (suffix ``-2``..``-10`` on same-second collisions, then
  ``BackupCollision``); only files matching our own naming pattern are
  pruned (rotation keeps the newest ``keep``), so backups made by other
  tooling are never touched;
- the write itself is tmp-file + ``os.replace`` on the same directory,
  i.e. atomic on POSIX and on Windows;
- content bytes are stored byte-exact — no newline/Unicode normalisation.
"""

from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path

CAP_BYTES = 65536

SOUL_FILENAME = "SOUL.md"
_BACKUP_INFIX = SOUL_FILENAME + ".bak-"
_BACKUP_RE = re.compile(r"^SOUL\.md\.bak-\d{8}-\d{6}(-\d+)?\.md$")
_MAX_SUFFIX = 10


class SoulNotFound(Exception):
    """SOUL.md does not exist in the profile home."""


class InvalidSoulContent(ValueError):
    """Content rejected before touching disk (empty/whitespace/oversize)."""


class SoulChangedSinceRead(Exception):
    """base_sha256 does not match the current file (optimistic-concurrency 409)."""

    def __init__(self, current_sha256: str):
        super().__init__("soul changed since read")
        self.current_sha256 = current_sha256


class SoulMissing(Exception):
    """base_sha256 was supplied but SOUL.md vanished; refuse, do not create."""


class BackupCollision(Exception):
    """Too many writes in the same second to name a backup; retry later."""


class SoulWriteFailed(OSError):
    """The atomic replace failed; the on-disk soul is byte-exact unchanged."""


def soul_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _backup_stamp() -> str:
    """UTC second-granularity stamp used for backup names (tests freeze it)."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _updated_at(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def read_soul(home: Path) -> dict:
    """Read SOUL.md from ``home`` byte-exact. Raises SoulNotFound if absent."""
    path = Path(home) / SOUL_FILENAME
    if not path.is_file():
        raise SoulNotFound(str(path))
    data = path.read_bytes()
    return {
        "content": data.decode("utf-8"),
        "sha256": soul_sha256(data),
        "bytes": len(data),
        "updated_at": _updated_at(path),
        "path": str(path),
    }


def _validate(content: str) -> bytes:
    if not isinstance(content, str) or not content.strip():
        raise InvalidSoulContent("content must be a non-empty string")
    data = content.encode("utf-8")
    if len(data) > CAP_BYTES:
        raise InvalidSoulContent(
            f"content is {len(data)} UTF-8 bytes, cap is {CAP_BYTES}")
    return data


def _backup_name(home: Path) -> str:
    stamp = _backup_stamp()
    name = f"{_BACKUP_INFIX}{stamp}.md"
    if not (home / name).exists():
        return name
    for n in range(2, _MAX_SUFFIX + 1):
        name = f"{_BACKUP_INFIX}{stamp}-{n}.md"
        if not (home / name).exists():
            return name
    raise BackupCollision("too many soul writes within one second")


def prune_backups(home: Path, keep: int = 50) -> int:
    """Delete the oldest of OUR backups, keeping the newest ``keep``.

    Only files matching the exact naming pattern of this module are
    considered; anything else (e.g. agent-made ``SOUL.md.bak-<date>``)
    is left alone. Returns the number of files deleted.
    """
    home = Path(home)
    ours = sorted(p for p in home.iterdir()
                  if p.is_file() and _BACKUP_RE.match(p.name))
    excess = ours[:-keep] if keep > 0 else ours
    for old in excess:
        try:
            old.unlink()
        except OSError:  # a backup we cannot delete is not worth failing the write
            pass
    return len(excess)


def write_soul(home: Path, content: str, base_sha256: str | None) -> dict:
    """Validate, hash-gate, back up, and atomically write SOUL.md.

    Returns ``{sha256, bytes, updated_at, backup}`` where ``backup`` is
    the created backup file name (None on create-from-nothing). Raises
    InvalidSoulContent / SoulChangedSinceRead / SoulMissing /
    BackupCollision / SoulWriteFailed — each of which the HTTP layer maps
    to the contract's status codes.
    """
    home = Path(home)
    data = _validate(content)
    path = home / SOUL_FILENAME
    existed = path.is_file()

    if base_sha256 is not None:
        if not existed:
            raise SoulMissing(str(path))
        current = soul_sha256(path.read_bytes())
        if current != base_sha256:
            raise SoulChangedSinceRead(current)

    backup_name: str | None = None
    if existed:
        backup_name = _backup_name(home)
        # Copy the current bytes to the backup name (same-directory copy).
        (home / backup_name).write_bytes(path.read_bytes())

    tmp = home / (SOUL_FILENAME + ".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except OSError as exc:
        try:
            tmp.unlink()  # best-effort: never leave a .tmp behind
        except OSError:
            pass
        raise SoulWriteFailed(f"atomic replace failed: {exc}") from exc

    prune_backups(home)
    return {
        "sha256": soul_sha256(data),
        "bytes": len(data),
        "updated_at": _updated_at(path),
        "backup": backup_name,
    }
