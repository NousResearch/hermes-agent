"""The logs an upgrade cell's machine writes, and what they gained since a point in the cell.

One reader for every upgrade test that checks log lines (``test_upgrade_path``, the handoff
scenario, the rehearsal cells), so coverage and rotation handling are the same everywhere.

Rotation: Hermes' handlers (stdlib ``RotatingFileHandler``; ``concurrent-log-handler`` on Windows,
without gzip) roll ``<log>`` over to ``<log>.1`` .. ``<log>.<n>``, so a line written after the
snapshot can sit in a rotated sibling by the time a check reads. The snapshot records each file's
identity (device, inode) with its size, and the reader finds a snapshotted file by identity
wherever it was renamed to.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_LOG_GLOBS = ("logs/*.log", "profiles/*/logs/*.log")


@dataclass(frozen=True)
class LogMark:
    """A log file at snapshot time: identity (``None`` where the OS reports none), size, first bytes.

    The first bytes tell a kept file from a new one that reuses the inode of a deleted backup.
    """

    identity: tuple[int, int] | None
    size: int
    head: bytes


_HEAD = 256


def _identity(st: os.stat_result) -> tuple[int, int] | None:
    # st_ino is 0 where the platform reports no file id (Windows before Python 3.12).
    return (st.st_dev, st.st_ino) if st.st_ino else None


def log_files(homes: Iterable[Path], extra: Iterable[Path] = ()) -> list[Path]:
    """Every log under the participating ``HERMES_HOME``s and the extra process logs that exist.

    Per home: ``logs/*.log``, every profile's ``profiles/<name>/logs/*.log`` and the kanban
    worker logs (``t_*.log``, anywhere below it), each with its rotated siblings ``<log>.<n>``.
    """
    found: set[Path] = set(extra)
    for home in homes:
        for pattern in _LOG_GLOBS:
            found.update(home.glob(pattern))
        found.update(home.rglob("t_*.log"))
    found.update(r for p in list(found) for r in p.parent.glob(f"{p.name}.[0-9]*"))
    return sorted(p for p in found if p.is_file())


def log_offsets(homes: Iterable[Path], extra: Iterable[Path] = ()) -> dict[Path, LogMark]:
    """Every log's identity, size and first bytes (``log_files``): what ``log_text_since`` reads after."""
    marks = {}
    for p in log_files(homes, extra):
        data = p.read_bytes()
        marks[p] = LogMark(_identity(p.stat()), len(data), data[:_HEAD])
    return marks


def log_text_since(offsets: dict[Path, LogMark], homes: Iterable[Path], extra: Iterable[Path] = ()) -> str:
    """What every log gained since ``offsets``, each under a ``--- <dir>/<dir>/<name> ---`` header.

    A file kept from the snapshot (found by identity, also after a rollover renamed it to
    ``<log>.<n>``; by name where the OS has no file id) is read from its snapshot size. Any other
    file was written whole since: a new log, the fresh file after a rollover, a truncated log.
    """
    by_identity = {m.identity: m for m in offsets.values() if m.identity is not None}
    parts = []
    for p in log_files(homes, extra):
        ident = _identity(p.stat())
        data = p.read_bytes()
        mark = by_identity.get(ident) if ident is not None else offsets.get(p)
        kept = mark is not None and len(data) >= mark.size and data.startswith(mark.head)
        text = data[mark.size:] if kept else data
        parts.append(f"--- {'/'.join(p.parts[-3:])} ---\n" + text.decode("utf-8", errors="replace"))
    return "\n".join(parts)
