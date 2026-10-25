"""Atomic file writes.

Same technique as ``utils.atomic_write_text`` in the main hermes-agent
package (temp file in the same directory, fsync, then ``os.replace``) —
reimplemented here, standalone, so this pipeline has no import dependency on
the hermes-agent package. A crash or kill mid-write leaves the previous file
version intact; a reader never observes a half-written file.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, content: str, *, mode: int | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, str(path))
        if mode is not None:
            try:
                os.chmod(path, mode)
            except OSError:
                pass
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    atomic_write_text(Path(path), json.dumps(data, indent=indent, ensure_ascii=False))


def append_line(path: Path, line: str, *, mode: int = 0o600) -> None:
    """Append one line to *path* via O_APPEND, atomic for small writes.

    Used for the append-only audit log: many short writes over time, where
    atomic-replace-the-whole-file would be wasteful and race-prone against
    concurrent readers.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (line if line.endswith("\n") else line + "\n").encode("utf-8", "replace")
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, mode)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)
