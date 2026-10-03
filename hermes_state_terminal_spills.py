"""Session-owned terminal spill references and post-delete cleanup.

Terminal spill files are durable transcript artifacts, not an age-based cache.  A
reference row is written in the same transaction as the tool message.  Session
foreign-key deletion queues a spill name only after its final owning row leaves;
filesystem cleanup then runs in a separate transaction so a failed session
commit can never leave a retained transcript pointing at an already-unlinked
file.
"""

from __future__ import annotations

import json
import logging
import os
import re
import stat
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("hermes_state")

_TERMINAL_SPILL_FIELD_RE = re.compile(
    r'"full_output_path"\s*:\s*("(?:[^"\\]|\\.)*")'
)


def _spill_paths_from_content(content: Any) -> list[str]:
    """Extract terminal ``full_output_path`` values from a stored tool result."""
    if not isinstance(content, str) or "full_output_path" not in content:
        return []
    try:
        decoded = json.loads(content)
    except (TypeError, ValueError):
        decoded = None
    if isinstance(decoded, dict) and isinstance(decoded.get("full_output_path"), str):
        return [decoded["full_output_path"]]

    paths: list[str] = []
    for match in _TERMINAL_SPILL_FIELD_RE.finditer(content):
        try:
            value = json.loads(match.group(1))
        except (TypeError, ValueError):
            continue
        if isinstance(value, str):
            paths.append(value)
    return paths


class SessionTerminalSpillsMixin:
    """Durable ownership accounting for ``cache/terminal-output`` files."""

    db_path: Path
    read_only: bool
    _read_one: Any
    _execute_write: Any

    def _terminal_spill_dir(self) -> Path:
        return Path(self.db_path).parent / "cache" / "terminal-output"

    def _owned_terminal_spill_name(self, raw_path: str) -> str | None:
        """Return the contained spill filename, rejecting traversal and non-files."""
        try:
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                return None
            root = self._terminal_spill_dir()
            if candidate.parent.resolve(strict=False) != root.resolve(strict=False):
                return None
            name = candidate.name
            if Path(name).name != name or not (name.startswith("out-") and name.endswith(".log")):
                return None
            info = os.lstat(candidate)
            if not stat.S_ISREG(info.st_mode):
                return None
            if os.name == "posix" and stat.S_IMODE(info.st_mode) != 0o600:
                os.chmod(candidate, 0o600)
            return name
        except (OSError, TypeError, ValueError):
            return None

    def _record_terminal_spill_references(
        self, conn, session_id: str, contents: Iterable[Any],
    ) -> int:
        """Own contained spills referenced by *contents* in the caller's write txn."""
        names = {
            name
            for content in contents
            for raw_path in _spill_paths_from_content(content)
            if (name := self._owned_terminal_spill_name(raw_path)) is not None
        }
        for name in names:
            conn.execute(
                "INSERT OR IGNORE INTO terminal_spill_references (session_id, spill_name) VALUES (?, ?)",
                (session_id, name),
            )
            # A new owner won the writer lock before cleanup; cancel the pending unlink.
            conn.execute("DELETE FROM terminal_spill_cleanup WHERE spill_name = ?", (name,))
        return len(names)

    @staticmethod
    def _safe_queued_spill_name(value: Any) -> str | None:
        name = str(value or "")
        if Path(name).name != name or not (name.startswith("out-") and name.endswith(".log")):
            return None
        return name

    def _drain_terminal_spill_cleanup(self) -> int:
        """Unlink queued spills with no owners, retaining failed unlinks for retry."""
        if getattr(self, "read_only", False) or self._read_one(
            "SELECT 1 FROM terminal_spill_cleanup LIMIT 1"
        ) is None:
            return 0

        root = self._terminal_spill_dir()

        def _do(conn) -> int:
            conn.execute(
                "DELETE FROM terminal_spill_cleanup WHERE EXISTS ("
                "SELECT 1 FROM terminal_spill_references r "
                "WHERE r.spill_name = terminal_spill_cleanup.spill_name)"
            )
            rows = conn.execute("SELECT spill_name FROM terminal_spill_cleanup").fetchall()
            removed = 0
            for row in rows:
                raw_name = row["spill_name"]
                name = self._safe_queued_spill_name(raw_name)
                if name is None:
                    conn.execute(
                        "DELETE FROM terminal_spill_cleanup WHERE spill_name = ?", (raw_name,),
                    )
                    continue
                path = root / name
                try:
                    info = os.lstat(path)
                    if stat.S_ISDIR(info.st_mode):
                        logger.warning("Refusing to remove terminal spill directory queued as a file: %s", path)
                        conn.execute(
                            "DELETE FROM terminal_spill_cleanup WHERE spill_name = ?", (name,),
                        )
                        continue
                    os.unlink(path)  # a symlink removes only the link, never its target
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.debug("Could not remove unreferenced terminal spill %s", path, exc_info=True)
                    continue
                conn.execute("DELETE FROM terminal_spill_cleanup WHERE spill_name = ?", (name,))
                removed += 1
            return removed

        return self._execute_write(_do)
