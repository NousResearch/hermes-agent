"""Active CLAUDE.md loader -- re-reads CLAUDE.md at each decision point.

Unlike other context files that are loaded once at startup, CLAUDE.md is
re-read on every refresh() call so that externally-made changes (e.g. user
edits while the agent is running) take effect immediately.

Sections are delimited by ``# `` headings.  get_relevant_context(task_type)
returns only the section(s) whose heading contains *task_type* (case-insensitive
substring match), falling back to the whole file if no section matches.
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default location of CLAUDE.md inside the hermes-agent config directory.
_DEFAULT_CLAUDE_MD = Path(".hermes") / "hermes-agent" / "CLAUDE.md"


def _find_claude_md() -> Optional[Path]:
    """Locate CLAUDE.md.  Checks HERMES_HOME, then project root."""
    from hermes_constants import get_hermes_home
    hermes_home = Path(get_hermes_home())
    candidates = [
        hermes_home / "CLAUDE.md",
        hermes_home / ".hermes" / "CLAUDE.md",
        hermes_home / "hermes-agent" / "CLAUDE.md",
        Path(__file__).parent.parent / "CLAUDE.md",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _parse_sections(content: str) -> dict[str, str]:
    """Split *content* into sectioned chunks keyed by lowercased heading.

    A section starts at a ``# `` heading (level-1 markdown) and ends just
    before the next ``# `` or at EOF.  The ``[global]`` pseudo-section holds
    any content before the first heading.
    """
    sections: dict[str, str] = {"[global]": ""}
    current_key = "[global]"
    for line in content.splitlines(keepends=True):
        if line.startswith("# ") and not line.startswith("## "):
            heading = line[2:].strip().lower()
            current_key = heading
            sections[current_key] = ""
        else:
            sections[current_key] += line
    # Strip trailing whitespace from each section
    return {k: v.rstrip() for k, v in sections.items()}


class CLAUDE_md_loader:
    """Re-reads CLAUDE.md on demand and exposes task-type-aware section lookup."""

    def __init__(self, path: Optional[Path] = None):
        self._path = path or _find_claude_md()
        self._last_read_mtime: Optional[float] = None
        self._last_read_content: str = ""
        self._sections: dict[str, str] = {}
        self._refresh_count = 0

    @property
    def path(self) -> Optional[Path]:
        return self._path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_refresh(self) -> bool:
        """Return True when the file's mtime is newer than our cached copy."""
        if self._path is None or not self._path.is_file():
            return False
        try:
            mtime = self._path.stat().st_mtime
        except OSError:
            return False
        if self._last_read_mtime is None:
            return True  # Never loaded — needs a first read
        return mtime > self._last_read_mtime

    def refresh(self) -> bool:
        """Re-read CLAUDE.md from disk if should_refresh() is True.

        Returns True when a new load actually happened.
        """
        if not self.should_refresh():
            return False

        if self._path is None or not self._path.is_file():
            self._last_read_content = ""
            self._sections = {}
            self._last_read_mtime = None
            logger.debug("CLAUDE.md not found — loader stayed empty")
            return False

        try:
            content = self._path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("Failed to read CLAUDE.md: %s", e)
            self._last_read_content = ""
            self._sections = {}
            self._last_read_mtime = None
            return False

        self._last_read_content = content
        self._sections = _parse_sections(content)
        self._last_read_mtime = self._path.stat().st_mtime
        self._refresh_count += 1
        logger.info(
            "CLAUDE.md loaded (mtime=%.0f, sections=%d, refresh_count=%d)",
            self._last_read_mtime,
            len(self._sections),
            self._refresh_count,
        )
        return True

    def get_relevant_context(self, task_type: str = "") -> str:
        """Return CLAUDE.md content relevant to *task_type*.

        If *task_type* is non-empty and a section heading contains it
        (case-insensitive substring), only that section is returned.
        Otherwise the entire file (or ``[global]`` if the file was
        sectioned) is returned.  An empty loader returns an empty string.
        """
        # Ensure we have a fresh read on every call
        self.refresh()

        if not self._sections:
            return ""

        if not task_type:
            # No task type — return everything except [global] if other sections exist
            non_global = {k: v for k, v in self._sections.items() if k != "[global]"}
            if non_global:
                return "\n\n".join(non_global.values())
            return self._sections.get("[global]", "")

        task_lower = task_type.lower()
        # Find any section whose heading contains the task_type substring
        matched = []
        for heading, body in self._sections.items():
            if heading != "[global]" and task_lower in heading:
                matched.append(body)

        if matched:
            return "\n\n".join(matched)

        # No match — fall back to global section
        return self._sections.get("[global]", "")

    def last_mtime(self) -> Optional[float]:
        """Return the mtime of the last successful load, or None."""
        return self._last_read_mtime

    def __repr__(self) -> str:
        return (
            f"CLAUDE_md_loader(path={str(self._path)!r}, "
            f"refresh_count={self._refresh_count}, "
            f"sections={len(self._sections)})"
        )
