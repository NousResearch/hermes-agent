"""Per-session on-disk artifacts removed alongside a deleted session row.

The transcript files of the sessions a destructive verb removes are unlinked after the deleting transaction commits. ``RoutingRepository.prune_never_active`` takes them as an explicit ``artifacts`` argument carrying the directory and the layout, never as a filesystem location the backend resolves on its own: profile scope (root AGENTS.md, never hardcode the home) is the caller's to bind. The other destructive verbs take no artifacts argument; they return the removed ids, and the caller removes the files after the commit (``session_retention.after_sessions_removed``).

Layout includes ``<id>.json`` and ``<id>.jsonl`` (the transcript pair), ``session_<id>.json`` (the snapshot), and every ``request_dump_<id>_*.json`` (gateway request dumps), with the id matched literally. Removal is best-effort by design: a missing file is nothing, an ``OSError`` is swallowed, and the store transition has already committed, so a filesystem hiccup never fails or rolls back a delete. Removal covers exactly the ids the transition removed (a cascaded delegate child's files go with it; a refused or unknown target's files stay).
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

from .identity import SessionId

__all__ = ["REQUEST_DUMP_PATTERN", "TRANSCRIPT_SUFFIXES", "SessionArtifacts"]

TRANSCRIPT_SUFFIXES: tuple[str, ...] = (".json", ".jsonl")
REQUEST_DUMP_PATTERN = "request_dump_{session_id}_*.json"


@dataclass(frozen=True, slots=True)
class SessionArtifacts:
    """Where a scope's per-session files live; passed by the caller to ``RoutingRepository.prune_never_active``."""

    transcript_dir: Path

    def paths_for(self, session_id: SessionId) -> tuple[Path, ...]:
        """Existing request dumps plus the three transcript and snapshot paths whether or not they exist."""
        sid = str(session_id)
        targets = [self.transcript_dir / f"{sid}{suffix}" for suffix in TRANSCRIPT_SUFFIXES]
        targets.append(self.transcript_dir / f"session_{sid}.json")
        try:
            targets.extend(sorted(self.transcript_dir.glob(REQUEST_DUMP_PATTERN.format(session_id=glob.escape(sid)))))
        except OSError:
            pass
        return tuple(targets)

    def remove_for(self, session_ids: Iterable[SessionId]) -> None:
        """Unlink every artifact of every id, best-effort: missing files and ``OSError`` are ignored."""
        for session_id in session_ids:
            for path in self.paths_for(session_id):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
