"""iterations_logger.py — First-class, queryable iterations.jsonl logger.

Promotes the procedural memory logging from StateLockingOptimizer into a
standalone, reusable component. Integrates with hermes_state.py for session
queries and supports compression/archival policies.

Usage:
    logger = IterationsLogger(log_path="~/.hermes/iterations/{session_id}.jsonl")
    logger.log_iteration(record)
    history = logger.get_history(limit=50)
"""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class IterationLogEntry:
    """Structured entry for iterations.jsonl (one line per iteration)."""

    timestamp: str
    session_id: str
    iteration: int
    primary_score: float
    action: str  # keep | revert | halt
    instrument_trust_score: float = 0.0  # dual scoring support (Phase 1)
    combined_score: float = 0.0
    locked_constraints: List[str] = field(default_factory=list)
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class IterationsLogger:
    """Production-grade logger for autonomous improvement loops."""

    def __init__(
        self,
        log_path: str | Path,
        session_id: str = "default",
        compress_after: int = 100,  # lines before rotating to .gz
        max_entries_in_memory: int = 500,
    ):
        self.log_path = Path(log_path).expanduser()
        self.session_id = session_id
        self.compress_after = compress_after
        self.max_entries_in_memory = max_entries_in_memory
        self._buffer: List[IterationLogEntry] = []
        self._ensure_parent()

    def _ensure_parent(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_iteration(
        self,
        primary_score: float,
        action: str,
        iteration: int,
        instrument_trust_score: float = 0.0,
        locked_constraints: Optional[List[str]] = None,
        explanation: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IterationLogEntry:
        """Append a new iteration record (supports dual scoring)."""
        combined = (primary_score * 0.7) + (instrument_trust_score * 0.3)  # simple weighted dual

        entry = IterationLogEntry(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            session_id=self.session_id,
            iteration=iteration,
            primary_score=round(primary_score, 4),
            instrument_trust_score=round(instrument_trust_score, 4),
            combined_score=round(combined, 4),
            action=action,
            locked_constraints=locked_constraints or [],
            explanation=explanation,
            metadata=metadata or {},
        )
        self._buffer.append(entry)

        # Write immediately (atomic append)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

        # Rotation / compression policy
        if self.log_path.exists() and self.log_path.stat().st_size > 0:
            line_count = sum(1 for _ in self.log_path.open())
            if line_count >= self.compress_after:
                self._rotate_and_compress()

        return entry

    def _rotate_and_compress(self) -> None:
        """Simple rotation: move current to .1.gz and start fresh."""
        if not self.log_path.exists():
            return
        archive = self.log_path.with_suffix(self.log_path.suffix + ".1.gz")
        with self.log_path.open("rb") as src, gzip.open(archive, "wb") as dst:
            dst.write(src.read())
        self.log_path.unlink()

    def get_history(self, limit: int = 100) -> List[IterationLogEntry]:
        """Return recent entries (from buffer + disk)."""
        entries: List[IterationLogEntry] = list(self._buffer[-limit:])
        if self.log_path.exists():
            try:
                with self.log_path.open("r", encoding="utf-8") as f:
                    lines = f.readlines()[-limit:]
                    for line in lines:
                        try:
                            data = json.loads(line.strip())
                            entries.append(IterationLogEntry(**data))
                        except Exception:
                            continue
            except Exception:
                pass
        # Dedup by (iteration, timestamp) and return most recent
        seen = set()
        unique = []
        for e in sorted(entries, key=lambda x: (x.iteration, x.timestamp), reverse=True):
            key = (e.iteration, e.timestamp)
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique[:limit]

    def query_by_session(self, session_id: str, limit: int = 200) -> List[IterationLogEntry]:
        """Future hook for hermes_state.py integration (FTS5 / SQL)."""
        # Placeholder — real implementation would join against state.db
        return [e for e in self.get_history(limit * 2) if e.session_id == session_id][:limit]

    def clear(self) -> None:
        """Clear buffer and truncate log (use with caution)."""
        self._buffer.clear()
        if self.log_path.exists():
            self.log_path.unlink()