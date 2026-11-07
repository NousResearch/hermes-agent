"""Lossless file offload of the compacted region before summarization.

Compaction summarizes the middle of a transcript; at commit the session DB archives
those rows (``active=0``) and ``session_search`` discovers them for as long as the
session's rows survive. A per-session JSONL file written BEFORE the summarizer call
adds a second, independent copy: greppable with ordinary tools, and immune to
row-level archive-coverage edge cases and session-row cleanup. This mirrors
deepagents' ``middleware/summarization.py``, which keeps evicted history in a
per-session file the summary points at.

The copy is byte-faithful (no redaction): it is the same sensitivity class as the
session DB, which already persists raw message bodies. A failed offload never
blocks compaction — the DB archive remains the primary store.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional, Sequence
from uuid import uuid4

logger = logging.getLogger(__name__)

# Newest N offload files per session are kept; compaction is recurring, an
# unbounded trail would grow with session length, not with need.
MAX_OFFLOAD_FILES_PER_SESSION = 10

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_session_dir_name(session_id: str) -> str:
    """Filesystem-safe, collision-resistant directory name for a session id."""
    cleaned = _SAFE_ID_RE.sub("_", session_id).strip("._") or "session"
    return f"{cleaned[:64]}-{uuid4().hex[:6]}" if len(cleaned) < len(session_id) else cleaned[:80]


def offload_dir_for(session_id: str) -> Path:
    """Per-session offload directory under the profile's Hermes home (HX001-safe)."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "offloads" / _safe_session_dir_name(session_id)


def _prune_old_offloads(directory: Path) -> None:
    """Keep the newest ``MAX_OFFLOAD_FILES_PER_SESSION`` files (names are UTC-sortable)."""
    files = sorted(p for p in directory.glob("evicted-*.jsonl"))
    for stale in files[:-MAX_OFFLOAD_FILES_PER_SESSION]:
        try:
            stale.unlink()
        except OSError:
            logger.debug("Could not prune stale compaction offload %s", stale)


def offload_evicted_region(session_id: str, turns: Sequence[Any]) -> Optional[str]:
    """Append the evicted region verbatim to the session's offload JSONL; return its path.

    One file per compaction event: a header record (session, count, UTC timestamp)
    then one JSON object per message, exactly as held in memory. Returns ``None``
    when there is nothing to write or any I/O step fails — compression always
    continues, since the DB archive remains the primary lossless store.
    """
    if not session_id or not turns:
        return None
    try:
        directory = offload_dir_for(session_id)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        path = directory / f"evicted-{stamp}-{uuid4().hex[:6]}.jsonl"
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            header = {
                "offload_meta": {
                    "session_id": session_id,
                    "message_count": len(turns),
                    "created_utc": stamp,
                }
            }
            handle.write(json.dumps(header, ensure_ascii=False, default=str) + "\n")
            for message in turns:
                handle.write(json.dumps(message, ensure_ascii=False, default=str) + "\n")
        tmp.replace(path)
        _prune_old_offloads(directory)
        return str(path)
    except Exception:
        logger.warning("Compaction offload write failed (compression continues)", exc_info=True)
        return None
