"""Per-entry provenance for the cognition memory provider.

Every committed entry carries mandatory provenance so a recalled fact can always
be traced back to the session, author, repo, and trace that produced it.
Eviction never destroys evidence: entries removed for any reason are appended
to an eviction-preserving archive (JSONL) first.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

# Mandatory provenance fields. An entry missing any of these is rejected at the
# gate (validate_provenance) and can never reach the committed store.
REQUIRED_FIELDS = (
    "source_session",
    "author",
    "created_at",
    "origin_repo",
    "trace_id",
)


class ProvenanceError(ValueError):
    """Raised when an entry's provenance is missing or malformed."""


def make_provenance(
    *,
    source_session: str,
    author: str,
    origin_repo: str,
    trace_id: str,
    created_at: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a complete provenance record (all mandatory fields present)."""
    return {
        "source_session": source_session,
        "author": author,
        "created_at": created_at if created_at is not None else time.time(),
        "origin_repo": origin_repo,
        "trace_id": trace_id,
    }


def validate_provenance(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Return the entry's provenance or raise ProvenanceError listing gaps.

    Empty-string values count as missing: a provenance field that names nothing
    proves nothing.
    """
    provenance = entry.get("provenance")
    if not isinstance(provenance, dict):
        raise ProvenanceError("entry has no provenance record")
    missing = [f for f in REQUIRED_FIELDS if not provenance.get(f)]
    if missing:
        raise ProvenanceError(
            "entry provenance missing mandatory fields: %s" % ", ".join(missing)
        )
    return provenance


def _atomic_append_jsonl(path: str, record: Dict[str, Any]) -> None:
    """Append one JSON record to a JSONL file (create dirs as needed)."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory or ".", prefix=".archive-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as existing:
                    handle.write(existing.read())
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def archive_entry(
    archive_path: str, entry: Dict[str, Any], *, reason: str, now: Optional[float] = None
) -> Dict[str, Any]:
    """Preserve an evicted entry in the archive; returns the archived record.

    The archive is append-only from the provider's perspective: eviction,
    expiry, and conflict-resolution losers all land here with a reason, and
    nothing in the normal lifecycle ever deletes from it.
    """
    validate_provenance(entry)  # archived evidence must itself be traceable
    record = {
        "ref": entry.get("ref"),
        "reason": reason,
        "archived_at": now if now is not None else time.time(),
        "entry": entry,
    }
    _atomic_append_jsonl(archive_path, record)
    return record


def load_archive(archive_path: str) -> List[Dict[str, Any]]:
    """Read every archived record (empty list when no archive exists yet)."""
    if not os.path.exists(archive_path):
        return []
    records = []
    with open(archive_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
