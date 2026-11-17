"""Client key/value metadata on ``sessions.metadata`` (a JSON object of string values).

External callers (CI bots, ticket trackers, orchestrators) stamp a session with their own
correlation keys and later find it again by them. The column is storage only: nothing here is
read into a prompt, so setting it never touches the cached prefix.

Writes are an RFC 7386 merge-patch over the whole compression lineage (a ``null`` value deletes
the key), the same lineage rule ``pinned``/``archived`` follow, and a compression continuation
inherits its parent's value, so a listing that projects a root onto its tip filters consistently.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

MAX_KEYS = 32
MAX_KEY_LEN = 40
MAX_VALUE_LEN = 500
_KEY_RE = re.compile(rf"[A-Za-z0-9_.-]{{1,{MAX_KEY_LEN}}}")


class SessionMetadataError(ValueError):
    """A metadata patch or filter outside the bounds; ``field`` names the offending input."""

    def __init__(self, field: str, message: str):
        super().__init__(message)
        self.field = field


def _check_key(key: Any, field: str) -> str:
    if not isinstance(key, str) or not _KEY_RE.fullmatch(key):
        raise SessionMetadataError(
            field, f"'{field}': keys must be 1-{MAX_KEY_LEN} characters of A-Z a-z 0-9 _ . -")
    return key


def _check_value(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise SessionMetadataError(field, f"'{field}' must be a string (or null to delete it)")
    if len(value) > MAX_VALUE_LEN:
        raise SessionMetadataError(field, f"'{field}' must be at most {MAX_VALUE_LEN} characters")
    return value


def parse_metadata_patch(patch: Any) -> dict[str, str | None]:
    """Validate a merge-patch body: an object of ``key -> string | null``."""
    if not isinstance(patch, dict):
        raise SessionMetadataError("metadata", "'metadata' must be an object")
    return {
        _check_key(key, "metadata"): None if value is None else _check_value(value, f"metadata.{key}")
        for key, value in patch.items()
    }


def parse_metadata_filter(pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    """``(key, value)`` pairs (``metadata.k=v`` query params, ``--metadata k=v``) -> exact-match filter.
    One key given two different values is an error: no row matches both, and keeping either would
    silently widen the filter."""
    parsed: dict[str, str] = {}
    for key, value in pairs:
        field = f"metadata.{_check_key(key, f'metadata.{key}')}"
        if parsed.setdefault(key, _check_value(value, field)) != value:
            raise SessionMetadataError(field, f"'{field}' was given more than one value")
    return parsed


def metadata_matches(raw: Any, metadata: Mapping[str, str]) -> bool:
    """Python twin of :func:`metadata_filter_clause` for a row already read."""
    stored = decode_metadata(raw)
    return all(stored.get(key) == value for key, value in metadata.items())


def normalize_imported_metadata(raw: Any) -> str | None:
    """An export's ``metadata`` (column text or object) -> column text, held to the write bounds."""
    merged = {k: v for k, v in parse_metadata_patch(decode_metadata(raw) if raw else {}).items() if v is not None}
    if len(merged) > MAX_KEYS:
        raise SessionMetadataError("metadata", f"'metadata' may hold at most {MAX_KEYS} keys")
    return json.dumps(merged, sort_keys=True) if merged else None


def decode_metadata(raw: Any) -> dict[str, str]:
    """The stored column as a dict; NULL or anything that is not a JSON object reads as ``{}``."""
    if isinstance(raw, str) and raw:
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return dict(raw) if isinstance(raw, dict) else {}


def metadata_filter_clause(metadata: Mapping[str, str]) -> tuple[str, list[str]]:
    """``sessions s`` WHERE fragment: every pair must match exactly (AND). Keys are validated, so
    the quoted JSON path (``$."a.b"`` keeps a dotted key one member) is bound, never interpolated."""
    safe = "(CASE WHEN json_valid(s.metadata) THEN s.metadata ELSE '{}' END)"
    clause = " AND ".join(f"json_extract({safe}, ?) = ?" for _ in metadata)
    params: list[str] = []
    for key, value in metadata.items():
        params += [f'$."{key}"', value]
    return f"({clause})", params


def merge_session_metadata(db, session_id: str, patch: Mapping[str, str | None]) -> dict[str, str] | None:
    """Apply *patch* (already validated) to *session_id*'s metadata in one write transaction; see
    :func:`merge_session_metadata_in_transaction`."""
    return db._execute_write(lambda conn: merge_session_metadata_in_transaction(db, conn, session_id, patch))


def merge_session_metadata_in_transaction(db, conn, session_id: str,
                                          patch: Mapping[str, str | None]) -> dict[str, str] | None:
    """Merge *patch* into *session_id*'s metadata and write it across the compression lineage on the
    caller's write *conn*. Returns the merged metadata, or ``None`` when the session does not exist.
    The key bound is checked after the merge, so it holds for the stored value no matter how many
    patches built it; raising aborts the caller's whole transaction."""
    row = conn.execute("SELECT metadata FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if row is None:
        return None
    merged = decode_metadata(row[0])
    for key, value in patch.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    if len(merged) > MAX_KEYS:
        raise SessionMetadataError("metadata", f"'metadata' may hold at most {MAX_KEYS} keys")
    db._set_lineage_column_in_transaction(
        conn, "metadata", session_id, json.dumps(merged, sort_keys=True) if merged else None)
    return merged
