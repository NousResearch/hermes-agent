"""Full raw transcript reconciliation for the opt-in DAG context engine.

The reconciler is deliberately additive: it mirrors transcript messages that are
missing from SQLite, records deterministic drift warnings, and only advances the
DAG checkpoint after all required ingestion succeeds.  It never rewrites raw
transcripts and never writes assembled/projection messages back as raw history.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
from typing import Any, Deque, Dict, List, Optional, Sequence

from agent.context_dag_store import ContextDAGStore


@dataclass
class TranscriptReconciliationResult:
    session_id: str
    scanned: int = 0
    inserted: int = 0
    duplicates_skipped: int = 0
    matched: int = 0
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    checkpoint_advanced: bool = False
    last_ingested_message_id: Optional[int] = None
    anchor_hash: Optional[str] = None
    covered_ordinals: List[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(w.get("severity") == "error" for w in self.warnings)


def _jsonish_value(value: Any) -> Any:
    """Return structured JSON values in their canonical Python shape.

    SessionDB serializes some structured columns (notably reasoning_details) as
    JSON strings but transcript callers pass the original list/dict.  Hashing
    must see both forms as identical so reconciliation is idempotent.
    """

    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{\"":
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _message_for_hash(message: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize either a transcript dict or DB row dict for stable hashing."""

    out: Dict[str, Any] = {
        "role": message.get("role"),
        "content": message.get("content"),
    }
    for key in (
        "tool_calls",
        "tool_call_id",
        "name",
        "tool_name",
        "reasoning",
        "reasoning_content",
        "reasoning_details",
    ):
        value = _jsonish_value(message.get(key))
        if value not in (None, "", [], {}):
            out[key] = value
    return out


def _append_raw_message(store: ContextDAGStore, session_id: str, message: Dict[str, Any]) -> int:
    """Append one transcript message to SQLite preserving known raw fields."""

    role = str(message.get("role") or "unknown")
    return store.db.append_message(
        session_id,
        role,
        message.get("content"),
        tool_name=message.get("tool_name") or message.get("name"),
        tool_calls=message.get("tool_calls"),
        tool_call_id=message.get("tool_call_id"),
        token_count=message.get("token_count"),
        finish_reason=message.get("finish_reason"),
        reasoning=message.get("reasoning") if role == "assistant" else None,
        reasoning_content=message.get("reasoning_content") if role == "assistant" else None,
        reasoning_details=message.get("reasoning_details") if role == "assistant" else None,
        codex_reasoning_items=message.get("codex_reasoning_items") if role == "assistant" else None,
        codex_message_items=message.get("codex_message_items") if role == "assistant" else None,
    )


def _warning(code: str, message: str, **extra: Any) -> Dict[str, Any]:
    payload = {"code": code, "message": message, "severity": extra.pop("severity", "warning")}
    payload.update(extra)
    return payload


def reconcile_full_transcript(
    store: ContextDAGStore,
    session_id: str,
    transcript_messages: Sequence[Dict[str, Any]],
    *,
    source: str = "transcript",
) -> TranscriptReconciliationResult:
    """Mirror a full raw transcript into DAG/SQLite state idempotently.

    Duplicate detection is occurrence-based and order-aware: each existing
    SQLite row can match at most one transcript occurrence, so repeated
    transcript messages are preserved instead of collapsed by hash. Existing
    rows are never edited or deleted.  If a previous checkpoint anchor cannot be
    found after a rewrite/drift event, the function returns a warning and leaves
    the checkpoint unchanged.
    """

    result = TranscriptReconciliationResult(session_id=session_id, scanned=len(transcript_messages))
    checkpoint = store.read_checkpoint(session_id)

    raw_rows = [dict(row) for row in store.db.get_messages(session_id)]
    db_hash_to_ids: Dict[str, Deque[int]] = {}
    db_hashes_by_ordinal: List[str] = []
    for row in raw_rows:
        digest = store.deterministic_message_hash(_message_for_hash(row))
        db_hashes_by_ordinal.append(digest)
        db_hash_to_ids.setdefault(digest, deque()).append(int(row["id"]))

    transcript_hashes = [store.deterministic_message_hash(_message_for_hash(m)) for m in transcript_messages]

    # Existing checkpoints are only advanced if their prior anchor is still
    # observable.  This catches destructive rewrite/retry/undo drift without
    # pretending a new tail is a continuation of old DAG state.
    if checkpoint and checkpoint.anchor_hash:
        if checkpoint.anchor_hash not in db_hash_to_ids and checkpoint.anchor_hash not in transcript_hashes:
            result.warnings.append(
                _warning(
                    "anchor_missing",
                    "Previous DAG checkpoint anchor is missing; leaving checkpoint unchanged.",
                    previous_anchor_hash=checkpoint.anchor_hash,
                    severity="error",
                )
            )
            return result

    latest_id = raw_rows[-1]["id"] if raw_rows else None

    for ordinal, (message, digest) in enumerate(zip(transcript_messages, transcript_hashes)):
        available_db_ids = db_hash_to_ids.get(digest)
        if available_db_ids:
            result.duplicates_skipped += 1
            result.matched += 1
            matched_id = available_db_ids.popleft()
            latest_id = matched_id
            result.covered_ordinals.append(ordinal)
            if ordinal < len(db_hashes_by_ordinal) and db_hashes_by_ordinal[ordinal] != digest:
                result.warnings.append(
                    _warning(
                        "out_of_order_message",
                        "Transcript message exists in SQLite at a different ordinal; no rewrite performed.",
                        transcript_ordinal=ordinal,
                        message_hash=digest,
                        db_message_ids=[matched_id, *list(available_db_ids)],
                    )
                )
            continue

        if ordinal < len(db_hashes_by_ordinal) and db_hashes_by_ordinal[ordinal] != digest:
            result.warnings.append(
                _warning(
                    "content_drift",
                    "Transcript message differs from SQLite at the same ordinal; mirroring additively.",
                    transcript_ordinal=ordinal,
                    message_hash=digest,
                    existing_hash=db_hashes_by_ordinal[ordinal],
                )
            )

        latest_id = _append_raw_message(store, session_id, message)
        db_hashes_by_ordinal.append(digest)
        result.inserted += 1
        result.covered_ordinals.append(ordinal)

    if latest_id is None:
        return result

    anchor_hash = transcript_hashes[-1] if transcript_hashes else None
    metadata = {
        "source": source,
        "transcript_message_count": len(transcript_messages),
        "unique_transcript_hash_count": len(set(transcript_hashes)),
        "inserted": result.inserted,
        "duplicates_skipped": result.duplicates_skipped,
        "matched": result.matched,
        "warnings": result.warnings,
    }
    try:
        store.write_checkpoint(
            session_id=session_id,
            last_ingested_message_id=int(latest_id),
            last_anchor_message_id=int(latest_id),
            anchor_hash=anchor_hash,
            metadata=metadata,
        )
    except Exception as exc:
        result.warnings.append(
            _warning(
                "checkpoint_write_failed",
                "DAG transcript checkpoint write failed after raw transcript reconciliation; raw rows were left intact.",
                error=str(exc),
                severity="error",
            )
        )
        result.last_ingested_message_id = int(latest_id)
        result.anchor_hash = anchor_hash
        return result
    result.checkpoint_advanced = True
    result.last_ingested_message_id = int(latest_id)
    result.anchor_hash = anchor_hash
    return result
