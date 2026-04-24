"""Hermes handoff record persistence and validation.

Manages small handoff/run records for Hermes <-> Code Crab coordination.
Records are stored as individual JSON files in ~/.hermes/handoffs/.

Each record tracks: handoff_id, origin metadata, requester, request summary,
acceptance criteria, queued next action, and status lifecycle.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

HANDOFFS_DIR = "handoffs"

# Valid status values and allowed transitions
VALID_STATUSES = {"requested", "in_progress", "done", "blocked", "failed"}
VALID_TRANSITIONS: Dict[str, set] = {
    "requested": {"in_progress", "blocked", "failed"},
    "in_progress": {"done", "blocked", "failed"},
    # Terminal states allow re-entry for idempotent callbacks
    "done": {"done"},
    "blocked": {"blocked", "in_progress"},
    "failed": {"failed", "in_progress"},
}


def _handoffs_dir() -> Path:
    """Return the handoffs storage directory, creating it if needed."""
    d = get_hermes_home() / HANDOFFS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _record_path(handoff_id: str) -> Path:
    """Return the file path for a handoff record."""
    # Sanitize ID to prevent path traversal
    safe_id = handoff_id.replace("/", "_").replace("..", "_")
    return _handoffs_dir() / f"{safe_id}.json"


def create_handoff(
    handoff_id: str,
    origin_platform: str = "",
    origin_channel_id: str = "",
    origin_thread_id: str = "",
    origin_message_id: str = "",
    requestor: str = "",
    request_summary: str = "",
    acceptance_criteria: str = "",
    done_when: str = "",
    next_action: str = "",
    next_action_safe: bool = False,
    callback_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create and persist a new handoff record.

    Raises ValueError if a record with this handoff_id already exists.
    """
    path = _record_path(handoff_id)
    if path.exists():
        raise ValueError(f"Handoff record already exists: {handoff_id}")

    record = {
        "handoff_id": handoff_id,
        "status": "requested",
        "origin": {
            "platform": origin_platform,
            "channel_id": origin_channel_id,
            "thread_id": origin_thread_id,
            "message_id": origin_message_id,
        },
        "requestor": requestor,
        "request_summary": request_summary,
        "acceptance_criteria": acceptance_criteria,
        "done_when": done_when,
        "next_action": next_action,
        "next_action_safe": next_action_safe,
        "callback_contract": callback_contract or {},
        "callback_received": None,
        "created_at": time.time(),
        "updated_at": time.time(),
        "return_message_id": None,
    }

    _write_record(path, record)
    logger.info("[handoffs] Created handoff record: %s", handoff_id)
    return record


def get_handoff(handoff_id: str) -> Optional[Dict[str, Any]]:
    """Load a handoff record by ID. Returns None if not found."""
    path = _record_path(handoff_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.error("[handoffs] Failed to read %s: %s", handoff_id, e)
        return None


def validate_status_transition(current: str, proposed: str) -> bool:
    """Check if a status transition is valid."""
    if current not in VALID_TRANSITIONS:
        return False
    return proposed in VALID_TRANSITIONS[current]


def update_handoff_status(
    handoff_id: str,
    new_status: str,
    callback_payload: Optional[Dict[str, Any]] = None,
    return_message_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Update a handoff record's status and optionally store callback data.

    Raises ValueError if handoff_id is unknown or transition is invalid.
    Idempotent: re-applying the same terminal status is a no-op.
    """
    record = get_handoff(handoff_id)
    if record is None:
        raise ValueError(f"Unknown handoff_id: {handoff_id}")

    current = record["status"]

    if not validate_status_transition(current, new_status):
        raise ValueError(
            f"Invalid status transition: {current} -> {new_status} "
            f"(allowed: {VALID_TRANSITIONS.get(current, set())})"
        )

    # Idempotent: same terminal status with existing callback = no-op
    if current == new_status and record.get("callback_received") is not None:
        logger.info(
            "[handoffs] Idempotent callback for %s (already %s)",
            handoff_id,
            current,
        )
        return record

    record["status"] = new_status
    record["updated_at"] = time.time()
    if callback_payload is not None:
        record["callback_received"] = callback_payload
    if return_message_id is not None:
        record["return_message_id"] = return_message_id

    _write_record(_record_path(handoff_id), record)
    logger.info(
        "[handoffs] Updated %s: %s -> %s", handoff_id, current, new_status
    )
    return record


def list_handoffs(status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all handoff records, optionally filtered by status."""
    records = []
    handoffs_dir = _handoffs_dir()
    for path in sorted(handoffs_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if status_filter is None or record.get("status") == status_filter:
                records.append(record)
        except (json.JSONDecodeError, OSError):
            continue
    return records


def render_origin_message(record: Dict[str, Any]) -> str:
    """Render a concise return message for the origin thread/channel.

    Includes status, summary, key artifacts, and verification results
    so Kevin (or whoever is watching the thread) can trust the outcome.
    """
    cb = record.get("callback_received") or {}
    status = record.get("status", "unknown")
    summary = cb.get("summary", record.get("request_summary", ""))

    parts = [f"**Handoff {record['handoff_id']}** — status: **{status}**"]

    if summary:
        parts.append(f"\n{summary}")

    # Artifacts
    artifacts = cb.get("artifacts", {})
    artifact_lines = []
    if artifacts.get("pr"):
        artifact_lines.append(f"PR: {artifacts['pr']}")
    if artifacts.get("branch"):
        artifact_lines.append(f"Branch: `{artifacts['branch']}`")
    if artifacts.get("commit"):
        artifact_lines.append(f"Commit: `{artifacts['commit'][:12]}`")
    if artifact_lines:
        parts.append("\n**Artifacts:** " + " | ".join(artifact_lines))

    # Verification
    verification = cb.get("verification", {})
    if verification.get("results"):
        parts.append(f"\n**Verification:** {verification['results']}")

    # Next action
    if cb.get("needs_kevin"):
        parts.append(
            f"\n:warning: **Needs Kevin** — {cb.get('next_recommended_action', 'review required')}"
        )
    elif cb.get("next_recommended_action"):
        parts.append(
            f"\n**Next:** {cb['next_recommended_action']}"
        )

    return "\n".join(parts)


def should_auto_resume(record: Dict[str, Any]) -> bool:
    """Determine if Hermes should auto-resume the queued next action.

    Auto-resume only when:
    - status is 'done'
    - needs_kevin is false
    - next_action is set and marked safe
    """
    cb = record.get("callback_received") or {}
    return (
        record.get("status") == "done"
        and not cb.get("needs_kevin", True)
        and bool(record.get("next_action"))
        and record.get("next_action_safe", False)
    )


def _write_record(path: Path, record: Dict[str, Any]) -> None:
    """Atomic write: temp file + rename to prevent partial reads."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(path))
