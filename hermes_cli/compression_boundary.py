"""Durable handoffs for terminal context-compression boundaries.

The conversation transcript is intentionally not copied across a terminal
compression boundary.  This module stores a small, redacted checkpoint in
``state_meta`` and builds the prompt that a fresh session can use to continue
from the workspace instead of replaying the poisoned transcript.
"""

from __future__ import annotations

import json
import time
from typing import Any

from agent.redact import redact_sensitive_text


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_META_PREFIX = "compression_boundary:"
_MAX_PROMPT_EXCERPT_CHARS = 1_600
_MAX_QUEUED_PROMPT_CHARS = 1_200
_MAX_GOAL_CHARS = 800
_MAX_ERROR_CHARS = 240
_MAX_RESUME_PROMPT_CHARS = 3_200


def checkpoint_meta_key(session_id: str) -> str:
    """Return the namespaced state-meta key for a session boundary."""
    return f"{CHECKPOINT_META_PREFIX}{str(session_id or '').strip()}"


def _redacted_excerpt(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    text = redact_sensitive_text(value, force=True).strip()
    if len(text) <= limit:
        return text
    head = max(1, int(limit * 0.68))
    tail = max(1, limit - head - 32)
    return f"{text[:head].rstrip()}\n[… excerpt shortened …]\n{text[-tail:].lstrip()}"


def build_compression_checkpoint(
    session_id: str,
    *,
    prompt: Any = None,
    queued_prompt: Any = None,
    goal: Any = None,
    cwd: Any = None,
    model: Any = None,
    error: Any = None,
) -> dict[str, Any]:
    """Build a bounded, redacted checkpoint for a failed conversation turn.

    The checkpoint deliberately carries task anchors, not transcript history.
    The old session remains available for inspection, while the fresh-session
    prompt directs the next model turn to inspect current durable state.
    """
    old_session_id = str(session_id or "").strip()
    prompt_excerpt = _redacted_excerpt(prompt, _MAX_PROMPT_EXCERPT_CHARS)
    queued_excerpt = _redacted_excerpt(queued_prompt, _MAX_QUEUED_PROMPT_CHARS)
    goal_excerpt = _redacted_excerpt(goal, _MAX_GOAL_CHARS)
    error_excerpt = _redacted_excerpt(error, _MAX_ERROR_CHARS)
    workspace = _redacted_excerpt(cwd, 500)
    model_name = _redacted_excerpt(model, 160)

    lines = [
        "Continue the interrupted task in this fresh Hermes session.",
        "The previous session hit terminal context-compression exhaustion and was sealed.",
        "Do not paste or replay the previous transcript.",
        "Start with a bounded read-only check of the current workspace and durable task files, then continue from the last verified step.",
    ]
    if goal_excerpt:
        lines.append(f"A standing goal was active: {goal_excerpt}")
    if prompt_excerpt:
        lines.append(
            "The last user request is included only as a bounded task anchor; do not treat it as a transcript to replay:\n"
            f"{prompt_excerpt}"
        )
    if queued_excerpt:
        lines.append(
            "A newer user message arrived during the failed turn and was preserved as an editable handoff draft:\n"
            f"{queued_excerpt}"
        )
    if workspace:
        lines.append(f"Workspace recorded at the boundary: {workspace}")
    lines.append(f"Sealed session id: {old_session_id}")
    resume_prompt = "\n\n".join(lines).strip()
    resume_prompt = _redacted_excerpt(resume_prompt, _MAX_RESUME_PROMPT_CHARS)

    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": "compression_boundary",
        "status": "pending",
        "reason": "compression_exhausted",
        "old_session_id": old_session_id,
        "created_at": time.time(),
        "prompt_excerpt": prompt_excerpt,
        "queued_prompt_excerpt": queued_excerpt,
        "goal_excerpt": goal_excerpt,
        "cwd": workspace,
        "model": model_name,
        "error_excerpt": error_excerpt,
        "resume_prompt": resume_prompt,
    }


def persist_compression_checkpoint(db: Any, checkpoint: dict[str, Any]) -> bool:
    """Persist one checkpoint atomically through the SessionDB meta API."""
    session_id = str(checkpoint.get("old_session_id") or "").strip()
    if not session_id or db is None or not hasattr(db, "set_meta"):
        return False
    db.set_meta(
        checkpoint_meta_key(session_id),
        json.dumps(checkpoint, ensure_ascii=False, separators=(",", ":")),
    )
    return True


def load_compression_checkpoint(db: Any, session_id: str) -> dict[str, Any] | None:
    """Load and validate one checkpoint, returning ``None`` on bad state."""
    if db is None or not hasattr(db, "get_meta"):
        return None
    raw = db.get_meta(checkpoint_meta_key(session_id))
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    if value.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        return None
    if value.get("kind") != "compression_boundary":
        return None
    if value.get("status") != "pending":
        return None
    if value.get("reason") != "compression_exhausted":
        return None
    if str(value.get("old_session_id") or "") != str(session_id or ""):
        return None
    if not isinstance(value.get("resume_prompt"), str) or not value["resume_prompt"].strip():
        return None
    return value


def compression_boundary_resume_payload(db: Any, session_id: str) -> dict[str, Any] | None:
    """Return a transcript-free resume response for a sealed session."""
    checkpoint = load_compression_checkpoint(db, session_id)
    if checkpoint is None:
        return None

    boundary = {
        "old_session_id": str(checkpoint["old_session_id"]),
        "reason": "compression_exhausted",
        "resume_prompt": str(checkpoint["resume_prompt"]).strip(),
    }
    return {
        "compression_boundary": boundary,
        "session_id": str(session_id),
        "resumed": str(session_id),
        "message_count": 0,
        "messages": [],
        "messages_omitted": True,
        "info": None,
        "inflight": None,
        "running": False,
        "session_key": str(session_id),
        "started_at": checkpoint.get("created_at"),
        "status": "idle",
    }
