"""Live LCM runtime snapshot helpers for the Context Cockpit.

The browser cockpit runs in a sidecar process, so it cannot access the live
in-memory LCM engine directly. The Hermes runtime writes a small sanitized JSON
snapshot here, and the cockpit reads it on refresh.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

SNAPSHOT_NAME = "context-cockpit-live-lcm.json"


def snapshot_path(profile_dir: Path) -> Path:
    return profile_dir / SNAPSHOT_NAME


def read_live_lcm_snapshot(profile_dir: Path) -> Dict[str, Any]:
    path = snapshot_path(profile_dir)
    if not path.exists():
        return {}
    try:
        with path.open() as fh:
            payload = json.load(fh)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def snapshot_is_bound(payload: Dict[str, Any]) -> bool:
    """True when the snapshot carries a session/conversation identity."""
    return bool(
        str(payload.get("session_id") or "").strip()
        or str(payload.get("conversation_id") or "").strip()
    )


def write_live_lcm_snapshot_for_engine(
    engine: Any,
    profile_dir: Path,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    """Write a minimal sanitized snapshot from the live Hermes runtime.

    This is read-only with respect to Hermes state: it only serializes runtime
    status into a sidecar JSON file for the cockpit.

    Unbound engines (empty session/conversation ids, typically the LCM plugin
    singleton) must not overwrite a previously bound per-turn snapshot. That
    overwrite is what blanked Live Status / Fresh Tail in the cockpit after
    ``/visor``.
    """
    status = engine.get_status() or {}
    rotate_preview = engine.rotate_active_session(apply=False) or {}
    now = time.time()

    payload = {
        "collected_at": now,
        "session_id": str(getattr(engine, "current_session_id", "") or ""),
        "conversation_id": str(getattr(engine, "current_conversation_id", "") or ""),
        "last_compression_status": str(status.get("last_compression_status", "") or ""),
        "last_compression_noop_reason": str(status.get("last_compression_noop_reason", "") or ""),
        "compression_count": int(status.get("compression_count", 0) or 0),
        "threshold_tokens": int(status.get("threshold_tokens", 0) or 0),
        "last_prompt_tokens": int(status.get("last_prompt_tokens", 0) or 0),
        "context_length": int(status.get("context_length", 0) or 0),
        "fresh_tail_count": int(getattr(getattr(engine, "_config", None), "fresh_tail_count", 0) or 0),
        "leaf_chunk_tokens": int(getattr(getattr(engine, "_config", None), "leaf_chunk_tokens", 0) or 0),
        "rotate_preview": {
            "ok": bool(rotate_preview.get("ok")),
            "noop": bool(rotate_preview.get("noop")),
            "reason": str(rotate_preview.get("reason", "") or ""),
            "total_message_count": int(rotate_preview.get("total_message_count", 0) or 0),
            "fresh_tail_count": int(rotate_preview.get("fresh_tail_count", 0) or 0),
            "pre_tail_message_count": int(rotate_preview.get("pre_tail_message_count", 0) or 0),
            "current_frontier_store_id": int(rotate_preview.get("current_frontier_store_id", 0) or 0),
            "new_frontier_store_id": int(rotate_preview.get("new_frontier_store_id", 0) or 0),
        },
    }

    if not force and not snapshot_is_bound(payload):
        existing = read_live_lcm_snapshot(profile_dir)
        if existing and snapshot_is_bound(existing):
            # Preserve the last bound per-turn proof; return it unchanged.
            return existing

    path = snapshot_path(profile_dir)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)
    return payload


def snapshot_matches_conversation(
    snapshot: Dict[str, Any],
    conversation_id: Optional[str],
) -> bool:
    """Match live snapshot to the cockpit's current session/conversation id."""
    target = str(conversation_id or "").strip()
    if not target or not snapshot:
        return False
    snap_conv = str(snapshot.get("conversation_id") or "").strip()
    snap_sess = str(snapshot.get("session_id") or "").strip()
    return target in {snap_conv, snap_sess} and bool(snap_conv or snap_sess)
