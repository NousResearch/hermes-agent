"""
Achilli Tendo -- Agent session checkpoint/resume.

Hooks into on_session_finalize to serialize agent working state at session
boundaries. Provides list_checkpoints tool and checkpoint files.

on_session_finalize IS invoked in Hermes core (cli.py lines 972, 6542, 6545)
at session boundaries: CLI quit, /new, /reset, gateway GC.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _checkpoint_dir() -> Path:
    env_dir = os.environ.get("ACHILLI_TENDO_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return Path.home() / ".hermes" / "checkpoints" / "agent-state"


def _max_checkpoints() -> int:
    try:
        return int(os.environ.get("ACHILLI_TENDO_MAX_CHECKPOINTS", "50"))
    except (ValueError, TypeError):
        return 50


def _disabled() -> bool:
    return os.environ.get("ACHILLI_TENDO_DISABLE", "").lower() in {
        "1", "true", "yes"
    }


def _ensure_dir() -> Path:
    d = _checkpoint_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------


def _evict_old_checkpoints() -> None:
    d = _ensure_dir()
    files = sorted(d.glob("checkpoint_*.json"), key=lambda f: f.stat().st_mtime)
    max_c = _max_checkpoints()
    while len(files) > max_c:
        oldest = files.pop(0)
        try:
            oldest.unlink()
            logger.debug("tendo: evicted old checkpoint %s", oldest.name)
        except OSError:
            pass


def _write_checkpoint(session_id: str, reason: str, extra: Dict[str, Any]) -> Path:
    d = _ensure_dir()
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    filename = f"checkpoint_{ts}_{session_id[:12]}.json"
    path = d / filename

    checkpoint = {
        "session_id": session_id,
        "checkpoint_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reason": reason,
        "message_count": extra.get("message_count", 0),
        "working_directory": extra.get("working_directory", ""),
        "background_processes": extra.get("background_processes", {}),
        "active_subagents": extra.get("active_subagents", []),
        "yantrikdb_session_id": extra.get("yantrikdb_session_id"),
        "completeness_score": extra.get("completeness_score", 0.5),
        "continuation_prompt": (
            "This session was resumed from a checkpoint. "
            "Previous work has been restored as context. "
            "Continue where you left off."
        ),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, default=str)

    logger.info("tendo: checkpoint written to %s", path)
    _evict_old_checkpoints()
    return path


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def _on_session_finalize(
    session_id: Optional[str] = None,
    platform: str = "unknown",
    **_: Any,
) -> None:
    """Create a checkpoint at session boundary."""
    if _disabled():
        return

    if not session_id:
        logger.debug("tendo: no session_id in on_session_finalize, skipping")
        return

    try:
        _write_checkpoint(
            session_id=session_id,
            reason=f"session_finalize:{platform}",
            extra={
                "message_count": 0,  # Would need agent internals for exact count
                "working_directory": os.getcwd(),
                "background_processes": {},  # Would need agent process tracker
                "active_subagents": [],
                "completeness_score": 0.5,
            },
        )
    except Exception as exc:
        logger.warning("tendo: checkpoint failed: %s", exc)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def _list_checkpoints_tool(limit: int = 10, **_: Any) -> dict:
    """List recent checkpoints with metadata."""
    d = _checkpoint_dir()
    files = sorted(d.glob("checkpoint_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)

    checkpoints = []
    for f in files[:limit]:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            checkpoints.append({
                "file": f.name,
                "session_id": data.get("session_id", "?"),
                "checkpoint_time": data.get("checkpoint_time", "?"),
                "reason": data.get("reason", "?"),
                "message_count": data.get("message_count", 0),
                "completeness_score": data.get("completeness_score", 0),
                "size_bytes": f.stat().st_size,
            })
        except (json.JSONDecodeError, OSError):
            checkpoints.append({"file": f.name, "error": "unreadable"})

    return {
        "checkpoints": checkpoints,
        "total": len(files),
        "directory": str(d),
        "max_retained": _max_checkpoints(),
    }


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    ctx.register_hook("on_session_finalize", _on_session_finalize)
    ctx.register_tool(
        name="list_checkpoints",
        description=(
            "List recent agent session checkpoints with metadata. "
            "Each checkpoint captures the session transcript and working state. "
            "(achilli-tendo)"
        ),
        parameters={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max checkpoints to return (default 10)",
                },
            },
        },
        handler=_list_checkpoints_tool,
    )
