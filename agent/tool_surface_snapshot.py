"""Cold-resume tool-surface snapshot — prompt-cache prefix stability (#103579 族).

Background (observed 2026-09-07 22:11:57, session 20260907_220529_5021de):
  ``agent.tools`` (the API-level ``tools[]`` array, located between the system
  prompt and the messages in the cache prefix) is rebuilt on every agent
  initialization by :func:`model_tools.get_tool_definitions`, whose output
  depends on the *registry state at build time*: the ``tool_search`` bridge
  description embeds the runtime deferrable set (``Search {N} additional
  tools`` + a full catalog listing), and the deferrable set follows plugin
  registration order / check_fn pass-through / MCP connection state. After a
  dashboard/gateway restart the cold-resumed agent therefore produces a
  different ``tools[]`` byte sequence than the previous process, and the
  prompt-cache prefix breaks right after the system prompt (observed hit
  residue == system-prompt tokens only, e.g. 14,592 for this host).

Fix semantics (content-addressed, mirroring the ``system_prompts`` table):
  * After a session's first build, store the final ``agent.tools`` bytes in
    the ``agent_tools`` table (hash = content hash of the tool surface);
    ``sessions`` records the ``tools_hash`` reference plus a
    ``tools_fingerprint`` derived from the **user-visible configuration only**
    (model + enabled/disabled toolsets).
  * Cold resume: fingerprint match → reuse the snapshot bytes unconditionally,
    ignoring the current registry (plugin registration order / check_fn
    pass-through / MCP connection state must not change the bytes);
    fingerprint mismatch or no snapshot → build live and store a new snapshot.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional


def tool_surface_fingerprint(
    model: Optional[str],
    enabled_toolsets: Optional[List[str]],
    disabled_toolsets: Optional[List[str]],
) -> str:
    """Fingerprint = f(user-visible configuration), **excluding** runtime registry state.

    This is the crux of the fix: only a model/toolset change invalidates the
    snapshot; plugin registration order, check_fn pass-through, and MCP
    connection state changes must never invalidate it.
    """
    payload = {
        "model": model or "",
        "enabled": sorted(enabled_toolsets or []),
        "disabled": sorted(disabled_toolsets or []),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _tools_hash(tools: List[Dict[str, Any]]) -> str:
    """Content hash of a tool-surface list (``agent_tools`` primary key)."""
    blob = json.dumps(tools, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_snapshot(
    session_db, session_id: str, fingerprint: str
) -> Optional[List[Dict[str, Any]]]:
    """Return the stored tool-surface bytes if the snapshot matches, else None.

    Match = sessions.tools_fingerprint == *fingerprint* (config unchanged) AND
    the referenced agent_tools row exists.
    """
    if session_db is None or not session_id:
        return None
    try:
        session = session_db.get_session(session_id)
        if not session:
            return None
        if session.get("tools_fingerprint") != fingerprint:
            return None
        tools_hash = session.get("tools_hash")
        if not tools_hash:
            return None
        return session_db.load_agent_tools(tools_hash)
    except Exception as exc:  # pragma: no cover — snapshot failure must not block agent build
        import logging

        logging.getLogger(__name__).warning(
            "Tool-surface snapshot load failed (%s); falling back to live build", exc
        )
        return None


def store_snapshot(
    session_db, session_id: str, fingerprint: str, tools: List[Dict[str, Any]]
) -> str:
    """Store the snapshot and link the session; returns the content hash."""
    if session_db is None or not session_id:
        return _tools_hash(tools)
    try:
        return session_db.store_agent_tools(session_id, fingerprint, tools)
    except Exception as exc:  # pragma: no cover — snapshot failure must not block agent build
        import logging

        logging.getLogger(__name__).info(
            "Tool-surface snapshot store skipped: %s", exc
        )
        return _tools_hash(tools)


def resolve_tool_surface(
    session_db,
    session_id: str,
    model: Optional[str],
    enabled_toolsets: Optional[List[str]],
    disabled_toolsets: Optional[List[str]],
    built_tools: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return the agent's final tool surface: snapshot on match, else store + use live build."""
    fp = tool_surface_fingerprint(model, enabled_toolsets, disabled_toolsets)
    cached = load_snapshot(session_db, session_id, fp)
    if cached is not None:
        return cached
    store_snapshot(session_db, session_id, fp, built_tools)
    return built_tools
