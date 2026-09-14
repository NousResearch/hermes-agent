"""Context Compiler for Hermes JIT Context Engine."""

from __future__ import annotations

import sqlite3
import time
from typing import Any, Dict, List, Optional

from .l0.overlay import ensure_session, get_active_overlays
from .l1.project_cache import get_project_summary
from .l1.scope import resolve_scope
from .renderer import render_capsule


class CapsuleResult(str):
    meta: Dict[str, Any]

    def __new__(cls, text: str, meta: Optional[Dict[str, Any]] = None):
        obj = super().__new__(cls, text)
        obj.meta = meta or {}
        return obj


def compile_context(
    conn: sqlite3.Connection,
    session_id: str,
    user_message: str,
    transcript_messages: Optional[List[Dict[str, Any]]] = None,
    active_scope: Optional[str] = None,
    default_scope: str = "general",
    session_cwd: Optional[str] = None,
    **kwargs: Any,
) -> CapsuleResult:
    """Compile L0 events, active overlays, and L1 scope into a lean <ONA_CONTEXT> capsule."""
    sess = ensure_session(conn, session_id, default_scope=default_scope)
    curr_scope = active_scope or sess.get("active_scope") or default_scope
    epoch = int(sess.get("scope_epoch", 1) or 1)

    resolved_scope, _, _, _ = resolve_scope(user_message, curr_scope)
    if resolved_scope != curr_scope:
        epoch += 1
        conn.execute(
            "UPDATE sessions SET active_scope = ?, scope_epoch = ?, updated_at = ? WHERE session_id = ?",
            (resolved_scope, epoch, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), session_id),
        )
        conn.commit()
        curr_scope = resolved_scope

    overlays = get_active_overlays(conn, session_id, limit=8)
    current_statements: List[str] = []
    if user_message and user_message.strip():
        current_statements.append(f"Goal: {user_message.strip()}")

    for ovl in overlays:
        val = ovl.get("value", "")
        if val and val not in current_statements:
            current_statements.append(val)

    # Extract project context from Obsidian SSOT or session_cwd
    project_summary = get_project_summary(resolved_scope, session_cwd=session_cwd)
    if not project_summary and session_cwd:
        project_summary = f"Working directory: {session_cwd}"

    rendered = render_capsule(
        scope=curr_scope,
        epoch=epoch,
        current_statements=current_statements,
        project_summary=project_summary,
    )
    return CapsuleResult(rendered, meta={"scope": curr_scope, "epoch": epoch})
