"""Session resources with an explicit project binding (``sessions.project_id``) for the API server.

``POST /api/sessions`` accepts ``project_id`` (or a ``project`` slug): the server resolves it in the
profile's projects.db and writes ``project_id`` + the project folder as ``cwd`` (with its git
branch / repo root) in the same single write as the row itself, so a client never holds a
homeless session. Every session payload carries ``project: {id, slug, name, icon, color, folder,
board_slug}`` (``null`` when unbound), and ``/v1/runs`` runs a bound session's turn in its folder.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Optional

from aiohttp import web

logger = logging.getLogger(__name__)

# Session columns that cross the client API (full prompts / model_config never do).
_SAFE_SESSION_KEYS = (
    "id", "source", "user_id", "model", "title", "started_at", "ended_at", "end_reason",
    "message_count", "tool_call_count", "input_tokens", "output_tokens",
    "cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "estimated_cost_usd",
    "actual_cost_usd", "api_call_count", "parent_session_id", "last_active", "preview",
    "_lineage_root_id", "pinned", "archived", "hidden", "project_id")


def session_response(session: dict[str, Any], projects: Optional[dict[str, dict]] = None) -> dict[str, Any]:
    """Return a stable, client-safe session representation; ``projects`` maps a bound
    ``project_id`` to its ``project`` object (``project`` is ``null`` for unbound rows)."""
    payload = {key: session.get(key) for key in _SAFE_SESSION_KEYS if key in session}
    # SQLite stores the flags as 0/1.
    payload.update(
        {f: bool(payload[f]) for f in ("pinned", "archived", "hidden") if f in payload})
    # Full system prompts / model_config never cross the client API; only their presence.
    payload["has_system_prompt"] = bool(session.get("system_prompt"))
    payload["has_model_config"] = bool(session.get("model_config"))
    raw_model_config = session.get("model_config")
    try:
        model_config = (
            json.loads(raw_model_config)
            if isinstance(raw_model_config, str)
            else raw_model_config
        )
    except (TypeError, json.JSONDecodeError):
        model_config = None
    # Exact-id consumers may inspect/resume delegate children even though
    # list endpoints intentionally omit them. Project only the provenance
    # bit the client needs so it cannot accidentally promote such a row
    # into an ordinary session list; never expose the model snapshot.
    payload["is_internal_child"] = bool(
        isinstance(model_config, dict)
        and model_config.get("_delegate_from") is not None
    )
    payload["project"] = (projects or {}).get(session.get("project_id") or "")
    return payload


def _load_project_refs(project_ids) -> dict[str, dict]:
    from hermes_cli.projects_db import session_project_refs
    try:
        return session_project_refs(project_ids)
    except Exception:  # health: allow BLE001 -- a broken projects.db must not fail session reads
        logger.warning("session project lookup failed; serving sessions without project", exc_info=True)
        return {}


async def session_payloads(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``session_response`` for each row with its ``project`` object resolved in ONE projects.db
    read, off the event loop (and only when some row is bound)."""
    ids = {s.get("project_id") for s in sessions if s.get("project_id")}
    projects = await asyncio.to_thread(_load_project_refs, ids) if ids else {}
    return [session_response(s, projects) for s in sessions]


async def session_payload(session: dict[str, Any]) -> dict[str, Any]:
    return (await session_payloads([session]))[0]


def _resolve_project(refs: list[str]):
    """The one ACTIVE project every given reference (id or slug) names, else ``None``."""
    from hermes_cli import projects_db as pdb
    with pdb.connect_closing() as conn:
        found = [pdb.resolve_session_project(conn, ref) for ref in refs]
    if any(p is None for p in found) or len({p.id for p in found if p is not None}) != 1:
        return None
    return found[0]


async def resolve_body_project(body: dict[str, Any], *, _api_server):
    """``(project, error_response)`` for a request body's ``project_id`` / ``project`` (slug).
    ``(None, None)`` when the body names no project; unknown, archived or contradicting
    references answer 400 ``invalid_project``."""
    refs = [body[key] for key in ("project_id", "project") if body.get(key) is not None]
    if not refs:
        return None, None
    invalid = _api_server._error_response
    if not all(isinstance(ref, str) and ref.strip() for ref in refs):
        return None, invalid("project_id / project must be a non-empty string", 400, code="invalid_project")
    try:
        project = await asyncio.to_thread(_resolve_project, [ref.strip() for ref in refs])
    except Exception:  # health: allow BLE001 -- answered as a 503, logged with traceback
        logger.exception("project resolution failed")
        return None, invalid("Project store unavailable", 503, code="project_store_unavailable")
    if project is None:
        return None, invalid(f"Unknown or archived project: {', '.join(refs)}", 400, code="invalid_project")
    return project, None


def project_workspace(project) -> dict[str, Optional[str]]:
    """``{project_id, cwd, git_branch, git_repo_root}`` a session bound to ``project`` starts with
    (blocking: git probes; call off the loop). A folder missing on this host keeps its path but
    has no git identity."""
    from hermes_cli.projects_db import project_folder
    from tui_gateway import git_probe
    folder = project_folder(project) or ""
    present = bool(folder) and os.path.isdir(folder)
    return {"project_id": project.id, "cwd": folder or None,
            "git_branch": (git_probe.branch(folder) or None) if present else None,
            "git_repo_root": (git_probe.common_repo_root(folder) or None) if present else None}


def _create_model_config(runtime_request: dict[str, Any]) -> Optional[dict[str, Any]]:
    requested = runtime_request.get("requested") or {}
    if not (requested.get("model") or requested.get("provider")):
        return None
    return {"browser_model_lock": {
        "provider": requested.get("provider") or "", "model": requested.get("model") or "",
        "model_options": runtime_request.get("model_options") or {},
        "route_source": runtime_request.get("route_source") or "",
        "confirmed": bool(runtime_request.get("require_model_lock")),
        "updated_at": time.time()}}


def _validated_session_id(self, body: dict[str, Any], _api_server) -> tuple[str, Optional[web.Response]]:
    raw_id = body.get("id") or body.get("session_id")
    session_id = str(raw_id).strip() if raw_id else f"api_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    from gateway.session import _is_path_unsafe
    if not session_id or re.search(r'[\r\n\x00]', session_id) or _is_path_unsafe(session_id):
        return session_id, _api_server._error_response("Invalid session ID", 400, code="invalid_session_id")
    if len(session_id) > self._MAX_SESSION_HEADER_LEN:
        return session_id, _api_server._error_response("Session ID too long", 400, code="invalid_session_id")
    return session_id, None


async def handle_create_session(self, request: web.Request, *, _api_server) -> web.Response:
    """POST /api/sessions -- create an empty Hermes session row. Existence check, insert, project
    binding and title handling run as ONE off-loop write so concurrent same-id creates can't both 201."""
    _error_response = _api_server._error_response
    body, err = await self._read_json_body(request)
    if err:
        return err
    from gateway.platforms.api_server_bot_chat import create_peer_bot_chat
    if (bot_chat := await create_peer_bot_chat(self, body)) is not None:
        return bot_chat  # the owner minted the canonical Bot Chat (#139370)
    db = await self._ensure_session_db_async()
    if db is None:
        return self._session_db_unavailable()
    session_id, err = _validated_session_id(self, body, _api_server)
    if err:
        return err
    system_prompt = body.get("system_prompt")
    if system_prompt is not None and not isinstance(system_prompt, str):
        return _error_response("system_prompt must be a string", 400, code="invalid_system_prompt")
    project, err = await resolve_body_project(body, _api_server=_api_server)
    if err:
        return err
    workspace = await asyncio.to_thread(project_workspace, project) if project is not None else {}
    source = self._normalize_session_source(body.get("source") or "api_server")
    runtime_request = self._session_runtime_request_from_body(body)
    lock_error = self._runtime_lock_error(runtime_request)
    if lock_error is not None:
        return lock_error
    # The normalized requested["model"] (prefix split, virtual alias nulled) — the raw body
    # would persist "hermes-agent" and later send it to the provider literally.
    model_name = self._clean_runtime_id((runtime_request.get("requested") or {}).get("model")) or None
    model_config = _create_model_config(runtime_request)
    title = body.get("title")

    def _atomic(conn):
        from hermes_state_mutation_retirement import create_conflict
        if conflict := create_conflict(conn, session_id):  # one BEGIN IMMEDIATE write: a racing create sees the row
            return None, conflict
        conn.execute(
            """INSERT INTO sessions (
               id, source, model, model_config, system_prompt, started_at,
               project_id, cwd, git_branch, git_repo_root
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, source, model_name, json.dumps(model_config) if model_config else None,
             system_prompt, time.time(), workspace.get("project_id"), workspace.get("cwd"),
             workspace.get("git_branch"), workspace.get("git_repo_root")))
        if title is not None:
            clean_title = db.sanitize_title(str(title))
            if clean_title:
                try:
                    db._resolve_title_conflict(conn, session_id, clean_title)
                except ValueError as exc:  # the DB's uniqueness rule; undo the INSERT
                    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                    return None, f"title:{exc}"
            conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (clean_title, session_id))
        session_row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return (dict(session_row) if session_row else {
            "id": session_id, "source": source, "model": model_name, "title": title, **workspace}), None
    session, err = await asyncio.to_thread(db._execute_write, _atomic)
    if isinstance(err, tuple):  # a live row, or a deleted id's fence: refused now, not on every turn
        return _error_response(f"{err[1]}: {session_id}", 409, code=err[0])
    if err and err.startswith("title:"):
        return _error_response(err[len("title:"):], 400, code="invalid_title")
    return web.json_response(
        {"object": "hermes.session", "session": await session_payload(session)}, status=201)


async def list_project_filter(request: web.Request) -> Optional[str]:
    """``GET /api/sessions?project_id=`` (an id or a slug) -> the project id to filter on. A
    reference that names no project is kept verbatim (it simply matches no session)."""
    ref = (request.query.get("project_id") or "").strip()
    if not ref:
        return None

    def _lookup() -> str:
        from hermes_cli import projects_db as pdb
        with pdb.connect_closing() as conn:
            project = pdb.get_project(conn, ref)
        return project.id if project is not None else ref
    try:
        return await asyncio.to_thread(_lookup)
    except Exception:  # health: allow BLE001 -- a raw id still filters correctly without the store
        logger.warning("project filter lookup failed; filtering on the raw id", exc_info=True)
        return ref


def stored_session_cwd(db, session_id: Optional[str]) -> str:
    """The folder a turn for ``session_id`` runs in (blocking DB read): the row's stored cwd,
    which a project binding sets to the project folder; ``""`` for a missing/cwd-less row."""
    from hermes_state_session_projects import session_run_cwd
    if db is None or not session_id:
        return ""
    try:
        return session_run_cwd(db.get_session(str(session_id)))
    except Exception:  # health: allow BLE001 -- a turn without a pinned cwd still runs
        logger.debug("session cwd lookup failed for %s", session_id, exc_info=True)
        return ""


async def session_cwd_for(adapter, session_id: Optional[str]) -> str:
    """:func:`stored_session_cwd` off the loop; ``""`` without a stored session (a fresh run id)."""
    if not session_id:
        return ""
    return await asyncio.to_thread(stored_session_cwd, await adapter._ensure_session_db_async(), session_id)


def pin_run_workspace(task_id: str, cwd: str) -> None:
    """Start this run's terminal/file tools in ``cwd`` (the session's folder). Registered only when
    the folder exists and differs from the task's current override, so a ``cd`` the agent made in
    an earlier turn of the same folder survives, and a workspace move re-anchors the next turn."""
    if not cwd or not task_id or not os.path.isdir(cwd):
        return
    from tools.terminal_tool import register_task_env_overrides, resolve_task_overrides
    current = resolve_task_overrides(task_id) or {}
    if current.get("cwd") != cwd:
        register_task_env_overrides(task_id, {**current, "cwd": cwd})
