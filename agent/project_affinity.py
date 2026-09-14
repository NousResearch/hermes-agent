"""Core-owned Session Project identity binding at system-prompt generation boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class ProjectAffinityCandidate:
    project_id: str
    project_root: str
    context_hash: str


def _project_context_hash(project_root: str, context_length: Optional[int] = None) -> str:
    """Hash the exact context-file bytes Core would build for this Project root."""
    from agent.prompt_builder import build_context_files_prompt

    context = build_context_files_prompt(
        cwd=project_root,
        skip_soul=True,
        context_length=context_length,
    )
    return "sha256:" + hashlib.sha256(context.encode("utf-8")).hexdigest()


def load_project_affinity_candidate(
    *, project_id: str, project_root: str, context_length: Optional[int] = None,
) -> Optional[ProjectAffinityCandidate]:
    """Load a complete Core-owned identity candidate from an explicit Project."""
    pid = (project_id or "").strip()
    root = str(Path(project_root).expanduser().resolve()) if str(project_root or "").strip() else ""
    if not pid or not root:
        return None
    return ProjectAffinityCandidate(
        project_id=pid,
        project_root=root,
        context_hash=_project_context_hash(root, context_length),
    )


def resolve_project_affinity_for_cwd(
    cwd: str, *, projects_conn, context_length: Optional[int] = None,
) -> Optional[ProjectAffinityCandidate]:
    """Resolve the innermost named Project owning cwd."""
    from hermes_cli import projects_db

    project = projects_db.project_for_path(projects_conn, cwd)
    if project is None or not project.primary_path:
        return None
    return load_project_affinity_candidate(
        project_id=project.id,
        project_root=project.primary_path,
        context_length=context_length,
    )


def _profile_home_for_agent(agent: Any) -> Optional[Path]:
    """Resolve the agent's profile authority without guessing from an arbitrary state path."""
    explicit = getattr(agent, "_profile_home", None) or getattr(agent, "profile_home", None)
    if explicit:
        return Path(explicit)
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except Exception:
        return None


def ensure_agent_project_affinity(agent: Any) -> dict[str, Any]:
    """Bind/refresh identity only at a system-prompt generation boundary.

    Existing complete identity is authoritative even when cwd drifts. A brand-new
    root session may auto-bind from cwd; parent-linked or non-empty sessions never
    infer a new owner. Context-file changes advance the generation only when the
    system prompt is already being rebuilt, preserving prompt-cache stability.
    """
    from hermes_cli.session_project_affinity import validate_project_affinity_row

    session_db = getattr(agent, "_session_db", None)
    session_id = str(getattr(agent, "session_id", "") or "")
    if session_db is None or not session_id or getattr(agent, "_persist_disabled", False):
        return {
            "status": "unavailable", "session_id": session_id,
            "project_id": "", "project_root": "", "project_generation": 0,
            "project_context_hash": "",
        }
    row = session_db.get_session(session_id)
    if not isinstance(row, Mapping):
        return validate_project_affinity_row(None, session_id=session_id, projects_db_path=None)

    context_length = int(
        getattr(getattr(agent, "context_compressor", None), "context_length", 0) or 0
    ) or None
    identity = tuple(str(row.get(key) or "").strip() for key in (
        "project_id", "project_root", "project_context_hash",
    ))
    candidate: Optional[ProjectAffinityCandidate] = None
    if all(identity):
        candidate = load_project_affinity_candidate(
            project_id=identity[0], project_root=identity[1], context_length=context_length,
        )
    elif not any(identity) and (
        int(row.get("message_count") or 0) == 0
        and not row.get("parent_session_id")
        and row.get("cwd")
    ):
        profile_home = _profile_home_for_agent(agent)
        projects_path = profile_home / "projects.db" if profile_home is not None else None
        if projects_path is not None and projects_path.exists():
            from hermes_cli import projects_db
            with projects_db.connect_closing(db_path=projects_path) as projects_conn:
                candidate = resolve_project_affinity_for_cwd(
                    str(row["cwd"]), projects_conn=projects_conn, context_length=context_length,
                )
    if candidate is not None:
        session_db.update_session_project_affinity(
            session_id,
            project_id=candidate.project_id,
            project_root=candidate.project_root,
            project_context_hash=candidate.context_hash,
        )
        row = session_db.get_session(session_id) or row

    profile_home = _profile_home_for_agent(agent)
    projects_path = profile_home / "projects.db" if profile_home is not None else None
    return validate_project_affinity_row(
        row,
        session_id=session_id,
        projects_db_path=projects_path,
    )
