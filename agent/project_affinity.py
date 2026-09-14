from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class ProjectAffinityCandidate:
    project_id: str
    project_root: str
    context: str
    context_hash: str


def render_project_affinity_context(candidate: ProjectAffinityCandidate, generation: int) -> str:
    """Turn-scoped context block; the hash marker makes replay/dedup deterministic."""
    body = candidate.context or "(No project context files are currently present at this Project root.)"
    return (
        f"<!-- hermes-project-affinity:{candidate.context_hash} -->\n"
        "# Runtime-confirmed Project Context\n\n"
        "This block supersedes any older Project Context in the cached system prompt or transcript.\n\n"
        f"Project ID: `{candidate.project_id}`\n"
        f"Project root: `{candidate.project_root}`\n"
        f"Affinity generation: `{generation}`\n\n"
        + body
    )


def _context_is_already_active(
    candidate: ProjectAffinityCandidate, active_system_prompt: str, messages: Sequence[dict],
) -> bool:
    if candidate.context and candidate.context in str(active_system_prompt or ""):
        return True
    return any(
        candidate.context_hash in str(message.get(key) or "")
        for message in messages if isinstance(message, dict)
        for key in ("api_content", "content")
    )


def load_project_affinity_candidate(
    *, project_id: str, project_root: str, context_length: Optional[int] = None,
) -> Optional[ProjectAffinityCandidate]:
    """Load a complete Runtime-owned affinity candidate from an explicit project."""
    pid = (project_id or "").strip()
    root = str(Path(project_root).expanduser().resolve()) if str(project_root or "").strip() else ""
    if not pid or not root:
        return None
    from agent.prompt_builder import build_context_files_prompt

    context = build_context_files_prompt(cwd=root, skip_soul=True, context_length=context_length)
    context_hash = "sha256:" + hashlib.sha256(context.encode("utf-8")).hexdigest()
    return ProjectAffinityCandidate(pid, root, context, context_hash)


def resolve_project_affinity_for_cwd(
    cwd: str, *, projects_conn, context_length: Optional[int] = None,
) -> Optional[ProjectAffinityCandidate]:
    """Resolve the innermost named Project owning cwd, then load its context bytes."""
    from hermes_cli import projects_db

    project = projects_db.project_for_path(projects_conn, cwd)
    if project is None or not project.primary_path:
        return None
    return load_project_affinity_candidate(
        project_id=project.id,
        project_root=project.primary_path,
        context_length=context_length,
    )


def collect_turn_project_affinity(
    agent: Any, *, messages: Sequence[dict], active_system_prompt: str,
) -> str:
    """Refresh/bind session affinity and return context missing from the active transcript.

    Auto-bind is limited to a brand-new, message-free session. Once a session has
    ownership, cwd drift never changes it; only an explicit project switch may do so.
    """
    session_db = getattr(agent, "_session_db", None)
    session_id = str(getattr(agent, "session_id", "") or "")
    if (
        session_db is None or not session_id
        or getattr(agent, "_persist_disabled", False)
        or getattr(agent, "skip_context_files", False)
    ):
        return ""
    row = session_db.get_session(session_id)
    if not isinstance(row, Mapping):
        return ""
    context_length = int(getattr(getattr(agent, "context_compressor", None), "context_length", 0) or 0) or None
    affinity = (row.get("project_id"), row.get("project_root"), row.get("project_context_hash"))
    candidate: Optional[ProjectAffinityCandidate] = None
    if all(affinity):
        candidate = load_project_affinity_candidate(
            project_id=str(affinity[0]), project_root=str(affinity[1]), context_length=context_length,
        )
    elif any(affinity):
        return ""  # fail closed on a legacy/corrupt partial tuple
    elif (
        int(row.get("message_count") or 0) == 0
        and not row.get("parent_session_id")
        and row.get("cwd")
    ):
        db_path = getattr(session_db, "db_path", None)
        projects_path = Path(db_path).parent / "projects.db" if db_path else None
        if projects_path is not None and projects_path.exists():
            from hermes_cli import projects_db
            with projects_db.connect_closing(db_path=projects_path) as projects_conn:
                candidate = resolve_project_affinity_for_cwd(
                    str(row["cwd"]), projects_conn=projects_conn, context_length=context_length,
                )
    if candidate is None:
        return ""
    generation = session_db.update_session_project_affinity(
        session_id,
        project_id=candidate.project_id,
        project_root=candidate.project_root,
        project_context_hash=candidate.context_hash,
    )
    if generation is None or _context_is_already_active(candidate, active_system_prompt, messages):
        return ""
    return render_project_affinity_context(candidate, generation)
