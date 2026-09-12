"""Bridge a dispatcher-owned worker session to its Kanban run receipt."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


log = logging.getLogger(__name__)
_CONTEXT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class WorkerReceiptState:
    task_id: str
    run_id: int
    worker_session_id: str
    workspace_path: Optional[str]


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _hash_json(value: Any) -> str:
    return _hash_text(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))


def _git_output(repo: Path, *args: str) -> Optional[bytes]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, check=False, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _worktree_fingerprint(path: str | Path | None) -> Optional[str]:
    """Return ``git-v1:<HEAD>:<delta digest>`` without retaining source content."""
    if not path:
        return None
    repo = Path(path).expanduser()
    head_raw = _git_output(repo, "rev-parse", "HEAD")
    status = _git_output(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    diff = _git_output(repo, "diff", "--binary", "HEAD", "--")
    untracked = _git_output(repo, "ls-files", "--others", "--exclude-standard", "-z")
    if head_raw is None or status is None or diff is None or untracked is None:
        return None
    digest = hashlib.sha256()
    digest.update(status)
    digest.update(diff)
    for raw_name in sorted(name for name in untracked.split(b"\0") if name):
        digest.update(b"\0untracked\0")
        digest.update(raw_name)
        candidate = repo / raw_name.decode("utf-8", errors="surrogateescape")
        try:
            if candidate.is_symlink():
                digest.update(os.readlink(candidate).encode("utf-8", errors="surrogateescape"))
            elif candidate.is_file():
                digest.update(candidate.read_bytes())
        except OSError:
            return None
    head = head_raw.decode("ascii", errors="strict").strip()
    return f"git-v1:{head}:{digest.hexdigest()}"


def _reasoning_config_label(config: Any) -> Optional[str]:
    if not isinstance(config, dict):
        return None
    if config.get("enabled") is False:
        return "disabled"
    value = config.get("effort")
    if value is not None:
        return str(value)
    return "enabled" if config.get("enabled") is True else None


def _requested_reasoning_label(cli: Any, task: Any) -> Optional[str]:
    if getattr(task, "reasoning_effort", None):
        return str(task.reasoning_effort)
    return _reasoning_config_label(getattr(cli, "reasoning_config", None))


def _env_identity() -> tuple[str, Optional[int]]:
    task_id = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    raw_run_id = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
    try:
        run_id = int(raw_run_id) if raw_run_id else None
    except ValueError:
        run_id = None
    return task_id, run_id


def begin_worker_receipt(cli: Any) -> Optional[WorkerReceiptState]:
    """Record the worker's actual session, route, and full-context baseline."""
    task_id, run_id = _env_identity()
    session_db = getattr(cli, "_session_db", None)
    agent = getattr(cli, "agent", None)
    session_id = str(getattr(agent, "session_id", None) or getattr(cli, "session_id", "") or "")
    if not task_id or run_id is None or not session_id or session_db is None:
        return None
    try:
        usage_start = session_db.session_lineage_usage_snapshot(session_id)
        lineage_root = session_db.get_conversation_root(session_id)
        with kbc.connect_closing() as conn:
            task = kb.get_task(conn, task_id)
            if task is None:
                return None
            workspace_path = (
                (os.environ.get("HERMES_KANBAN_WORKSPACE") or "").strip()
                or task.workspace_path
            )
            context = kb.build_worker_context(conn, task_id)
            profile = (os.environ.get("HERMES_PROFILE") or task.assignee or "").strip() or None
            requested_reasoning = _requested_reasoning_label(cli, task)
            ok = kb.record_run_session(
                conn, task_id, run_id,
                worker_session_id=session_id,
                session_lineage_root=lineage_root,
                profile=profile,
                requested_provider=(task.provider_override or getattr(cli, "requested_provider", None)),
                requested_model=(task.model_override or getattr(cli, "model", None)),
                requested_reasoning=requested_reasoning,
                effective_provider=getattr(agent, "provider", None),
                effective_model=getattr(agent, "model", None),
                effective_reasoning=_reasoning_config_label(
                    getattr(agent, "reasoning_config", None)
                ),
                fresh_or_resumed=("resumed" if getattr(cli, "_resumed", False) else "fresh"),
                usage_start=usage_start,
            )
            if not ok:
                return None
            kb.record_run_context_receipt(
                conn, task_id, run_id,
                context_schema_version=_CONTEXT_SCHEMA_VERSION,
                context_fingerprint=_hash_text(context),
                context_chars=len(context),
                system_prompt_hash=_hash_text(
                    str(getattr(agent, "_cached_system_prompt", "") or "")
                ),
                toolset_hash=_hash_json(sorted(getattr(cli, "enabled_toolsets", None) or [])),
                skills_hash=_hash_json(sorted(task.skills or [])),
            )
            kb.record_run_runner_handle(
                conn, task_id, run_id,
                runner=(
                    os.environ.get("HERMES_KANBAN_RUNNER")
                    or ("herdr" if os.environ.get("HERDR_ENV") == "1" else "hermes")
                ),
                runner_metadata={
                    "workspace_id": os.environ.get("HERMES_HERDR_WORKSPACE_ID"),
                    "pane_id": os.environ.get("HERMES_HERDR_PANE_ID"),
                    "agent_id": os.environ.get("HERMES_HERDR_AGENT_ID"),
                },
                worktree_start_fingerprint=_worktree_fingerprint(workspace_path),
            )
        return WorkerReceiptState(task_id, run_id, session_id, workspace_path)
    except Exception:
        log.debug("kanban run receipt start failed", exc_info=True)
        return None


def finalize_worker_receipt(cli: Any, state: Optional[WorkerReceiptState]) -> bool:
    """Flush token accounting and finalize one exact worker/run receipt."""
    if state is None:
        return False
    session_db = getattr(cli, "_session_db", None)
    if session_db is None:
        return False
    try:
        current_session_id = str(
            getattr(getattr(cli, "agent", None), "session_id", None)
            or state.worker_session_id
        )
        usage_end = session_db.session_lineage_usage_snapshot(current_session_id)
        with kbc.connect_closing() as conn:
            return kb.finalize_run_usage(
                conn, state.task_id, state.run_id,
                worker_session_id=state.worker_session_id,
                usage_end=usage_end,
                worktree_end_fingerprint=_worktree_fingerprint(state.workspace_path),
            )
    except Exception:
        log.debug("kanban run receipt finalization failed", exc_info=True)
        return False
