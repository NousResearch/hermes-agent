"""Kanban ↔ Linear issue linkage (idempotent create + parent/child inheritance)."""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

from hermes_cli import kanban_db as kb

logger = logging.getLogger(__name__)

LINEAR_API_URL = "https://api.linear.app/graphql"


class LinearKanbanError(RuntimeError):
    """Linear API or configuration failure for kanban linkage."""


@dataclass(frozen=True)
class LinearKanbanConfig:
    """Resolved ``kanban.linear`` settings plus env fallbacks."""

    enabled: bool
    team: str
    default_priority: Optional[int] = None

    @classmethod
    def from_runtime_config(cls, config: Optional[dict] = None) -> "LinearKanbanConfig":
        if config is None:
            from hermes_cli.config import load_config

            config = load_config()
        kanban_cfg = (config or {}).get("kanban") or {}
        linear_cfg = kanban_cfg.get("linear") or {}
        if not isinstance(linear_cfg, dict):
            linear_cfg = {}
        enabled = linear_cfg.get("enabled")
        if enabled is None:
            enabled = bool(os.environ.get("LINEAR_API_KEY", "").strip())
        else:
            enabled = bool(enabled)
        team = (
            str(linear_cfg.get("team") or linear_cfg.get("team_id") or "")
            .strip()
        )
        prio = linear_cfg.get("default_priority")
        default_priority = int(prio) if prio is not None else None
        return cls(enabled=enabled, team=team, default_priority=default_priority)

    def api_key(self) -> str:
        return os.environ.get("LINEAR_API_KEY", "").strip()


def _gql(query: str, variables: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    key = os.environ.get("LINEAR_API_KEY", "").strip()
    if not key:
        raise LinearKanbanError(
            "LINEAR_API_KEY is not set (Linear Settings → API → personal key)"
        )
    payload: dict[str, Any] = {"query": query}
    if variables:
        payload["variables"] = variables
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        LINEAR_API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": key,
            "User-Agent": "hermes-agent-kanban-linear/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise LinearKanbanError(f"Linear HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LinearKanbanError(f"Linear network error: {exc}") from exc

    result = json.loads(body)
    if result.get("errors"):
        raise LinearKanbanError(
            f"Linear GraphQL errors: {json.dumps(result['errors'], ensure_ascii=False)}"
        )
    return result.get("data") or {}


def _resolve_team_id(team_ref: str) -> str:
    """Resolve a team key (ENG) or UUID to Linear team id."""
    ref = team_ref.strip()
    if not ref:
        raise LinearKanbanError(
            "kanban.linear.team is not configured (set team key, e.g. ENG)"
        )
    if len(ref) == 36 and ref.count("-") == 4:
        return ref
    q = """query($key: String!) {
      teams(filter: { key: { eq: $key } }, first: 1) {
        nodes { id key name }
      }
    }"""
    nodes = _gql(q, {"key": ref.upper()}).get("teams", {}).get("nodes") or []
    if not nodes:
        raise LinearKanbanError(f"Linear team not found for key {ref!r}")
    return str(nodes[0]["id"])


def _create_linear_issue(
    *,
    title: str,
    description: Optional[str],
    team_ref: str,
    priority: Optional[int] = None,
) -> tuple[str, str]:
    team_id = _resolve_team_id(team_ref)
    inp: dict[str, Any] = {"title": title.strip(), "teamId": team_id}
    if description:
        inp["description"] = description
    if priority is not None:
        inp["priority"] = int(priority)
    q = """mutation($input: IssueCreateInput!) {
      issueCreate(input: $input) {
        success
        issue { id identifier title url }
      }
    }"""
    payload = _gql(q, {"input": inp}).get("issueCreate") or {}
    if not payload.get("success"):
        raise LinearKanbanError("Linear issueCreate returned success=false")
    issue = payload.get("issue") or {}
    issue_id = issue.get("id")
    issue_url = issue.get("url")
    if not issue_id or not issue_url:
        raise LinearKanbanError("Linear issueCreate missing id or url")
    return str(issue_id), str(issue_url)


def ensure_linear_issue_for_task(
    conn,
    task_id: str,
    *,
    config: Optional[LinearKanbanConfig] = None,
    force_create: bool = False,
) -> kb.Task:
    """Ensure ``task_id`` has a Linear issue link (reuse, inherit, or create).

    When ``force_create`` is false (default), an existing ``linear_issue_id``
    on the task is returned as-is. When true, a missing link still follows
    inherit-then-create, but an existing link is not replaced.
    """
    cfg = config or LinearKanbanConfig.from_runtime_config()
    task = kb.get_task(conn, task_id)
    if task is None:
        raise ValueError(f"unknown task: {task_id}")
    if task.linear_issue_id and not force_create:
        return task

    for parent_id in kb.parent_ids(conn, task_id):
        parent = kb.get_task(conn, parent_id)
        if parent and parent.linear_issue_id and parent.linear_issue_url:
            updated = kb.set_linear_link(
                conn,
                task_id,
                parent.linear_issue_id,
                parent.linear_issue_url,
                propagate=False,
                source="inherited",
            )
            if updated:
                logger.info(
                    "kanban linear: task %s inherited issue from parent %s",
                    task_id,
                    parent_id,
                )
            out = kb.get_task(conn, task_id)
            return out if out is not None else task

    if not cfg.enabled:
        return task
    if not cfg.api_key():
        raise LinearKanbanError("Linear linkage is enabled but LINEAR_API_KEY is unset")
    if not cfg.team:
        raise LinearKanbanError("kanban.linear.team is required to create issues")

    body = task.body or ""
    desc_parts = [body] if body.strip() else []
    desc_parts.append(f"Hermes kanban task: `{task.id}`")
    description = "\n\n".join(desc_parts)

    issue_id, issue_url = _create_linear_issue(
        title=task.title,
        description=description,
        team_ref=cfg.team,
        priority=cfg.default_priority,
    )
    updated = kb.set_linear_link(
        conn,
        task_id,
        issue_id,
        issue_url,
        propagate=True,
        source="created",
    )
    if updated:
        logger.info(
            "kanban linear: created issue for task %s → %s",
            task_id,
            issue_url,
        )
    out = kb.get_task(conn, task_id)
    return out if out is not None else task
