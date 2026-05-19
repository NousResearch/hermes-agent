"""Nubbi Linear project polling automation.

Polls Linear for actionable issues and mirrors them into Hermes Kanban so the
existing dispatcher can safely start work without duplicate claims.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from hermes_constants import get_hermes_home
from utils import atomic_replace

logger = logging.getLogger(__name__)

LINEAR_API_URL = "https://api.linear.app/graphql"
STATE_RELATIONS = {"blocked", "blockedBy", "blocking"}


@dataclass(frozen=True)
class NubbiTickResult:
    created: int = 0
    skipped_duplicates: int = 0
    skipped_blocked: int = 0
    skipped_non_actionable: int = 0
    synced: int = 0


class LinearAPIError(RuntimeError):
    """Raised when Linear returns an HTTP or GraphQL error."""


class LinearClient:
    def __init__(self, api_key: str, *, api_url: str = LINEAR_API_URL, timeout: int = 30):
        self.api_key = api_key
        self.api_url = api_url
        self.timeout = timeout

    def _graphql(self, query: str, variables: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        payload = json.dumps(
            {"query": query, "variables": variables or {}},
            ensure_ascii=False,
        ).encode("utf-8")
        req = urllib.request.Request(
            self.api_url,
            data=payload,
            method="POST",
            headers={
                "Authorization": self.api_key,
                "Content-Type": "application/json",
                "User-Agent": "Hermes-Nubbi-Automation/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise LinearAPIError(f"Linear HTTP {exc.code}: {body[:500]}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LinearAPIError(f"Linear request failed: {exc}") from exc
        if result.get("errors"):
            raise LinearAPIError(f"Linear GraphQL errors: {json.dumps(result['errors'])[:500]}")
        data = result.get("data")
        if not isinstance(data, dict):
            raise LinearAPIError("Linear response missing data")
        return data

    def list_project_issues(
        self,
        project_name: str,
        state_names: Iterable[str],
    ) -> list[dict[str, Any]]:
        query = """
        query NubbiIssues($projectName: String!, $stateNames: [String!]) {
          issues(
            first: 100,
            filter: {
              project: { name: { eq: $projectName } },
              state: { name: { in: $stateNames } },
              archivedAt: { null: true }
            }
          ) {
            nodes {
              id
              identifier
              title
              description
              url
              blocked
              state { name }
              project { name }
              relations(first: 25) {
                nodes {
                  type
                  relatedIssue { id identifier state { name type } }
                }
              }
            }
          }
        }
        """
        data = self._graphql(
            query,
            {"projectName": project_name, "stateNames": list(state_names)},
        )
        return list(data.get("issues", {}).get("nodes") or [])

    def add_comment(self, issue_id: str, body: str) -> None:
        self._graphql(
            """
            mutation NubbiComment($issueId: String!, $body: String!) {
              commentCreate(input: { issueId: $issueId, body: $body }) {
                success
              }
            }
            """,
            {"issueId": issue_id, "body": body},
        )

    def transition_issue(self, issue_id: str, state_name: str) -> None:
        state_id = self._state_id_for_issue(issue_id, state_name)
        self._graphql(
            """
            mutation NubbiTransition($issueId: String!, $stateId: String!) {
              issueUpdate(id: $issueId, input: { stateId: $stateId }) {
                success
              }
            }
            """,
            {"issueId": issue_id, "stateId": state_id},
        )

    def _state_id_for_issue(self, issue_id: str, state_name: str) -> str:
        data = self._graphql(
            """
            query NubbiIssueTeam($issueId: String!, $stateName: String!) {
              issue(id: $issueId) {
                team {
                  states(filter: { name: { eq: $stateName } }) {
                    nodes { id name }
                  }
                }
              }
            }
            """,
            {"issueId": issue_id, "stateName": state_name},
        )
        nodes = (
            data.get("issue", {})
            .get("team", {})
            .get("states", {})
            .get("nodes")
            or []
        )
        if not nodes:
            raise LinearAPIError(f"Linear state not found for issue {issue_id}: {state_name}")
        return str(nodes[0]["id"])


def _state_path() -> Path:
    return get_hermes_home() / "nubbi" / "linear_automation_state.json"


def _load_state() -> dict[str, Any]:
    path = _state_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"issues": {}}
    if not isinstance(raw, dict):
        return {"issues": {}}
    raw.setdefault("issues", {})
    if not isinstance(raw["issues"], dict):
        raw["issues"] = {}
    return raw


def _save_state(state: dict[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    atomic_replace(tmp, path)


def _issue_state_name(issue: dict[str, Any]) -> str:
    state = issue.get("state") if isinstance(issue.get("state"), dict) else {}
    return str(state.get("name") or "").strip()


def _issue_key(issue: dict[str, Any]) -> str:
    return str(issue.get("id") or "").strip()


def _is_blocked(issue: dict[str, Any]) -> bool:
    if bool(issue.get("blocked")):
        return True
    relations = issue.get("relations")
    nodes = []
    if isinstance(relations, dict):
        nodes = relations.get("nodes") or []
    for rel in nodes:
        if not isinstance(rel, dict):
            continue
        rel_type = str(rel.get("type") or "").strip()
        if rel_type not in STATE_RELATIONS:
            continue
        related = rel.get("relatedIssue") if isinstance(rel.get("relatedIssue"), dict) else {}
        state = related.get("state") if isinstance(related.get("state"), dict) else {}
        if str(state.get("type") or "").lower() not in {"completed", "canceled"}:
            return True
    return False


def _task_title(issue: dict[str, Any]) -> str:
    identifier = str(issue.get("identifier") or "").strip()
    title = str(issue.get("title") or "").strip()
    return f"{identifier}: {title}" if identifier else title


def _task_body(issue: dict[str, Any]) -> str:
    parts = [
        "Imported from Linear by Nubbi automation.",
        f"Linear issue: {issue.get('url') or issue.get('identifier') or issue.get('id')}",
    ]
    description = str(issue.get("description") or "").strip()
    if description:
        parts.extend(["", description])
    return "\n".join(parts).strip()


def _find_existing_task(conn: Any, issue_id: str) -> Optional[Any]:
    from hermes_cli import kanban_db

    row = conn.execute(
        "SELECT id FROM tasks WHERE idempotency_key = ? AND status != 'archived' "
        "ORDER BY created_at DESC LIMIT 1",
        (f"linear:{issue_id}",),
    ).fetchone()
    if not row:
        return None
    return kanban_db.get_task(conn, row["id"])


def _sync_linear_state_for_task(
    task: Any,
    issue_id: str,
    *,
    current_linear_state: str,
    config: dict[str, Any],
    linear_client: Any,
    state: dict[str, Any],
) -> bool:
    target = None
    if task.status in {"ready", "running"}:
        target = config.get("start_state", "In Progress")
    elif task.status == "review":
        target = config.get("review_state", "In Review")
    elif task.status == "done":
        target = config.get("done_state", "Done")
    if not target:
        return False

    issue_state = state["issues"].setdefault(issue_id, {})
    if current_linear_state == target:
        issue_state["linear_state"] = target
        issue_state["last_synced_at"] = int(time.time())
        return False
    linear_client.transition_issue(issue_id, target)
    issue_state["linear_state"] = target
    issue_state["last_synced_at"] = int(time.time())
    return True


def _linear_config_from_root(root_config: dict[str, Any]) -> dict[str, Any]:
    nubbi_cfg = root_config.get("nubbi") if isinstance(root_config, dict) else {}
    linear_cfg = nubbi_cfg.get("linear") if isinstance(nubbi_cfg, dict) else {}
    return dict(linear_cfg) if isinstance(linear_cfg, dict) else {}


def tick_from_root_config(
    root_config: dict[str, Any],
    *,
    tick_fn: Any = None,
    linear_client: Optional[Any] = None,
) -> NubbiTickResult:
    runner = tick_fn or tick
    return runner(
        config=_linear_config_from_root(root_config),
        linear_client=linear_client,
    )


async def watch_from_config(
    *,
    load_config_fn: Any,
    running_fn: Any,
    tick_fn: Any = None,
    sleep_fn: Any = None,
) -> None:
    """Poll Nubbi Linear automation until ``running_fn`` returns false."""
    runner = tick_fn or tick_from_root_config
    sleeper = sleep_fn or asyncio.sleep

    while running_fn():
        interval = 300
        try:
            cfg = load_config_fn()
            linear_cfg = _linear_config_from_root(cfg)
            try:
                interval = int(linear_cfg.get("poll_interval_seconds", 300) or 300)
            except (TypeError, ValueError):
                interval = 300
            interval = max(interval, 60)
            if linear_cfg.get("enabled", False):
                await asyncio.to_thread(runner, cfg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Nubbi Linear automation tick failed: %s", exc)

        slept = 0.0
        while slept < interval and running_fn():
            step = min(1.0, interval - slept)
            await sleeper(step)
            slept += step


def tick(
    *,
    config: Optional[dict[str, Any]] = None,
    linear_client: Optional[Any] = None,
) -> NubbiTickResult:
    config = dict(config or {})
    if not config.get("enabled", False):
        return NubbiTickResult()

    api_key = os.getenv("LINEAR_API_KEY", "").strip()
    if not api_key and linear_client is None:
        logger.info("Nubbi Linear automation disabled: LINEAR_API_KEY is not set")
        return NubbiTickResult()

    project_name = str(config.get("project_name") or "Nubbi Command Center").strip()
    source_states = tuple(config.get("source_states") or ("Todo", "Backlog"))
    active_states = tuple(config.get("active_states") or ())
    query_states = tuple(dict.fromkeys(source_states + active_states))
    if not project_name:
        logger.warning("Nubbi Linear automation disabled: project_name is empty")
        return NubbiTickResult()

    client = linear_client or LinearClient(api_key)
    issues = client.list_project_issues(project_name, query_states)

    from hermes_cli import kanban_db

    board = config.get("kanban_board") or "default"
    conn = kanban_db.connect(board=board)
    state = _load_state()
    created = skipped_duplicates = skipped_blocked = skipped_non_actionable = synced = 0
    try:
        for issue in issues:
            issue_id = _issue_key(issue)
            if not issue_id:
                skipped_non_actionable += 1
                continue

            existing = _find_existing_task(conn, issue_id)
            if existing is not None:
                skipped_duplicates += 1
                if _sync_linear_state_for_task(
                    existing,
                    issue_id,
                    current_linear_state=_issue_state_name(issue),
                    config=config,
                    linear_client=client,
                    state=state,
                ):
                    synced += 1
                continue

            if _issue_state_name(issue) not in source_states:
                skipped_non_actionable += 1
                continue
            if _is_blocked(issue):
                skipped_blocked += 1
                continue

            task_id = kanban_db.create_task(
                conn,
                title=_task_title(issue),
                body=_task_body(issue),
                assignee=config.get("assignee") or "default",
                created_by="nubbi-linear-automation",
                workspace_kind=config.get("workspace_kind") or "scratch",
                workspace_path=config.get("workspace_path") or None,
                tenant="linear",
                idempotency_key=f"linear:{issue_id}",
                board=board,
            )
            kanban_db.add_comment(
                conn,
                task_id,
                author="nubbi-linear-automation",
                body=f"Imported from Linear issue {issue.get('identifier') or issue_id}.",
            )
            client.add_comment(
                issue_id,
                f"Nubbi automation started work by creating Hermes Kanban task {task_id}.",
            )
            start_state = config.get("start_state", "In Progress")
            client.transition_issue(issue_id, start_state)
            state["issues"][issue_id] = {
                "kanban_task_id": task_id,
                "linear_state": start_state,
                "identifier": issue.get("identifier"),
                "url": issue.get("url"),
                "last_synced_at": int(time.time()),
            }
            created += 1
    finally:
        conn.close()
    _save_state(state)
    return NubbiTickResult(
        created=created,
        skipped_duplicates=skipped_duplicates,
        skipped_blocked=skipped_blocked,
        skipped_non_actionable=skipped_non_actionable,
        synced=synced,
    )
