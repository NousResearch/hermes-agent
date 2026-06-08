"""Safe-local SEO Mission Control payloads for Agents OS."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from hermes_cli.agents_os import AgentsOSPaths, connect, row_to_dict

SEO_GOAL_WORKFLOWS = {"seo-goal"}
KEYWORD_WORKFLOWS = {"keyword-research"}
DRAFT_WORKFLOWS = {"seo-draft"}
REVIEW_WORKFLOWS = {"publish-gate", "outreach-gate"}


def _path_info(path: str) -> dict[str, Any]:
    p = Path(path)
    return {
        "path": str(p),
        "exists": p.exists(),
        "size_bytes": p.stat().st_size if p.exists() and p.is_file() else None,
    }


def _artifact_items(paths: AgentsOSPaths, workflows: set[str]) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in workflows)
    if not placeholders:
        return []
    query = f"SELECT id,kind,title,path,task_id,workflow,created_at FROM artifacts WHERE workflow IN ({placeholders}) ORDER BY created_at DESC"
    with connect(paths) as conn:
        rows = conn.execute(query, tuple(workflows)).fetchall()
    items = []
    for row in rows:
        item = row_to_dict(row)
        item.update(_path_info(item["path"]))
        item["draft_only"] = True
        item["publish_blocked_until_approval"] = True
        items.append(item)
    return items


def seo_mission_control_payload(paths: AgentsOSPaths) -> dict[str, Any]:
    """Return a side-effect-free SEO/AISO dashboard payload.

    v0 intentionally supports only draft/review state. Live metrics, outreach,
    and publishing remain approval-gated and disabled.
    """
    with connect(paths) as conn:
        goal_rows = conn.execute(
            "SELECT id,title,status,workflow,priority,created_at,updated_at,notes,route,approval_required FROM tasks WHERE workflow='seo-goal' ORDER BY created_at DESC"
        ).fetchall()
    goals = [row_to_dict(row) for row in goal_rows]
    for goal in goals:
        goal["draft_only"] = True
        goal["publish_blocked_until_approval"] = True
    return {
        "local_only": True,
        "publish_enabled": False,
        "outreach_enabled": False,
        "credentials_required_for_live_metrics": True,
        "goals": goals,
        "keyword_queue": _artifact_items(paths, KEYWORD_WORKFLOWS),
        "draft_queue": _artifact_items(paths, DRAFT_WORKFLOWS),
        "review_gates": _artifact_items(paths, REVIEW_WORKFLOWS),
        "analytics_evidence_slot": {
            "enabled": False,
            "reason": "Search Console/Analytics credentials are approval-gated and not connected in v0.",
        },
        "approval_gates": [
            "publish",
            "outreach",
            "credentials",
            "analytics",
            "deploy",
            "cross_agent_memory_merge",
        ],
        "lane_contract": {
            "goal": "SEO goal → keyword research → draft → review gate → optional approved publish.",
            "default_status": "draft_only",
            "external_actions": "disabled_until_explicit_approval",
        },
    }
