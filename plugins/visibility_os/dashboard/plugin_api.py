from __future__ import annotations

from typing import Any
import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from plugins.visibility_os.core.actions import (
    approve_action,
    create_action,
    edit_action,
    get_action,
    list_actions,
    list_audit_log,
    reject_action,
    save_for_later,
)
from plugins.visibility_os.core.executors import execute_approved_action
from plugins.visibility_os.core.impact_picker import generate_daily_plan
from plugins.visibility_os.core.opportunities import get_opportunity, list_opportunities
from plugins.visibility_os.core.opportunity_actions import build_opportunity_detail, draft_action_from_opportunity
from plugins.visibility_os.core.workstreams import get_workstream, latest_for_opportunity, list_workstreams
from plugins.visibility_os.core.board import get_board_states, set_board_state
from plugins.visibility_os.core.pr_audit import audit_pr_from_opportunity, deep_review_pr_from_opportunity
from plugins.visibility_os.core.scanner import scan_github
from plugins.visibility_os.core.connectors.github import GitHubConnector
from plugins.visibility_os.core import db
from plugins.visibility_os.core.config import get_visibility_config

router = APIRouter()

class ActionCreate(BaseModel):
    proposed_by_agent: str
    action_type: str
    target_system: str
    target_location: str
    title: str
    summary: str
    proposed_payload: dict[str, Any]
    evidence_links: list[dict[str, Any]] = Field(default_factory=list)
    risk_level: str = "low"
    opportunity_id: str | None = None
    impact_score: int | None = None
    visibility_score: int | None = None
    effort_score: int | None = None
    approval_reason: str | None = None

class ActorBody(BaseModel):
    actor: str = "human"
    execute_immediately: bool = False

class EditBody(BaseModel):
    final_payload: dict[str, Any]
    actor: str = "human"

class RejectBody(BaseModel):
    reason: str
    actor: str = "human"

class ScanBody(BaseModel):
    repo: str | None = None

class OpportunityDraftBody(BaseModel):
    action_kind: str
    target_location: str | None = None
    actor: str = "visibility_os"

class OpportunityAuditBody(BaseModel):
    actor: str = "visibility_os"

class BoardStateBody(BaseModel):
    item_kind: str
    item_id: str
    board_state: str
    actor: str = "human"

def _is_github_actions_run_url(url: str | None) -> bool:
    return bool(url and "/actions/runs/" in url)


def _is_github_issue_url(url: str | None) -> bool:
    return bool(url and "/issues/" in url)


def _workstream_rollup(opportunity_id: str | None) -> dict[str, Any]:
    if not opportunity_id:
        return {
            "workstream_id": None,
            "workstream_status": None,
            "workstream_stage": None,
            "agent_has_worked_on_this": False,
            "last_agent_activity_at": None,
            "pending_human_action": None,
        }
    ws = latest_for_opportunity(opportunity_id)
    if not ws:
        return {
            "workstream_id": None,
            "workstream_status": None,
            "workstream_stage": None,
            "agent_has_worked_on_this": False,
            "last_agent_activity_at": None,
            "pending_human_action": None,
        }
    return {
        "workstream_id": ws.get("id"),
        "workstream_status": ws.get("status"),
        "workstream_stage": ws.get("stage"),
        "agent_has_worked_on_this": True,
        "current_step": ws.get("current_step"),
        "progress_percent": ws.get("progress_percent"),
        "last_agent_activity_at": ws.get("updated_at"),
        "pending_human_action": ws.get("pending_human_action"),
    }


def _feed_action(action: dict[str, Any]) -> dict[str, Any]:
    item = {"kind": "action", **action}
    opportunity_id = action.get("opportunity_id")
    item.update(_workstream_rollup(opportunity_id))
    if not opportunity_id:
        return item
    try:
        opportunity = get_opportunity(opportunity_id)
    except KeyError:
        return item
    source_url = opportunity.get("source_url")
    item["opportunity_source_url"] = source_url
    item["opportunity_title"] = opportunity.get("title")
    item["can_diagnose_ci"] = _is_github_actions_run_url(source_url)
    item["can_fix_issue"] = _is_github_issue_url(source_url)
    return item


def _derived_board_state(item: dict[str, Any]) -> str:
    if item.get("action_type") == "github_push_branch" and item.get("status") in {"queued", "edited_by_human"}:
        return "in_review"
    if item.get("pending_human_action"):
        return "in_review"
    stage = item.get("workstream_stage")
    if stage in {"ready_for_push", "review_ready", "self_auditing", "independent_reviewing"}:
        return "in_review"
    if stage in {"pushed", "completed"} or item.get("workstream_status") == "completed":
        return "done"
    if item.get("kind") == "action" and item.get("status") in {"executed", "completed", "rejected", "failed"}:
        return "done"
    if item.get("kind") == "opportunity" and item.get("status") in {"resolved", "closed", "done"}:
        return "done"
    if item.get("workstream_id") and item.get("workstream_status") == "active":
        return "in_progress"
    if item.get("kind") == "action" and item.get("status") in {"approved"}:
        return "in_progress"
    return "todo"


def _apply_board_state(items: list[dict[str, Any]], include_archived: bool = False) -> list[dict[str, Any]]:
    overrides = get_board_states()
    out = []
    for item in items:
        kind = str(item.get("kind") or "")
        item_id = str(item.get("id") or "")
        override = overrides.get((kind, item_id))
        board_state = override["board_state"] if override else _derived_board_state(item)
        item["board_state"] = board_state
        item["board_state_updated_at"] = override.get("updated_at") if override else None
        item["board_state_actor"] = override.get("actor") if override else None
        if include_archived or board_state != "archived":
            out.append(item)
    return out


def _is_needs_decision(item: dict[str, Any]) -> bool:
    # Decision cards should be the actual approval/rejectable action, not both the
    # action and its source opportunity, otherwise the command center feels noisy.
    if item.get("kind") != "action":
        return False
    if item.get("action_type") in {"github_push_branch", "github_pr_comment", "github_issue_comment", "slack_message"}:
        return item.get("status") in {"queued", "edited_by_human", "needs_review", "drafted"}
    return item.get("status") in {"queued", "edited_by_human", "needs_review", "drafted"} and bool(item.get("approval_required"))


def _is_agent_working(item: dict[str, Any]) -> bool:
    return item.get("workstream_status") == "active" and (item.get("board_state") == "in_progress" or item.get("workstream_stage") not in {None, "queued", "ready_for_push", "review_ready", "completed", "failed", "cancelled", "pushed"})


def _section_item(item: dict[str, Any]) -> dict[str, Any]:
    # Keep the same card shape so the dashboard can open section cards exactly like Kanban cards.
    return dict(item)


def _build_sections(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    sections = {
        "needs_decision": [],
        "agent_working_now": [],
        "open_opportunities": [],
        "completed_recently": [],
    }
    for item in items:
        state = item.get("board_state") or "todo"
        if _is_needs_decision(item):
            sections["needs_decision"].append(_section_item(item))
        if _is_agent_working(item):
            sections["agent_working_now"].append(_section_item(item))
        if item.get("kind") == "opportunity" and state == "todo" and not item.get("agent_has_worked_on_this"):
            sections["open_opportunities"].append(_section_item(item))
        if state == "done" or (item.get("kind") == "action" and item.get("status") in {"executed", "completed", "rejected", "failed"}):
            sections["completed_recently"].append(_section_item(item))
    return {key: value[:12] for key, value in sections.items()}


def _github_repo_allowed(repo: str | None) -> bool:
    return get_visibility_config().github_repo_allowed(repo)


@router.get("/config")
async def visibility_config() -> dict[str, Any]:
    cfg = get_visibility_config()
    return {
        "company_name": cfg.company_name,
        "github_orgs": sorted(cfg.github_orgs),
        "github_repos": cfg.github_repos,
        "default_slack_channel": cfg.default_slack_channel,
    }


@router.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "plugin": "visibility-os"}

@router.get("/feed")
async def feed(include_archived: bool = False) -> dict[str, Any]:
    actions = list_actions()
    opportunities = list_opportunities(limit=100)
    items = []
    for a in actions:
        items.append(_feed_action(a))
    for o in opportunities:
        source_url = o.get("source_url")
        items.append({"kind": "opportunity", "can_diagnose_ci": _is_github_actions_run_url(source_url), "can_fix_issue": _is_github_issue_url(source_url), **_workstream_rollup(o.get("id")), **o})
    items = _apply_board_state(items, include_archived=include_archived)
    items.sort(key=lambda x: (x.get("created_at") or x.get("updated_at") or ""), reverse=True)
    counts_by_state: dict[str, int] = {"todo": 0, "in_progress": 0, "in_review": 0, "done": 0, "archived": 0}
    for item in items:
        state = item.get("board_state") or "todo"
        counts_by_state[state] = counts_by_state.get(state, 0) + 1
    sections = _build_sections(items)
    counts_by_section = {key: len(value) for key, value in sections.items()}
    return {
        "items": items,
        "sections": sections,
        "counts": {
            "actions": len(actions),
            "opportunities": len(opportunities),
            "board": counts_by_state,
            "sections": counts_by_section,
        },
    }

@router.post("/board-state")
async def update_board_state(body: BoardStateBody) -> dict[str, Any]:
    try:
        return set_board_state(item_kind=body.item_kind, item_id=body.item_id, board_state=body.board_state, actor=body.actor)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.get("/actions")
async def actions(status: str | None = None) -> dict[str, Any]:
    return {"actions": list_actions(status=status)}

@router.post("/actions")
async def post_action(body: ActionCreate) -> dict[str, Any]:
    return create_action(**body.model_dump())

@router.get("/actions/{action_id}")
async def action_detail(action_id: str) -> dict[str, Any]:
    try:
        return get_action(action_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

@router.post("/actions/{action_id}/approve")
async def approve(action_id: str, body: ActorBody) -> dict[str, Any]:
    try:
        approved = approve_action(action_id, actor=body.actor)
        if body.execute_immediately:
            return execute_approved_action(action_id, actor=body.actor)
        return approved
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/actions/{action_id}/edit")
async def edit(action_id: str, body: EditBody) -> dict[str, Any]:
    try:
        return edit_action(action_id, final_payload=body.final_payload, actor=body.actor)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/actions/{action_id}/reject")
async def reject(action_id: str, body: RejectBody) -> dict[str, Any]:
    try:
        return reject_action(action_id, reason=body.reason, actor=body.actor)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/actions/{action_id}/save")
async def save(action_id: str, body: ActorBody) -> dict[str, Any]:
    return save_for_later(action_id, actor=body.actor)

@router.post("/actions/{action_id}/execute")
async def execute(action_id: str, body: ActorBody) -> dict[str, Any]:
    try:
        return execute_approved_action(action_id, actor=body.actor)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.get("/audit-log")
async def audit_log(action_id: str | None = None) -> dict[str, Any]:
    return {"events": list_audit_log(action_id=action_id)}

@router.get("/opportunities")
async def opportunities() -> dict[str, Any]:
    return {"opportunities": list_opportunities(limit=100)}

@router.get("/workstreams")
async def workstreams(status: str | None = None) -> dict[str, Any]:
    return {"workstreams": list_workstreams(status=status)}

@router.get("/workstreams/{workstream_id}")
async def workstream_detail(workstream_id: str) -> dict[str, Any]:
    try:
        return get_workstream(workstream_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

@router.get("/opportunities/{opportunity_id}/workstreams")
async def opportunity_workstreams(opportunity_id: str) -> dict[str, Any]:
    try:
        get_opportunity(opportunity_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"workstreams": list_workstreams(opportunity_id=opportunity_id)}

@router.get("/opportunities/{opportunity_id}")
async def opportunity_detail(opportunity_id: str) -> dict[str, Any]:
    try:
        return build_opportunity_detail(opportunity_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

@router.post("/opportunities/{opportunity_id}/draft-action")
async def draft_opportunity_action(opportunity_id: str, body: OpportunityDraftBody) -> dict[str, Any]:
    try:
        return draft_action_from_opportunity(opportunity_id, action_kind=body.action_kind, target_location=body.target_location, actor=body.actor)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/opportunities/{opportunity_id}/fix-ci")
async def fix_ci_opportunity(opportunity_id: str, body: ActorBody) -> dict[str, Any]:
    try:
        action = draft_action_from_opportunity(opportunity_id, action_kind="ci_fix_lane", actor=body.actor)
        approve_action(action["id"], actor=body.actor)
        return execute_approved_action(action["id"], actor=body.actor)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/opportunities/{opportunity_id}/fix-issue")
async def fix_issue_opportunity(opportunity_id: str, body: ActorBody) -> dict[str, Any]:
    try:
        action = draft_action_from_opportunity(opportunity_id, action_kind="github_issue_fix_lane", actor=body.actor)
        approve_action(action["id"], actor=body.actor)
        return execute_approved_action(action["id"], actor=body.actor)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/opportunities/{opportunity_id}/audit-pr")
async def audit_pr_opportunity(opportunity_id: str, body: OpportunityAuditBody) -> dict[str, Any]:
    try:
        return audit_pr_from_opportunity(opportunity_id, actor=body.actor)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/opportunities/{opportunity_id}/deep-review-pr")
async def deep_review_pr_opportunity(opportunity_id: str, body: OpportunityAuditBody) -> dict[str, Any]:
    try:
        return deep_review_pr_from_opportunity(opportunity_id, actor=body.actor)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/scan/github")
async def scan(body: ScanBody) -> dict[str, Any]:
    cfg = get_visibility_config()
    if not _github_repo_allowed(body.repo):
        raise HTTPException(status_code=400, detail=f"Visibility OS is restricted to configured GitHub organisations: {cfg.github_scope_label}")
    try:
        return {"opportunities": scan_github(GitHubConnector(repo=body.repo))}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/scan/github/all")
async def scan_all_github() -> dict[str, Any]:
    db.init_db()
    cfg = get_visibility_config()
    repos = cfg.github_repos
    if not repos:
        with db.connect() as conn:
            row = conn.execute("SELECT state_payload FROM connector_state WHERE connector_name = ?", ("github_repos",)).fetchone()
        if row:
            repos = json.loads(row["state_payload"]).get("repos") or []
    repos = [repo for repo in repos if _github_repo_allowed(repo)]
    if not repos:
        raise HTTPException(status_code=400, detail="No configured GitHub repos found. Set VISIBILITY_OS_GITHUB_REPOS and VISIBILITY_OS_GITHUB_ORGS in .env.")
    results = []
    for repo in repos:
        try:
            opportunities = scan_github(GitHubConnector(repo=repo))
            results.append({"repo": repo, "ok": True, "count": len(opportunities)})
        except Exception as exc:
            results.append({"repo": repo, "ok": False, "error": str(exc)})
    return {"results": results, "opportunities": list_opportunities(limit=100)}

@router.get("/connectors")
async def connectors() -> dict[str, Any]:
    db.init_db()
    with db.connect() as conn:
        rows = conn.execute("SELECT connector_name, state_payload, updated_at FROM connector_state ORDER BY connector_name").fetchall()
    return {"connectors": [{"name": r["connector_name"], "state": json.loads(r["state_payload"]), "updated_at": r["updated_at"]} for r in rows]}

@router.post("/daily-plan")
async def daily_plan() -> dict[str, Any]:
    return generate_daily_plan()
