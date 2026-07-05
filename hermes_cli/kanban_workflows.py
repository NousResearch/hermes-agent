"""Standard Kanban workflow templates.

These helpers write durable task graphs into the existing Kanban kernel. They
do not introduce a second scheduler: parent/child links keep downstream steps in
``todo`` until the dispatcher promotes them to ``ready`` after upstream
completion.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import json
import sqlite3
from typing import Any, Optional

from hermes_cli import kanban_db as kb

TEMPLATE_CODER_QA_PM_REVIEW = "coder_qa_pm_review"
WORKFLOW_COMMENT_PREFIX = "[workflow:coder_qa_pm_review] "


@dataclass(frozen=True)
class CoderQaPmWorkflowCreated:
    """IDs produced by :func:`create_coder_qa_pm_review_workflow`."""

    root_id: str
    coder_task_id: str
    qa_task_id: str
    pm_review_task_id: str
    template_id: str = TEMPLATE_CODER_QA_PM_REVIEW
    subscriptions: dict[str, bool] = field(default_factory=dict)

    @property
    def task_ids(self) -> list[str]:
        return [self.coder_task_id, self.qa_task_id, self.pm_review_task_id]

    def as_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "root_id": self.root_id,
            "coder_task_id": self.coder_task_id,
            "qa_task_id": self.qa_task_id,
            "pm_review_task_id": self.pm_review_task_id,
            "task_ids": self.task_ids,
            "subscriptions": dict(self.subscriptions),
        }


def _require_text(value: Optional[str], field_name: str) -> str:
    text = (value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _workflow_context(root_id: str, title: str, body: Optional[str]) -> str:
    spec = body.strip() if body else "(no extra body provided)"
    return (
        "\n\n## Standard workflow\n"
        f"- Template: {TEMPLATE_CODER_QA_PM_REVIEW}\n"
        f"- Workflow root: `{root_id}`.\n"
        "- Handoff through Kanban completion summaries and metadata.\n"
        "- Do not skip downstream gates; Coder completes for QA, QA completes for PM review.\n"
        "- Keep releases, pushes, deploys, and irreversible external actions "
        "behind PM review unless the task explicitly authorizes them.\n"
        f"\n## Original request\nTitle: {title}\n\n{spec}\n"
    )


def _latest_topology(conn: sqlite3.Connection, root_id: str) -> dict[str, Any]:
    rows = kb.list_comments(conn, root_id)
    for comment in reversed(rows):
        body = comment.body or ""
        if not body.startswith(WORKFLOW_COMMENT_PREFIX):
            continue
        try:
            payload = json.loads(body[len(WORKFLOW_COMMENT_PREFIX):])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _subscribe_all(
    task_ids: list[str],
    subscribe_task: Optional[Callable[[str], bool]],
) -> dict[str, bool]:
    if subscribe_task is None:
        return {task_id: False for task_id in task_ids}
    subscribed: dict[str, bool] = {}
    for task_id in task_ids:
        try:
            subscribed[task_id] = bool(subscribe_task(task_id))
        except Exception:
            subscribed[task_id] = False
    return subscribed


def create_coder_qa_pm_review_workflow(
    conn: sqlite3.Connection,
    *,
    title: str,
    body: Optional[str] = None,
    coder_assignee: str = "coder-codex",
    qa_assignee: str = "qa-minimax",
    pm_assignee: str = "pm-deepseek",
    tenant: Optional[str] = None,
    created_by: str = "workflow-orchestrator",
    workspace_kind: str = "scratch",
    workspace_path: Optional[str] = None,
    project_id: Optional[str] = None,
    priority: int = 0,
    idempotency_key: Optional[str] = None,
    session_id: Optional[str] = None,
    subscribe_task: Optional[Callable[[str], bool]] = None,
) -> CoderQaPmWorkflowCreated:
    """Create the standard ``coder -> qa -> pm review`` Kanban workflow.

    The returned graph is immediately dispatchable. The root card is completed
    as an audit anchor, the coder card is ``ready``, QA waits on coder, and PM
    review waits on QA. If ``subscribe_task`` is provided, it is called for all
    three executable cards so the creating PM gets completion notifications for
    each handoff.
    """

    title = _require_text(title, "title")
    coder_assignee = _require_text(coder_assignee, "coder_assignee")
    qa_assignee = _require_text(qa_assignee, "qa_assignee")
    pm_assignee = _require_text(pm_assignee, "pm_assignee")
    created_by = _require_text(created_by, "created_by")

    root = kb.create_task(
        conn,
        title=f"Workflow: {title[:80]}",
        body=(
            "Standard coder -> qa -> pm review workflow root. This card is "
            "completed immediately so the first executable step can dispatch, "
            "while the card remains the audit anchor for the pipeline.\n\n"
            f"Original request:\n{body or title}"
        ),
        assignee=created_by,
        created_by=created_by,
        tenant=tenant,
        priority=priority,
        idempotency_key=idempotency_key,
        workspace_kind=workspace_kind,
        workspace_path=workspace_path,
        project_id=project_id,
        session_id=session_id,
        workflow_template_id=TEMPLATE_CODER_QA_PM_REVIEW,
        current_step_key="root",
    )

    existing = _latest_topology(conn, root)
    if existing:
        coder_id = existing.get("coder_task_id")
        qa_id = existing.get("qa_task_id")
        pm_review_id = existing.get("pm_review_task_id")
        if coder_id and qa_id and pm_review_id:
            task_ids = [str(coder_id), str(qa_id), str(pm_review_id)]
            return CoderQaPmWorkflowCreated(
                root_id=root,
                coder_task_id=str(coder_id),
                qa_task_id=str(qa_id),
                pm_review_task_id=str(pm_review_id),
                subscriptions=_subscribe_all(task_ids, subscribe_task),
            )

    kb.complete_task(
        conn,
        root,
        summary="Workflow topology planned; executable steps are linked by dependencies.",
        metadata={"template_id": TEMPLATE_CODER_QA_PM_REVIEW, "title": title},
    )

    context = _workflow_context(root, title, body)
    coder = kb.create_task(
        conn,
        title=f"Coder: {title}",
        body=(
            "Implement the requested work and complete this card with a concise "
            "handoff for QA: changed files, verification commands, and residual "
            "risks."
            + context
        ),
        assignee=coder_assignee,
        created_by=created_by,
        parents=[root],
        tenant=tenant,
        priority=priority,
        workspace_kind=workspace_kind,
        workspace_path=workspace_path,
        project_id=project_id,
        session_id=session_id,
        workflow_template_id=TEMPLATE_CODER_QA_PM_REVIEW,
        current_step_key="coder",
    )

    qa = kb.create_task(
        conn,
        title=f"QA: {title}",
        body=(
            "Review the coder handoff and verify the acceptance criteria. "
            "Complete this card only when QA evidence is ready for PM review; "
            "otherwise block with the exact missing input or failing evidence."
            + context
        ),
        assignee=qa_assignee,
        created_by=created_by,
        parents=[coder],
        tenant=tenant,
        priority=priority,
        workspace_kind=workspace_kind,
        workspace_path=workspace_path,
        project_id=project_id,
        session_id=session_id,
        skills=["requesting-code-review"],
        workflow_template_id=TEMPLATE_CODER_QA_PM_REVIEW,
        current_step_key="qa",
    )

    pm_review = kb.create_task(
        conn,
        title=f"PM review: {title}",
        body=(
            "Review the coder and QA handoffs, decide whether the workflow is "
            "accepted, and perform or explicitly authorize any final human-gated "
            "action."
            + context
        ),
        assignee=pm_assignee,
        created_by=created_by,
        parents=[qa],
        tenant=tenant,
        priority=priority,
        workspace_kind=workspace_kind,
        workspace_path=workspace_path,
        project_id=project_id,
        session_id=session_id,
        workflow_template_id=TEMPLATE_CODER_QA_PM_REVIEW,
        current_step_key="pm_review",
    )

    task_ids = [coder, qa, pm_review]
    subscriptions = _subscribe_all(task_ids, subscribe_task)
    created = CoderQaPmWorkflowCreated(
        root_id=root,
        coder_task_id=coder,
        qa_task_id=qa,
        pm_review_task_id=pm_review,
        subscriptions=subscriptions,
    )
    kb.add_comment(
        conn,
        root,
        author=created_by,
        body=WORKFLOW_COMMENT_PREFIX
        + json.dumps(created.as_dict() | {"title": title}, sort_keys=True),
    )
    return created
