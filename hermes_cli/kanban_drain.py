"""Safe Kanban drain controller slices.

This module intentionally implements only the narrow review-packet
PASS/FAIL consumption path.  It plans by default and mutates only when
``apply=True`` so operators can inspect the same report shape before
committing changes.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Optional

from hermes_cli import kanban_db as kb

DRAIN_AUTHOR = "kanban-drain-controller"
REVIEW_PASS = "consume_review_pass"
REVIEW_FAIL = "consume_review_fail"
ROUTE_REVIEW = "route_to_review"
RESLICE_TIMEOUT_GAVE_UP = "reslice_timeout_gave_up"
ARCHIVE_SUPERSEDED_DUPLICATE = "archive_superseded_duplicate"
TERMINAL_PARENT_STATUSES = {"done", "archived"}
_HOLD_RE = re.compile(
    r"\b("
    r"operator|credential|credentials|secret|secrets|oauth|"
    r"production|prod|deploy|deployment|rollout|payment|payments|pii|"
    r"public[- ]posting|approval|approve|decision"
    r")\b",
    re.IGNORECASE,
)
_CREDENTIAL_RE = re.compile(
    r"\b("
    r"credential|credentials|secret|secrets|oauth|mfa|auth|"
    r"github auth|private repo|bad credentials"
    r")\b",
    re.IGNORECASE,
)
_PROD_RISK_RE = re.compile(
    r"\b("
    r"production|prod|deploy|deployment|rollout|payment|payments|pii|"
    r"public[- ]posting|public action|billing|customer data"
    r")\b",
    re.IGNORECASE,
)
_HUMAN_HOLD_RE = re.compile(
    r"\b(operator|approval|approve|decision|owner/release-owner)\b",
    re.IGNORECASE,
)
_RUNTIME_INFRA_RE = re.compile(
    r"\b("
    r"pid \d+ not alive|protocol violation|missing tools?|terminal|"
    r"gateway|session|cron|runtime|worker exited|no terminal|no filesystem"
    r")\b",
    re.IGNORECASE,
)
_SCOPE_TOO_LARGE_RE = re.compile(
    r"\b(iteration budget exhausted|max[- ]runtime|too large|reslice|smaller child)\b",
    re.IGNORECASE,
)
_SOURCE_OF_TRUTH_RE = re.compile(
    r"\b("
    r"source[- ]of[- ]truth|missing spec|spec gate|adr|approved contract|"
    r"domain|authority|prerequisite docs|dependency/contract"
    r")\b",
    re.IGNORECASE,
)
_PRODUCT_SCOPE_RE = re.compile(
    r"\b(product|scope|acceptance criteria|ux|copy|business decision|priority)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DrainTaskClassification:
    task_id: str
    title: str
    status: str
    assignee: Optional[str]
    inferred_class: str
    owner: str
    eligible_action: str
    reason: str
    evidence: list[str]
    nonterminal_parents: list[str]


@dataclass(frozen=True)
class ReviewPacket:
    review_task_id: str
    source_task_id: str
    approved: bool
    evidence: str
    reviewer: str
    review_evidence: dict[str, Any]
    authority_boundary: str
    findings: list[str]
    required_followups: list[str]
    run_id: Optional[int] = None
    comment_id: Optional[int] = None


@dataclass(frozen=True)
class InvalidReviewPacket:
    review_task_id: str
    evidence: str
    reason: str
    source_task_id: Optional[str] = None
    run_id: Optional[int] = None
    comment_id: Optional[int] = None


def classify_board_for_drain(conn: sqlite3.Connection) -> dict[str, Any]:
    """Classify non-terminal board rows for safe drain reporting.

    This function is intentionally read-only. It reports drain classes and
    owners so operators can choose the next profile-specific batch without
    promoting unknown, human-held, credential-held, prod-risk, or parent-gated
    work.
    """
    tasks = [
        task
        for task in kb.list_tasks(conn, include_archived=False)
        if task.status in {"todo", "blocked", "review"}
    ]
    classified = [_classify_task_for_drain(conn, task) for task in tasks]
    by_status = Counter(item.status for item in classified)
    by_class = Counter(item.inferred_class for item in classified)
    by_owner = Counter(item.owner for item in classified)

    owner_classes: dict[str, Counter[str]] = defaultdict(Counter)
    owner_samples: dict[str, list[str]] = defaultdict(list)
    for item in classified:
        owner_classes[item.owner][item.inferred_class] += 1
        if len(owner_samples[item.owner]) < 5:
            owner_samples[item.owner].append(item.task_id)

    profile_queues = [
        {
            "owner": owner,
            "count": by_owner[owner],
            "classes": dict(sorted(owner_classes[owner].items())),
            "sample_task_ids": owner_samples[owner],
        }
        for owner in sorted(by_owner)
    ]

    return {
        "summary": {
            "total": len(classified),
            "by_status": dict(sorted(by_status.items())),
            "by_class": dict(sorted(by_class.items())),
            "by_owner": dict(sorted(by_owner.items())),
            "parent_gated_todo": by_class.get("parent_gated", 0),
            "review_required": by_class.get("review_required", 0),
            "classification_debt": by_class.get("classification_debt", 0),
        },
        "profile_queues": profile_queues,
        "tasks": [asdict(item) for item in classified],
    }


def drain_review_packets(
    conn: sqlite3.Connection,
    *,
    apply: bool = False,
    limit: Optional[int] = None,
) -> dict[str, Any]:
    """Plan or consume completed Review Packet PASS/FAIL rows.

    The only mutating classes are:
    - PASS: audit-comment the blocked review-required source and complete it.
    - FAIL: audit-comment the source and create/identify one idempotent rework
      task for the original source assignee.

    Human/credential/prod-risk block text, non-review-required blocks, missing
    parents, malformed packets, and source-card state drift are reported as
    refusals/skips rather than mutated.
    """
    packet_items = _completed_review_packets(conn)
    if limit is not None:
        packet_items = packet_items[: max(0, int(limit))]

    actions: list[dict[str, Any]] = []
    for packet in packet_items:
        if isinstance(packet, InvalidReviewPacket):
            action = _build_invalid_packet_action(packet)
        else:
            action = _build_action(conn, packet)
        if apply and action["status"] == "planned" and isinstance(packet, ReviewPacket):
            action = _apply_action(conn, packet, action)
        actions.append(action)

    summary = {
        "total_packets": len(packet_items),
        "planned": sum(1 for a in actions if a["status"] == "planned"),
        "applied": sum(1 for a in actions if a["status"] == "applied"),
        "already_applied": sum(1 for a in actions if a["status"] == "already_applied"),
        "refused": sum(1 for a in actions if a["status"] == "refused"),
        "skipped": sum(1 for a in actions if a["status"] == "skipped"),
    }
    return {
        "dry_run": not apply,
        "class": "review_packets",
        "summary": summary,
        "classification": classify_board_for_drain(conn),
        "actions": actions,
    }


def drain_review_required(
    conn: sqlite3.Connection,
    *,
    apply: bool = False,
    limit: Optional[int] = None,
) -> dict[str, Any]:
    """Plan or route safe review-required blocked cards into review lane."""
    candidates = _review_required_candidates(conn)
    if limit is not None:
        candidates = candidates[: max(0, int(limit))]

    actions: list[dict[str, Any]] = []
    for task in candidates:
        action = _build_review_required_action(conn, task)
        if apply and action["status"] == "planned":
            action = _apply_review_required_action(conn, task, action)
        actions.append(action)

    summary = {
        "total_candidates": len(candidates),
        "planned": sum(1 for a in actions if a["status"] == "planned"),
        "applied": sum(1 for a in actions if a["status"] == "applied"),
        "already_applied": sum(1 for a in actions if a["status"] == "already_applied"),
        "refused": sum(1 for a in actions if a["status"] == "refused"),
        "skipped": sum(1 for a in actions if a["status"] == "skipped"),
    }
    return {
        "dry_run": not apply,
        "class": "review_required",
        "summary": summary,
        "classification": classify_board_for_drain(conn),
        "actions": actions,
    }


def drain_timeout_gave_up(
    conn: sqlite3.Connection,
    *,
    apply: bool = False,
    limit: Optional[int] = None,
) -> dict[str, Any]:
    """Plan or reslice structured timeout/gave-up blocked cards."""
    candidates = _timeout_gave_up_candidates(conn)
    if limit is not None:
        candidates = candidates[: max(0, int(limit))]

    actions: list[dict[str, Any]] = []
    for task in candidates:
        action = _build_timeout_gave_up_action(conn, task)
        if apply and action["status"] == "planned":
            action = _apply_timeout_gave_up_action(conn, task, action)
        actions.append(action)

    summary = {
        "total_candidates": len(candidates),
        "planned": sum(1 for a in actions if a["status"] == "planned"),
        "applied": sum(1 for a in actions if a["status"] == "applied"),
        "already_applied": sum(1 for a in actions if a["status"] == "already_applied"),
        "refused": sum(1 for a in actions if a["status"] == "refused"),
        "skipped": sum(1 for a in actions if a["status"] == "skipped"),
    }
    return {
        "dry_run": not apply,
        "class": "timeout_gave_up",
        "summary": summary,
        "classification": classify_board_for_drain(conn),
        "actions": actions,
    }


def drain_superseded_duplicates(
    conn: sqlite3.Connection,
    *,
    apply: bool = False,
    limit: Optional[int] = None,
) -> dict[str, Any]:
    """Plan or archive structured superseded duplicate cards."""
    candidates = _superseded_duplicate_candidates(conn)
    if limit is not None:
        candidates = candidates[: max(0, int(limit))]

    actions: list[dict[str, Any]] = []
    for task in candidates:
        action = _build_superseded_duplicate_action(conn, task)
        if apply and action["status"] == "planned":
            action = _apply_superseded_duplicate_action(conn, task, action)
        actions.append(action)

    summary = {
        "total_candidates": len(candidates),
        "planned": sum(1 for a in actions if a["status"] == "planned"),
        "applied": sum(1 for a in actions if a["status"] == "applied"),
        "already_applied": sum(1 for a in actions if a["status"] == "already_applied"),
        "refused": sum(1 for a in actions if a["status"] == "refused"),
        "skipped": sum(1 for a in actions if a["status"] == "skipped"),
    }
    return {
        "dry_run": not apply,
        "class": "superseded_duplicates",
        "summary": summary,
        "classification": classify_board_for_drain(conn),
        "actions": actions,
    }


def _review_required_candidates(conn: sqlite3.Connection) -> list[kb.Task]:
    candidates: list[kb.Task] = []
    tasks = [
        task
        for task in kb.list_tasks(conn, include_archived=False)
        if task.status in {"blocked", "review"}
    ]
    for task in tasks:
        latest_block = _latest_block_reason(conn, task.id) or ""
        if (
            task.block_class in {"review_required", "human_hold", "credential_hold", "prod_risk_hold"}
            or latest_block.lower().startswith("review-required:")
            or (task.status == "review" and _comment_exists(conn, task.id, ROUTE_REVIEW))
        ):
            candidates.append(task)
    return candidates


def _superseded_duplicate_candidates(conn: sqlite3.Connection) -> list[kb.Task]:
    return [
        task
        for task in kb.list_tasks(conn, include_archived=True)
        if task.block_class == "superseded_duplicate"
    ]


def _timeout_gave_up_candidates(conn: sqlite3.Connection) -> list[kb.Task]:
    candidates: list[kb.Task] = []
    tasks = [
        task
        for task in kb.list_tasks(conn, include_archived=False)
        if task.status in {"blocked", "review", "ready"}
    ]
    for task in tasks:
        if task.block_class == "timeout_gave_up" or (
            task.block_class in {"human_hold", "credential_hold", "prod_risk_hold"}
            and _has_timeout_gave_up_signal(task)
        ):
            candidates.append(task)
    return candidates


def _build_timeout_gave_up_action(conn: sqlite3.Connection, task: kb.Task) -> dict[str, Any]:
    action_id = _timeout_gave_up_action_id(task)
    idempotency_key = _timeout_gave_up_idempotency_key(task)
    action = {
        "drain_action_id": action_id,
        "action": RESLICE_TIMEOUT_GAVE_UP,
        "source_task_id": task.id,
        "status": "planned",
        "evidence": _timeout_gave_up_evidence(task),
        "reslice_idempotency_key": idempotency_key,
    }
    existing_child = _find_task_by_idempotency_key(conn, idempotency_key)
    if existing_child:
        action["child_task_id"] = existing_child

    if task.status != "blocked":
        action.update(status="skipped", reason=f"source task is {task.status!r}, not 'blocked'")
        return action
    if not _parents_terminal(conn, task.id):
        action.update(status="refused", reason="source task has unsatisfied parent dependencies")
        return action
    if task.block_class in {"human_hold", "credential_hold", "prod_risk_hold"}:
        action.update(status="refused", reason="source task has explicit human/credential/prod-risk block_class")
        return action
    if task.block_class != "timeout_gave_up":
        action.update(status="refused", reason="source task lacks structured timeout_gave_up block_class")
        return action
    hold_class = _hold_class(" ".join([task.last_failure_error or "", _latest_block_reason(conn, task.id) or ""]))
    if hold_class:
        action.update(status="refused", reason=f"timeout/gave-up evidence looks like {hold_class}")
        return action

    reslice, reason = _timeout_gave_up_reslice_spec(task)
    if reslice is None:
        action.update(status="refused", reason=reason)
        return action
    action["reslice"] = reslice

    if existing_child and _comment_exists(conn, task.id, action_id):
        action.update(status="already_applied")
    return action


def _apply_timeout_gave_up_action(
    conn: sqlite3.Connection,
    task: kb.Task,
    action: dict[str, Any],
) -> dict[str, Any]:
    current = kb.get_task(conn, task.id)
    if current is None:
        action.update(status="refused", reason="source task disappeared before apply")
        return action
    rebuilt = _build_timeout_gave_up_action(conn, current)
    if rebuilt["status"] != "planned":
        return rebuilt

    reslice = dict(rebuilt["reslice"])
    action_id = str(rebuilt["drain_action_id"])
    idempotency_key = str(rebuilt["reslice_idempotency_key"])
    child_id = _find_task_by_idempotency_key(conn, idempotency_key)
    if child_id is None:
        child_id = kb.create_task(
            conn,
            title=reslice["title"],
            body=_timeout_gave_up_child_body(current, reslice, action_id),
            assignee=reslice.get("assignee") or current.assignee,
            created_by=DRAIN_AUTHOR,
            priority=current.priority,
            parents=[current.id],
            idempotency_key=idempotency_key,
        )
    comment_existed = _comment_exists(conn, current.id, action_id)
    if not comment_existed:
        kb.add_comment(
            conn,
            current.id,
            DRAIN_AUTHOR,
            _timeout_gave_up_audit_comment(current, reslice, action_id, child_id),
        )
        kb._append_event(
            conn,
            current.id,
            "drain_resliced_timeout_gave_up",
            {
                "drain_action_id": action_id,
                "action": RESLICE_TIMEOUT_GAVE_UP,
                "source_task_id": current.id,
                "child_task_id": child_id,
                "reslice_idempotency_key": idempotency_key,
            },
        )

    rebuilt["child_task_id"] = child_id
    if comment_existed:
        rebuilt.update(status="already_applied")
    else:
        rebuilt.update(status="applied")
    return rebuilt


def _build_review_required_action(conn: sqlite3.Connection, task: kb.Task) -> dict[str, Any]:
    action_id = _route_review_action_id(task)
    action = {
        "drain_action_id": action_id,
        "action": ROUTE_REVIEW,
        "source_task_id": task.id,
        "status": "planned",
        "evidence": _route_review_evidence(task),
    }
    if task.status == "review" and _comment_exists(conn, task.id, ROUTE_REVIEW):
        action.update(status="already_applied")
        return action
    if task.status != "blocked":
        action.update(status="skipped", reason=f"source task is {task.status!r}, not 'blocked'")
        return action
    if not _parents_terminal(conn, task.id):
        action.update(status="refused", reason="source task has unsatisfied parent dependencies")
        return action
    if task.block_class in {"human_hold", "credential_hold", "prod_risk_hold"}:
        action.update(status="refused", reason="source task has explicit human/credential/prod-risk block_class")
        return action
    latest_block = _latest_block_reason(conn, task.id) or ""
    if task.block_class != "review_required" and not latest_block.lower().startswith("review-required:"):
        action.update(status="refused", reason="latest source block is not review-required")
        return action
    if _HOLD_RE.search(latest_block):
        action.update(status="refused", reason="latest source block text looks like a human/credential/prod-risk hold")
        return action
    if _comment_exists(conn, task.id, action_id):
        action.update(status="already_applied")
    return action


def _apply_review_required_action(
    conn: sqlite3.Connection,
    task: kb.Task,
    action: dict[str, Any],
) -> dict[str, Any]:
    action_id = str(action["drain_action_id"])
    current = kb.get_task(conn, task.id)
    if current is None:
        action.update(status="refused", reason="source task disappeared before apply")
        return action
    rebuilt = _build_review_required_action(conn, current)
    if rebuilt["status"] != "planned":
        return rebuilt

    comment_body = _route_review_audit_comment(current, action_id, action.get("evidence", []))
    with kb.write_txn(conn):
        if not _comment_exists(conn, current.id, action_id):
            conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) "
                "VALUES (?, ?, ?, ?)",
                (current.id, DRAIN_AUTHOR, comment_body, int(time.time())),
            )
        cur = conn.execute(
            "UPDATE tasks SET status = 'review' WHERE id = ? AND status = 'blocked'",
            (current.id,),
        )
        if cur.rowcount != 1:
            action.update(status="skipped", reason="source task status changed before apply")
            return action
        kb._append_event(
            conn,
            current.id,
            "drain_routed_review",
            {
                "drain_action_id": action_id,
                "action": ROUTE_REVIEW,
                "source_task_id": current.id,
                "evidence": action.get("evidence", []),
            },
        )
    action.update(status="applied")
    return action


def _route_review_action_id(task: kb.Task) -> str:
    payload = {
        "action": ROUTE_REVIEW,
        "source_task_id": task.id,
        "block_class": task.block_class,
        "block_metadata": task.block_metadata,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:24]


def _route_review_evidence(task: kb.Task) -> list[str]:
    evidence: list[str] = []
    if task.block_class:
        evidence.append(f"block_class:{task.block_class}")
    metadata = task.block_metadata or {}
    for item in metadata.get("evidence") or []:
        text = str(item).strip()
        if text:
            evidence.append(text)
    if not evidence:
        evidence.append("latest_block_reason")
    return evidence


def _route_review_audit_comment(task: kb.Task, action_id: str, evidence: list[str]) -> str:
    body = {
        "drain_action_id": action_id,
        "action": ROUTE_REVIEW,
        "source_task_id": task.id,
        "previous_status": "blocked",
        "new_status": "review",
        "evidence": evidence,
    }
    return "kanban drain review-required routing:\n" + json.dumps(
        body,
        indent=2,
        sort_keys=True,
    )


def _has_timeout_gave_up_signal(task: kb.Task) -> bool:
    metadata = task.block_metadata or {}
    trigger = str(metadata.get("trigger_outcome") or "").strip().lower()
    if trigger in {"timed_out", "timeout", "gave_up"}:
        return True
    text = " ".join(
        str(part)
        for part in [
            task.last_failure_error or "",
            metadata.get("reason") or "",
            metadata.get("source") or "",
        ]
        if part
    ).lower()
    return any(marker in text for marker in ("timed_out", "timeout", "gave_up", "iteration budget"))


def _timeout_gave_up_action_id(task: kb.Task) -> str:
    payload = {
        "action": RESLICE_TIMEOUT_GAVE_UP,
        "source_task_id": task.id,
        "block_class": task.block_class,
        "reslice": (task.block_metadata or {}).get("reslice"),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:24]


def _timeout_gave_up_idempotency_key(task: kb.Task) -> str:
    return f"kanban-drain:timeout-gave-up-reslice:{task.id}"


def _timeout_gave_up_evidence(task: kb.Task) -> list[str]:
    evidence: list[str] = []
    if task.block_class:
        evidence.append(f"block_class:{task.block_class}")
    metadata = task.block_metadata or {}
    trigger = metadata.get("trigger_outcome")
    if isinstance(trigger, str) and trigger.strip():
        evidence.append(f"trigger_outcome:{trigger.strip()}")
    for item in metadata.get("evidence") or []:
        text = str(item).strip()
        if text:
            evidence.append(text)
    if task.last_failure_error:
        evidence.append("last_failure_error")
    if task.consecutive_failures:
        evidence.append(f"consecutive_failures:{task.consecutive_failures}")
    if not evidence:
        evidence.append("timeout_gave_up_metadata")
    return evidence


def _timeout_gave_up_reslice_spec(task: kb.Task) -> tuple[Optional[dict[str, Any]], str]:
    metadata = task.block_metadata or {}
    raw = metadata.get("reslice")
    if not isinstance(raw, dict):
        return None, "timeout/gave-up block_metadata.reslice object missing"
    title = raw.get("title")
    scope = raw.get("scope")
    if not isinstance(title, str) or not title.strip():
        return None, "timeout/gave-up reslice title missing"
    if not isinstance(scope, str) or not scope.strip():
        return None, "timeout/gave-up reslice explicit scope missing"
    acceptance_raw = raw.get("acceptance") or raw.get("acceptance_criteria") or []
    if isinstance(acceptance_raw, str) and acceptance_raw.strip():
        acceptance = [acceptance_raw.strip()]
    elif isinstance(acceptance_raw, list):
        acceptance = [str(item).strip() for item in acceptance_raw if str(item).strip()]
    else:
        acceptance = []
    assignee = raw.get("assignee")
    reslice = {
        "title": title.strip(),
        "scope": scope.strip(),
        "acceptance": acceptance,
    }
    if isinstance(assignee, str) and assignee.strip():
        reslice["assignee"] = assignee.strip()
    return reslice, ""


def _timeout_gave_up_child_body(
    source: kb.Task,
    reslice: dict[str, Any],
    action_id: str,
) -> str:
    acceptance = "\n".join(f"- {item}" for item in reslice.get("acceptance") or [])
    if not acceptance:
        acceptance = "- Complete the explicit scope and end with review-required evidence."
    return (
        "Timeout/gave-up reslice created by Kanban drain controller.\n\n"
        f"Source card: {source.id}\n"
        f"Drain action id: {action_id}\n\n"
        "Scope:\n"
        f"{reslice['scope']}\n\n"
        "Acceptance:\n"
        f"{acceptance}\n\n"
        "Boundary: do not clear or mutate the source card failure counters; "
        "finish this child with a review-required handoff."
    )


def _timeout_gave_up_audit_comment(
    source: kb.Task,
    reslice: dict[str, Any],
    action_id: str,
    child_id: str,
) -> str:
    body = {
        "drain_action_id": action_id,
        "action": RESLICE_TIMEOUT_GAVE_UP,
        "source_task_id": source.id,
        "child_task_id": child_id,
        "reslice_idempotency_key": _timeout_gave_up_idempotency_key(source),
        "source_status_preserved": source.status,
        "source_consecutive_failures_preserved": source.consecutive_failures,
        "source_last_failure_error_preserved": source.last_failure_error,
        "reslice": reslice,
    }
    return "kanban drain timeout/gave-up reslice:\n" + json.dumps(
        body,
        indent=2,
        sort_keys=True,
    )


def _build_superseded_duplicate_action(conn: sqlite3.Connection, task: kb.Task) -> dict[str, Any]:
    action_id = _superseded_duplicate_action_id(task)
    action = {
        "drain_action_id": action_id,
        "action": ARCHIVE_SUPERSEDED_DUPLICATE,
        "source_task_id": task.id,
        "status": "planned",
        "evidence": _superseded_duplicate_evidence(task),
    }
    metadata = task.block_metadata or {}
    canonical_task_id = _first_text(metadata, "canonical_task_id", "canonical", "superseded_by")
    if canonical_task_id:
        action["canonical_task_id"] = canonical_task_id

    if task.status == "archived" and _comment_exists(conn, task.id, action_id):
        action.update(status="already_applied")
        return action
    if task.status != "blocked":
        action.update(status="skipped", reason=f"source task is {task.status!r}, not 'blocked'")
        return action
    if task.block_class != "superseded_duplicate":
        action.update(status="refused", reason="source task lacks structured superseded_duplicate block_class")
        return action
    if not canonical_task_id:
        action.update(status="refused", reason="canonical task ID missing")
        return action
    if canonical_task_id == task.id:
        action.update(status="refused", reason="canonical task cannot be the source task")
        return action
    canonical = kb.get_task(conn, canonical_task_id)
    if canonical is None:
        action.update(status="refused", reason="canonical task not found")
        return action
    if canonical.status == "archived":
        action.update(status="refused", reason="canonical task is archived")
        return action
    canonical_evidence = _metadata_text_list(metadata.get("canonical_evidence"))
    duplicate_evidence = _metadata_text_list(metadata.get("duplicate_evidence"))
    if not canonical_evidence:
        action.update(status="refused", reason="canonical evidence missing")
        return action
    if not duplicate_evidence:
        action.update(status="refused", reason="duplicate evidence missing")
        return action
    unique_acceptance = _metadata_text_list(metadata.get("unique_acceptance"))
    if unique_acceptance:
        action.update(status="refused", reason="source has unique acceptance criteria")
        return action

    action["canonical_evidence"] = canonical_evidence
    action["duplicate_evidence"] = duplicate_evidence
    return action


def _apply_superseded_duplicate_action(
    conn: sqlite3.Connection,
    task: kb.Task,
    action: dict[str, Any],
) -> dict[str, Any]:
    current = kb.get_task(conn, task.id)
    if current is None:
        action.update(status="refused", reason="source task disappeared before apply")
        return action
    rebuilt = _build_superseded_duplicate_action(conn, current)
    if rebuilt["status"] != "planned":
        return rebuilt

    action_id = str(rebuilt["drain_action_id"])
    comment_existed = _comment_exists(conn, current.id, action_id)
    if not comment_existed:
        kb.add_comment(
            conn,
            current.id,
            DRAIN_AUTHOR,
            _superseded_duplicate_audit_comment(current, rebuilt),
        )
    archived = kb.archive_task(conn, current.id)
    if not archived:
        after_archive = kb.get_task(conn, current.id)
        if after_archive and after_archive.status == "archived" and _comment_exists(conn, current.id, action_id):
            rebuilt.update(status="already_applied")
            return rebuilt
        rebuilt.update(status="refused", reason="source task could not be archived")
        return rebuilt
    with kb.write_txn(conn):
        kb._append_event(
            conn,
            current.id,
            "drain_archived_superseded_duplicate",
            {
                "drain_action_id": action_id,
                "action": ARCHIVE_SUPERSEDED_DUPLICATE,
                "source_task_id": current.id,
                "canonical_task_id": rebuilt.get("canonical_task_id"),
                "canonical_evidence": rebuilt.get("canonical_evidence", []),
                "duplicate_evidence": rebuilt.get("duplicate_evidence", []),
            },
        )
    rebuilt.update(status="applied")
    return rebuilt


def _superseded_duplicate_action_id(task: kb.Task) -> str:
    metadata = task.block_metadata or {}
    payload = {
        "action": ARCHIVE_SUPERSEDED_DUPLICATE,
        "source_task_id": task.id,
        "canonical_task_id": metadata.get("canonical_task_id") or metadata.get("canonical") or metadata.get("superseded_by"),
        "canonical_evidence": metadata.get("canonical_evidence"),
        "duplicate_evidence": metadata.get("duplicate_evidence"),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:24]


def _superseded_duplicate_evidence(task: kb.Task) -> list[str]:
    metadata = task.block_metadata or {}
    evidence = [f"block_class:{task.block_class}"] if task.block_class else []
    canonical_task_id = metadata.get("canonical_task_id") or metadata.get("canonical") or metadata.get("superseded_by")
    if canonical_task_id:
        evidence.append("canonical_task_id")
    if metadata.get("canonical_evidence"):
        evidence.append("canonical_evidence")
    if metadata.get("duplicate_evidence"):
        evidence.append("duplicate_evidence")
    return evidence or ["superseded_duplicate_metadata"]


def _superseded_duplicate_audit_comment(task: kb.Task, action: dict[str, Any]) -> str:
    body = {
        "drain_action_id": action["drain_action_id"],
        "action": ARCHIVE_SUPERSEDED_DUPLICATE,
        "source_task_id": task.id,
        "canonical_task_id": action.get("canonical_task_id"),
        "canonical_evidence": action.get("canonical_evidence", []),
        "duplicate_evidence": action.get("duplicate_evidence", []),
        "unique_acceptance": [],
    }
    return "kanban drain superseded duplicate archive:\n" + json.dumps(
        body,
        indent=2,
        sort_keys=True,
    )


def _metadata_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _classify_task_for_drain(
    conn: sqlite3.Connection,
    task: kb.Task,
) -> DrainTaskClassification:
    parent_rows = _parent_status_rows(conn, task.id)
    nonterminal_parents = [
        row["parent_id"]
        for row in parent_rows
        if row["status"] not in TERMINAL_PARENT_STATUSES
    ]
    latest_block = _latest_block_reason(conn, task.id) or ""
    failure = task.last_failure_error or ""
    text = " ".join(part for part in [latest_block, failure, task.title] if part)
    evidence = _classification_evidence(task, latest_block, failure, nonterminal_parents)
    structured_class = task.block_class if task.block_class in kb.VALID_BLOCK_CLASSES else None

    inferred_class = "classification_debt"
    owner = "factory-orchestrator"
    eligible_action = "report_only"
    reason = "needs classification before drain"

    if task.status == "todo":
        if nonterminal_parents:
            inferred_class = "parent_gated"
            reason = "todo card has nonterminal parent dependencies"
        else:
            inferred_class = "todo_ready_candidate"
            reason = "todo card has no nonterminal parent dependency"
        return DrainTaskClassification(
            task_id=task.id,
            title=task.title,
            status=task.status,
            assignee=task.assignee,
            inferred_class=inferred_class,
            owner=owner,
            eligible_action=eligible_action,
            reason=reason,
            evidence=evidence,
            nonterminal_parents=nonterminal_parents,
        )

    if task.status == "review":
        return DrainTaskClassification(
            task_id=task.id,
            title=task.title,
            status=task.status,
            assignee=task.assignee,
            inferred_class="review_lane",
            owner="pr-reviewer",
            eligible_action="review_packet_required",
            reason="card is already in review lane",
            evidence=evidence,
            nonterminal_parents=nonterminal_parents,
        )

    hold_class = (
        structured_class
        if structured_class in {"human_hold", "credential_hold", "prod_risk_hold"}
        else _hold_class(text)
    )
    if hold_class:
        inferred_class = hold_class
        owner = "human/operator"
        reason = f"{hold_class} requires explicit non-agent authority"
    elif structured_class == "review_required" or latest_block.lower().startswith("review-required:"):
        inferred_class = "review_required"
        owner = "pr-reviewer"
        eligible_action = "route_to_review"
        reason = "review-required handoff can move to review after evidence check"
    elif latest_block.lower().startswith("changes-required:"):
        inferred_class = "changes_required"
        owner = "pr-reviewer"
        reason = "review failure needs actionable follow-up routing"
    elif "superseded" in text.lower():
        inferred_class = "superseded_duplicate"
        owner = "factory-orchestrator"
        reason = "possible superseded duplicate needs canonical evidence"
    elif structured_class == "superseded_duplicate":
        inferred_class = "superseded_duplicate"
        owner = "factory-orchestrator"
        reason = "superseded duplicate needs canonical evidence before archive"
    elif structured_class == "parent_gated" or "dependency-not-ready" in text.lower() or nonterminal_parents:
        inferred_class = "parent_gated"
        reason = "blocked by dependency or parent gate"
    elif structured_class == "timeout_gave_up":
        inferred_class = "timeout_gave_up"
        owner = "product-manager"
        reason = "timeout/gave-up work needs safe reslice classification"
    elif _SCOPE_TOO_LARGE_RE.search(text):
        inferred_class = "scope_too_large"
        owner = "product-manager"
        reason = "work likely exceeded worker envelope and needs reslicing"
    elif _RUNTIME_INFRA_RE.search(text):
        inferred_class = "runtime_infra"
        owner = "devops-engineer"
        reason = "worker/runtime evidence needs infrastructure classification"
    elif _SOURCE_OF_TRUTH_RE.search(text):
        inferred_class = "source_of_truth_gap"
        owner = "design-with-docs"
        reason = "missing source-of-truth or domain authority"
    elif _PRODUCT_SCOPE_RE.search(text):
        inferred_class = "product_scope"
        owner = "product-manager"
        reason = "product or acceptance decision needed"

    return DrainTaskClassification(
        task_id=task.id,
        title=task.title,
        status=task.status,
        assignee=task.assignee,
        inferred_class=inferred_class,
        owner=owner,
        eligible_action=eligible_action,
        reason=reason,
        evidence=evidence,
        nonterminal_parents=nonterminal_parents,
    )


def _hold_class(text: str) -> Optional[str]:
    if _CREDENTIAL_RE.search(text):
        return "credential_hold"
    if _PROD_RISK_RE.search(text):
        return "prod_risk_hold"
    if _HUMAN_HOLD_RE.search(text):
        return "human_hold"
    return None


def _classification_evidence(
    task: kb.Task,
    latest_block: str,
    failure: str,
    nonterminal_parents: list[str],
) -> list[str]:
    evidence: list[str] = []
    if latest_block:
        evidence.append("latest_block_reason")
    if task.block_class:
        evidence.append(f"block_class:{task.block_class}")
    if failure:
        evidence.append("last_failure_error")
    if task.consecutive_failures:
        evidence.append(f"consecutive_failures:{task.consecutive_failures}")
    if nonterminal_parents:
        evidence.append(f"nonterminal_parents:{len(nonterminal_parents)}")
    return evidence


def _completed_review_packets(conn: sqlite3.Connection) -> list[ReviewPacket | InvalidReviewPacket]:
    packets: list[ReviewPacket | InvalidReviewPacket] = []
    seen: set[tuple[str, str]] = set()

    done_tasks = conn.execute(
        "SELECT id FROM tasks WHERE status = 'done' ORDER BY completed_at ASC, id ASC"
    ).fetchall()
    for task_row in done_tasks:
        review_task_id = task_row["id"]
        for run in kb.list_runs(conn, review_task_id, include_active=False):
            if run.metadata:
                packet = _packet_from_dict(
                    run.metadata,
                    review_task_id=review_task_id,
                    evidence=f"run:{run.id}",
                    run_id=run.id,
                )
                if packet and _packet_seen_key(packet) not in seen:
                    packets.append(packet)
                    seen.add(_packet_seen_key(packet))
        for comment in kb.list_comments(conn, review_task_id):
            parsed = _json_from_text(comment.body)
            if parsed:
                packet = _packet_from_dict(
                    parsed,
                    review_task_id=review_task_id,
                    evidence=f"comment:{comment.id}",
                    comment_id=comment.id,
                )
                if packet and _packet_seen_key(packet) not in seen:
                    packets.append(packet)
                    seen.add(_packet_seen_key(packet))
    return packets


def _packet_from_dict(
    data: dict[str, Any],
    *,
    review_task_id: str,
    evidence: str,
    run_id: Optional[int] = None,
    comment_id: Optional[int] = None,
) -> ReviewPacket | InvalidReviewPacket | None:
    if data.get("drain_action") or data.get("drain_action_id"):
        return None

    packet_data = data
    nested = data.get("review_packet") or data.get("review_packet_metadata")
    if isinstance(nested, dict):
        packet_data = nested
    if not _looks_like_review_packet(data, packet_data, nested):
        return None

    source_task_id = _first_text(
        packet_data,
        "source_task_id",
        "source_card",
        "source_card_id",
        "source_task",
    )
    errors: list[str] = []
    if not source_task_id:
        errors.append("source card ID missing")

    approved = _approved_value(packet_data)
    if approved is None:
        errors.append("PASS or FAIL verdict missing")

    reviewer = _first_text(packet_data, "reviewer", "reviewer_identity", "reviewed_by")
    if not reviewer:
        errors.append("reviewer identity missing")

    evidence_raw = packet_data.get("evidence")
    evidence_data = evidence_raw if isinstance(evidence_raw, dict) else {}
    if not evidence_data:
        errors.append("review evidence object missing")
    if not _reviewed_target(evidence_data):
        errors.append("diff, branch, PR, or workspace path reviewed missing")
    if not _tests_or_checks_inspected(evidence_data):
        errors.append("tests or checks inspected missing")

    findings_raw, findings_present = _packet_findings(packet_data, evidence_data)
    findings: list[str]
    if isinstance(findings_raw, list):
        findings = [str(item) for item in findings_raw if str(item).strip()]
    else:
        findings = []
        if not findings_present:
            errors.append("findings list missing")
        else:
            errors.append("findings must be a list")

    authority_boundary = _first_text(packet_data, "authority_boundary", "authority_boundary_statement")
    if not authority_boundary:
        errors.append("authority-boundary statement missing")

    required_followups_raw = packet_data.get("required_followups")
    required_followups = (
        [str(item) for item in required_followups_raw if str(item).strip()]
        if isinstance(required_followups_raw, list)
        else []
    )

    if errors:
        return InvalidReviewPacket(
            review_task_id=review_task_id,
            source_task_id=source_task_id,
            evidence=evidence,
            reason="; ".join(errors),
            run_id=run_id,
            comment_id=comment_id,
        )

    return ReviewPacket(
        review_task_id=review_task_id,
        source_task_id=source_task_id or "",
        approved=bool(approved),
        evidence=evidence,
        reviewer=reviewer or "",
        review_evidence=dict(evidence_data),
        authority_boundary=authority_boundary or "",
        findings=findings,
        required_followups=required_followups,
        run_id=run_id,
        comment_id=comment_id,
    )


def _packet_seen_key(packet: ReviewPacket | InvalidReviewPacket) -> tuple[str, str]:
    if isinstance(packet, ReviewPacket):
        return (packet.review_task_id, packet.source_task_id)
    return (packet.review_task_id, packet.evidence)


def _looks_like_review_packet(
    original: dict[str, Any],
    packet_data: dict[str, Any],
    nested: Any,
) -> bool:
    if isinstance(nested, dict):
        return True
    keys = {
        "review_packet_version",
        "source_task_id",
        "source_card",
        "source_card_id",
        "source_task",
        "approved",
        "verdict",
        "reviewer",
        "reviewer_identity",
        "reviewed_by",
        "authority_boundary",
        "authority_boundary_statement",
        "required_followups",
    }
    return bool(keys.intersection(original) or keys.intersection(packet_data))


def _reviewed_target(evidence: dict[str, Any]) -> Optional[str]:
    return _first_text(
        evidence,
        "diff_path",
        "branch",
        "pr_url",
        "pr",
        "workspace_path",
        "reviewed_path",
        "path",
    )


def _tests_or_checks_inspected(evidence: dict[str, Any]) -> bool:
    for key in ("tests_run", "checks_inspected", "tests", "checks"):
        value = evidence.get(key)
        if isinstance(value, list) and any(str(item).strip() for item in value):
            return True
        if isinstance(value, str) and value.strip():
            return True
    return False


def _packet_findings(
    packet_data: dict[str, Any],
    evidence_data: dict[str, Any],
) -> tuple[Any, bool]:
    for source in (packet_data, evidence_data):
        for key in ("findings", "blocking_findings"):
            if key in source:
                return source.get(key), True
    return None, False


def _approved_value(data: dict[str, Any]) -> Optional[bool]:
    if "approved" in data:
        value = data.get("approved")
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "pass", "passed", "approved"}:
                return True
            if lowered in {"false", "no", "fail", "failed", "rejected"}:
                return False
    verdict = data.get("verdict")
    if isinstance(verdict, str):
        normalized = verdict.strip().upper()
        if normalized == "PASS":
            return True
        if normalized == "FAIL":
            return False
    return None


def _first_text(data: dict[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _json_from_text(body: str) -> Optional[dict[str, Any]]:
    stripped = body.strip()
    if not stripped:
        return None
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if 0 <= start < end:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _build_action(conn: sqlite3.Connection, packet: ReviewPacket) -> dict[str, Any]:
    action_name = REVIEW_PASS if packet.approved else REVIEW_FAIL
    action_id = _drain_action_id(packet, action_name)
    action = {
        "drain_action_id": action_id,
        "action": action_name,
        "source_task_id": packet.source_task_id,
        "review_task_id": packet.review_task_id,
        "approved": packet.approved,
        "reviewer": packet.reviewer,
        "evidence": [packet.evidence],
        "review_evidence": packet.review_evidence,
        "status": "planned",
    }
    source = kb.get_task(conn, packet.source_task_id)
    if source is None:
        action.update(status="refused", reason="source task not found")
        return action
    review = kb.get_task(conn, packet.review_task_id)
    if review is None or review.status != "done":
        action.update(status="refused", reason="review task is not completed")
        return action
    if not _parents_terminal(conn, packet.source_task_id):
        action.update(status="refused", reason="source task has unsatisfied parent dependencies")
        return action

    latest_block = _latest_block_reason(conn, packet.source_task_id)
    if source.status == "done" and _comment_exists(conn, packet.source_task_id, action_id):
        action.update(status="already_applied")
        return action
    if source.status != "blocked":
        action.update(status="skipped", reason=f"source task is {source.status!r}, not 'blocked'")
        return action
    latest_block_class = source.block_class
    if not latest_block and latest_block_class != "review_required":
        action.update(status="refused", reason="source task has no sticky block event")
        return action
    if latest_block_class in {"human_hold", "credential_hold", "prod_risk_hold"}:
        action.update(status="refused", reason="source task has explicit human/credential/prod-risk block_class")
        return action
    if latest_block_class != "review_required" and not latest_block.lower().startswith("review-required:"):
        action.update(status="refused", reason="latest source block is not review-required")
        return action
    if _HOLD_RE.search(latest_block):
        action.update(status="refused", reason="latest source block text looks like a human/credential/prod-risk hold")
        return action

    if packet.approved:
        if _comment_exists(conn, packet.source_task_id, action_id) and source.status == "done":
            action.update(status="already_applied")
    else:
        rework_key = _rework_idempotency_key(packet)
        action["rework_idempotency_key"] = rework_key
        existing = _find_task_by_idempotency_key(conn, rework_key)
        if existing:
            action["rework_task_id"] = existing
        if _comment_exists(conn, packet.source_task_id, action_id) and existing:
            action.update(status="already_applied")
    return action


def _build_invalid_packet_action(packet: InvalidReviewPacket) -> dict[str, Any]:
    payload = {
        "action": "consume_review_packet",
        "review_task_id": packet.review_task_id,
        "source_task_id": packet.source_task_id,
        "evidence": packet.evidence,
        "reason": packet.reason,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "drain_action_id": digest[:24],
        "action": "consume_review_packet",
        "source_task_id": packet.source_task_id,
        "review_task_id": packet.review_task_id,
        "evidence": [packet.evidence],
        "status": "refused",
        "reason": f"invalid review packet: {packet.reason}",
    }


def _apply_action(
    conn: sqlite3.Connection,
    packet: ReviewPacket,
    action: dict[str, Any],
) -> dict[str, Any]:
    action_id = str(action["drain_action_id"])
    source = kb.get_task(conn, packet.source_task_id)
    if source is None:
        action.update(status="refused", reason="source task disappeared before apply")
        return action

    if packet.approved:
        comment_existed = _comment_exists(conn, packet.source_task_id, action_id)
        if not comment_existed:
            kb.add_comment(
                conn,
                packet.source_task_id,
                DRAIN_AUTHOR,
                _audit_comment(packet, REVIEW_PASS, action_id),
            )
        completed = False
        current = kb.get_task(conn, packet.source_task_id)
        if current and current.status == "blocked":
            completed = kb.complete_task(
                conn,
                packet.source_task_id,
                summary=(
                    "Review-approved and drain-consumed: "
                    f"review task {packet.review_task_id} approved this source card."
                ),
                metadata={
                    "drain_action": REVIEW_PASS,
                    "drain_action_id": action_id,
                    "review_task_id": packet.review_task_id,
                    "review_evidence": [packet.evidence],
                    "review_packet_evidence": packet.review_evidence,
                    "reviewer": packet.reviewer,
                    "authority_boundary": packet.authority_boundary,
                    "findings": packet.findings,
                    "required_followups": packet.required_followups,
                    "source_task_id": packet.source_task_id,
                    "approved": True,
                },
            )
        if completed or not comment_existed:
            action.update(status="applied")
        else:
            action.update(status="already_applied")
        return action

    rework_key = _rework_idempotency_key(packet)
    existing_rework = _find_task_by_idempotency_key(conn, rework_key)
    comment_existed = _comment_exists(conn, packet.source_task_id, action_id)
    if existing_rework is None:
        rework_id = kb.create_task(
            conn,
            title=f"rework after review FAIL: {source.title}",
            body=_rework_body(source, packet, action_id),
            assignee=source.assignee,
            created_by=DRAIN_AUTHOR,
            priority=source.priority,
            idempotency_key=rework_key,
        )
    else:
        rework_id = existing_rework
    if not comment_existed:
        kb.add_comment(
            conn,
            packet.source_task_id,
            DRAIN_AUTHOR,
            _audit_comment(packet, REVIEW_FAIL, action_id, rework_task_id=rework_id),
        )
    action["rework_task_id"] = rework_id
    if comment_existed and existing_rework is not None:
        action.update(status="already_applied")
    else:
        action.update(status="applied")
    return action


def _latest_block_reason(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    row = conn.execute(
        "SELECT kind, payload FROM task_events "
        "WHERE task_id = ? AND kind IN ('blocked', 'unblocked') "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if not row or row["kind"] != "blocked":
        return None
    try:
        payload = json.loads(row["payload"]) if row["payload"] else {}
    except Exception:
        payload = {}
    reason = payload.get("reason") if isinstance(payload, dict) else None
    return reason if isinstance(reason, str) else ""


def _parents_terminal(conn: sqlite3.Connection, task_id: str) -> bool:
    return all(
        row["status"] in TERMINAL_PARENT_STATUSES
        for row in _parent_status_rows(conn, task_id)
    )


def _parent_status_rows(conn: sqlite3.Connection, task_id: str) -> list[sqlite3.Row]:
    rows = conn.execute(
        "SELECT t.id AS parent_id, t.status FROM tasks t "
        "JOIN task_links l ON l.parent_id = t.id "
        "WHERE l.child_id = ?",
        (task_id,),
    ).fetchall()
    return list(rows)


def _comment_exists(conn: sqlite3.Connection, task_id: str, marker: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM task_comments WHERE task_id = ? AND body LIKE ? LIMIT 1",
        (task_id, f"%{marker}%"),
    ).fetchone()
    return row is not None


def _find_task_by_idempotency_key(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute(
        "SELECT id FROM tasks WHERE idempotency_key = ? AND status != 'archived' "
        "ORDER BY created_at DESC, id DESC LIMIT 1",
        (key,),
    ).fetchone()
    return row["id"] if row else None


def _drain_action_id(packet: ReviewPacket, action_name: str) -> str:
    payload = {
        "action": action_name,
        "review_task_id": packet.review_task_id,
        "source_task_id": packet.source_task_id,
        "approved": packet.approved,
        "evidence": packet.evidence,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:24]


def _rework_idempotency_key(packet: ReviewPacket) -> str:
    return f"kanban-drain:review-fail:{packet.source_task_id}:{packet.review_task_id}"


def _audit_comment(
    packet: ReviewPacket,
    action_name: str,
    action_id: str,
    *,
    rework_task_id: Optional[str] = None,
) -> str:
    body = {
        "drain_action_id": action_id,
        "action": action_name,
        "source_task_id": packet.source_task_id,
        "review_task_id": packet.review_task_id,
        "approved": packet.approved,
        "reviewer": packet.reviewer,
        "evidence": [packet.evidence],
        "review_packet_evidence": packet.review_evidence,
        "authority_boundary": packet.authority_boundary,
        "findings": packet.findings,
        "required_followups": packet.required_followups,
    }
    if rework_task_id:
        body["rework_task_id"] = rework_task_id
    return "kanban drain review-packet consumption:\n" + json.dumps(
        body,
        indent=2,
        sort_keys=True,
    )


def _rework_body(source: kb.Task, packet: ReviewPacket, action_id: str) -> str:
    findings = "\n".join(f"- {finding}" for finding in packet.findings) or "- Review failed; see review task evidence."
    followups = "\n".join(f"- {followup}" for followup in packet.required_followups) or "- Address review findings."
    reviewed_target = _reviewed_target(packet.review_evidence) or "review evidence target missing"
    return (
        "Review Packet FAIL rework created by Kanban drain controller.\n\n"
        f"Source card: {packet.source_task_id}\n"
        f"Review task: {packet.review_task_id}\n"
        f"Reviewer: {packet.reviewer}\n"
        f"Reviewed target: {reviewed_target}\n"
        f"Drain action id: {action_id}\n\n"
        "Findings:\n"
        f"{findings}\n\n"
        "Required follow-ups:\n"
        f"{followups}\n\n"
        "Acceptance: address the review findings, run the relevant verification, "
        "and end with review-required for a fresh independent review."
    )
