from __future__ import annotations

import argparse
import json
import os
import re
import shlex
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from hermes_cli import kanban_db as kb
from hermes_constants import get_hermes_home

SECRET_PATTERNS = [
    re.compile(r"(?i)api[_-]?key\s*[:=]\s*\S+"),
    re.compile(r"(?i)token\s*[:=]\s*\S+"),
    re.compile(r"(?i)authorization\s*[:=]\s*\S+"),
    re.compile(r"(?i)bearer\s+[a-z0-9._\-]+"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]

DEFAULT_LINEAR_PENDING_REASON = (
    "Linear mirror unavailable or disabled; Kanban and local closeout artifacts remain the source of truth."
)


@dataclass(frozen=True)
class CloseThreadOptions:
    invoked_as: str
    mode: str
    scope: str
    task_id: str | None
    include_memory: bool
    post_human_review: bool


class CloseThreadError(ValueError):
    pass


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "close-thread"


def _redact_text(value: str) -> str:
    redacted = value
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    if isinstance(value, tuple):
        return [_redact(v) for v in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _thread_labels_from_source(source: Any) -> list[dict[str, Any]]:
    """Return machine-readable Discord thread labels from source metadata/name."""
    if not isinstance(source, dict):
        return []
    labels = source.get("thread_labels")
    if isinstance(labels, list):
        return _redact(labels)
    try:
        from importlib import import_module

        extract_thread_labels = import_module(
            "plugins.platforms.discord.thread_labels"
        ).extract_thread_labels
        return _redact(extract_thread_labels(source.get("thread_name")))
    except Exception:
        return []


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="/close-thread", add_help=False)
    parser.add_argument("task_id_arg", nargs="?")
    parser.add_argument("--mode", choices=("dry_run", "close"), default="dry_run")
    parser.add_argument(
        "--scope",
        choices=("current_thread", "current_channel_session", "referenced_task"),
        default="current_thread",
    )
    parser.add_argument("--task-id", dest="task_id_flag")
    parser.add_argument("--include-memory", dest="include_memory", action="store_true", default=True)
    parser.add_argument("--no-include-memory", dest="include_memory", action="store_false")
    parser.add_argument("--post-human-review", dest="post_human_review", action="store_true", default=True)
    parser.add_argument("--no-post-human-review", dest="post_human_review", action="store_false")
    return parser


def parse_close_thread_command(command_text: str) -> CloseThreadOptions:
    raw = (command_text or "").strip()
    if not raw:
        raw = "/close-thread"
    parts = shlex.split(raw)
    if not parts:
        parts = ["/close-thread"]
    invoked_as = parts[0]
    if invoked_as not in ("/close-thread", "/close-tread", "close-thread", "close-tread"):
        raise CloseThreadError(f"unsupported close-thread command: {invoked_as}")
    parser = _build_parser()
    try:
        args = parser.parse_args(parts[1:])
    except SystemExit as exc:
        raise CloseThreadError("invalid /close-thread arguments") from exc
    task_id = args.task_id_flag or args.task_id_arg or os.getenv("HERMES_KANBAN_TASK") or None
    return CloseThreadOptions(
        invoked_as=invoked_as if invoked_as.startswith("/") else f"/{invoked_as}",
        mode=args.mode,
        scope=args.scope,
        task_id=task_id,
        include_memory=bool(args.include_memory),
        post_human_review=bool(args.post_human_review),
    )


def closeout_root() -> Path:
    configured = os.getenv("FOUNDATION_DISCORD_OFFICE_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return get_hermes_home() / "workspaces" / "business-os" / "01-command-center" / "foundation-discord-office"


def _closeout_dir(now: datetime | None = None) -> Path:
    current = now or datetime.now().astimezone()
    return closeout_root() / "closeouts" / current.strftime("%Y-%m-%d")


def _task_ref(task: kb.Task | None) -> dict[str, Any] | None:
    if not task:
        return None
    return {
        "id": task.id,
        "title": _redact_text(task.title),
        "status": task.status,
        "assignee": task.assignee,
        "priority": task.priority,
        "workspace_kind": task.workspace_kind,
        "workspace_path": task.workspace_path,
        "tenant": task.tenant,
    }


def _run_ref(run: kb.Run) -> dict[str, Any]:
    return {
        "id": run.id,
        "status": run.status,
        "outcome": run.outcome,
        "profile": run.profile,
        "started_at": run.started_at,
        "ended_at": run.ended_at,
        "summary": _redact(run.summary),
        "metadata": _redact(deepcopy(run.metadata)),
        "error": _redact(run.error),
    }


def _comment_ref(comment: kb.Comment) -> dict[str, Any]:
    return {
        "id": comment.id,
        "author": comment.author,
        "created_at": comment.created_at,
        "body": _redact_text(comment.body),
    }


def _event_ref(event: kb.Event) -> dict[str, Any]:
    return {
        "id": event.id,
        "kind": event.kind,
        "created_at": event.created_at,
        "run_id": event.run_id,
        "payload": _redact(deepcopy(event.payload)),
    }


def _extract_decisions(task: kb.Task | None, comments: list[kb.Comment], runs: list[kb.Run]) -> list[str]:
    decisions: list[str] = []
    if task and task.assignee:
        decisions.append(f"Task remains assigned to {task.assignee} with durable status {task.status}.")
    for run in reversed(runs):
        if run.summary:
            decisions.append(_redact_text(run.summary.strip().splitlines()[0]))
        if len(decisions) >= 3:
            break
    for comment in reversed(comments):
        lowered = comment.body.lower()
        if any(word in lowered for word in ("decision", "decided", "approved", "reject", "accept")):
            decisions.append(_redact_text(comment.body.strip().splitlines()[0]))
        if len(decisions) >= 5:
            break
    out: list[str] = []
    seen: set[str] = set()
    for item in decisions:
        cleaned = item.strip()
        if cleaned and cleaned not in seen:
            out.append(cleaned)
            seen.add(cleaned)
    return out


def _extract_actions(task: kb.Task | None, children: list[kb.Task], runs: list[kb.Run]) -> list[str]:
    actions: list[str] = []
    if task:
        actions.append(f"Durable task anchor {task.id} reviewed with status {task.status}.")
    for child in children[:5]:
        actions.append(f"Linked child task {child.id} ({child.status}) remains tracked under Kanban.")
    for run in reversed(runs):
        if run.outcome == "completed" and run.summary:
            actions.append(_redact_text(run.summary.strip()))
        if len(actions) >= 6:
            break
    return actions[:6]


def _extract_risks(task: kb.Task | None, comments: list[kb.Comment], latest_run: kb.Run | None) -> list[str]:
    risks: list[str] = []
    if task and task.status not in {"done", "archived"}:
        risks.append(f"Task {task.id} is not complete (status={task.status}).")
    if latest_run and latest_run.outcome in {"blocked", "crashed", "timed_out", "spawn_failed"}:
        label = latest_run.error or latest_run.summary or latest_run.outcome
        risks.append(_redact_text(str(label)))
    for comment in reversed(comments):
        lowered = comment.body.lower()
        if any(word in lowered for word in ("risk", "block", "waiting", "todo", "follow-up")):
            risks.append(_redact_text(comment.body.strip().splitlines()[0]))
        if len(risks) >= 5:
            break
    out: list[str] = []
    seen: set[str] = set()
    for item in risks:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _score_closeout(
    *,
    has_anchor: bool,
    decisions: list[str],
    task: kb.Task | None,
    blockers: list[dict[str, Any]],
    evidence_count: int,
    human_review_count: int,
    include_memory: bool,
) -> tuple[dict[str, int], int, str]:
    dimensions = {
        "durable_anchor": 2 if has_anchor else 0,
        "decisions": 2 if decisions else 0,
        "tasks": 2 if task else 0,
        "blockers": 2 if blockers else (1 if task and task.status in {"done", "archived"} else 0),
        "evidence": 2 if evidence_count >= 2 else (1 if evidence_count == 1 else 0),
        "human_review": 2 if human_review_count == 0 else 1,
        "linear": 1,
        "memory": 2 if include_memory else 1,
        "safety": 2,
        "human_response": 2,
    }
    total = sum(dimensions.values())
    blocked = (not has_anchor) or human_review_count > 0
    if blocked:
        status = "BLOCKED"
    elif total >= 18:
        status = "GREEN"
    elif total >= 14:
        status = "YELLOW"
    elif total >= 8:
        status = "RED"
    else:
        status = "BLOCKED"
    return dimensions, total, status


def _write_packet_files(packet: dict[str, Any], response_text: str, slug: str, now: datetime) -> tuple[str, str]:
    out_dir = _closeout_dir(now)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%dT%H%M%S%z")
    json_path = out_dir / f"close-thread-{slug}-{stamp}.json"
    md_path = out_dir / f"close-thread-{slug}-{stamp}.md"
    json_path.write_text(json.dumps(packet, indent=2, ensure_ascii=False) + "\n")
    md_lines = [
        "# CLOSE_THREAD_PACKET",
        "",
        f"Status: {packet['status']}",
        f"Score: {packet['score']['total']}/{packet['score']['max']}",
        f"Durable anchor: {response_text.splitlines()[3].split(': ', 1)[1] if len(response_text.splitlines()) > 3 else 'unknown'}",
        "",
        "## Summary",
        packet["summary"]["closeout_summary"],
        "",
        "## Decisions",
    ]
    decisions = packet["summary"].get("decisions") or []
    md_lines.extend([f"- {item}" for item in decisions] or ["- none"])
    md_lines.extend([
        "",
        "## Actions completed",
    ])
    actions = packet["summary"].get("actions_completed") or []
    md_lines.extend([f"- {item}" for item in actions] or ["- none"])
    md_lines.extend([
        "",
        "## Unresolved risks",
    ])
    risks = packet["summary"].get("unresolved_risks") or []
    md_lines.extend([f"- {item}" for item in risks] or ["- none"])
    md_lines.extend([
        "",
        "## Human response",
        response_text,
        "",
        "## Artifact paths",
        f"- {json_path}",
        f"- {md_path}",
    ])
    md_path.write_text("\n".join(md_lines) + "\n")
    return str(json_path), str(md_path)


def build_close_thread_packet(
    command_text: str,
    *,
    source: dict[str, Any] | None = None,
    session: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> tuple[dict[str, Any], str]:
    opts = parse_close_thread_command(command_text)
    now_dt = now or datetime.now().astimezone()
    invoked_at = now_dt.isoformat()
    source = _redact(deepcopy(source or {}))
    session = _redact(deepcopy(session or {}))

    kb.init_db()
    conn = kb.connect()
    try:
        task = kb.get_task(conn, opts.task_id) if opts.task_id else None
        comments = kb.list_comments(conn, task.id) if task else []
        events = kb.list_events(conn, task.id) if task else []
        runs = kb.list_runs(conn, task.id) if task else []
        latest_run = kb.latest_run(conn, task.id) if task else None
        parent_tasks = [kb.get_task(conn, pid) for pid in (kb.parent_ids(conn, task.id) if task else [])]
        child_tasks = [kb.get_task(conn, cid) for cid in (kb.child_ids(conn, task.id) if task else [])]
    finally:
        conn.close()

    parent_tasks = [item for item in parent_tasks if item]
    child_tasks = [item for item in child_tasks if item]

    has_anchor = task is not None
    decisions = _extract_decisions(task, comments, runs)
    actions = _extract_actions(task, child_tasks, runs)
    risks = _extract_risks(task, comments, latest_run)
    blocker_refs: list[dict[str, Any]] = []
    if task and task.status == "blocked":
        blocker_refs.append({
            "task_id": task.id,
            "status": task.status,
            "reason": _redact_text((latest_run.summary or latest_run.error or task.result or "Task is blocked").strip()),
        })
    human_review_packets = {"emitted": [], "queued": [], "not_emitted_reason": None}
    if not has_anchor:
        human_review_packets["queued"].append({
            "packet_type": "BLOCKED_NEEDS_HUMAN",
            "requested_action": "provide durable task/session anchor for /close-thread",
            "source_task": opts.task_id,
            "delivery": "queued_or_blocked" if opts.post_human_review else "suppressed",
        })
    elif opts.mode == "close":
        human_review_packets["queued"].append({
            "packet_type": "APPROVAL_NEEDED",
            "requested_action": "approve any live Discord archive/lock behavior before enabling close mode mutations",
            "source_task": task.id,
            "delivery": "queued_or_blocked" if opts.post_human_review else "suppressed",
        })
    elif not opts.post_human_review:
        human_review_packets["not_emitted_reason"] = "post_human_review=false"

    evidence_paths: list[str] = []
    if task and task.workspace_path:
        evidence_paths.append(task.workspace_path)
    tests_or_smokes = []
    for run in runs:
        meta = run.metadata or {}
        if isinstance(meta, dict):
            if meta.get("tests_run") is not None:
                tests_or_smokes.append({
                    "run_id": run.id,
                    "tests_run": meta.get("tests_run"),
                    "tests_passed": meta.get("tests_passed"),
                })
            changed = meta.get("changed_files")
            if isinstance(changed, list):
                evidence_paths.extend(str(item) for item in changed[:10])

    dimensions, total_score, status = _score_closeout(
        has_anchor=has_anchor,
        decisions=decisions,
        task=task,
        blockers=blocker_refs,
        evidence_count=len(evidence_paths) + len(tests_or_smokes),
        human_review_count=len(human_review_packets["emitted"]) + len(human_review_packets["queued"]),
        include_memory=opts.include_memory,
    )

    if has_anchor:
        durable_anchor = f"Kanban {task.id}"
    elif opts.task_id:
        durable_anchor = f"missing task {opts.task_id}"
    else:
        durable_anchor = "missing durable anchor"

    canonical_note = ""
    if opts.invoked_as == "/close-tread":
        canonical_note = " Alias /close-tread accepted; canonical command is /close-thread."

    closeout_summary = (
        f"Reviewed durable state for {durable_anchor}. "
        f"Captured {len(decisions)} decision(s), {len(actions)} action summary item(s), and {len(risks)} unresolved risk(s)."
        f"{canonical_note}"
    ).strip()

    thread_labels = _thread_labels_from_source(source)
    thread_label_ids = [
        label["id"]
        for label in thread_labels
        if isinstance(label, dict) and isinstance(label.get("id"), str)
    ]

    packet: dict[str, Any] = {
        "schema_version": "CLOSE_THREAD_PACKET.v1",
        "command": {
            "canonical": "/close-thread",
            "invoked_as": opts.invoked_as,
            "mode": opts.mode,
            "scope": opts.scope,
            "requested_by": source.get("requested_by") or source.get("user_id") or "[REDACTED]",
            "invoked_at": invoked_at,
        },
        "status": status,
        "score": {
            "total": total_score,
            "max": 20,
            "dimensions": dimensions,
        },
        "source_refs": {
            "discord": {
                "guild_id": source.get("guild_id"),
                "guild_name": source.get("guild_name"),
                "channel_id": source.get("channel_id"),
                "channel_name": source.get("channel_name"),
                "thread_id": source.get("thread_id"),
                "thread_name": source.get("thread_name"),
                "thread_labels": thread_labels,
                "message_id": source.get("message_id"),
            },
            "hermes": {
                "run_id": session.get("run_id"),
                "profile": session.get("profile"),
                "command_surface": session.get("command_surface") or "gateway",
                "request_id": session.get("request_id"),
            },
            "kanban": {
                "task_ids": [task.id] if task else ([opts.task_id] if opts.task_id else []),
                "run_ids": [run.id for run in runs],
                "status": task.status if task else None,
                "parents": [_task_ref(item) for item in parent_tasks],
                "children": [_task_ref(item) for item in child_tasks],
            },
            "linear": {
                "issues": [],
                "mirror_status": "pending",
                "pending_reason": DEFAULT_LINEAR_PENDING_REASON,
                "linear_mirror_pending": True,
            },
        },
        "summary": {
            "human_request": _redact_text(command_text.strip() or "/close-thread"),
            "closeout_summary": closeout_summary,
            "decisions": decisions,
            "actions_completed": actions,
            "unresolved_risks": risks,
        },
        "machine_metadata": {
            "thread_label_ids": thread_label_ids,
            "thread_labels": thread_labels,
        },
        "tasks": {
            "created": [],
            "updated": [],
            "existing_relevant": [_task_ref(item) for item in ([task] if task else []) + child_tasks],
        },
        "blockers": {
            "created": [],
            "updated": [],
            "remaining": blocker_refs,
        },
        "human_review_packets": human_review_packets,
        "evidence": {
            "artifact_paths": sorted(dict.fromkeys(evidence_paths)),
            "tests_or_smokes": _redact(tests_or_smokes),
            "reviewer_refs": [],
            "missing_evidence": [] if evidence_paths or tests_or_smokes else ["No changed_files or test metadata recorded on related runs."],
        },
        "memory": {
            "updates": [] if opts.include_memory else [],
            "skipped": [] if opts.include_memory else ["include_memory=false"],
            "memory_refs": [] if opts.include_memory else [],
        },
        "watchers_or_crons": {
            "created_or_updated": [],
            "missing_or_required": [],
        },
        "safety": {
            "secrets_redacted": True,
            "public_discord_mutations": [],
            "gateway_restart_requested": False,
            "approval_boundaries_triggered": [
                packet["packet_type"]
                for packet in human_review_packets["queued"]
                if packet.get("packet_type") == "APPROVAL_NEEDED"
            ],
        },
        "next_action": {
            "owner_profile": task.assignee if task else "human",
            "action": (
                "Provide the correct task/session anchor to rerun /close-thread."
                if not has_anchor
                else (
                    "Review the queued approval packet before enabling live close behavior."
                    if opts.mode == "close"
                    else "Review the closeout artifact and proceed with any remaining Kanban follow-up."
                )
            ),
            "due_by": None,
        },
        "closeout_artifacts": {"json_path": None, "markdown_path": None},
        "kanban_refs": {
            "task": _task_ref(task),
            "comments": [_comment_ref(comment) for comment in comments[-10:]],
            "events": [_event_ref(event) for event in events[-10:]],
            "runs": [_run_ref(run) for run in runs[-5:]],
        },
        "linear_refs": {
            "issues": [],
            "linear_mirror_pending": True,
            "pending_reason": DEFAULT_LINEAR_PENDING_REASON,
        },
        "artifact_refs": sorted(dict.fromkeys(evidence_paths)),
        "memory_refs": [],
        "missing_durable_anchor": not has_anchor,
    }

    response_lines = [
        "CLOSE_THREAD_PACKET",
        f"Status: {status}",
        f"Score: {total_score}/20",
        f"Durable anchor: {durable_anchor}",
        f"Decisions: {len(decisions)}",
        "Tasks created/updated: none created; existing tracked in packet",
        f"Blockers: {len(blocker_refs)}",
        f"Evidence: {len(packet['evidence']['artifact_paths'])} artifact ref(s), {len(packet['evidence']['tests_or_smokes'])} test/smoke ref(s)",
        (
            "Human review: none"
            if not human_review_packets["queued"] and not human_review_packets["emitted"]
            else f"Human review: {len(human_review_packets['emitted']) + len(human_review_packets['queued'])} packet(s) queued"
        ),
        f"Next owner/action: {packet['next_action']['owner_profile']} — {packet['next_action']['action']}",
        "Artifact: pending write",
        "Safety: no secrets; no unapproved Discord/gateway mutation",
    ]
    response_text = "\n".join(response_lines)
    slug = _slugify(task.id if task else (opts.task_id or "unanchored"))
    json_path, md_path = _write_packet_files(packet, response_text, slug, now_dt)
    packet["closeout_artifacts"] = {"json_path": json_path, "markdown_path": md_path}
    response_lines[10] = f"Artifact: {json_path}"
    response_text = "\n".join(response_lines)
    Path(md_path).write_text(
        Path(md_path)
        .read_text(encoding="utf-8")
        .replace("Artifact: pending write", f"Artifact: {json_path}"),
        encoding="utf-8",
    )
    Path(json_path).write_text(json.dumps(packet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return packet, response_text


def run_close_thread(
    command_text: str,
    *,
    source: dict[str, Any] | None = None,
    session: dict[str, Any] | None = None,
) -> str:
    _packet, response = build_close_thread_packet(command_text, source=source, session=session)
    return response
