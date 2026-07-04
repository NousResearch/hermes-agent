"""Kanban auxiliary coordinator launcher.

This module is deliberately a thin edge helper on top of the existing Kanban
kernel.  It does not add a second dispatcher or spawn path; it writes one
idempotent coordinator card that the normal gateway dispatcher can pick up as a
regular Hermes profile-lane task.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Iterable, Optional

from hermes_cli import kanban_db as kb

DEFAULT_AUX_ASSIGNEE_CANDIDATES = ("kanban-aux", "autopilot", "default")
AUX_CREATED_BY = "kanban-aux-launcher"
DEFAULT_MAX_CREATED_CARDS = 3
DEFAULT_LANE = "coordination"


@dataclass(frozen=True)
class EvidencePointer:
    path: Optional[Path]
    sha256: Optional[str]

    def as_dict(self) -> dict[str, Optional[str]]:
        return {
            "path": str(self.path) if self.path else None,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class AuxiliaryLaunch:
    """Result returned by :func:`launch_auxiliary_agent`."""

    task_id: Optional[str]
    title: str
    status: str
    assignee: str
    idempotency_key: str
    reused: bool
    dry_run: bool
    round_id: str
    source_task_id: Optional[str]
    parent_ids: list[str]
    evidence: EvidencePointer
    steward_plan_path: Optional[str]
    body: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "status": self.status,
            "assignee": self.assignee,
            "idempotency_key": self.idempotency_key,
            "reused": self.reused,
            "dry_run": self.dry_run,
            "round_id": self.round_id,
            "source_task_id": self.source_task_id,
            "parent_ids": list(self.parent_ids),
            "evidence": self.evidence.as_dict(),
            "steward_plan_path": self.steward_plan_path,
            "body": self.body,
        }


def _compact_slug(value: Optional[str], fallback: str) -> str:
    text = (value or "").strip()
    if not text:
        return fallback
    base = Path(text).name if any(sep in text for sep in ("/", "\\")) else text
    slug = re.sub(r"[^a-z0-9]+", "-", base.casefold()).strip("-") or fallback
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"{slug[:48]}-{digest}"


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in values:
        value = (raw or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _parent_context_key(parent_ids: Iterable[str]) -> str:
    parents = _dedupe_preserve_order(parent_ids)
    if not parents:
        return "root"
    if len(parents) == 1:
        return parents[0]
    canonical = "\n".join(sorted(parents))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"parents-{digest}"


def build_auxiliary_idempotency_key(
    *,
    source_task_id: Optional[str] = None,
    parent_ids: Iterable[str] = (),
    project: Optional[str] = None,
    micro_goal: Optional[str] = None,
    lane: str = DEFAULT_LANE,
) -> str:
    """Return the stable key used to prevent duplicate aux-card launches.

    ``--source-task`` is context/comment metadata only.  Dependency context is
    the first key segment so identical project/micro-goal launches under
    different parents produce distinct coordinator cards.
    """

    _ = source_task_id
    parent_context = _parent_context_key(parent_ids)
    project_key = _compact_slug(project, "none")
    goal_key = _compact_slug(micro_goal, "board-triage")
    lane_key = _compact_slug(lane, DEFAULT_LANE)
    return f"kanban-aux:{parent_context}:{project_key}:{lane_key}:{goal_key}:v1"


def _existing_task_id_for_key(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute(
        "SELECT id FROM tasks WHERE idempotency_key = ? "
        "AND status != 'archived' ORDER BY created_at DESC LIMIT 1",
        (key,),
    ).fetchone()
    return str(row["id"]) if row else None


def _select_assignee(requested: Optional[str]) -> str:
    available = set(kb.list_profiles_on_disk())
    if requested:
        candidate = requested.strip()
        if not candidate:
            raise ValueError("--assignee requires a non-empty profile name")
        if candidate not in available:
            known = ", ".join(sorted(available)) or "(none)"
            raise ValueError(
                f"auxiliary assignee profile {candidate!r} does not exist on disk; "
                f"known profiles: {known}"
            )
        return candidate

    for candidate in DEFAULT_AUX_ASSIGNEE_CANDIDATES:
        if candidate in available:
            return candidate
    known = ", ".join(sorted(available)) or "(none)"
    raise ValueError(
        "no auxiliary assignee profile is available. Create or choose an existing "
        "profile with `--assignee <profile>`; known profiles: " + known
    )


def _default_evidence_root() -> Path:
    """Pick a local evidence root without creating user-specific surface.

    Gustavo's `.omo` workspace is preferred when present because the auxiliary
    agent spec stores detailed evidence there.  Generic installs fall back to
    the Hermes home so the command remains useful outside that workspace.
    """

    omo = Path.home() / ".omo"
    if omo.is_dir():
        return omo / "evidence" / "kanban-aux"
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home() / "kanban" / "evidence" / "kanban-aux"
    except Exception:
        return Path.home() / ".hermes" / "kanban" / "evidence" / "kanban-aux"


def _default_steward_plan_path() -> Optional[Path]:
    candidate = Path.home() / ".omo" / "plans" / "hermes-project-steward-runtime.md"
    return candidate if candidate.is_file() else None


def _resolve_steward_plan(
    *,
    steward_plan: Optional[str],
    include_steward_plan: bool,
) -> Optional[Path]:
    if not include_steward_plan:
        return None
    if steward_plan:
        path = Path(steward_plan).expanduser()
        if not path.is_file():
            raise ValueError(f"steward plan not found: {path}")
        return path
    return _default_steward_plan_path()


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _round_id(key: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    nonce = hashlib.sha256(f"{key}:{time.time_ns()}".encode("utf-8")).hexdigest()[:8]
    return f"kbaux-{stamp}-{nonce}"


def build_auxiliary_task_body(
    *,
    source_task_id: Optional[str],
    project: Optional[str],
    micro_goal: Optional[str],
    max_created_cards: int,
    evidence_root: Path,
    round_id: str,
    idempotency_key: str,
    steward_plan_path: Optional[Path],
) -> str:
    goal = (micro_goal or "board triage / evidence cleanup").strip()
    source = source_task_id or "board-scan"
    project_text = project or "not specified"
    steward_section = ""
    if steward_plan_path:
        steward_section = f"""
## Project Steward/.omo integration
Detected plan: `{steward_plan_path}`.

Explicit pending work to preserve while routing:
- MVP remains local-only, foreground, one selected project/fixture, one card, one micro-goal, one serial cycle.
- Do not create new profiles/skills for the Steward MVP; use existing profile lanes and Kanban summary-level state.
- Detailed evidence belongs under `.omo/evidence/hermes-steward/` or `.omo/evidence/kanban-aux/` by local path + sha256, not pasted into Kanban.
- Pending Steward artifacts are operator runbook, `steward doctor/run/status/tail/stop/resume/qa-final` command contract, append-only state, project lock, evidence manifest, optional visual manifest, review pack, and final executable QA gate.
- If this board only has a single Steward MVP card/micro-goal, record `no_action_needed` instead of fanning out.
""".strip()
    else:
        steward_section = """
## Optional .omo/Steward integration
No `~/.omo/plans/hermes-project-steward-runtime.md` plan was detected or supplied. Run the generic Kanban auxiliary pass only; keep evidence local and summary-level.
""".strip()

    return f"""
Kanban auxiliary coordination pass. This card is a regular Hermes profile-lane task; the existing gateway dispatcher spawns it. It is not a new dispatcher, daemon, cron job, or recursive agent factory.

## Inputs
- source: `{source}`
- project/fixture: `{project_text}`
- micro-goal: `{goal}`
- round id: `{round_id}`
- launcher idempotency key: `{idempotency_key}`
- evidence root: `{evidence_root}`

## Operating contract
1. Call `kanban_show()` first and read parent handoffs/comments.
2. Run exactly one coordination pass: snapshot board pressure, dedupe similar cards, repair obvious parent links, and create only small follow-up cards with clear acceptance/evidence.
3. Hard cap: create at most {max_created_cards} new cards in this pass. If more are needed, complete/block with a prioritized recommendation instead of launching another auxiliary card.
4. Use existing profile assignees only. Block if the target profile does not exist or the next step requires credentials, 2FA/captcha, production mutation, external egress, destructive operations, or a human scope decision.
5. Do not call `hermes kanban aux`, do not create another auxiliary coordinator card, and do not create profiles, skills, cron jobs, gateways, or a Steward foreground runner from this pass.
6. Use deterministic child idempotency keys shaped like `kanban-aux:<parent_id|root>:<project>:<lane>:<normalized_micro_goal>:v1`.
7. Keep raw stdout, screenshots, browser payloads, tokens, cookies, PHI, and full logs out of Kanban. Store detailed evidence locally and cite only path + sha256 + run_id.

{steward_section}

## Completion contract
Complete with metadata containing `created_task_ids`, `reused_task_ids`, `dedupe_decisions`, `evidence_manifest_path`, `evidence_sha256`, and `no_action_needed` when applicable. Block with a specific one-line reason if a human decision is genuinely required.
""".strip()


def _write_evidence(
    *,
    evidence_root: Path,
    round_id: str,
    event: dict[str, Any],
) -> EvidencePointer:
    round_dir = evidence_root.expanduser() / round_id
    round_dir.mkdir(parents=True, exist_ok=True)
    path = round_dir / "events.jsonl"
    path.write_text(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return EvidencePointer(path=path, sha256=digest)


def _launch_comment(result: AuxiliaryLaunch) -> str:
    action = "reused" if result.reused else "created"
    evidence = result.evidence.as_dict()
    return (
        f"kanban-aux {result.round_id}: {action} auxiliary coordinator\n"
        f"- source: {result.source_task_id or 'board-scan'}\n"
        f"- task: {result.task_id}\n"
        f"- assignee: {result.assignee}\n"
        f"- idempotency_key: {result.idempotency_key}\n"
        f"- evidence: {evidence.get('path')}, sha256={evidence.get('sha256')}\n"
        "- omitted: raw stdout/screenshots/secrets/PHI kept out of Kanban\n"
        "- next: dispatcher runs the regular Hermes profile lane; do not launch a recursive aux card"
    )


def launch_auxiliary_agent(
    conn: sqlite3.Connection,
    *,
    source_task_id: Optional[str] = None,
    project: Optional[str] = None,
    micro_goal: Optional[str] = None,
    assignee: Optional[str] = None,
    parents: Iterable[str] = (),
    tenant: Optional[str] = None,
    priority: int = 0,
    max_runtime_seconds: Optional[int] = None,
    max_created_cards: int = DEFAULT_MAX_CREATED_CARDS,
    evidence_root: Optional[str | Path] = None,
    steward_plan: Optional[str] = None,
    include_steward_plan: bool = True,
    created_by: str = AUX_CREATED_BY,
    idempotency_key: Optional[str] = None,
    dry_run: bool = False,
) -> AuxiliaryLaunch:
    if max_created_cards < 0:
        raise ValueError("--max-created must be >= 0")

    source_task_id = (source_task_id or "").strip() or None
    if source_task_id and not kb.get_task(conn, source_task_id):
        raise ValueError(f"unknown source task: {source_task_id}")

    parent_ids = _dedupe_preserve_order(parents)
    for parent_id in parent_ids:
        if not kb.get_task(conn, parent_id):
            raise ValueError(f"unknown parent task: {parent_id}")

    chosen_assignee = _select_assignee(assignee)
    evidence_root_path = Path(evidence_root).expanduser() if evidence_root else _default_evidence_root()
    steward_plan_path = _resolve_steward_plan(
        steward_plan=steward_plan,
        include_steward_plan=include_steward_plan,
    )
    key = idempotency_key or build_auxiliary_idempotency_key(
        source_task_id=source_task_id,
        parent_ids=parent_ids,
        project=project,
        micro_goal=micro_goal,
    )
    round_id = _round_id(key)
    title_goal = (micro_goal or "board triage").strip().splitlines()[0]
    title = f"Kanban auxiliary: {title_goal[:80]}"
    body = build_auxiliary_task_body(
        source_task_id=source_task_id,
        project=project,
        micro_goal=micro_goal,
        max_created_cards=max_created_cards,
        evidence_root=evidence_root_path,
        round_id=round_id,
        idempotency_key=key,
        steward_plan_path=steward_plan_path,
    )

    existing_id = _existing_task_id_for_key(conn, key)
    if dry_run:
        dry_parent_ids = kb.parent_ids(conn, existing_id) if existing_id else parent_ids
        return AuxiliaryLaunch(
            task_id=existing_id,
            title=title,
            status="dry-run",
            assignee=chosen_assignee,
            idempotency_key=key,
            reused=bool(existing_id),
            dry_run=True,
            round_id=round_id,
            source_task_id=source_task_id,
            parent_ids=dry_parent_ids,
            evidence=EvidencePointer(path=evidence_root_path / round_id / "events.jsonl", sha256=None),
            steward_plan_path=str(steward_plan_path) if steward_plan_path else None,
            body=body,
        )

    task_id = kb.create_task(
        conn,
        title=title,
        body=body,
        assignee=chosen_assignee,
        created_by=created_by or AUX_CREATED_BY,
        tenant=tenant,
        priority=priority,
        parents=parent_ids,
        idempotency_key=key,
        max_runtime_seconds=max_runtime_seconds,
        skills=["kanban-orchestrator"],
    )
    task = kb.get_task(conn, task_id)
    if not task:
        raise RuntimeError(f"created auxiliary task disappeared: {task_id}")
    reused = bool(existing_id and existing_id == task_id)
    actual_parent_ids = kb.parent_ids(conn, task_id)

    event = {
        "ts": _iso_now(),
        "agent": "kanban-aux-launcher",
        "round_id": round_id,
        "source_task_id": source_task_id,
        "event": "launch",
        "decision": "reuse_auxiliary_task" if reused else "create_auxiliary_task",
        "reason": "idempotent launcher for one bounded Kanban coordination pass",
        "created_task_ids": [] if reused else [task_id],
        "reused_task_ids": [task_id] if reused else [],
        "parents": actual_parent_ids,
        "assignee": chosen_assignee,
        "idempotency_key": key,
        "project": project,
        "micro_goal": micro_goal,
        "steward_plan_path": str(steward_plan_path) if steward_plan_path else None,
        "redaction": "pass",
        "next_step": "wait_for_dispatcher_profile_lane",
    }
    evidence = _write_evidence(evidence_root=evidence_root_path, round_id=round_id, event=event)
    result = AuxiliaryLaunch(
        task_id=task_id,
        title=task.title,
        status=task.status,
        assignee=task.assignee or chosen_assignee,
        idempotency_key=key,
        reused=reused,
        dry_run=False,
        round_id=round_id,
        source_task_id=source_task_id,
        parent_ids=actual_parent_ids,
        evidence=evidence,
        steward_plan_path=str(steward_plan_path) if steward_plan_path else None,
        body=task.body or body,
    )

    comment = _launch_comment(result)
    kb.add_comment(conn, task_id, author=created_by or AUX_CREATED_BY, body=comment)
    if source_task_id and source_task_id != task_id:
        kb.add_comment(conn, source_task_id, author=created_by or AUX_CREATED_BY, body=comment)
    return result
