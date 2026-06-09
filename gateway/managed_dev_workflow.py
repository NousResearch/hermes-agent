from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

MANAGED_DEV_SKILL = "managed-dev-workflow"

# Dev channel that requires every operational message's first line to begin
# with the supervisor mention. Kept here so both the gateway dispatch path and
# the message formatter agree on the contract.
DEV_CHANNEL_ID = "1513170795847749794"
DEV_CHANNEL_MENTION = "<@330355303738769409>"

# Event types the supervisor flags as user-notifiable (mirrors
# src.models.NOTIFIABLE_EVENT_TYPES on the supervisor side).
NOTIFIABLE_EVENT_TYPES = (
    "WAITING_INPUT_DETECTED",
    "BLOCKED_DETECTED",
    "FAILED_DETECTED",
    "COMPLETED_DETECTED",
)

_logger = logging.getLogger(__name__)
_DEFAULT_REPO_DIR = os.environ.get(
    "HERMES_MANAGED_DEV_SUPERVISOR_DIR",
    os.path.expanduser("~/workspace/claude-code-supervisor"),
)

_APPROVE_RE = re.compile(r"^\s*(승인|approve|ok|오케이|진행|start|시작)\s*$", re.IGNORECASE)
_STOP_RE = re.compile(r"^\s*(중단|취소|stop|cancel|그만)\s*$", re.IGNORECASE)
_STATUS_RE = re.compile(r"^\s*(상태|status|진행상황|progress)\s*\??\s*$", re.IGNORECASE)
_EDIT_PREFIXES = (
    "수정:",
    "변경:",
    "revise:",
    "edit:",
    "update:",
)
_PLAN_PATH_RE = re.compile(r"(?P<path>(?:~|/)[^\s'\"]+\.md)")
_PLAN_HEADINGS = (
    "## 개발 계획",
    "### 1. 현재 이해한 내용",
    "### 2. 코드 구조 확인 결과",
    "### 3. 구현 계획",
    "### 4. 변경 예상 범위",
    "### 5. 위험 요소",
    "### 6. 확인 필요한 사항",
)


class ManagedDevWorkflowError(RuntimeError):
    pass


def normalize_auto_skills(auto_skill: Any) -> List[str]:
    if not auto_skill:
        return []
    if isinstance(auto_skill, str):
        return [auto_skill]
    if isinstance(auto_skill, (list, tuple, set)):
        return [str(item) for item in auto_skill if item]
    return [str(auto_skill)]


def is_managed_dev_workflow_enabled(auto_skill: Any) -> bool:
    return MANAGED_DEV_SKILL in normalize_auto_skills(auto_skill)


def default_supervisor_repo() -> Path:
    return Path(_DEFAULT_REPO_DIR).expanduser().resolve()


def supervisor_db_path(repo_dir: Path | str) -> Path:
    return Path(repo_dir) / "supervisor.db"


def tasks_dir(repo_dir: Path | str) -> Path:
    return Path(repo_dir) / "tasks"


def task_plan_file(repo_dir: Path | str, task_id: str) -> Path:
    return tasks_dir(repo_dir) / f"{task_id}-plan.md"


def task_prompt_file(repo_dir: Path | str, task_id: str) -> Path:
    return tasks_dir(repo_dir) / f"{task_id}.md"


def task_plan_request_file(repo_dir: Path | str, task_id: str) -> Path:
    return tasks_dir(repo_dir) / f"{task_id}-plan-request.md"


def classify_turn(message: str, task_status: Optional[str]) -> str:
    text = (message or "").strip()
    if not text:
        return "status"
    if _STOP_RE.match(text):
        return "stop"
    if _STATUS_RE.match(text):
        return "status"
    if _APPROVE_RE.match(text):
        return "approve_start"
    lowered = text.lower()
    if lowered.startswith(_EDIT_PREFIXES):
        return "revise_plan"
    if task_status in {"WAITING_INPUT", "BLOCKED"}:
        return "reply"
    if task_status in {"WAITING_APPROVAL", "APPROVED"}:
        return "revise_plan"
    if task_status in {"RUNNING"}:
        return "status"
    return "plan"


def build_execution_prompt(task_id: str, plan_text: str, user_message: str = "") -> str:
    parts = [
        f"Task ID: {task_id}",
        "",
        "You are Claude Code running under supervisor control.",
        "Implement the approved plan below exactly, then verify the result with real commands/tests before declaring completion.",
        "If you need a choice, clarification, credential, or external decision, stop and ask a concise blocking question.",
        "",
        "Approved plan:",
        plan_text.strip(),
    ]
    user_message = (user_message or "").strip()
    if user_message:
        parts.extend([
            "",
            "Latest planning note from Hermes:",
            user_message,
        ])
    parts.extend([
        "",
        "Execution rules:",
        "1. Do the work; do not just restate the plan.",
        "2. Run relevant tests or verification commands.",
        "3. If blocked, ask one concrete question with the minimum missing info.",
        "4. When complete, summarize changed files and exact verification results.",
    ])
    return "\n".join(parts).strip() + "\n"


def build_planning_request(user_message: str, existing_plan: str = "") -> str:
    user_message = (user_message or "").strip()
    existing_plan = (existing_plan or "").strip()
    parts = [
        "당장 구현하지 말고, 먼저 개발 계획만 작성해.",
        "",
        "다음 형식을 지켜.",
        "",
        "## 개발 계획",
        "",
        "### 1. 현재 이해한 내용",
        "- {요청 이해}",
        "",
        "### 2. 코드 구조 확인 결과",
        "- {확인한 파일/구조}",
        "",
        "### 3. 구현 계획",
        "1. {1단계}",
        "2. {2단계}",
        "3. {3단계}",
        "",
        "### 4. 변경 예상 범위",
        "- 수정 파일",
        "- DB 변경 여부",
        "- 설정 변경 여부",
        "- 기존 기능 영향 여부",
        "",
        "### 5. 위험 요소",
        "- 위험 1",
        "- 위험 2",
        "",
        "### 6. 확인 필요한 사항",
        "- 사용자 결정 필요 사항",
        "",
        "중요:",
        "- 아직 구현 시작 금지",
        "- 아직 파일 수정 금지",
        "- 계획만 답변",
        "",
        "사용자 요청:",
        user_message or "(없음)",
    ]
    if existing_plan:
        parts.extend([
            "",
            "기존 계획 초안:",
            existing_plan,
            "",
            "위 기존 계획을 그대로 반복하지 말고, 사용자의 최신 수정 요청을 반영해서 전체 계획을 다시 써.",
        ])
    return "\n".join(parts).strip() + "\n"


def _read_plan_from_output(stdout: str) -> str:
    text = (stdout or "").strip()
    if not text:
        return ""
    match = _PLAN_PATH_RE.search(text)
    if match:
        path = Path(match.group("path")).expanduser()
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    return text


def _validate_plan_text(plan_text: str) -> None:
    missing = [heading for heading in _PLAN_HEADINGS if heading not in plan_text]
    if missing:
        raise ManagedDevWorkflowError(
            "plan output missing required sections: " + ", ".join(missing)
        )


def collect_plan(
    repo_dir: Path | str,
    task_id: str,
    user_message: str,
    *,
    timeout: int = 300,
) -> Dict[str, str]:
    repo_dir = Path(repo_dir).expanduser().resolve()
    _tasks_dir = tasks_dir(repo_dir)
    _tasks_dir.mkdir(parents=True, exist_ok=True)

    existing_plan = ""
    existing_plan_path = task_plan_file(repo_dir, task_id)
    if existing_plan_path.exists():
        existing_plan = existing_plan_path.read_text(encoding="utf-8")

    request_text = build_planning_request(user_message=user_message, existing_plan=existing_plan)
    request_file = task_plan_request_file(repo_dir, task_id)
    request_file.write_text(request_text, encoding="utf-8")

    cmd = [
        "zsh",
        "-ic",
        f"cd {shlex.quote(str(repo_dir))} && cld-d -p --permission-mode plan",
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=request_text,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ManagedDevWorkflowError(f"plan collection timed out after {timeout}s") from exc

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        raise ManagedDevWorkflowError(
            f"cld-d plan command failed (exit {proc.returncode}): {stderr or stdout or 'no output'}"
        )

    plan_text = _read_plan_from_output(proc.stdout)
    if not plan_text:
        raise ManagedDevWorkflowError("cld-d returned empty plan output")
    _validate_plan_text(plan_text)
    return {
        "plan_text": plan_text.rstrip() + "\n",
        "request_file": str(request_file),
    }


def write_task_files(
    repo_dir: Path | str,
    task_id: str,
    plan_text: str,
    user_message: str = "",
) -> Dict[str, str]:
    repo_dir = Path(repo_dir)
    _tasks_dir = tasks_dir(repo_dir)
    _tasks_dir.mkdir(parents=True, exist_ok=True)
    plan_file = task_plan_file(repo_dir, task_id)
    prompt_file = task_prompt_file(repo_dir, task_id)
    plan_file.write_text(plan_text.rstrip() + "\n", encoding="utf-8")
    prompt_file.write_text(
        build_execution_prompt(task_id=task_id, plan_text=plan_text, user_message=user_message),
        encoding="utf-8",
    )
    return {
        "plan_file": str(plan_file),
        "prompt_file": str(prompt_file),
    }


def _parse_json_output(stdout: str, stderr: str) -> Dict[str, Any]:
    text = (stdout or "").strip()
    if not text:
        raise ManagedDevWorkflowError(f"supervisor returned empty output: {stderr.strip()}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ManagedDevWorkflowError(
            f"failed to parse supervisor JSON: {exc}; stdout={text[:500]!r}; stderr={stderr[:200]!r}"
        ) from exc


def run_supervisor(repo_dir: Path | str, *args: str) -> Dict[str, Any]:
    repo_dir = Path(repo_dir).expanduser().resolve()
    if not repo_dir.is_dir():
        raise ManagedDevWorkflowError(f"supervisor repo not found: {repo_dir}")
    db_path = supervisor_db_path(repo_dir)
    cmd = [sys.executable, "-m", "src.main", "--db", str(db_path), *args]
    proc = subprocess.run(
        cmd,
        cwd=str(repo_dir),
        text=True,
        capture_output=True,
        check=False,
    )
    data = _parse_json_output(proc.stdout, proc.stderr)
    if proc.returncode != 0 or not data.get("success", False):
        error = data.get("error") or proc.stderr.strip() or f"exit {proc.returncode}"
        raise ManagedDevWorkflowError(error)
    return data


def get_task_status(repo_dir: Path | str, task_id: str) -> Optional[Dict[str, Any]]:
    repo_dir = Path(repo_dir).expanduser().resolve()
    db_path = supervisor_db_path(repo_dir)
    if not db_path.exists():
        return None
    cmd = [
        sys.executable,
        "-m",
        "src.main",
        "--db",
        str(db_path),
        "status",
        "--task-id",
        task_id,
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(repo_dir),
        text=True,
        capture_output=True,
        check=False,
    )
    data = _parse_json_output(proc.stdout, proc.stderr)
    if data.get("success"):
        return data
    if "task not found" in str(data.get("error", "")):
        return None
    raise ManagedDevWorkflowError(data.get("error") or proc.stderr.strip() or "status failed")


def save_plan(
    repo_dir: Path | str,
    task_id: str,
    plan_text: str,
    *,
    title: Optional[str] = None,
    discord_channel_id: Optional[str] = None,
    user_message: str = "",
) -> Dict[str, Any]:
    files = write_task_files(repo_dir, task_id, plan_text, user_message=user_message)
    args = [
        "plan-save",
        "--task-id",
        task_id,
        "--plan-file",
        files["plan_file"],
        "--status",
        "WAITING_APPROVAL",
    ]
    if title:
        args.extend(["--title", title])
    if discord_channel_id:
        args.extend(["--discord-channel-id", str(discord_channel_id)])
    result = run_supervisor(repo_dir, *args)
    result.update(files)
    return result


def approve_and_start(
    repo_dir: Path | str,
    task_id: str,
    *,
    title: Optional[str] = None,
    discord_channel_id: Optional[str] = None,
) -> Dict[str, Any]:
    approve = run_supervisor(repo_dir, "approve", "--task-id", task_id)
    prompt_file = str(task_prompt_file(repo_dir, task_id))
    args = ["start", "--task-id", task_id, "--prompt-file", prompt_file]
    if title:
        args.extend(["--title", title])
    if discord_channel_id:
        args.extend(["--discord-channel-id", str(discord_channel_id)])
    start = run_supervisor(repo_dir, *args)
    return {
        "approve": approve,
        "start": start,
        "prompt_file": prompt_file,
    }


def forward_reply(repo_dir: Path | str, task_id: str, message: str) -> Dict[str, Any]:
    return run_supervisor(repo_dir, "reply", "--task-id", task_id, "--message", message)


def stop_task(repo_dir: Path | str, task_id: str) -> Dict[str, Any]:
    return run_supervisor(repo_dir, "stop", "--task-id", task_id)


def format_status_reply(task: Optional[Dict[str, Any]], task_id: str) -> str:
    if not task:
        return "아직 저장된 작업이 없어. 새로 계획부터 잡으면 된다."
    status = task.get("status") or "UNKNOWN"
    lines = [f"현재 상태: {status}", f"task_id: {task_id}"]
    log_file = task.get("log_file")
    if log_file:
        lines.append(f"log: {log_file}")
    events = task.get("latest_events") or []
    if events:
        latest = events[0]
        summary = latest.get("summary") or latest.get("event_type") or "최근 이벤트 없음"
        lines.append(f"최근 이벤트: {summary}")
    if status == "WAITING_APPROVAL":
        lines.append("답장: 승인 / 수정: ... / 중단")
    elif status in {"WAITING_INPUT", "BLOCKED"}:
        lines.append("지금 답장을 보내면 supervisor 쪽으로 그대로 전달한다.")
    elif status == "RUNNING":
        lines.append("지금은 실행 중이야. blocker가 뜨면 그때 답하면 된다.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Supervisor notification delivery (Hermes -> Discord)
# ---------------------------------------------------------------------------
#
# The supervisor never sends Discord itself. It records notifiable events
# (WAITING_INPUT / BLOCKED / FAILED / COMPLETED) and exposes them read-only via
# `notifications --task-id`. Hermes polls, renders a user-facing message,
# delivers it, and only then calls `notify-ack --event-id` so a failed send
# leaves the notification pending for retry (no loss, no double-send).

# Headline templates keyed by supervisor event_type. Each gives the user a
# scannable, plain-language status instead of a raw payload dump.
_EVENT_HEADLINES = {
    "COMPLETED_DETECTED": "✅ 작업 완료",
    "FAILED_DETECTED": "❌ 작업 실패",
    "BLOCKED_DETECTED": "⛔ 작업이 막혔어",
    "WAITING_INPUT_DETECTED": "❓ 입력이 필요해",
}

# Closing call-to-action per event type.
_EVENT_FOOTERS = {
    "COMPLETED_DETECTED": "확인하고 이어서 할 일이 있으면 알려줘.",
    "FAILED_DETECTED": "로그를 확인하고, 다시 시도하려면 답장 줘.",
    "BLOCKED_DETECTED": "아래 내용 확인하고 답장으로 막힌 부분을 풀어줘.",
    "WAITING_INPUT_DETECTED": "답장을 보내면 supervisor 쪽으로 그대로 전달한다.",
}


def list_pending_notifications(repo_dir: Path | str, task_id: str) -> List[Dict[str, Any]]:
    """Return pending (unsent) supervisor notification payloads for ``task_id``.

    Read-only: calls the supervisor ``notifications`` command, which never
    marks anything sent. Returns ``[]`` when there is nothing pending.
    """
    data = run_supervisor(repo_dir, "notifications", "--task-id", task_id)
    payloads = data.get("notifications") or []
    return [p for p in payloads if isinstance(p, dict)]


def ack_notification(repo_dir: Path | str, task_id: str, event_id: int) -> Dict[str, Any]:
    """Mark a notification sent via the supervisor ``notify-ack`` command.

    Idempotent on the supervisor side (UNIQUE(task_id, event_id)). Only call
    this AFTER a successful Discord send.
    """
    return run_supervisor(
        repo_dir, "notify-ack", "--task-id", task_id, "--event-id", str(event_id)
    )


def tasks_with_pending_notifications(repo_dir: Path | str) -> List[str]:
    """Enumerate task_ids that have at least one un-acked notifiable event.

    The supervisor exposes no "list tasks" command, so this reads the DB
    directly (read-only) to avoid shelling out ``notifications`` for every
    task. Best-effort: returns ``[]`` if the DB is missing or unreadable.
    """
    db_path = supervisor_db_path(Path(repo_dir).expanduser().resolve())
    if not db_path.exists():
        return []
    placeholders = ",".join("?" for _ in NOTIFIABLE_EVENT_TYPES)
    query = (
        "SELECT DISTINCT e.task_id "
        "FROM task_events e "
        "LEFT JOIN notifications n "
        "  ON n.task_id = e.task_id AND n.event_id = e.id "
        f"WHERE e.event_type IN ({placeholders}) AND n.id IS NULL "
        "ORDER BY e.task_id"
    )
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
        try:
            rows = conn.execute(query, NOTIFIABLE_EVENT_TYPES).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        _logger.warning("managed-dev-workflow: cannot enumerate pending tasks: %s", exc)
        return []
    return [r[0] for r in rows if r and r[0]]


def format_notification_message(payload: Dict[str, Any]) -> str:
    """Render a supervisor notification payload as a user-facing Discord message.

    Produces structured, plain-language text (never a raw JSON dump). When the
    target channel is the dev channel, the first line begins with the required
    supervisor mention so the operational message is delivered correctly.
    """
    event_type = str(payload.get("event_type") or "")
    headline = _EVENT_HEADLINES.get(event_type, "ℹ️ 작업 알림")

    title = (payload.get("title") or "").strip()
    if title:
        headline = f"{headline} — {title}"

    lines = [headline]

    summary = (payload.get("summary") or "").strip()
    if summary:
        lines.append("")
        lines.append(summary)

    excerpt = (payload.get("raw_excerpt") or "").strip()
    if excerpt and excerpt != summary:
        lines.append("")
        lines.append("```")
        # Keep excerpts bounded so a runaway log never floods the channel.
        lines.append(excerpt[:1500])
        lines.append("```")

    log_file = (payload.get("log_file") or "").strip()
    if log_file:
        lines.append(f"log: {log_file}")

    footer = _EVENT_FOOTERS.get(event_type)
    if footer:
        lines.append("")
        lines.append(footer)

    message = "\n".join(lines).strip()

    if str(payload.get("discord_channel_id") or "") == DEV_CHANNEL_ID:
        message = f"{DEV_CHANNEL_MENTION} {message}"

    return message


# Type alias for the async Discord send callback the gateway supplies.
SendCallable = Callable[[str, str], Awaitable[bool]]


async def dispatch_pending_notifications(
    repo_dir: Path | str,
    send: SendCallable,
    *,
    task_ids: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
    _list_pending: Callable[..., List[Dict[str, Any]]] = list_pending_notifications,
    _ack: Callable[..., Dict[str, Any]] = ack_notification,
    _enumerate: Callable[..., List[str]] = tasks_with_pending_notifications,
) -> Dict[str, int]:
    """Poll pending supervisor notifications and deliver them via ``send``.

    ``send(channel_id, text)`` is an async callable returning ``True`` on a
    confirmed delivery. On success the matching event is acked exactly once; on
    failure (``False`` or exception) the notification is left pending for retry.
    Per-task and per-notification errors are isolated so one failure never
    aborts the rest of the sweep.

    Returns counts: ``{"sent", "acked", "failed", "skipped"}``.
    """
    log = logger or _logger
    summary = {"sent": 0, "acked": 0, "failed": 0, "skipped": 0}

    if task_ids is None:
        try:
            task_ids = await asyncio.to_thread(_enumerate, repo_dir)
        except Exception as exc:  # noqa: BLE001 - never break the gateway loop
            log.warning("managed-dev-workflow: task enumeration failed: %s", exc)
            return summary

    for task_id in task_ids or []:
        try:
            payloads = await asyncio.to_thread(_list_pending, repo_dir, task_id)
        except Exception as exc:  # noqa: BLE001 - isolate per-task failure
            log.warning(
                "managed-dev-workflow: pending fetch failed for %s: %s", task_id, exc
            )
            continue

        for payload in payloads:
            event_id = payload.get("event_id")
            channel_id = payload.get("discord_channel_id")
            if not channel_id or event_id is None:
                # No route or no id — can't deliver or ack safely. Leave it.
                summary["skipped"] += 1
                continue
            try:
                text = format_notification_message(payload)
                ok = await send(str(channel_id), text)
            except Exception as exc:  # noqa: BLE001 - send raised: leave pending
                log.warning(
                    "managed-dev-workflow: send failed for %s/%s: %s",
                    task_id, event_id, exc,
                )
                summary["failed"] += 1
                continue
            if not ok:
                # Delivery not confirmed — leave pending for the next sweep.
                summary["failed"] += 1
                continue
            summary["sent"] += 1
            try:
                await asyncio.to_thread(_ack, repo_dir, task_id, event_id)
                summary["acked"] += 1
            except Exception as exc:  # noqa: BLE001 - sent but ack failed
                # Message was delivered but ack failed. The supervisor ack is
                # idempotent, so a re-send on the next sweep is at worst one
                # duplicate; surface it loudly but don't crash the loop.
                log.warning(
                    "managed-dev-workflow: ack failed after send for %s/%s: %s",
                    task_id, event_id, exc,
                )

    return summary


# ===========================================================================
# Deterministic `/wf-dev` control plane
# ===========================================================================
#
# Everything below implements the explicit Discord slash-command router for
# managed-dev workflows (see ``deterministic-discord-router-plan.md``). The
# free-text path above (``classify_turn`` + the run.py dispatch) is preserved
# untouched; this is a separate, *deterministic* surface:
#
#   Discord /wf-dev <sub> ...  ->  WfDevCommand (parsed, never LLM-interpreted)
#                              ->  handle_wf_dev_command (preconditions + state)
#                              ->  supervisor wrappers (authoritative state)
#                              ->  WfDevResponse (text or Embed spec)
#
# Design invariants enforced here:
#   * Supervisor remains the source of truth for task status.
#   * `/wf-dev` fails CLOSED — a handler error returns an explicit error
#     response, never a silent fall-through to the generic agent path.
#   * Operator-facing responses identify a workflow by BOTH ``ref`` and
#     ``task_id``.
#   * Thread-local commands resolve the active binding so ``task_id`` may be
#     omitted in the common case.

# Public slash namespace and the subcommands it accepts. Anything outside this
# set is rejected deterministically (never guessed at).
WF_DEV_GROUP = "wf-dev"
WF_DEV_SUBCOMMANDS = (
    "plan",
    "revise",
    "approve",
    "start",
    "reply",
    "status",
    "list",
    "stop",
    "summary",
    "help",
)

# Metadata envelope key the Discord adapter stamps onto a MessageEvent so the
# gateway can recognize a structured `/wf-dev` invocation without parsing free
# text. See ``is_wf_dev_event`` / ``parse_wf_dev_event``.
WF_DEV_METADATA_KEY = "wf_dev"

# Supervisor status sets that gate each transition. Kept as data (not scattered
# `if`s) so the policy is auditable in one place and easy to test.
_TERMINAL_STATUSES = {"COMPLETED", "FAILED", "STOPPED"}
_REPLYABLE_STATUSES = {"WAITING_INPUT", "BLOCKED"}
_REVISABLE_STATUSES = {"WAITING_APPROVAL", "APPROVED"}
_APPROVABLE_STATUSES = {"WAITING_APPROVAL"}
_STARTABLE_STATUSES = {"APPROVED"}

# Next-action hints surfaced to the operator so the control panel is
# self-documenting inside Discord.
_NEXT_BY_STATUS = {
    "WAITING_APPROVAL": "/wf-dev approve",
    "APPROVED": "/wf-dev start",
    "RUNNING": "/wf-dev status",
    "WAITING_INPUT": "/wf-dev reply message:...",
    "BLOCKED": "/wf-dev reply message:...",
    "COMPLETED": "/wf-dev summary",
    "FAILED": "/wf-dev summary",
    "STOPPED": "/wf-dev plan request:...",
}


class WfDevError(ManagedDevWorkflowError):
    """A deterministic `/wf-dev` precondition or routing failure.

    Distinct from the generic :class:`ManagedDevWorkflowError` so the router
    can tell an *expected* fail-closed rejection (bad state, missing binding)
    apart from an unexpected internal fault, while both still fail closed.
    """


class AmbiguousBindingError(WfDevError):
    """Raised when a thread/channel has more than one active workflow and the
    command omitted ``task_id`` so the target cannot be resolved unambiguously.
    """


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Structured command + response data model
# ---------------------------------------------------------------------------


@dataclass
class WfDevCommand:
    """A parsed, deterministic `/wf-dev` invocation.

    Built from the structured metadata envelope the Discord adapter produces —
    never from free-text reconstruction. Optional fields default so each
    subcommand only reads what it needs.
    """

    subcommand: str
    task_id: Optional[str] = None
    ref: Optional[str] = None
    request: Optional[str] = None
    message: Optional[str] = None
    reason: Optional[str] = None
    title: Optional[str] = None
    scope: str = "thread"
    status_filter: str = "active"
    page: int = 1
    auto_start: bool = True
    # Discord routing context (from the interaction).
    platform: str = "discord"
    channel_id: Optional[str] = None
    thread_id: Optional[str] = None
    guild_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None


@dataclass
class WorkflowBinding:
    """Maps a supervisor ``task_id`` to a user-facing ``ref`` and the Discord
    context (channel/thread/session) that owns it.

    ``ref`` is a gateway-side, human-friendly identifier (e.g. ``DEV-142``);
    the supervisor knows only ``task_id``. ``is_active`` enforces the
    one-active-workflow-per-thread policy and powers ``task_id`` omission.
    """

    task_id: str
    ref: str
    platform: str = "discord"
    channel_id: Optional[str] = None
    thread_id: Optional[str] = None
    guild_id: Optional[str] = None
    session_id: Optional[str] = None
    title: Optional[str] = None
    status: Optional[str] = None
    plan_version: int = 1
    is_active: bool = True
    created_at: str = ""
    updated_at: str = ""


@dataclass
class WfDevResponse:
    """The deterministic result of a `/wf-dev` command.

    ``text`` is always populated (operator message / plain-text fallback).
    ``embed`` carries a platform-neutral Embed spec for ``/wf-dev list`` that
    the Discord adapter renders as a real Embed; other surfaces fall back to
    ``text``. ``ok`` is False for every fail-closed rejection.
    """

    ok: bool
    text: str
    embed: Optional[Dict[str, Any]] = None
    ephemeral: bool = False
    subcommand: str = ""
    task_id: Optional[str] = None
    ref: Optional[str] = None


# ---------------------------------------------------------------------------
# Parsing / normalization
# ---------------------------------------------------------------------------


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _coerce_int(value: Any, default: int = 1) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return default
    return out if out >= 1 else default


def _clean_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_wf_dev_command(envelope: Dict[str, Any]) -> WfDevCommand:
    """Normalize a structured `/wf-dev` metadata envelope into a WfDevCommand.

    Fails closed: an unknown/missing subcommand raises :class:`WfDevError`
    rather than guessing. String fields are stripped (empty → None), ``page``
    and ``auto_start`` are coerced to their declared types.
    """
    if not isinstance(envelope, dict):
        raise WfDevError("wf-dev envelope must be a mapping")
    sub = _clean_str(envelope.get("subcommand"))
    if not sub:
        raise WfDevError("wf-dev command missing subcommand")
    sub = sub.lower()
    if sub not in WF_DEV_SUBCOMMANDS:
        raise WfDevError(
            f"unknown /wf-dev subcommand: {sub!r} "
            f"(expected one of: {', '.join(WF_DEV_SUBCOMMANDS)})"
        )
    return WfDevCommand(
        subcommand=sub,
        task_id=_clean_str(envelope.get("task_id")),
        ref=_clean_str(envelope.get("ref")),
        request=_clean_str(envelope.get("request")),
        message=_clean_str(envelope.get("message")),
        reason=_clean_str(envelope.get("reason")),
        title=_clean_str(envelope.get("title")),
        scope=(_clean_str(envelope.get("scope")) or "thread").lower(),
        status_filter=(_clean_str(envelope.get("status_filter")) or "active").lower(),
        page=_coerce_int(envelope.get("page"), default=1),
        auto_start=_coerce_bool(envelope.get("auto_start"), default=True),
        platform=(_clean_str(envelope.get("platform")) or "discord").lower(),
        channel_id=_clean_str(envelope.get("channel_id")),
        thread_id=_clean_str(envelope.get("thread_id")),
        guild_id=_clean_str(envelope.get("guild_id")),
        session_id=_clean_str(envelope.get("session_id")),
        user_id=_clean_str(envelope.get("user_id")),
    )


def is_wf_dev_event(event: Any) -> bool:
    """Whether ``event`` carries a structured `/wf-dev` metadata envelope.

    Tolerant of events that predate the ``metadata`` field — returns False
    rather than raising so the normal agent path is never disturbed for
    non-`/wf-dev` traffic.
    """
    metadata = getattr(event, "metadata", None)
    if not isinstance(metadata, dict):
        return False
    return isinstance(metadata.get(WF_DEV_METADATA_KEY), dict)


def parse_wf_dev_event(event: Any) -> WfDevCommand:
    """Extract and parse the `/wf-dev` envelope carried on ``event``."""
    metadata = getattr(event, "metadata", None) or {}
    envelope = dict(metadata.get(WF_DEV_METADATA_KEY) or {})
    return parse_wf_dev_command(envelope)


# ---------------------------------------------------------------------------
# Binding persistence (gateway-side SQLite)
# ---------------------------------------------------------------------------


def default_binding_db_path() -> Path:
    """Gateway-side binding DB path. Separate from supervisor.db so workflow
    bindings never depend on the supervisor repo being writable.
    """
    override = os.environ.get("HERMES_MANAGED_DEV_BINDINGS_DB")
    if override:
        return Path(override).expanduser()
    return Path(os.path.expanduser("~/.hermes")) / "managed_dev_bindings.db"


_BINDINGS_DDL = """
CREATE TABLE IF NOT EXISTS workflow_bindings (
    task_id          TEXT PRIMARY KEY,
    ref              TEXT NOT NULL,
    platform         TEXT NOT NULL DEFAULT 'discord',
    channel_id       TEXT,
    thread_id        TEXT,
    guild_id         TEXT,
    session_id       TEXT,
    title            TEXT,
    status           TEXT,
    plan_version     INTEGER NOT NULL DEFAULT 1,
    is_active        INTEGER NOT NULL DEFAULT 1,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_wb_thread ON workflow_bindings(thread_id);
CREATE INDEX IF NOT EXISTS idx_wb_channel ON workflow_bindings(channel_id);
CREATE INDEX IF NOT EXISTS idx_wb_active ON workflow_bindings(is_active);
"""


class WorkflowBindingStore:
    """SQLite-backed persistence for `/wf-dev` workflow bindings.

    Owns the gateway-side mapping that lets thread-local commands omit
    ``task_id`` and powers ``/wf-dev list``. The supervisor remains the source
    of truth for *status*; this store only caches ``last_known_status`` plus
    the user-facing ``ref`` the supervisor does not track.
    """

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else default_binding_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_BINDINGS_DDL)
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _row_to_binding(row: sqlite3.Row) -> WorkflowBinding:
        return WorkflowBinding(
            task_id=row["task_id"],
            ref=row["ref"],
            platform=row["platform"],
            channel_id=row["channel_id"],
            thread_id=row["thread_id"],
            guild_id=row["guild_id"],
            session_id=row["session_id"],
            title=row["title"],
            status=row["status"],
            plan_version=row["plan_version"],
            is_active=bool(row["is_active"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def next_ref(self) -> str:
        """Allocate the next human-facing ``DEV-N`` ref (monotonic by count)."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM workflow_bindings"
        ).fetchone()
        return f"DEV-{(row['n'] if row else 0) + 1}"

    # -- writes -------------------------------------------------------------

    def save(self, binding: WorkflowBinding) -> WorkflowBinding:
        """Insert or update a binding, keyed by ``task_id``.

        On insert, missing timestamps are stamped. On update, ``created_at`` is
        preserved and ``updated_at`` advanced.
        """
        now = _now_iso()
        existing = self.get(binding.task_id)
        created_at = binding.created_at or (existing.created_at if existing else now)
        binding.created_at = created_at
        binding.updated_at = now
        self._conn.execute(
            """
            INSERT INTO workflow_bindings (
                task_id, ref, platform, channel_id, thread_id, guild_id,
                session_id, title, status, plan_version, is_active,
                created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(task_id) DO UPDATE SET
                ref=excluded.ref,
                platform=excluded.platform,
                channel_id=excluded.channel_id,
                thread_id=excluded.thread_id,
                guild_id=excluded.guild_id,
                session_id=excluded.session_id,
                title=excluded.title,
                status=excluded.status,
                plan_version=excluded.plan_version,
                is_active=excluded.is_active,
                updated_at=excluded.updated_at
            """,
            (
                binding.task_id, binding.ref, binding.platform,
                binding.channel_id, binding.thread_id, binding.guild_id,
                binding.session_id, binding.title, binding.status,
                int(binding.plan_version), 1 if binding.is_active else 0,
                created_at, now,
            ),
        )
        self._conn.commit()
        return binding

    def update_status(
        self,
        task_id: str,
        status: Optional[str],
        *,
        is_active: Optional[bool] = None,
        plan_version: Optional[int] = None,
    ) -> Optional[WorkflowBinding]:
        """Patch a binding's cached status / active flag / plan version."""
        binding = self.get(task_id)
        if binding is None:
            return None
        if status is not None:
            binding.status = status
        if plan_version is not None:
            binding.plan_version = plan_version
        if is_active is not None:
            binding.is_active = is_active
        return self.save(binding)

    def deactivate(self, task_id: str) -> Optional[WorkflowBinding]:
        return self.update_status(task_id, None, is_active=False)

    # -- reads --------------------------------------------------------------

    def get(self, task_id: str) -> Optional[WorkflowBinding]:
        row = self._conn.execute(
            "SELECT * FROM workflow_bindings WHERE task_id = ?", (task_id,)
        ).fetchone()
        return self._row_to_binding(row) if row else None

    def resolve_active(
        self,
        *,
        thread_id: Optional[str] = None,
        channel_id: Optional[str] = None,
    ) -> Optional[WorkflowBinding]:
        """Return the single active binding for the current thread/channel.

        Resolution priority (per the design doc): ``thread_id`` first, then
        ``channel_id``. Returns None when nothing is bound. Raises
        :class:`AmbiguousBindingError` when more than one active workflow
        matches — the caller must then require an explicit ``task_id``.
        """
        for column, value in (("thread_id", thread_id), ("channel_id", channel_id)):
            if not value:
                continue
            rows = self._conn.execute(
                f"SELECT * FROM workflow_bindings "
                f"WHERE {column} = ? AND is_active = 1 "
                f"ORDER BY updated_at DESC",
                (value,),
            ).fetchall()
            if len(rows) == 1:
                return self._row_to_binding(rows[0])
            if len(rows) > 1:
                refs = ", ".join(r["ref"] for r in rows)
                raise AmbiguousBindingError(
                    f"{len(rows)} active workflows are bound here ({refs}); "
                    f"pass task_id to disambiguate"
                )
        return None

    def list_recent(
        self,
        *,
        thread_id: Optional[str] = None,
        channel_id: Optional[str] = None,
        scope: str = "thread",
        status_filter: str = "active",
        page: int = 1,
        page_size: int = 10,
    ) -> List[WorkflowBinding]:
        """List recent bindings for the list view, newest first.

        ``scope`` selects thread- vs channel-local rows; ``status_filter`` is
        one of ``active`` / ``all`` / ``completed`` / ``failed``. Pagination is
        a simple 1-based ``page`` over ``page_size`` rows.
        """
        clauses: List[str] = []
        params: List[Any] = []
        if scope == "thread" and thread_id:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        elif channel_id:
            clauses.append("channel_id = ?")
            params.append(channel_id)
        elif thread_id:
            clauses.append("thread_id = ?")
            params.append(thread_id)

        sf = (status_filter or "active").lower()
        if sf == "active":
            clauses.append("is_active = 1")
        elif sf == "completed":
            clauses.append("status = 'COMPLETED'")
        elif sf == "failed":
            clauses.append("status = 'FAILED'")
        # "all" → no status clause.

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        offset = max(0, (max(1, page) - 1) * page_size)
        rows = self._conn.execute(
            f"SELECT * FROM workflow_bindings{where} "
            f"ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (*params, page_size, offset),
        ).fetchall()
        return [self._row_to_binding(r) for r in rows]


# ---------------------------------------------------------------------------
# Supervisor wrappers (added for the deterministic surface)
# ---------------------------------------------------------------------------


def run_start(repo_dir: Path | str, task_id: str, *, discord_channel_id: Optional[str] = None) -> Dict[str, Any]:
    """Start an already-APPROVED task without re-approving (manual `start`)."""
    prompt_file = str(task_prompt_file(repo_dir, task_id))
    args = ["start", "--task-id", task_id, "--prompt-file", prompt_file]
    if discord_channel_id:
        args.extend(["--discord-channel-id", str(discord_channel_id)])
    result = run_supervisor(repo_dir, *args)
    result["prompt_file"] = prompt_file
    return result


def run_summary(repo_dir: Path | str, task_id: str) -> Dict[str, Any]:
    """Fetch the supervisor's completion/partial summary payload for a task."""
    return run_supervisor(repo_dir, "summary", "--task-id", task_id)


# ---------------------------------------------------------------------------
# Response formatting
# ---------------------------------------------------------------------------


def _identity_lines(ref: Optional[str], task_id: Optional[str]) -> List[str]:
    """The ref/task_id header every workflow-identifying response carries."""
    lines = []
    if ref:
        lines.append(f"- ref: {ref}")
    if task_id:
        lines.append(f"- task_id: {task_id}")
    return lines


def format_status_text(
    *,
    ref: Optional[str],
    task_id: str,
    status: Optional[str],
    plan_version: Optional[int] = None,
    latest_event: Optional[str] = None,
    question: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """Compact, operational status block — ref + task_id always shown."""
    lines = ["Current workflow status."]
    lines.extend(_identity_lines(ref, task_id))
    if title:
        lines.append(f"- title: {title}")
    lines.append(f"- status: {status or 'UNKNOWN'}")
    if plan_version is not None:
        lines.append(f"- plan_version: {plan_version}")
    if latest_event:
        lines.append(f"- latest_event: {latest_event}")
    if question:
        lines.append(f"- question: {question}")
    nxt = _NEXT_BY_STATUS.get(status or "")
    if nxt:
        lines.append(f"- next: {nxt}")
    return "\n".join(lines)


def _truncate(text: str, width: int) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def _short_time(ts: Optional[str]) -> str:
    """Render an ISO timestamp as HH:MM; fall back to the raw/empty string."""
    if not ts:
        return "-"
    try:
        return datetime.fromisoformat(ts).strftime("%H:%M")
    except (ValueError, TypeError):
        return str(ts)[:16]


def format_list_embed(
    bindings: List[WorkflowBinding],
    *,
    scope: str = "thread",
    page: int = 1,
) -> Dict[str, Any]:
    """Build a platform-neutral Embed spec for ``/wf-dev list``.

    The description is a monospace code-block pseudo-table (Discord has no real
    table widget). Every row shows BOTH ``ref`` and ``task_id`` so an operator
    can scan by the short ref but still copy the precise task_id.
    """
    title = (
        "Active workflows in this thread"
        if scope == "thread"
        else "Recent workflows"
    )

    header = f"{'#':<2} {'Ref':<8} {'Task ID':<22} {'Title':<22} {'Status':<14} {'Updated':<6}"
    rows = [header]
    if not bindings:
        rows.append("(none)")
    for i, b in enumerate(bindings, start=1):
        rows.append(
            f"{i:<2} {_truncate(b.ref, 8):<8} {_truncate(b.task_id, 22):<22} "
            f"{_truncate(b.title or '-', 22):<22} {_truncate(b.status or '-', 14):<14} "
            f"{_short_time(b.updated_at):<6}"
        )
    description = "```\n" + "\n".join(rows) + "\n```"

    return {
        "title": title,
        "description": description,
        "fields": [
            {
                "name": "Use",
                "value": (
                    "/wf-dev status · /wf-dev approve · "
                    "/wf-dev summary task_id:<id>"
                ),
                "inline": False,
            },
            {
                "name": "Tip",
                "value": "Default target is this thread's active workflow.",
                "inline": False,
            },
        ],
        "footer": f"page {page} · scope={scope}",
    }


def format_list_text(bindings: List[WorkflowBinding], *, scope: str = "thread", page: int = 1) -> str:
    """Plain-text fallback for surfaces that can't render an Embed."""
    embed = format_list_embed(bindings, scope=scope, page=page)
    return f"{embed['title']}\n{embed['description']}"


WF_DEV_HELP_TEXT = "\n".join(
    [
        "/wf-dev commands",
        "- plan: 새 개발 계획 생성 (request:...)",
        "- revise: 기존 계획 수정 (request:... [task_id:...])",
        "- approve: 승인 후 실행 ([task_id:...] [auto_start:true])",
        "- start: 승인된 작업 수동 시작 ([task_id:...])",
        "- reply: blocker/질문에 답변 (message:... [task_id:...])",
        "- status: 현재 상태 조회 ([task_id:...])",
        "- list: 최근/활성 workflow 목록 (Embed)",
        "- stop: 작업 중단 ([task_id:...])",
        "- summary: 요약 조회 ([task_id:...])",
        "- help: 이 도움말",
        "",
        "task_id를 생략하면 현재 thread의 active workflow를 대상으로 한다.",
    ]
)


# ---------------------------------------------------------------------------
# Deterministic router
# ---------------------------------------------------------------------------


def _resolve_target(
    cmd: WfDevCommand,
    store: WorkflowBindingStore,
    *,
    required: bool = True,
) -> Optional[WorkflowBinding]:
    """Resolve the binding a control command targets.

    Explicit ``task_id`` wins over the thread/channel binding (design §9.4).
    When ``task_id`` is omitted, fall back to the single active binding for the
    current thread/channel. Fails closed with an explicit message when nothing
    can be resolved and ``required`` is True.
    """
    if cmd.task_id:
        binding = store.get(cmd.task_id)
        if binding is None:
            # Allow operating on a task the gateway has no binding row for
            # (e.g. created out-of-band) — synthesize a minimal binding so the
            # supervisor call can still proceed with the explicit task_id.
            return WorkflowBinding(task_id=cmd.task_id, ref=cmd.ref or cmd.task_id)
        return binding
    binding = store.resolve_active(thread_id=cmd.thread_id, channel_id=cmd.channel_id)
    if binding is None and required:
        raise WfDevError(
            "No active workflow is bound to this thread. "
            "Use /wf-dev plan to create one, or pass task_id."
        )
    return binding


def _fetch_status(
    repo_dir: Path | str,
    task_id: str,
    *,
    status_fn: Callable[..., Optional[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    return status_fn(repo_dir, task_id)


def _latest_event_summary(task: Optional[Dict[str, Any]]) -> tuple[Optional[str], Optional[str]]:
    """Return ``(latest_event_type, pending_question)`` from a status payload."""
    if not task:
        return None, None
    events = task.get("latest_events") or []
    if not events:
        return None, None
    latest = events[0]
    event_type = latest.get("event_type")
    question = None
    if latest.get("requires_user_response") and not latest.get("resolved"):
        question = latest.get("summary")
    return event_type, question


def handle_wf_dev_command(
    cmd: WfDevCommand,
    *,
    repo_dir: Path | str,
    store: WorkflowBindingStore,
    collect_plan_fn: Callable[..., Dict[str, str]] = collect_plan,
    save_plan_fn: Callable[..., Dict[str, Any]] = save_plan,
    approve_and_start_fn: Callable[..., Dict[str, Any]] = approve_and_start,
    approve_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    start_fn: Callable[..., Dict[str, Any]] = run_start,
    reply_fn: Callable[..., Dict[str, Any]] = forward_reply,
    stop_fn: Callable[..., Dict[str, Any]] = stop_task,
    status_fn: Callable[..., Optional[Dict[str, Any]]] = get_task_status,
    summary_fn: Callable[..., Dict[str, Any]] = run_summary,
) -> WfDevResponse:
    """Route one parsed `/wf-dev` command to the supervisor, fail-closed.

    Every dependency is injectable (defaults to the module functions) so the
    whole router is unit-testable with fakes — no real supervisor or ``cld-d``
    needed. The function NEVER raises: expected precondition failures and
    unexpected faults alike become ``ok=False`` responses, guaranteeing the
    caller never silently falls through to the generic agent path.
    """
    try:
        return _dispatch_wf_dev(
            cmd,
            repo_dir=repo_dir,
            store=store,
            collect_plan_fn=collect_plan_fn,
            save_plan_fn=save_plan_fn,
            approve_and_start_fn=approve_and_start_fn,
            approve_fn=approve_fn,
            start_fn=start_fn,
            reply_fn=reply_fn,
            stop_fn=stop_fn,
            status_fn=status_fn,
            summary_fn=summary_fn,
        )
    except WfDevError as exc:
        return WfDevResponse(
            ok=False,
            text=f"/wf-dev {cmd.subcommand} 거부됨: {exc}",
            ephemeral=True,
            subcommand=cmd.subcommand,
            task_id=cmd.task_id,
            ref=cmd.ref,
        )
    except Exception as exc:  # noqa: BLE001 - fail closed, never reach the LLM
        _logger.warning("wf-dev handler error (%s): %s", cmd.subcommand, exc)
        return WfDevResponse(
            ok=False,
            text=f"/wf-dev {cmd.subcommand} 실패: {exc}",
            ephemeral=True,
            subcommand=cmd.subcommand,
            task_id=cmd.task_id,
            ref=cmd.ref,
        )


def _dispatch_wf_dev(
    cmd: WfDevCommand,
    *,
    repo_dir: Path | str,
    store: WorkflowBindingStore,
    collect_plan_fn: Callable[..., Dict[str, str]],
    save_plan_fn: Callable[..., Dict[str, Any]],
    approve_and_start_fn: Callable[..., Dict[str, Any]],
    approve_fn: Optional[Callable[..., Dict[str, Any]]],
    start_fn: Callable[..., Dict[str, Any]],
    reply_fn: Callable[..., Dict[str, Any]],
    stop_fn: Callable[..., Dict[str, Any]],
    status_fn: Callable[..., Optional[Dict[str, Any]]],
    summary_fn: Callable[..., Dict[str, Any]],
) -> WfDevResponse:
    sub = cmd.subcommand

    if sub == "help":
        return WfDevResponse(ok=True, text=WF_DEV_HELP_TEXT, ephemeral=True, subcommand=sub)

    if sub == "list":
        bindings = store.list_recent(
            thread_id=cmd.thread_id,
            channel_id=cmd.channel_id,
            scope=cmd.scope,
            status_filter=cmd.status_filter,
            page=cmd.page,
        )
        embed = format_list_embed(bindings, scope=cmd.scope, page=cmd.page)
        return WfDevResponse(
            ok=True,
            text=format_list_text(bindings, scope=cmd.scope, page=cmd.page),
            embed=embed,
            subcommand=sub,
        )

    if sub == "plan":
        return _handle_plan(cmd, repo_dir=repo_dir, store=store,
                            collect_plan_fn=collect_plan_fn, save_plan_fn=save_plan_fn)

    if sub == "revise":
        return _handle_revise(cmd, repo_dir=repo_dir, store=store,
                              collect_plan_fn=collect_plan_fn, save_plan_fn=save_plan_fn,
                              status_fn=status_fn)

    # All remaining subcommands operate on an existing workflow.
    binding = _resolve_target(cmd, store, required=True)
    task_id = binding.task_id

    if sub == "status":
        task = _fetch_status(repo_dir, task_id, status_fn=status_fn)
        if task is None:
            raise WfDevError(f"task not found: {task_id}")
        status = task.get("status")
        if status in _TERMINAL_STATUSES:
            store.deactivate(task_id)
        else:
            store.update_status(task_id, status)
        event_type, question = _latest_event_summary(task)
        return WfDevResponse(
            ok=True,
            text=format_status_text(
                ref=binding.ref, task_id=task_id, status=status,
                plan_version=binding.plan_version, latest_event=event_type,
                question=question, title=binding.title,
            ),
            subcommand=sub, task_id=task_id, ref=binding.ref,
        )

    if sub == "approve":
        task = _fetch_status(repo_dir, task_id, status_fn=status_fn)
        current = (task or {}).get("status")
        if current not in _APPROVABLE_STATUSES:
            raise WfDevError(
                f"approve rejected — status is {current}, "
                f"expected {', '.join(sorted(_APPROVABLE_STATUSES))}"
            )
        if cmd.auto_start:
            result = approve_and_start_fn(repo_dir, task_id, discord_channel_id=cmd.channel_id)
            new_status = (result.get("start") or {}).get("status", "RUNNING")
            headline = "Approved and started."
        else:
            approve = approve_fn or (lambda rd, tid: run_supervisor(rd, "approve", "--task-id", tid))
            approve(repo_dir, task_id)
            new_status = "APPROVED"
            headline = "Approved (not started)."
        store.update_status(task_id, new_status)
        lines = [headline, *_identity_lines(binding.ref, task_id), f"- status: {new_status}"]
        nxt = _NEXT_BY_STATUS.get(new_status)
        if nxt:
            lines.append(f"- next: {nxt}")
        return WfDevResponse(ok=True, text="\n".join(lines), subcommand=sub,
                             task_id=task_id, ref=binding.ref)

    if sub == "start":
        task = _fetch_status(repo_dir, task_id, status_fn=status_fn)
        current = (task or {}).get("status")
        if current not in _STARTABLE_STATUSES:
            raise WfDevError(
                f"start rejected — status is {current}, expected APPROVED"
            )
        result = start_fn(repo_dir, task_id, discord_channel_id=cmd.channel_id)
        new_status = result.get("status", "RUNNING")
        store.update_status(task_id, new_status)
        lines = ["Execution started.", *_identity_lines(binding.ref, task_id),
                 f"- status: {new_status}", "- next: /wf-dev status"]
        return WfDevResponse(ok=True, text="\n".join(lines), subcommand=sub,
                             task_id=task_id, ref=binding.ref)

    if sub == "reply":
        if not cmd.message:
            raise WfDevError("reply requires a message")
        task = _fetch_status(repo_dir, task_id, status_fn=status_fn)
        current = (task or {}).get("status")
        if current not in _REPLYABLE_STATUSES:
            raise WfDevError(
                f"reply rejected — status is {current}, "
                f"allowed: {', '.join(sorted(_REPLYABLE_STATUSES))}"
            )
        result = reply_fn(repo_dir, task_id, cmd.message)
        new_status = result.get("status", "RUNNING")
        store.update_status(task_id, new_status)
        lines = ["Reply sent.", *_identity_lines(binding.ref, task_id),
                 f"- previous_status: {current}", f"- current_status: {new_status}"]
        return WfDevResponse(ok=True, text="\n".join(lines), subcommand=sub,
                             task_id=task_id, ref=binding.ref)

    if sub == "stop":
        result = stop_fn(repo_dir, task_id)
        new_status = result.get("status", "STOPPED")
        store.update_status(task_id, new_status, is_active=False)
        lines = ["Workflow stopped.", *_identity_lines(binding.ref, task_id),
                 f"- status: {new_status}"]
        return WfDevResponse(ok=True, text="\n".join(lines), subcommand=sub,
                             task_id=task_id, ref=binding.ref)

    if sub == "summary":
        result = summary_fn(repo_dir, task_id)
        summary = result.get("summary") or {}
        status = summary.get("status") or binding.status
        if status in _TERMINAL_STATUSES:
            store.update_status(task_id, status, is_active=False)
        lines = ["Workflow summary.", *_identity_lines(binding.ref, task_id),
                 f"- status: {status or 'UNKNOWN'}"]
        final = (summary.get("final_summary") or "").strip()
        if final:
            lines.append(f"- summary: {_truncate(final, 500)}")
        verification = (summary.get("verification_summary") or "").strip()
        if verification:
            lines.append(f"- verification: {_truncate(verification, 300)}")
        return WfDevResponse(ok=True, text="\n".join(lines), subcommand=sub,
                             task_id=task_id, ref=binding.ref)

    # Should be unreachable — parse_wf_dev_command validates the subcommand.
    raise WfDevError(f"unhandled subcommand: {sub}")


def _handle_plan(
    cmd: WfDevCommand,
    *,
    repo_dir: Path | str,
    store: WorkflowBindingStore,
    collect_plan_fn: Callable[..., Dict[str, str]],
    save_plan_fn: Callable[..., Dict[str, Any]],
) -> WfDevResponse:
    if not cmd.request:
        raise WfDevError("plan requires a request")
    # One active workflow per thread (design §6.1 policy).
    existing = store.resolve_active(thread_id=cmd.thread_id, channel_id=cmd.channel_id)
    if existing is not None:
        raise WfDevError(
            f"This thread already has an active workflow (ref: {existing.ref}, "
            f"task_id: {existing.task_id}). Use /wf-dev status or /wf-dev stop first."
        )
    task_id = cmd.task_id or _new_task_id(cmd)
    collected = collect_plan_fn(repo_dir, task_id, cmd.request)
    saved = save_plan_fn(
        repo_dir, task_id, collected["plan_text"],
        title=cmd.title, discord_channel_id=cmd.channel_id, user_message=cmd.request,
    )
    ref = store.next_ref()
    plan_version = saved.get("plan_version", 1)
    status = saved.get("status", "WAITING_APPROVAL")
    store.save(WorkflowBinding(
        task_id=task_id, ref=ref, platform=cmd.platform,
        channel_id=cmd.channel_id, thread_id=cmd.thread_id, guild_id=cmd.guild_id,
        session_id=cmd.session_id, title=cmd.title, status=status,
        plan_version=plan_version, is_active=True,
    ))
    lines = ["Plan registered.", *_identity_lines(ref, task_id),
             f"- status: {status}", f"- plan_version: {plan_version}",
             "- next: /wf-dev approve", "- revise: /wf-dev revise request:..."]
    return WfDevResponse(ok=True, text="\n".join(lines), subcommand="plan",
                         task_id=task_id, ref=ref)


def _handle_revise(
    cmd: WfDevCommand,
    *,
    repo_dir: Path | str,
    store: WorkflowBindingStore,
    collect_plan_fn: Callable[..., Dict[str, str]],
    save_plan_fn: Callable[..., Dict[str, Any]],
    status_fn: Callable[..., Optional[Dict[str, Any]]],
) -> WfDevResponse:
    if not cmd.request:
        raise WfDevError("revise requires a request")
    binding = _resolve_target(cmd, store, required=True)
    task_id = binding.task_id
    task = status_fn(repo_dir, task_id)
    current = (task or {}).get("status") or binding.status
    if current not in _REVISABLE_STATUSES:
        raise WfDevError(
            f"revise rejected — status is {current}, "
            f"allowed: {', '.join(sorted(_REVISABLE_STATUSES))}"
        )
    collected = collect_plan_fn(repo_dir, task_id, cmd.request)
    saved = save_plan_fn(
        repo_dir, task_id, collected["plan_text"],
        title=binding.title or cmd.title, discord_channel_id=cmd.channel_id,
        user_message=cmd.request,
    )
    plan_version = saved.get("plan_version", binding.plan_version + 1)
    status = saved.get("status", "WAITING_APPROVAL")
    store.update_status(task_id, status, plan_version=plan_version)
    lines = ["Plan revised.", *_identity_lines(binding.ref, task_id),
             f"- status: {status}", f"- plan_version: {plan_version}",
             "- next: /wf-dev approve"]
    return WfDevResponse(ok=True, text="\n".join(lines), subcommand="revise",
                         task_id=task_id, ref=binding.ref)


def _new_task_id(cmd: WfDevCommand) -> str:
    """Derive a stable task_id for a new plan.

    Prefers the Discord session id (keeps parity with the free-text path where
    task_id == session_id); otherwise derives one from the thread/channel and a
    timestamp so it is unique and operator-traceable.
    """
    if cmd.session_id:
        return cmd.session_id
    scope = cmd.thread_id or cmd.channel_id or "dev"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"dev-{scope}-{stamp}"
