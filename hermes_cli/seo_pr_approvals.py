"""Durable SEO PR approval cards and slash-command helpers.

This module deliberately lives behind a CLI/slash surface rather than a model
tool. Approval cards are durable control-plane state: Telegram buttons can be
clicked hours later and still resolve the exact stored PR action instead of
reading current chat context.
"""

from __future__ import annotations

import json
import secrets
import shlex
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from hermes_constants import get_hermes_home

_PENDING = "pending"
_CONSUMED = "consumed"
_APPROVING = "approving"
_HELD = "held"
_REVISE = "revise_requested"
_EXPIRED = "expired"
_FAILED = "failed"

_SECRET_KEY_FRAGMENTS = ("secret", "token", "password", "passwd", "api_key", "apikey")
_DEFAULT_TTL_DAYS = 14


@dataclass(frozen=True)
class SEOApprovalItem:
    approval_id: str
    created_at: str
    updated_at: str
    status: str
    source_platform: Optional[str]
    source_chat_id: Optional[str]
    source_thread_id: Optional[str]
    source_user_id: Optional[str]
    site: str
    route: str
    target_keyword: Optional[str]
    repo: str
    pr_url: str
    pr_number: int
    branch: str
    preview_url: Optional[str]
    checks_summary: Optional[str]
    checks_status: Optional[str]
    merge_payload: dict[str, Any]
    expires_at: Optional[str]
    invalid_reason: Optional[str]
    consumed_at: Optional[str]
    result_summary: Optional[str]


@dataclass(frozen=True)
class ActionResult:
    ok: bool
    approval_id: str
    status: str
    message: str
    result_summary: Optional[str] = None


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    message: str
    details: tuple[str, ...] = ()


class GitHubCLIExecutor:
    """Small subprocess wrapper for GitHub PR validation and merge.

    The executor always uses argument lists, never a shell. It does not read or
    print credentials; gh handles its own auth state.
    """

    def run_json(self, args: Sequence[str]) -> dict[str, Any]:
        completed = subprocess.run(
            list(args),
            capture_output=True,
            text=True,
            check=False,
            timeout=90,
        )
        if completed.returncode != 0:
            err = (completed.stderr or completed.stdout or "gh command failed").strip()
            raise RuntimeError(err[:500])
        if not completed.stdout.strip():
            return {}
        return json.loads(completed.stdout)

    def run(self, args: Sequence[str]) -> str:
        completed = subprocess.run(
            list(args),
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
        if completed.returncode != 0:
            err = (completed.stderr or completed.stdout or "gh command failed").strip()
            raise RuntimeError(err[:500])
        return (completed.stdout or completed.stderr or "").strip()


class SEOApprovalStore:
    """SQLite-backed durable store for Kavera SEO PR approvals."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        if db_path is None:
            db_path = get_hermes_home() / "approvals" / "seo_pr_approvals.sqlite3"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS seo_pr_approvals (
                    approval_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_platform TEXT,
                    source_chat_id TEXT,
                    source_thread_id TEXT,
                    source_user_id TEXT,
                    site TEXT NOT NULL,
                    route TEXT NOT NULL,
                    target_keyword TEXT,
                    repo TEXT NOT NULL,
                    pr_url TEXT NOT NULL,
                    pr_number INTEGER NOT NULL,
                    branch TEXT NOT NULL,
                    preview_url TEXT,
                    checks_summary TEXT,
                    checks_status TEXT,
                    merge_payload TEXT NOT NULL,
                    expires_at TEXT,
                    invalid_reason TEXT,
                    consumed_at TEXT,
                    result_summary TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_seo_pr_approvals_status "
                "ON seo_pr_approvals(status, created_at)"
            )

    def create(self, payload: dict[str, Any]) -> str:
        now = _now_iso()
        approval_id = str(payload.get("approval_id") or _new_approval_id())
        expires_at = payload.get("expires_at")
        if expires_at is None:
            expires_at = (datetime.now(timezone.utc) + timedelta(days=_DEFAULT_TTL_DAYS)).isoformat()

        merge_payload = _sanitize_payload(dict(payload.get("merge_payload") or {}))
        values = {
            "approval_id": approval_id,
            "created_at": str(payload.get("created_at") or now),
            "updated_at": now,
            "status": str(payload.get("status") or _PENDING),
            "source_platform": _optional_str(payload.get("source_platform")),
            "source_chat_id": _optional_str(payload.get("source_chat_id")),
            "source_thread_id": _optional_str(payload.get("source_thread_id")),
            "source_user_id": _optional_str(payload.get("source_user_id")),
            "site": _required_str(payload, "site"),
            "route": _required_str(payload, "route"),
            "target_keyword": _optional_str(payload.get("target_keyword")),
            "repo": _required_str(payload, "repo"),
            "pr_url": _required_str(payload, "pr_url"),
            "pr_number": int(payload.get("pr_number")),
            "branch": _required_str(payload, "branch"),
            "preview_url": _optional_str(payload.get("preview_url")),
            "checks_summary": _optional_str(payload.get("checks_summary")),
            "checks_status": _optional_str(payload.get("checks_status")),
            "merge_payload": json.dumps(merge_payload, sort_keys=True),
            "expires_at": _optional_str(expires_at),
            "invalid_reason": _optional_str(payload.get("invalid_reason")),
            "consumed_at": _optional_str(payload.get("consumed_at")),
            "result_summary": _optional_str(payload.get("result_summary")),
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO seo_pr_approvals (
                    approval_id, created_at, updated_at, status,
                    source_platform, source_chat_id, source_thread_id, source_user_id,
                    site, route, target_keyword, repo, pr_url, pr_number, branch,
                    preview_url, checks_summary, checks_status, merge_payload,
                    expires_at, invalid_reason, consumed_at, result_summary
                ) VALUES (
                    :approval_id, :created_at, :updated_at, :status,
                    :source_platform, :source_chat_id, :source_thread_id, :source_user_id,
                    :site, :route, :target_keyword, :repo, :pr_url, :pr_number, :branch,
                    :preview_url, :checks_summary, :checks_status, :merge_payload,
                    :expires_at, :invalid_reason, :consumed_at, :result_summary
                )
                """,
                values,
            )
        return approval_id

    def get(self, approval_id: str) -> Optional[SEOApprovalItem]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM seo_pr_approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
        return _row_to_item(row) if row else None

    def list(self, status: Optional[str] = _PENDING) -> list[SEOApprovalItem]:
        with self._connect() as conn:
            if status in (None, "all"):
                rows = conn.execute(
                    "SELECT * FROM seo_pr_approvals ORDER BY created_at ASC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM seo_pr_approvals WHERE status = ? ORDER BY created_at ASC",
                    (status,),
                ).fetchall()
        return [_row_to_item(row) for row in rows]

    def approve(self, approval_id: str, *, actor: Optional[str] = None,
                executor: Optional[GitHubCLIExecutor] = None) -> ActionResult:
        now = _now_iso()
        item: Optional[SEOApprovalItem]
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM seo_pr_approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
            item = _row_to_item(row) if row else None
            if item is None:
                conn.rollback()
                return ActionResult(False, approval_id, "missing", f"Approval {approval_id} was not found.")
            if item.status != _PENDING:
                conn.rollback()
                return ActionResult(False, approval_id, item.status, f"Approval {approval_id} already {item.status}.")
            if _is_expired(item, now):
                message = "Approval expired before it was acted on."
                conn.execute(
                    "UPDATE seo_pr_approvals SET status = ?, updated_at = ?, invalid_reason = ? "
                    "WHERE approval_id = ?",
                    (_EXPIRED, now, message, approval_id),
                )
                conn.commit()
                return ActionResult(False, approval_id, _EXPIRED, message)

            if bool(item.merge_payload.get("dry_run")):
                result = _dry_run_summary(item)
                conn.execute(
                    "UPDATE seo_pr_approvals SET status = ?, updated_at = ?, consumed_at = ?, "
                    "result_summary = ?, invalid_reason = NULL WHERE approval_id = ?",
                    (_CONSUMED, now, now, result, approval_id),
                )
                conn.commit()
                actor_note = f" by {actor}" if actor else ""
                return ActionResult(True, approval_id, _CONSUMED, f"Approved {approval_id}{actor_note}: {result}", result)

            conn.execute(
                "UPDATE seo_pr_approvals SET status = ?, updated_at = ? WHERE approval_id = ?",
                (_APPROVING, now, approval_id),
            )
            conn.commit()

        assert item is not None
        try:
            result = _approve_live_pr(item, executor or GitHubCLIExecutor())
        except Exception as exc:
            message = f"Approval failed validation or merge: {exc}"
            self._mark_terminal(approval_id, _FAILED, message, invalid_reason=message)
            return ActionResult(False, approval_id, _FAILED, message)

        self._mark_terminal(approval_id, _CONSUMED, result, consumed_at=_now_iso())
        return ActionResult(True, approval_id, _CONSUMED, f"Approved {approval_id}: {result}", result)

    def hold(self, approval_id: str, reason: Optional[str] = None, *, actor: Optional[str] = None) -> ActionResult:
        return self._mark_operator_decision(approval_id, _HELD, reason, actor=actor)

    def revise(self, approval_id: str, reason: Optional[str] = None, *, actor: Optional[str] = None) -> ActionResult:
        return self._mark_operator_decision(approval_id, _REVISE, reason, actor=actor)

    def _mark_operator_decision(
        self,
        approval_id: str,
        status: str,
        reason: Optional[str],
        *,
        actor: Optional[str] = None,
    ) -> ActionResult:
        now = _now_iso()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM seo_pr_approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
            item = _row_to_item(row) if row else None
            if item is None:
                conn.rollback()
                return ActionResult(False, approval_id, "missing", f"Approval {approval_id} was not found.")
            if item.status != _PENDING:
                conn.rollback()
                return ActionResult(False, approval_id, item.status, f"Approval {approval_id} already {item.status}.")
            summary = _decision_summary(status, reason, actor)
            conn.execute(
                "UPDATE seo_pr_approvals SET status = ?, updated_at = ?, result_summary = ?, "
                "invalid_reason = ? WHERE approval_id = ?",
                (status, now, summary, reason, approval_id),
            )
            conn.commit()
        return ActionResult(True, approval_id, status, f"Approval {approval_id} {status.replace('_', ' ')}: {summary}", summary)

    def _mark_terminal(
        self,
        approval_id: str,
        status: str,
        result_summary: str,
        *,
        invalid_reason: Optional[str] = None,
        consumed_at: Optional[str] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE seo_pr_approvals SET status = ?, updated_at = ?, consumed_at = ?, "
                "result_summary = ?, invalid_reason = ? WHERE approval_id = ?",
                (status, _now_iso(), consumed_at, result_summary, invalid_reason, approval_id),
            )


def validate_pr_snapshot(
    item: SEOApprovalItem,
    snapshot: dict[str, Any],
    files: Optional[Iterable[Any]] = None,
) -> ValidationResult:
    """Validate that a live PR snapshot still matches the stored approval."""

    problems: list[str] = []
    payload = item.merge_payload or {}

    expected_repo = str(payload.get("expected_repo") or item.repo or "").strip()
    snapshot_repo = _repo_from_snapshot(snapshot)
    if expected_repo and snapshot_repo and snapshot_repo.lower() != expected_repo.lower():
        problems.append(f"target repo changed from {expected_repo} to {snapshot_repo}")

    try:
        if int(snapshot.get("number")) != int(item.pr_number):
            problems.append(f"PR number changed from {item.pr_number} to {snapshot.get('number')}")
    except (TypeError, ValueError):
        problems.append("live PR number is missing or invalid")

    live_url = str(snapshot.get("url") or "").rstrip("/")
    if live_url and live_url != item.pr_url.rstrip("/"):
        problems.append("live PR URL no longer matches the stored approval")

    state = str(snapshot.get("state") or "").upper()
    if state and state != "OPEN":
        problems.append(f"PR is not open (state={state.lower()})")
    if snapshot.get("mergedAt"):
        problems.append("PR is already merged")
    if bool(snapshot.get("isDraft")):
        problems.append("PR is still a draft")

    live_branch = str(snapshot.get("headRefName") or "").strip()
    if live_branch != item.branch:
        problems.append(f"branch changed from {item.branch} to {live_branch or '(missing)'}")

    expected_base = str(payload.get("expected_base") or "main").strip()
    live_base = str(snapshot.get("baseRefName") or "").strip()
    if expected_base and live_base and live_base != expected_base:
        problems.append(f"base branch is {live_base}, expected {expected_base}")

    file_paths = tuple(_file_path(f) for f in (files or snapshot.get("files") or []) if _file_path(f))
    blocked_paths = tuple(p for p in file_paths if _is_blocked_pr_path(p))
    if blocked_paths and not bool(payload.get("allow_env_or_vercel_changes")):
        sample = ", ".join(blocked_paths[:3])
        problems.append(f"PR changes production env/Vercel settings ({sample})")

    if not bool(payload.get("allow_non_passing_checks")):
        checks_problem = _checks_problem(item, snapshot)
        if checks_problem:
            problems.append(checks_problem)

    if problems:
        return ValidationResult(False, "; ".join(problems), tuple(problems))
    return ValidationResult(True, "PR snapshot matches the stored approval.")


def run_slash(text: str) -> str:
    """Run the /approvals or /seo-approve slash surface."""

    text = _strip_command_prefix(text or "")
    try:
        tokens = shlex.split(text)
    except ValueError as exc:
        return f"Could not parse approvals command: {exc}"

    if not tokens:
        return _usage()

    action = tokens[0].lower().replace("_", "-")
    args = tokens[1:]
    store = SEOApprovalStore()

    if action in {"list", "pending", "ls"}:
        status = args[0].lower() if args else _PENDING
        if status == "pending":
            status = _PENDING
        return format_approval_list(store.list(status=status))

    if action == "show":
        if not args:
            return "Usage: /approvals show <approval_id>"
        item = store.get(args[0])
        return format_approval_detail(item) if item else f"Approval {args[0]} was not found."

    if action in {"demo", "create-demo"}:
        if args and args[0].lower() not in {"create", "new"}:
            return "Usage: /approvals demo create"
        approval_id = create_demo_approval(store)
        return (
            f"Created demo approval {approval_id}.\n"
            f"List it with /approvals list, approve it with /approvals approve {approval_id}, "
            f"or hold it with /approvals hold {approval_id}."
        )

    if action == "approve":
        if not args:
            return "Usage: /approvals approve <approval_id...|all>"
        return _format_batch_results(_approve_args(store, args))

    if action in {"hold", "revise"}:
        if not args:
            return f"Usage: /approvals {action} <approval_id...|all> [reason]"
        ids, reason = _split_ids_and_reason(args)
        if not ids:
            return f"Usage: /approvals {action} <approval_id...|all> [reason]"
        targets = [item.approval_id for item in store.list(status=_PENDING)] if ids == ["all"] else ids
        results = [getattr(store, action)(approval_id, reason, actor="slash") for approval_id in targets]
        return _format_batch_results(results, verb=action.capitalize())

    return _usage()


def create_demo_approval(store: Optional[SEOApprovalStore] = None) -> str:
    store = store or SEOApprovalStore()
    suffix = secrets.token_hex(3)
    return store.create(
        {
            "site": "Kavera SEO Demo",
            "route": f"/services/demo-{suffix}",
            "target_keyword": "demo local SEO page",
            "repo": "KaveraAI/kavera-seo-pages",
            "pr_url": f"https://github.com/KaveraAI/kavera-seo-pages/pull/{int(suffix, 16) % 9000 + 1000}",
            "pr_number": int(suffix, 16) % 9000 + 1000,
            "branch": f"seo/demo-{suffix}",
            "preview_url": f"https://demo-{suffix}.vercel.app/services/demo-{suffix}",
            "checks_summary": "demo checks passed (dry-run)",
            "checks_status": "passed",
            "merge_payload": {"dry_run": True, "merge_method": "squash"},
        }
    )


def format_approval_list(items: Sequence[SEOApprovalItem]) -> str:
    if not items:
        return "No SEO PR approvals found."
    lines = [f"SEO PR approvals ({len(items)}):"]
    for item in items:
        lines.append(
            f"- {item.approval_id} [{item.status}] {item.site} {item.route} "
            f"PR #{item.pr_number} ({item.branch}) checks={item.checks_summary or item.checks_status or 'unknown'}"
        )
    return "\n".join(lines)


def format_approval_detail(item: Optional[SEOApprovalItem]) -> str:
    if item is None:
        return "Approval not found."
    lines = [
        f"SEO PR approval {item.approval_id}",
        f"Status: {item.status}",
        f"Site: {item.site}",
        f"Route: {item.route}",
        f"Target keyword: {item.target_keyword or '(none)'}",
        f"Repo: {item.repo}",
        f"PR: {item.pr_url}",
        f"Branch: {item.branch}",
        f"Preview: {item.preview_url or '(none)'}",
        f"Checks: {item.checks_summary or item.checks_status or 'unknown'}",
        f"Expires: {item.expires_at or '(none)'}",
    ]
    if item.invalid_reason:
        lines.append(f"Invalid reason: {item.invalid_reason}")
    if item.result_summary:
        lines.append(f"Result: {item.result_summary}")
    return "\n".join(lines)


def format_approval_card_text(item: SEOApprovalItem, *, command_prefix: str = "/") -> str:
    prefix = command_prefix or "/"
    command = f"{prefix}approvals" if prefix.endswith("/") else f"{prefix}approvals"
    return "\n".join(
        [
            "🧾 Kavera SEO PR approval",
            f"Site: {item.site}",
            f"Route: {item.route}",
            f"Target keyword: {item.target_keyword or '(none)'}",
            f"PR: {item.pr_url}",
            f"Preview: {item.preview_url or '(none)'}",
            f"Checks: {item.checks_summary or item.checks_status or 'unknown'}",
            f"Approval ID: {item.approval_id}",
            "",
            f"Approve: {command} approve {item.approval_id}",
            f"Revise: {command} revise {item.approval_id} <reason>",
            f"Hold: {command} hold {item.approval_id} <reason>",
        ]
    )


def handle_callback_action(action: str, approval_id: str, *, actor: Optional[str] = None) -> ActionResult:
    store = SEOApprovalStore()
    normalized = action.strip().lower()
    if normalized == "approve":
        return store.approve(approval_id, actor=actor)
    if normalized == "hold":
        return store.hold(approval_id, "Held from Telegram", actor=actor)
    if normalized == "revise":
        return store.revise(approval_id, "Revision requested from Telegram", actor=actor)
    return ActionResult(False, approval_id, "invalid", f"Unknown approval action: {action}")


def _approve_args(store: SEOApprovalStore, args: Sequence[str]) -> list[ActionResult]:
    ids = [arg for arg in args if not arg.startswith("--")]
    targets = [item.approval_id for item in store.list(status=_PENDING)] if ids == ["all"] else ids
    return [store.approve(approval_id, actor="slash") for approval_id in targets]


def _format_batch_results(results: Sequence[ActionResult], *, verb: str = "Approved") -> str:
    if not results:
        return f"{verb} 0 approvals."
    ok_count = sum(1 for result in results if result.ok)
    lines = [f"{verb} {ok_count} approval{'s' if ok_count != 1 else ''}."]
    for result in results:
        marker = "✓" if result.ok else "!"
        lines.append(f"- {marker} {result.approval_id}: {result.message}")
    return "\n".join(lines)


def _split_ids_and_reason(args: Sequence[str]) -> tuple[list[str], Optional[str]]:
    if not args:
        return [], None
    if args[0].lower() == "all":
        return ["all"], " ".join(args[1:]).strip() or None
    ids: list[str] = []
    reason_parts: list[str] = []
    for arg in args:
        if not reason_parts and (arg.startswith("apr_") or arg == "all"):
            ids.append(arg)
            continue
        reason_parts.append(arg)
    return ids, " ".join(reason_parts).strip() or None


def _approve_live_pr(item: SEOApprovalItem, executor: GitHubCLIExecutor) -> str:
    fields = "number,url,state,isDraft,headRefName,baseRefName,mergedAt,statusCheckRollup,files"
    snapshot = executor.run_json([
        "gh", "pr", "view", str(item.pr_number),
        "--repo", item.repo,
        "--json", fields,
    ])
    validation = validate_pr_snapshot(item, snapshot, files=snapshot.get("files") or [])
    if not validation.ok:
        raise RuntimeError(validation.message)

    merge_method = str(item.merge_payload.get("merge_method") or "squash").lower()
    if merge_method not in {"squash", "merge", "rebase"}:
        merge_method = "squash"
    merge_flag = f"--{merge_method}"
    output = executor.run([
        "gh", "pr", "merge", str(item.pr_number),
        "--repo", item.repo,
        merge_flag,
        "--delete-branch",
    ])
    return output or f"Merged PR #{item.pr_number} with {merge_method}."


def _checks_problem(item: SEOApprovalItem, snapshot: dict[str, Any]) -> Optional[str]:
    status = (item.checks_status or "").strip().lower()
    rollup = snapshot.get("statusCheckRollup") or []
    if not rollup:
        if status in {"passed", "pass", "success", "successful", "green"}:
            return None
        return "checks are not confirmed as passing"

    bad: list[str] = []
    for check in rollup:
        if not isinstance(check, dict):
            continue
        conclusion = str(check.get("conclusion") or check.get("state") or "").upper()
        check_status = str(check.get("status") or "").upper()
        name = str(check.get("name") or check.get("context") or "check")
        if conclusion in {"SUCCESS", "SKIPPED", "NEUTRAL"}:
            continue
        if check_status and check_status != "COMPLETED":
            bad.append(f"{name} is {check_status.lower()}")
            continue
        if conclusion:
            bad.append(f"{name} concluded {conclusion.lower()}")
    if bad:
        return "checks are not passing: " + ", ".join(bad[:5])
    return None


def _repo_from_snapshot(snapshot: dict[str, Any]) -> Optional[str]:
    repo = snapshot.get("repository")
    if isinstance(repo, dict):
        name_with_owner = repo.get("nameWithOwner")
        if name_with_owner:
            return str(name_with_owner)
    return None


def _file_path(file_entry: Any) -> Optional[str]:
    if isinstance(file_entry, str):
        return file_entry
    if isinstance(file_entry, dict):
        for key in ("path", "filename", "name"):
            value = file_entry.get(key)
            if value:
                return str(value)
    return None


def _is_blocked_pr_path(path: str) -> bool:
    lowered = path.strip().lower()
    if not lowered:
        return False
    if lowered == "vercel.json" or lowered.startswith(".vercel/"):
        return True
    if "vercel" in lowered and (lowered.endswith(".json") or lowered.endswith(".yaml") or lowered.endswith(".yml")):
        return True
    name = lowered.rsplit("/", 1)[-1]
    if name.startswith(".env") or name in {"env.production", "production.env"}:
        return True
    return False


def _is_expired(item: SEOApprovalItem, now_iso: str) -> bool:
    if not item.expires_at:
        return False
    expiry = _parse_dt(item.expires_at)
    now = _parse_dt(now_iso)
    return expiry is not None and now is not None and expiry <= now


def _dry_run_summary(item: SEOApprovalItem) -> str:
    return f"dry-run merge recorded for {item.repo} PR #{item.pr_number} ({item.branch})"


def _decision_summary(status: str, reason: Optional[str], actor: Optional[str]) -> str:
    label = "held" if status == _HELD else "revision requested"
    actor_note = f" by {actor}" if actor else ""
    reason_note = f": {reason}" if reason else ""
    return f"{label}{actor_note}{reason_note}"


def _row_to_item(row: sqlite3.Row) -> SEOApprovalItem:
    payload_text = row["merge_payload"] or "{}"
    try:
        merge_payload = json.loads(payload_text)
    except json.JSONDecodeError:
        merge_payload = {}
    return SEOApprovalItem(
        approval_id=row["approval_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        status=row["status"],
        source_platform=row["source_platform"],
        source_chat_id=row["source_chat_id"],
        source_thread_id=row["source_thread_id"],
        source_user_id=row["source_user_id"],
        site=row["site"],
        route=row["route"],
        target_keyword=row["target_keyword"],
        repo=row["repo"],
        pr_url=row["pr_url"],
        pr_number=int(row["pr_number"]),
        branch=row["branch"],
        preview_url=row["preview_url"],
        checks_summary=row["checks_summary"],
        checks_status=row["checks_status"],
        merge_payload=merge_payload,
        expires_at=row["expires_at"],
        invalid_reason=row["invalid_reason"],
        consumed_at=row["consumed_at"],
        result_summary=row["result_summary"],
    )


def _sanitize_payload(value: Any, *, key: str = "") -> Any:
    lowered_key = key.lower()
    if any(fragment in lowered_key for fragment in _SECRET_KEY_FRAGMENTS):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): _sanitize_payload(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_payload(v, key=key) for v in value]
    return value


def _strip_command_prefix(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("/"):
        stripped = stripped[1:]
    for command in ("approvals", "seo-approve"):
        if stripped == command:
            return ""
        if stripped.startswith(command + " "):
            return stripped[len(command):].strip()
    return stripped


def _usage() -> str:
    return (
        "Usage: /approvals list|show <id>|approve <id...|all>|hold <id...|all> [reason]|"
        "revise <id...|all> [reason]|demo create"
    )


def _new_approval_id() -> str:
    return "apr_" + secrets.token_urlsafe(10).rstrip("=")


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"Missing required approval field: {key}")
    return value


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    raw = str(value).strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
