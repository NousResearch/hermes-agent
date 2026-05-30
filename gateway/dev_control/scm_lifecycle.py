"""SCM lifecycle control plane for Dev PR review and guarded merges."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from gateway.dev_control.ci_status import fetch_ci_status


GITHUB_API_BASE = "https://api.github.com"
CODE_REVIEW_OBJECT = "hermes.dev_code_review_result"
CODE_REVIEW_RE = re.compile(
    r"```(?:json)?\s*(?:DEV_CODE_REVIEW_RESULT|dev_code_review_result)?\s*(\{.*?\})\s*```",
    re.IGNORECASE | re.DOTALL,
)
VALID_REVIEW_VERDICTS = {"approved", "changes_requested", "commented", "unknown"}
BLOCKING_VERIFICATION_VERDICTS = {"failed", "needs_review"}


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dev_pr_states (
    pr_key TEXT PRIMARY KEY,
    repo TEXT NOT NULL,
    plan_id TEXT,
    task_id TEXT,
    branch TEXT,
    pr_number INTEGER,
    pr_url TEXT,
    head_sha TEXT,
    ci_state TEXT,
    ci_status TEXT,
    review_state TEXT,
    mergeable INTEGER,
    merge_state TEXT,
    warnings TEXT,
    raw TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dev_pr_states_plan
    ON dev_pr_states(plan_id, task_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS dev_code_review_runs (
    review_run_id TEXT PRIMARY KEY,
    repo TEXT NOT NULL,
    plan_id TEXT,
    task_id TEXT,
    pr_number INTEGER,
    head_sha TEXT,
    status TEXT NOT NULL,
    verdict TEXT NOT NULL,
    findings TEXT NOT NULL,
    summary TEXT,
    evidence_refs TEXT NOT NULL,
    profile_id TEXT,
    permissions TEXT,
    prompt TEXT,
    warnings TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dev_code_review_runs_pr
    ON dev_code_review_runs(repo, pr_number, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_dev_code_review_runs_task
    ON dev_code_review_runs(plan_id, task_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS dev_merge_readiness (
    readiness_id TEXT PRIMARY KEY,
    repo TEXT NOT NULL,
    plan_id TEXT,
    task_id TEXT,
    pr_number INTEGER,
    head_sha TEXT,
    ready INTEGER NOT NULL,
    blocked_by TEXT NOT NULL,
    gates TEXT NOT NULL,
    pr_state TEXT NOT NULL,
    verification TEXT NOT NULL,
    code_review TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dev_merge_readiness_pr
    ON dev_merge_readiness(repo, pr_number, created_at DESC);

CREATE TABLE IF NOT EXISTS dev_merge_approvals (
    approval_id TEXT PRIMARY KEY,
    repo TEXT NOT NULL,
    plan_id TEXT,
    task_id TEXT,
    pr_number INTEGER NOT NULL,
    head_sha TEXT NOT NULL,
    status TEXT NOT NULL,
    gate_snapshot TEXT NOT NULL,
    requested_by TEXT,
    approved_by TEXT,
    created_at REAL NOT NULL,
    approved_at REAL,
    consumed_at REAL,
    invalidated_at REAL,
    message TEXT
);
CREATE INDEX IF NOT EXISTS idx_dev_merge_approvals_pr
    ON dev_merge_approvals(repo, pr_number, created_at DESC);

CREATE TABLE IF NOT EXISTS dev_merge_audits (
    audit_id TEXT PRIMARY KEY,
    approval_id TEXT,
    repo TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    head_sha TEXT,
    result TEXT NOT NULL,
    reason TEXT,
    approval_snapshot TEXT NOT NULL,
    execution_snapshot TEXT NOT NULL,
    executor TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dev_merge_audits_pr
    ON dev_merge_audits(repo, pr_number, created_at DESC);
"""


class DevSCMLifecycleStore:
    """SQLite state for PR, review, readiness, merge approval, and audit rows."""

    def __init__(self, db_path: Optional[Path | str] = None):
        self.db_path = Path(db_path or os.path.expanduser("~/.hermes/state.db"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA_SQL)

    def upsert_pr_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        now = time.time()
        repo = _require_text(state.get("repo"), "repo")
        pr_number = _require_int(state.get("pr_number"), "pr_number")
        pr_key = _pr_key(repo, pr_number)
        created_at = float(state.get("created_at") or now)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO dev_pr_states (
                    pr_key, repo, plan_id, task_id, branch, pr_number, pr_url,
                    head_sha, ci_state, ci_status, review_state, mergeable,
                    merge_state, warnings, raw, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pr_key) DO UPDATE SET
                    plan_id = excluded.plan_id,
                    task_id = excluded.task_id,
                    branch = excluded.branch,
                    pr_url = excluded.pr_url,
                    head_sha = excluded.head_sha,
                    ci_state = excluded.ci_state,
                    ci_status = excluded.ci_status,
                    review_state = excluded.review_state,
                    mergeable = excluded.mergeable,
                    merge_state = excluded.merge_state,
                    warnings = excluded.warnings,
                    raw = excluded.raw,
                    updated_at = excluded.updated_at
                """,
                (
                    pr_key,
                    repo,
                    _optional_text(state.get("plan_id")),
                    _optional_text(state.get("task_id")),
                    _optional_text(state.get("branch")),
                    pr_number,
                    _optional_text(state.get("pr_url")),
                    _optional_text(state.get("head_sha")),
                    _optional_text(state.get("ci_state")),
                    json.dumps(state.get("ci_status") or {}, ensure_ascii=False),
                    _optional_text(state.get("review_state")),
                    _bool_to_db(state.get("mergeable")),
                    _optional_text(state.get("merge_state")),
                    json.dumps(state.get("warnings") or [], ensure_ascii=False),
                    json.dumps(state.get("raw") or {}, ensure_ascii=False),
                    created_at,
                    now,
                ),
            )
        return self.get_pr_state(repo=repo, pr_number=pr_number) or dict(state)

    def get_pr_state(self, *, repo: str, pr_number: int) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM dev_pr_states WHERE pr_key = ?",
            (_pr_key(repo, pr_number),),
        ).fetchone()
        return _pr_state_from_row(row)

    def record_code_review(self, result: Dict[str, Any]) -> Dict[str, Any]:
        now = time.time()
        review_run_id = str(result.get("review_run_id") or f"devreview-{uuid.uuid4().hex[:10]}")
        repo = _require_text(result.get("repo"), "repo")
        pr_number = _require_int(result.get("pr_number"), "pr_number")
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO dev_code_review_runs (
                    review_run_id, repo, plan_id, task_id, pr_number, head_sha,
                    status, verdict, findings, summary, evidence_refs, profile_id,
                    permissions, prompt, warnings, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review_run_id,
                    repo,
                    _optional_text(result.get("plan_id")),
                    _optional_text(result.get("task_id")),
                    pr_number,
                    _optional_text(result.get("head_sha")),
                    str(result.get("status") or "completed"),
                    _normalize_review_verdict(result.get("verdict")),
                    json.dumps(result.get("findings") or [], ensure_ascii=False),
                    _optional_text(result.get("summary")),
                    json.dumps(result.get("evidence_refs") or [], ensure_ascii=False),
                    str(result.get("profile_id") or "review"),
                    str(result.get("permissions") or "review_only"),
                    _optional_text(result.get("prompt")),
                    json.dumps(result.get("warnings") or [], ensure_ascii=False),
                    now,
                    now,
                ),
            )
        return self.get_code_review(review_run_id) or dict(result)

    def get_code_review(self, review_run_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM dev_code_review_runs WHERE review_run_id = ?",
            (str(review_run_id or "").strip(),),
        ).fetchone()
        return _code_review_from_row(row)

    def latest_code_review(
        self,
        *,
        repo: str,
        pr_number: int,
        head_sha: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        params: list[Any] = [repo, int(pr_number)]
        sql = """
            SELECT *
            FROM dev_code_review_runs
            WHERE repo = ? AND pr_number = ?
        """
        if head_sha:
            sql += " AND head_sha = ?"
            params.append(head_sha)
        sql += " ORDER BY updated_at DESC LIMIT 1"
        row = self._conn.execute(sql, tuple(params)).fetchone()
        return _code_review_from_row(row)

    def list_code_reviews(
        self,
        *,
        repo: Optional[str] = None,
        pr_number: Optional[int] = None,
        limit: int = 50,
    ) -> list[Dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if repo:
            clauses.append("repo = ?")
            params.append(str(repo).strip())
        if pr_number:
            clauses.append("pr_number = ?")
            params.append(int(pr_number))
        sql = "SELECT * FROM dev_code_review_runs"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, min(int(limit or 50), 200)))
        rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [item for row in rows if (item := _code_review_from_row(row))]

    def record_readiness(self, readiness: Dict[str, Any]) -> Dict[str, Any]:
        now = time.time()
        readiness_id = str(readiness.get("readiness_id") or f"devmerge-ready-{uuid.uuid4().hex[:10]}")
        repo = _require_text(readiness.get("repo"), "repo")
        pr_number = _require_int(readiness.get("pr_number"), "pr_number")
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO dev_merge_readiness (
                    readiness_id, repo, plan_id, task_id, pr_number, head_sha,
                    ready, blocked_by, gates, pr_state, verification, code_review,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    readiness_id,
                    repo,
                    _optional_text(readiness.get("plan_id")),
                    _optional_text(readiness.get("task_id")),
                    pr_number,
                    _optional_text(readiness.get("head_sha")),
                    1 if readiness.get("ready") else 0,
                    json.dumps(readiness.get("blocked_by") or [], ensure_ascii=False),
                    json.dumps(readiness.get("gates") or {}, ensure_ascii=False),
                    json.dumps(readiness.get("pr_state") or {}, ensure_ascii=False),
                    json.dumps(readiness.get("verification") or {}, ensure_ascii=False),
                    json.dumps(readiness.get("code_review") or {}, ensure_ascii=False),
                    now,
                ),
            )
        return self.get_readiness(readiness_id) or dict(readiness)

    def get_readiness(self, readiness_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM dev_merge_readiness WHERE readiness_id = ?",
            (str(readiness_id or "").strip(),),
        ).fetchone()
        return _readiness_from_row(row)

    def latest_readiness(self, *, limit: int = 10) -> list[Dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT *
            FROM dev_merge_readiness
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (max(1, min(int(limit or 10), 50)),),
        ).fetchall()
        return [item for row in rows if (item := _readiness_from_row(row))]

    def create_merge_approval(
        self,
        *,
        repo: str,
        pr_number: int,
        head_sha: str,
        gate_snapshot: Dict[str, Any],
        plan_id: Optional[str] = None,
        task_id: Optional[str] = None,
        requested_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not str(head_sha or "").strip():
            raise ValueError("head_sha is required for merge approval.")
        now = time.time()
        approval_id = f"devmerge-appr-{uuid.uuid4().hex[:10]}"
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO dev_merge_approvals (
                    approval_id, repo, plan_id, task_id, pr_number, head_sha,
                    status, gate_snapshot, requested_by, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    approval_id,
                    repo,
                    _optional_text(plan_id),
                    _optional_text(task_id),
                    int(pr_number),
                    str(head_sha).strip(),
                    "pending",
                    json.dumps(gate_snapshot or {}, ensure_ascii=False),
                    _optional_text(requested_by),
                    now,
                ),
            )
        approval = self.get_merge_approval(approval_id)
        if not approval:
            raise RuntimeError("Merge approval was not persisted")
        return approval

    def get_merge_approval(self, approval_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM dev_merge_approvals WHERE approval_id = ?",
            (str(approval_id or "").strip(),),
        ).fetchone()
        return _approval_from_row(row)

    def approve_merge_approval(
        self,
        approval_id: str,
        *,
        approved_by: Optional[str],
        message: Optional[str] = None,
    ) -> Dict[str, Any]:
        approval = self.get_merge_approval(approval_id)
        if not approval:
            raise KeyError(f"Merge approval not found: {approval_id}")
        if approval["status"] != "pending":
            raise ValueError(f"Merge approval is {approval['status']}, not pending.")
        now = time.time()
        with self._conn:
            self._conn.execute(
                """
                UPDATE dev_merge_approvals
                SET status = ?, approved_by = ?, approved_at = ?, message = ?
                WHERE approval_id = ?
                """,
                ("approved", _optional_text(approved_by), now, _optional_text(message), approval_id),
            )
        return self.get_merge_approval(approval_id) or approval

    def mark_merge_approval(
        self,
        approval_id: str,
        *,
        status: str,
        message: Optional[str],
    ) -> Dict[str, Any]:
        if status not in {"consumed", "invalidated", "denied"}:
            raise ValueError(f"Unsupported merge approval status: {status}")
        approval = self.get_merge_approval(approval_id)
        if not approval:
            raise KeyError(f"Merge approval not found: {approval_id}")
        now = time.time()
        timestamp_column = "consumed_at" if status == "consumed" else "invalidated_at"
        with self._conn:
            self._conn.execute(
                f"""
                UPDATE dev_merge_approvals
                SET status = ?, {timestamp_column} = ?, message = ?
                WHERE approval_id = ?
                """,
                (status, now, _optional_text(message), approval_id),
            )
        return self.get_merge_approval(approval_id) or approval

    def record_merge_audit(
        self,
        *,
        approval_id: Optional[str],
        repo: str,
        pr_number: int,
        head_sha: Optional[str],
        result: str,
        reason: Optional[str],
        approval_snapshot: Dict[str, Any],
        execution_snapshot: Dict[str, Any],
        executor: Optional[str],
    ) -> Dict[str, Any]:
        audit_id = f"devmerge-audit-{uuid.uuid4().hex[:10]}"
        now = time.time()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO dev_merge_audits (
                    audit_id, approval_id, repo, pr_number, head_sha, result,
                    reason, approval_snapshot, execution_snapshot, executor, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit_id,
                    _optional_text(approval_id),
                    repo,
                    int(pr_number),
                    _optional_text(head_sha),
                    result,
                    _optional_text(reason),
                    json.dumps(approval_snapshot or {}, ensure_ascii=False),
                    json.dumps(execution_snapshot or {}, ensure_ascii=False),
                    _optional_text(executor),
                    now,
                ),
            )
        row = self._conn.execute("SELECT * FROM dev_merge_audits WHERE audit_id = ?", (audit_id,)).fetchone()
        return _audit_from_row(row) or {"audit_id": audit_id, "result": result}


def fetch_pr_state(
    *,
    repo: str,
    pr_number: int,
    plan_id: Optional[str] = None,
    task_id: Optional[str] = None,
    ci_status_fetcher: Callable[..., Dict[str, Any]] = fetch_ci_status,
    opener: Any = None,
    timeout_seconds: float = 8.0,
) -> Dict[str, Any]:
    """Fetch GitHub PR state and CI status; fail open to unknown on errors."""

    repo = str(repo or "").strip()
    try:
        pr_number = int(pr_number)
    except Exception:
        return _unknown_pr_state(repo=repo, pr_number=0, warning="pr_number is required.")
    if not _valid_repo(repo) or pr_number <= 0:
        return _unknown_pr_state(repo=repo, pr_number=pr_number, warning="repo must be owner/name and pr_number is required.")
    try:
        payload = _github_get_json(
            f"{GITHUB_API_BASE}/repos/{repo}/pulls/{pr_number}",
            opener=opener or urllib.request.urlopen,
            timeout_seconds=timeout_seconds,
        )
        head = payload.get("head") if isinstance(payload, dict) else {}
        branch = str((head or {}).get("ref") or "").strip()
        head_sha = str((head or {}).get("sha") or "").strip()
        ci_status = ci_status_fetcher(repo=repo, ref=head_sha) if head_sha else _unknown_ci("PR head_sha unavailable.")
        return {
            "ok": True,
            "object": "hermes.dev_pr_state",
            "repo": repo,
            "plan_id": plan_id,
            "task_id": task_id,
            "branch": branch,
            "pr_number": pr_number,
            "pr_url": payload.get("html_url"),
            "head_sha": head_sha,
            "ci_state": str((ci_status or {}).get("state") or "unknown"),
            "ci_status": ci_status,
            "review_state": _review_state(payload),
            "mergeable": payload.get("mergeable"),
            "merge_state": payload.get("mergeable_state") or payload.get("state"),
            "warnings": list((ci_status or {}).get("warnings") or []),
            "raw": {
                "state": payload.get("state"),
                "draft": payload.get("draft"),
                "mergeable_state": payload.get("mergeable_state"),
            },
        }
    except Exception as exc:
        return _unknown_pr_state(repo=repo, pr_number=pr_number, warning=f"PR state unavailable: {exc}")


def parse_code_review_result(text_or_payload: Any) -> Dict[str, Any]:
    """Parse a structured review result from a dict or worker output."""

    warnings: list[str] = []
    if isinstance(text_or_payload, dict):
        payload = dict(text_or_payload)
    else:
        text = str(text_or_payload or "")
        match = CODE_REVIEW_RE.search(text)
        if not match:
            payload = _extract_unfenced_code_review_payload(text)
            if payload:
                warnings.append("DEV_CODE_REVIEW_RESULT block was missing; recovered review JSON from transcript.")
            else:
                warnings.append("DEV_CODE_REVIEW_RESULT block was missing.")
        else:
            try:
                payload = json.loads(match.group(1))
            except Exception as exc:
                payload = _extract_unfenced_code_review_payload(text)
                if payload:
                    warnings.append(f"DEV_CODE_REVIEW_RESULT JSON was invalid: {exc}; recovered review JSON from transcript.")
                else:
                    payload = {}
                    warnings.append(f"DEV_CODE_REVIEW_RESULT JSON was invalid: {exc}")
    payload = _unwrap_code_review_payload(payload)
    verdict = _normalize_review_verdict(payload.get("verdict"))
    findings = payload.get("findings") if isinstance(payload.get("findings"), list) else []
    if payload.get("object") and payload.get("object") != CODE_REVIEW_OBJECT:
        warnings.append("Unexpected code-review result object type.")
    return {
        "object": CODE_REVIEW_OBJECT,
        "verdict": verdict,
        "findings": [_normalize_finding(item) for item in findings if isinstance(item, dict)],
        "summary": str(payload.get("summary") or "").strip(),
        "evidence_refs": payload.get("evidence_refs") if isinstance(payload.get("evidence_refs"), list) else [],
        "warnings": [*warnings, *[str(item) for item in payload.get("warnings", []) if str(item).strip()]]
        if isinstance(payload.get("warnings"), list)
        else warnings,
    }


def _extract_unfenced_code_review_payload(text: str) -> Dict[str, Any]:
    """Recover a code-review JSON object that was emitted without the fence."""

    if CODE_REVIEW_OBJECT not in text:
        return {}
    payload = _extract_json_object_containing(text, CODE_REVIEW_OBJECT)
    if payload:
        return _unwrap_code_review_payload(payload)
    verdict_match = re.search(r'"verdict"\s*:\s*"(approved|changes_requested|commented|unknown)"', text, re.IGNORECASE)
    if not verdict_match:
        return {}
    summary = ""
    summary_match = re.search(
        r'"summary"\s*:\s*"(?P<summary>.*?)(?<!\\)"\s*,\s*"evidence_refs"',
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if summary_match:
        summary = re.sub(r"\s+", " ", summary_match.group("summary")).strip()
    evidence_refs: list[str] = []
    evidence_match = re.search(r'"evidence_refs"\s*:\s*\[(?P<refs>.*?)\]', text, re.IGNORECASE | re.DOTALL)
    if evidence_match:
        evidence_refs = [
            re.sub(r"\s+", " ", item).strip()
            for item in re.findall(r'"(.*?)(?<!\\)"', evidence_match.group("refs"), re.DOTALL)
            if item.strip()
        ]
    return {
        "object": CODE_REVIEW_OBJECT,
        "verdict": verdict_match.group(1).lower(),
        "findings": [],
        "summary": summary,
        "evidence_refs": evidence_refs,
    }


def _unwrap_code_review_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    wrapped = payload.get(CODE_REVIEW_OBJECT)
    if isinstance(wrapped, dict):
        return {"object": CODE_REVIEW_OBJECT, **wrapped}
    return payload


def _extract_json_object_containing(text: str, marker: str) -> Dict[str, Any]:
    marker_index = text.rfind(marker)
    if marker_index < 0:
        return {}
    starts = [match.start() for match in re.finditer(r"\{", text[:marker_index])]
    for start in reversed(starts):
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:index + 1]
                    try:
                        payload = json.loads(candidate)
                    except Exception:
                        break
                    if payload.get("object") == marker or isinstance(payload.get(marker), dict):
                        return payload
                    break
    return {}


def build_code_review_prompt(*, plan: Dict[str, Any], pr_state: Dict[str, Any]) -> str:
    title = str(plan.get("title") or plan.get("plan_id") or "Dev plan").strip()
    repo = str(pr_state.get("repo") or "").strip()
    pr_number = str(pr_state.get("pr_number") or "").strip()
    criteria = []
    for task in plan.get("tasks") or []:
        for criterion in task.get("acceptance_criteria") or []:
            criteria.append(str(criterion))
    return "\n".join([
        "You are the independent Dev code-review worker. Perform a direct PR diff review only.",
        "Profile: review; permissions: review_only.",
        "Review the PR diff against the approved plan intent and acceptance criteria using only direct GitHub diff evidence.",
        "Do not edit files, create commits, approve a merge, comment on GitHub, or change branch state.",
        "Do not run slash commands such as /review. Do not run CodeRabbit, coderabbit, review agents, tests, builds, or linters.",
        "Allowed evidence commands, if you need shell evidence, are exactly:",
        f"- gh pr view {pr_number} --repo {repo} --json number,title,headRefOid,headRefName,baseRefName,isDraft,state,files,commits",
        f"- gh pr diff {pr_number} --repo {repo} --name-only",
        f"- gh pr diff {pr_number} --repo {repo} --patch",
        "After inspecting the diff, answer directly with the required JSON block. Do not start background tools.",
        f"Repository: {repo}",
        f"PR: #{pr_number} ({pr_state.get('pr_url') or 'no URL'})",
        f"Head SHA: {pr_state.get('head_sha')}",
        f"Plan: {title}",
        "Acceptance criteria:",
        *[f"- {item}" for item in criteria[:30]],
        "Return a final fenced DEV_CODE_REVIEW_RESULT JSON block with object "
        f"{CODE_REVIEW_OBJECT!r}, verdict approved|changes_requested|commented, "
        "findings, summary, and evidence_refs.",
    ])


def compose_merge_readiness(
    *,
    repo: str,
    pr_number: int,
    pr_state: Dict[str, Any],
    draft_status: Optional[str],
    verification: Optional[Dict[str, Any]],
    code_review: Optional[Dict[str, Any]],
    plan_id: Optional[str] = None,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    blocked_by: list[Dict[str, str]] = []
    gates: Dict[str, Dict[str, Any]] = {}

    draft_ok = True if not plan_id else str(draft_status or "").lower() == "approved_for_launch"
    gates["draft"] = {"state": "not_applicable" if not plan_id else (draft_status or "unknown"), "ok": draft_ok}
    if not draft_ok:
        blocked_by.append({"gate": "draft", "reason": "Execution plan draft is not approved for launch."})

    ci_state = str(pr_state.get("ci_state") or (pr_state.get("ci_status") or {}).get("state") or "unknown").lower()
    ci_ok = ci_state == "success"
    gates["ci"] = {"state": ci_state, "ok": ci_ok}
    if not ci_ok:
        blocked_by.append({"gate": "ci", "reason": f"CI state is {ci_state}, not success."})

    verification_verdict = str((verification or {}).get("verdict") or "unknown").lower()
    verification_ok = verification_verdict not in BLOCKING_VERIFICATION_VERDICTS
    gates["verification"] = {"state": verification_verdict, "ok": verification_ok}
    if not verification_ok:
        blocked_by.append({
            "gate": "verification",
            "reason": f"Verification verdict is {verification_verdict}.",
        })

    review_verdict = str((code_review or {}).get("verdict") or "unknown").lower()
    review_ok = review_verdict == "approved"
    gates["code_review"] = {"state": review_verdict, "ok": review_ok}
    if not review_ok:
        blocked_by.append({"gate": "code_review", "reason": "Independent code review has not approved the PR."})

    mergeable = pr_state.get("mergeable")
    mergeable_ok = mergeable is True
    gates["mergeable"] = {"state": mergeable, "ok": mergeable_ok}
    if not mergeable_ok:
        blocked_by.append({"gate": "mergeable", "reason": "GitHub does not report the PR as mergeable."})

    ready = not blocked_by
    return {
        "ok": True,
        "object": "hermes.dev_merge_readiness",
        "repo": repo,
        "plan_id": plan_id,
        "task_id": task_id,
        "pr_number": int(pr_number),
        "head_sha": pr_state.get("head_sha"),
        "ready": ready,
        "blocked_by": blocked_by,
        "gates": gates,
        "pr_state": pr_state,
        "verification": verification or {},
        "code_review": code_review or {},
    }


def request_merge_approval(
    *,
    store: DevSCMLifecycleStore,
    readiness: Dict[str, Any],
    requested_by: Optional[str] = None,
) -> Dict[str, Any]:
    if not readiness.get("ready"):
        raise ValueError("Merge approval requires a fully green merge-readiness snapshot.")
    approval = store.create_merge_approval(
        repo=_require_text(readiness.get("repo"), "repo"),
        pr_number=_require_int(readiness.get("pr_number"), "pr_number"),
        head_sha=_require_text(readiness.get("head_sha"), "head_sha"),
        plan_id=_optional_text(readiness.get("plan_id")),
        task_id=_optional_text(readiness.get("task_id")),
        gate_snapshot=readiness,
        requested_by=requested_by,
    )
    return {"ok": True, "object": "hermes.dev_merge_approval_request", "approval": approval}


def execute_merge(
    *,
    store: DevSCMLifecycleStore,
    approval_id: str,
    live_readiness: Dict[str, Any],
    executor: Optional[Callable[..., Dict[str, Any]]] = None,
    executor_name: str = "gh",
    merge_method: str = "squash",
    executor_enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    approval = store.get_merge_approval(approval_id)
    if not approval:
        raise KeyError(f"Merge approval not found: {approval_id}")
    if approval.get("status") != "approved":
        return _refuse_merge(
            store=store,
            approval=approval,
            live_readiness=live_readiness,
            reason=f"Merge approval is {approval.get('status')}, not approved.",
            invalidate=False,
            executor_name=executor_name,
        )
    if str(approval.get("head_sha") or "") != str(live_readiness.get("head_sha") or ""):
        return _refuse_merge(
            store=store,
            approval=approval,
            live_readiness=live_readiness,
            reason="PR head_sha changed since approval; a new single-use approval is required.",
            invalidate=True,
            executor_name=executor_name,
        )
    if not live_readiness.get("ready"):
        return _refuse_merge(
            store=store,
            approval=approval,
            live_readiness=live_readiness,
            reason="One or more merge gates regressed after approval.",
            invalidate=True,
            executor_name=executor_name,
        )
    enabled = _merge_executor_enabled() if executor_enabled is None else bool(executor_enabled)
    if not enabled:
        return _refuse_merge(
            store=store,
            approval=approval,
            live_readiness=live_readiness,
            reason="Merge executor is disabled until branch protection is confirmed.",
            invalidate=False,
            executor_name=executor_name,
        )
    executor = executor or gh_merge_executor
    result = executor(
        repo=approval["repo"],
        pr_number=int(approval["pr_number"]),
        merge_method=merge_method,
        head_sha=approval["head_sha"],
    )
    consumed = store.mark_merge_approval(
        approval_id,
        status="consumed",
        message=f"Merged by {executor_name} using {merge_method}.",
    )
    audit = store.record_merge_audit(
        approval_id=approval_id,
        repo=approval["repo"],
        pr_number=int(approval["pr_number"]),
        head_sha=approval["head_sha"],
        result="merged",
        reason=None,
        approval_snapshot=approval.get("gate_snapshot") or {},
        execution_snapshot=live_readiness,
        executor=executor_name,
    )
    return {
        "ok": True,
        "object": "hermes.dev_merge_execution",
        "merged": True,
        "result": "merged",
        "approval": consumed,
        "executor_result": result,
        "audit": audit,
    }


def gh_merge_executor(*, repo: str, pr_number: int, merge_method: str, head_sha: str) -> Dict[str, Any]:
    method_flag = {"squash": "--squash", "merge": "--merge", "rebase": "--rebase"}.get(merge_method)
    if not method_flag:
        raise ValueError(f"Unsupported merge method: {merge_method}")
    command = ["gh", "pr", "merge", str(int(pr_number)), "--repo", repo, method_flag, "--match-head-commit", head_sha]
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "gh pr merge failed").strip())
    return {
        "command": " ".join(command),
        "exit_code": result.returncode,
        "output": (result.stdout or result.stderr or "").strip()[:2000],
    }


def _refuse_merge(
    *,
    store: DevSCMLifecycleStore,
    approval: Dict[str, Any],
    live_readiness: Dict[str, Any],
    reason: str,
    invalidate: bool,
    executor_name: str,
) -> Dict[str, Any]:
    updated = approval
    if invalidate:
        updated = store.mark_merge_approval(approval["approval_id"], status="invalidated", message=reason)
    audit = store.record_merge_audit(
        approval_id=approval.get("approval_id"),
        repo=approval["repo"],
        pr_number=int(approval["pr_number"]),
        head_sha=approval.get("head_sha"),
        result="refused",
        reason=reason,
        approval_snapshot=approval.get("gate_snapshot") or {},
        execution_snapshot=live_readiness,
        executor=executor_name,
    )
    return {
        "ok": True,
        "object": "hermes.dev_merge_execution",
        "merged": False,
        "result": "refused",
        "reason": reason,
        "approval": updated,
        "audit": audit,
    }


def _github_get_json(url: str, *, opener: Any, timeout_seconds: float) -> Dict[str, Any]:
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or _gh_auth_token()
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "HermesDevSCMLifecycle",
        **({"Authorization": f"Bearer {token.strip()}"} if token else {}),
    }
    request = urllib.request.Request(url, headers=headers)
    try:
        with opener(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(body or f"GitHub API returned HTTP {exc.code}") from exc


def _gh_auth_token() -> str:
    try:
        result = subprocess.run(["gh", "auth", "token"], check=False, capture_output=True, text=True, timeout=3)
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _merge_executor_enabled() -> bool:
    return (
        str(os.getenv("HERMES_DEV_MERGE_EXECUTOR_ENABLED") or "").lower() in {"1", "true", "yes", "on"}
        and str(os.getenv("HERMES_DEV_BRANCH_PROTECTION_CONFIRMED") or "").lower() in {"1", "true", "yes", "on"}
    )


def _unknown_pr_state(*, repo: str, pr_number: int, warning: str) -> Dict[str, Any]:
    return {
        "ok": True,
        "object": "hermes.dev_pr_state",
        "repo": repo,
        "pr_number": pr_number,
        "branch": None,
        "pr_url": None,
        "head_sha": None,
        "ci_state": "unknown",
        "ci_status": _unknown_ci(warning),
        "review_state": "unknown",
        "mergeable": None,
        "merge_state": "unknown",
        "warnings": [warning],
        "raw": {},
    }


def _unknown_ci(warning: str) -> Dict[str, Any]:
    return {"state": "unknown", "total": 0, "failing": [], "checks": [], "warnings": [warning]}


def _review_state(payload: Dict[str, Any]) -> str:
    if payload.get("draft"):
        return "draft"
    if payload.get("state") == "closed":
        return "closed"
    return "open"


def _normalize_review_verdict(value: Any) -> str:
    verdict = str(value or "unknown").strip().lower()
    return verdict if verdict in VALID_REVIEW_VERDICTS else "unknown"


def _normalize_finding(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "severity": str(item.get("severity") or "info").strip().lower(),
        "file": _optional_text(item.get("file")),
        "line": item.get("line") if isinstance(item.get("line"), int) else None,
        "note": str(item.get("note") or "").strip(),
    }


def _pr_state_from_row(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    item = dict(row)
    item["object"] = "hermes.dev_pr_state"
    item["mergeable"] = _db_to_bool(item.get("mergeable"))
    for key, default in (("ci_status", {}), ("warnings", []), ("raw", {})):
        item[key] = _json_value(item.get(key), default)
    return item


def _code_review_from_row(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    item = dict(row)
    item["object"] = "hermes.dev_code_review_run"
    for key, default in (("findings", []), ("evidence_refs", []), ("warnings", [])):
        item[key] = _json_value(item.get(key), default)
    return item


def _readiness_from_row(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    item = dict(row)
    item["object"] = "hermes.dev_merge_readiness"
    item["ready"] = bool(item.get("ready"))
    for key, default in (
        ("blocked_by", []),
        ("gates", {}),
        ("pr_state", {}),
        ("verification", {}),
        ("code_review", {}),
    ):
        item[key] = _json_value(item.get(key), default)
    return item


def _approval_from_row(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    item = dict(row)
    item["object"] = "hermes.dev_merge_approval"
    item["gate_snapshot"] = _json_value(item.get("gate_snapshot"), {})
    return item


def _audit_from_row(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    item = dict(row)
    item["object"] = "hermes.dev_merge_audit"
    item["approval_snapshot"] = _json_value(item.get("approval_snapshot"), {})
    item["execution_snapshot"] = _json_value(item.get("execution_snapshot"), {})
    return item


def _json_value(raw: Any, default: Any) -> Any:
    try:
        return json.loads(raw or json.dumps(default))
    except Exception:
        return default


def _bool_to_db(value: Any) -> Optional[int]:
    if value is None:
        return None
    return 1 if bool(value) else 0


def _db_to_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    return bool(value)


def _pr_key(repo: str, pr_number: int) -> str:
    return f"{str(repo).strip()}#{int(pr_number)}"


def _valid_repo(repo: str) -> bool:
    return bool(re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", repo or ""))


def _require_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required.")
    return text


def _require_int(value: Any, field: str) -> int:
    try:
        number = int(value)
    except Exception as exc:
        raise ValueError(f"{field} is required.") from exc
    if number <= 0:
        raise ValueError(f"{field} is required.")
    return number


def _optional_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None
