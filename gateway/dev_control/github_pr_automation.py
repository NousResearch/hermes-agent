"""GitHub webhook driven PR automation for the Dev control plane."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import sqlite3
import subprocess
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

from gateway.dev_control.scm_lifecycle import (
    DevSCMLifecycleStore,
    compose_merge_readiness,
    execute_merge,
    fetch_pr_state,
    request_merge_approval,
)
from gateway.dev_worker_runtimes import WorkerRuntimeRouter
from gateway.dev_execution import build_profiled_prompt, resolve_launch_defaults


MANAGED_REPOS = ("Felippen/Oryn", "Felippen/hermes-agent", "Felippen/hermes-ops")
AUTO_FIX_LABEL = "hermes:auto-fix"
AUTO_MERGE_LABEL = "hermes:auto-merge"
AUTO_RELEASE_LABEL = "hermes:auto-release"
FIX_ACTIONS = {"fix_ci", "fix_review_comments"}
COPILOT_LOGINS = {"copilot-pull-request-reviewer", "copilot"}
GITHUB_API_BASE = "https://api.github.com"
DEFAULT_PROJECT_IDS = {
    "Felippen/Oryn": "OrynPlatform",
    "Felippen/hermes-agent": "HermesAgent",
    "Felippen/hermes-ops": "OrynPlatform",
}


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dev_github_webhook_events (
    delivery_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    action TEXT,
    repo TEXT,
    pr_number INTEGER,
    accepted INTEGER NOT NULL,
    status TEXT NOT NULL,
    result TEXT NOT NULL,
    payload TEXT NOT NULL,
    warnings TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dev_github_webhook_events_pr
    ON dev_github_webhook_events(repo, pr_number, created_at DESC);

CREATE TABLE IF NOT EXISTS dev_pr_automation_runs (
    run_id TEXT PRIMARY KEY,
    repo TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    head_sha TEXT,
    action TEXT NOT NULL,
    status TEXT NOT NULL,
    labels TEXT NOT NULL,
    reason TEXT,
    result TEXT NOT NULL,
    warnings TEXT NOT NULL,
    ao_session_id TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dev_pr_automation_runs_pr
    ON dev_pr_automation_runs(repo, pr_number, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_dev_pr_automation_runs_head_action
    ON dev_pr_automation_runs(repo, pr_number, head_sha, action, created_at DESC);
"""


class DevGitHubPRAutomationStore:
    """SQLite state for GitHub webhook deliveries and PR automation attempts."""

    def __init__(self, db_path: Optional[Path | str] = None):
        self.db_path = Path(db_path or os.path.expanduser("~/.hermes/state.db"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA_SQL)

    def record_webhook_event(
        self,
        *,
        delivery_id: str,
        event_type: str,
        action: Optional[str],
        repo: Optional[str],
        pr_number: Optional[int],
        accepted: bool,
        status: str,
        result: Dict[str, Any],
        payload: Dict[str, Any],
        warnings: Iterable[str] = (),
    ) -> Dict[str, Any]:
        now = time.time()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO dev_github_webhook_events (
                    delivery_id, event_type, action, repo, pr_number, accepted,
                    status, result, payload, warnings, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(delivery_id) DO NOTHING
                """,
                (
                    delivery_id,
                    event_type,
                    action,
                    repo,
                    pr_number,
                    1 if accepted else 0,
                    status,
                    json.dumps(result or {}, ensure_ascii=False),
                    json.dumps(payload or {}, ensure_ascii=False),
                    json.dumps(list(warnings or []), ensure_ascii=False),
                    now,
                ),
            )
        return self.get_webhook_event(delivery_id) or {
            "delivery_id": delivery_id,
            "duplicate": True,
            "status": "duplicate",
        }

    def get_webhook_event(self, delivery_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM dev_github_webhook_events WHERE delivery_id = ?",
            (str(delivery_id or "").strip(),),
        ).fetchone()
        return _webhook_from_row(row)

    def has_delivery(self, delivery_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM dev_github_webhook_events WHERE delivery_id = ?",
            (str(delivery_id or "").strip(),),
        ).fetchone()
        return row is not None

    def record_run(self, run: Dict[str, Any]) -> Dict[str, Any]:
        now = time.time()
        run_id = str(run.get("run_id") or f"devprauto-{uuid.uuid4().hex[:10]}")
        repo = _require_text(run.get("repo"), "repo")
        pr_number = _require_int(run.get("pr_number"), "pr_number")
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO dev_pr_automation_runs (
                    run_id, repo, pr_number, head_sha, action, status, labels,
                    reason, result, warnings, ao_session_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    repo,
                    pr_number,
                    _optional_text(run.get("head_sha")),
                    _require_text(run.get("action"), "action"),
                    str(run.get("status") or "completed"),
                    json.dumps(run.get("labels") or [], ensure_ascii=False),
                    _optional_text(run.get("reason")),
                    json.dumps(run.get("result") or {}, ensure_ascii=False),
                    json.dumps(run.get("warnings") or [], ensure_ascii=False),
                    _optional_text(run.get("ao_session_id")),
                    now,
                    now,
                ),
            )
        return self.get_run(run_id) or dict(run)

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM dev_pr_automation_runs WHERE run_id = ?",
            (str(run_id or "").strip(),),
        ).fetchone()
        return _run_from_row(row)

    def list_runs(
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
        sql = "SELECT * FROM dev_pr_automation_runs"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, min(int(limit or 50), 200)))
        rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [item for row in rows if (item := _run_from_row(row))]

    def latest_runs_by_pr(self, *, limit: int = 25) -> list[Dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT *
            FROM dev_pr_automation_runs
            WHERE run_id IN (
                SELECT run_id
                FROM dev_pr_automation_runs r2
                WHERE r2.repo = dev_pr_automation_runs.repo
                  AND r2.pr_number = dev_pr_automation_runs.pr_number
                ORDER BY r2.created_at DESC
                LIMIT 1
            )
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (max(1, min(int(limit or 25), 100)),),
        ).fetchall()
        return [item for row in rows if (item := _run_from_row(row))]

    def count_fix_attempts(self, *, repo: str, pr_number: int, head_sha: str) -> int:
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM dev_pr_automation_runs
            WHERE repo = ? AND pr_number = ? AND head_sha = ? AND action IN ('fix_ci', 'fix_review_comments')
            """,
            (repo, int(pr_number), head_sha),
        ).fetchone()
        return int((row or {})["count"] or 0)

    def has_action(
        self,
        *,
        repo: str,
        pr_number: int,
        action: str,
        head_sha: Optional[str] = None,
        statuses: Optional[Iterable[str]] = None,
    ) -> bool:
        clauses = ["repo = ?", "pr_number = ?", "action = ?"]
        params: list[Any] = [repo, int(pr_number), action]
        if head_sha:
            clauses.append("head_sha = ?")
            params.append(head_sha)
        if statuses:
            status_values = [str(item) for item in statuses if str(item)]
            if status_values:
                clauses.append("status IN (" + ", ".join("?" for _ in status_values) + ")")
                params.extend(status_values)
        row = self._conn.execute(
            "SELECT 1 FROM dev_pr_automation_runs WHERE " + " AND ".join(clauses) + " LIMIT 1",
            tuple(params),
        ).fetchone()
        return row is not None


def verify_github_signature(*, body: bytes, signature_header: str, secret: str) -> bool:
    if not secret or not signature_header:
        return False
    prefix = "sha256="
    if not signature_header.startswith(prefix):
        return False
    expected = prefix + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header)


def managed_repos() -> set[str]:
    raw = os.getenv("HERMES_DEV_GITHUB_MANAGED_REPOS")
    if not raw:
        return set(MANAGED_REPOS)
    return {item.strip() for item in raw.split(",") if item.strip()}


def process_github_webhook(
    *,
    store: DevGitHubPRAutomationStore,
    scm_store: DevSCMLifecycleStore,
    delivery_id: str,
    event_type: str,
    payload: Dict[str, Any],
    router: Any = None,
    command_runner: Optional[Callable[..., Dict[str, Any]]] = None,
    pr_state_fetcher: Callable[..., Dict[str, Any]] = fetch_pr_state,
    release_dispatcher: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Persist and react to one verified GitHub webhook delivery."""

    if store.has_delivery(delivery_id):
        event = store.get_webhook_event(delivery_id)
        return {"ok": True, "object": "hermes.dev_github_webhook_result", "duplicate": True, "event": event}

    normalized = normalize_webhook_event(event_type=event_type, payload=payload)
    repo = normalized.get("repo")
    pr_number = normalized.get("pr_number")
    warnings = list(normalized.get("warnings") or [])
    if repo not in managed_repos():
        result = {"decision": "ignored", "reason": f"Repository {repo or 'unknown'} is not managed."}
        event = store.record_webhook_event(
            delivery_id=delivery_id,
            event_type=event_type,
            action=normalized.get("action"),
            repo=repo,
            pr_number=pr_number,
            accepted=False,
            status="ignored",
            result=result,
            payload=payload,
            warnings=warnings,
        )
        return {"ok": True, "object": "hermes.dev_github_webhook_result", "event": event, **result}

    actions: list[Dict[str, Any]] = []
    pr_state: Dict[str, Any] = {}
    readiness: Dict[str, Any] = {}
    labels = set(normalized.get("labels") or [])
    if repo and pr_number:
        if not labels:
            labels = set(fetch_pr_labels(repo=repo, pr_number=int(pr_number)))
        pr_state = pr_state_fetcher(repo=repo, pr_number=int(pr_number))
        if pr_state.get("pr_number"):
            pr_state = scm_store.upsert_pr_state({**pr_state, "repo": repo, "pr_number": int(pr_number)})
        review_gate = fetch_github_review_gate(repo=repo, pr_number=int(pr_number))
        readiness = scm_store.record_readiness(compose_merge_readiness(
            repo=repo,
            pr_number=int(pr_number),
            pr_state=pr_state,
            draft_status="approved_for_launch" if not normalized.get("plan_id") else normalized.get("draft_status"),
            verification={},
            code_review=review_gate,
            plan_id=normalized.get("plan_id"),
            task_id=normalized.get("task_id"),
        ))
        if should_request_copilot_review(event_type=event_type, normalized=normalized, pr_state=pr_state):
            actions.append(request_copilot_review(
                store=store,
                repo=repo,
                pr_number=int(pr_number),
                head_sha=pr_state.get("head_sha"),
                labels=labels,
                command_runner=command_runner,
            ))
        if should_auto_fix(event_type=event_type, normalized=normalized, labels=labels, pr_state=pr_state):
            actions.append(delegate_pr_fix(
                store=store,
                repo=repo,
                pr_number=int(pr_number),
                pr_state=pr_state,
                labels=labels,
                reason=fix_reason(event_type=event_type, normalized=normalized),
                router=router,
                command_runner=command_runner,
            ))
        if should_auto_merge(labels=labels, readiness=readiness, normalized=normalized):
            actions.append(auto_merge_pr(
                store=store,
                scm_store=scm_store,
                repo=repo,
                pr_number=int(pr_number),
                readiness=readiness,
                labels=labels,
            ))
        if should_auto_release(labels=labels, normalized=normalized):
            actions.append(auto_release_oryn_workspace(
                store=store,
                repo=repo,
                pr_number=int(pr_number),
                labels=labels,
                release_dispatcher=release_dispatcher,
                command_runner=command_runner,
            ))
    result = {
        "decision": "processed",
        "repo": repo,
        "pr_number": pr_number,
        "labels": sorted(labels),
        "pr_state": pr_state,
        "readiness": readiness,
        "actions": actions,
    }
    event = store.record_webhook_event(
        delivery_id=delivery_id,
        event_type=event_type,
        action=normalized.get("action"),
        repo=repo,
        pr_number=pr_number,
        accepted=True,
        status="processed",
        result=result,
        payload=payload,
        warnings=warnings,
    )
    return {"ok": True, "object": "hermes.dev_github_webhook_result", "event": event, **result}


def reconcile_github_pr_automation(
    *,
    store: DevGitHubPRAutomationStore,
    scm_store: DevSCMLifecycleStore,
    repos: Optional[Iterable[str]] = None,
    limit: int = 50,
    router: Any = None,
    command_runner: Optional[Callable[..., Dict[str, Any]]] = None,
    pr_state_fetcher: Callable[..., Dict[str, Any]] = fetch_pr_state,
    release_dispatcher: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Poll managed GitHub PRs and apply trusted-label PR automation decisions."""

    target_repos = [repo for repo in (repos or sorted(managed_repos())) if repo in managed_repos()]
    reports: list[Dict[str, Any]] = []
    for repo in target_repos:
        open_prs, open_warnings = fetch_open_prs(repo=repo, limit=limit, command_runner=command_runner)
        merged_prs, merged_warnings = fetch_recent_merged_prs(
            repo=repo,
            limit=max(5, min(limit, 50)),
            command_runner=command_runner,
        )
        repo_actions: list[Dict[str, Any]] = []
        pr_reports: list[Dict[str, Any]] = []
        for pr in open_prs:
            result = reconcile_open_pr(
                store=store,
                scm_store=scm_store,
                repo=repo,
                pr=pr,
                router=router,
                command_runner=command_runner,
                pr_state_fetcher=pr_state_fetcher,
            )
            pr_reports.append(result)
            repo_actions.extend(result.get("actions") or [])
        for pr in merged_prs:
            result = reconcile_merged_pr_release(
                store=store,
                repo=repo,
                pr=pr,
                release_dispatcher=release_dispatcher,
                command_runner=command_runner,
            )
            pr_reports.append(result)
            repo_actions.extend(result.get("actions") or [])
        reports.append({
            "repo": repo,
            "open_pr_count": len(open_prs),
            "merged_pr_count": len(merged_prs),
            "actions": repo_actions,
            "prs": pr_reports,
            "warnings": [*open_warnings, *merged_warnings],
        })
    return {
        "ok": True,
        "object": "hermes.dev_pr_automation_reconcile_result",
        "repos": reports,
        "action_count": sum(len(report.get("actions") or []) for report in reports),
    }


def reconcile_open_pr(
    *,
    store: DevGitHubPRAutomationStore,
    scm_store: DevSCMLifecycleStore,
    repo: str,
    pr: Dict[str, Any],
    router: Any = None,
    command_runner: Optional[Callable[..., Dict[str, Any]]] = None,
    pr_state_fetcher: Callable[..., Dict[str, Any]] = fetch_pr_state,
) -> Dict[str, Any]:
    pr_number = int(pr.get("number") or 0)
    labels = set(_label_names(pr.get("labels") or []))
    normalized = {
        "event_type": "poll",
        "action": "poll",
        "repo": repo,
        "pr_number": pr_number,
        "labels": sorted(labels),
        "head_sha": str(pr.get("headRefOid") or "").strip(),
        "branch": pr.get("headRefName"),
        "state": "open",
        "draft": bool(pr.get("isDraft")),
    }
    pr_state = pr_state_fetcher(repo=repo, pr_number=pr_number)
    if pr_state.get("pr_number"):
        pr_state = scm_store.upsert_pr_state({**pr_state, "repo": repo, "pr_number": pr_number})
    review_gate = fetch_github_review_gate(repo=repo, pr_number=pr_number)
    readiness = scm_store.record_readiness(compose_merge_readiness(
        repo=repo,
        pr_number=pr_number,
        pr_state=pr_state,
        draft_status="approved_for_launch",
        verification={},
        code_review=review_gate,
    ))
    actions: list[Dict[str, Any]] = []
    if _poll_should_request_copilot(
        store=store,
        repo=repo,
        pr_number=pr_number,
        head_sha=pr_state.get("head_sha"),
        labels=labels,
        review_gate=review_gate,
        draft=bool(pr.get("isDraft")),
    ):
        actions.append(request_copilot_review(
            store=store,
            repo=repo,
            pr_number=pr_number,
            head_sha=pr_state.get("head_sha"),
            labels=labels,
            command_runner=command_runner,
        ))
    if _poll_should_auto_fix(
        store=store,
        repo=repo,
        pr_number=pr_number,
        head_sha=pr_state.get("head_sha"),
        labels=labels,
        pr=pr,
        draft=bool(pr.get("isDraft")),
    ):
        actions.append(delegate_pr_fix(
            store=store,
            repo=repo,
            pr_number=pr_number,
            pr_state=pr_state,
            labels=labels,
            reason=_poll_fix_reason(pr),
            router=router,
            command_runner=command_runner,
        ))
    if should_auto_merge(labels=labels, readiness=readiness, normalized=normalized):
        actions.append(auto_merge_pr(
            store=store,
            scm_store=scm_store,
            repo=repo,
            pr_number=pr_number,
            readiness=readiness,
            labels=labels,
        ))
    return {
        "repo": repo,
        "pr_number": pr_number,
        "labels": sorted(labels),
        "pr_state": pr_state,
        "readiness": readiness,
        "actions": actions,
    }


def reconcile_merged_pr_release(
    *,
    store: DevGitHubPRAutomationStore,
    repo: str,
    pr: Dict[str, Any],
    release_dispatcher: Optional[Callable[..., Dict[str, Any]]] = None,
    command_runner: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    pr_number = int(pr.get("number") or 0)
    labels = set(_label_names(pr.get("labels") or []))
    normalized = {
        "event_type": "pull_request",
        "action": "closed",
        "repo": repo,
        "pr_number": pr_number,
        "labels": sorted(labels),
        "merged": True,
    }
    actions: list[Dict[str, Any]] = []
    if should_auto_release(labels=labels, normalized=normalized) and not store.has_action(
        repo=repo,
        pr_number=pr_number,
        action="release",
        statuses={"completed", "needs_human"},
    ):
        actions.append(auto_release_oryn_workspace(
            store=store,
            repo=repo,
            pr_number=pr_number,
            labels=labels,
            release_dispatcher=release_dispatcher,
            command_runner=command_runner,
        ))
    return {"repo": repo, "pr_number": pr_number, "labels": sorted(labels), "actions": actions}


def normalize_webhook_event(*, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    repo = ((payload.get("repository") or {}).get("full_name") or "").strip()
    pull_request = payload.get("pull_request") if isinstance(payload.get("pull_request"), dict) else {}
    issue = payload.get("issue") if isinstance(payload.get("issue"), dict) else {}
    check_run = payload.get("check_run") if isinstance(payload.get("check_run"), dict) else {}
    check_suite = payload.get("check_suite") if isinstance(payload.get("check_suite"), dict) else {}
    check_prs = check_run.get("pull_requests") or check_suite.get("pull_requests") or []
    check_pr_number = None
    if check_prs and isinstance(check_prs[0], dict):
        check_pr_number = check_prs[0].get("number")
    pr_number = pull_request.get("number") or (issue.get("number") if issue.get("pull_request") else None) or check_pr_number
    labels_source = pull_request.get("labels") or issue.get("labels") or []
    labels = _label_names(labels_source)
    head = pull_request.get("head") if isinstance(pull_request.get("head"), dict) else {}
    return {
        "event_type": event_type,
        "action": str(payload.get("action") or "").strip(),
        "repo": repo,
        "pr_number": int(pr_number) if pr_number else None,
        "labels": labels,
        "head_sha": head.get("sha"),
        "branch": head.get("ref"),
        "merged": bool(pull_request.get("merged")),
        "state": pull_request.get("state") or issue.get("state"),
        "draft": bool(pull_request.get("draft")),
        "conclusion": check_run.get("conclusion") or check_suite.get("conclusion") or payload.get("state"),
        "status": check_run.get("status") or check_suite.get("status") or payload.get("state"),
        "sender": (payload.get("sender") or {}).get("login"),
        "warnings": [],
    }


def should_request_copilot_review(*, event_type: str, normalized: Dict[str, Any], pr_state: Dict[str, Any]) -> bool:
    if event_type != "pull_request":
        return False
    if normalized.get("draft") or normalized.get("state") == "closed":
        return False
    return normalized.get("action") in {"opened", "reopened", "ready_for_review", "synchronize", "labeled"}


def request_copilot_review(
    *,
    store: DevGitHubPRAutomationStore,
    repo: str,
    pr_number: int,
    head_sha: Optional[str],
    labels: Iterable[str],
    command_runner: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    runner = command_runner or _run_command
    try:
        result = runner(["gh", "pr", "edit", str(int(pr_number)), "--repo", repo, "--add-reviewer", "copilot"], timeout=60)
        status = "completed" if int(result.get("exit_code") or 0) == 0 else "needs_human"
        warnings = [] if status == "completed" else [str(result.get("output") or "Copilot review request failed.")]
    except Exception as exc:
        result = {"error": str(exc)}
        status = "needs_human"
        warnings = [f"Copilot review request failed: {exc}"]
    return store.record_run({
        "repo": repo,
        "pr_number": pr_number,
        "head_sha": head_sha,
        "action": "request_copilot_review",
        "status": status,
        "labels": sorted(set(labels)),
        "reason": "Ensure managed PR receives Copilot review.",
        "result": result,
        "warnings": warnings,
    })


def should_auto_fix(*, event_type: str, normalized: Dict[str, Any], labels: set[str], pr_state: Dict[str, Any]) -> bool:
    if AUTO_FIX_LABEL not in labels:
        return False
    if normalized.get("state") == "closed" or normalized.get("draft"):
        return False
    if event_type in {"pull_request_review", "pull_request_review_comment", "issue_comment"}:
        return True
    if event_type in {"check_run", "check_suite", "status"}:
        return _event_failed(normalized)
    return False


def fix_reason(*, event_type: str, normalized: Dict[str, Any]) -> str:
    if event_type in {"check_run", "check_suite", "status"}:
        return "CI reported a failing check/status."
    return "Review comment or review event needs follow-up."


def delegate_pr_fix(
    *,
    store: DevGitHubPRAutomationStore,
    repo: str,
    pr_number: int,
    pr_state: Dict[str, Any],
    labels: Iterable[str],
    reason: str,
    router: Any = None,
    command_runner: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    head_sha = str(pr_state.get("head_sha") or "").strip()
    if not head_sha:
        return _record_needs_human(store, repo, pr_number, head_sha, "fix_ci", labels, "PR head SHA is unavailable.")
    if store.count_fix_attempts(repo=repo, pr_number=pr_number, head_sha=head_sha) >= _max_fix_attempts():
        return _record_needs_human(store, repo, pr_number, head_sha, "fix_ci", labels, "Fix attempt limit reached for this head SHA.")
    files = fetch_pr_changed_files(repo=repo, pr_number=pr_number, command_runner=command_runner)
    if not files:
        return _record_needs_human(store, repo, pr_number, head_sha, "fix_ci", labels, "No PR changed files were available for scope bounding.")
    prompt = build_pr_fix_prompt(repo=repo, pr_number=pr_number, pr_state=pr_state, reason=reason, allowed_files=files)
    profile = resolve_launch_defaults(profile_id="workspace.implement", project_id=project_id_for_repo(repo))
    profiled_prompt = build_profiled_prompt(
        prompt,
        goal=f"Fix PR #{pr_number} review/CI feedback",
        profile=profile,
        acceptance_criteria=[
            "Only edit files already touched by the PR plus directly related tests/docs.",
            "Push fixes to the existing PR branch.",
            "Do not merge, release, publish, or change branch protection.",
        ],
    )
    try:
        runtime_router = router or WorkerRuntimeRouter()
        session = runtime_router.spawn(
            profile.get("runtime") or "ao",
            project_id=profile["project_id"],
            prompt=profiled_prompt,
            branch=pr_state.get("branch"),
            agent=profile.get("agent"),
            model=profile.get("model"),
            reasoning_effort=profile.get("reasoning_effort"),
        )
        return store.record_run({
            "repo": repo,
            "pr_number": pr_number,
            "head_sha": head_sha,
            "action": "fix_ci" if "CI" in reason else "fix_review_comments",
            "status": "launched",
            "labels": sorted(set(labels)),
            "reason": reason,
            "result": {
                "session_id": getattr(session, "id", None),
                "project_id": profile.get("project_id"),
                "branch": pr_state.get("branch"),
                "allowed_files": files,
            },
            "ao_session_id": getattr(session, "id", None),
        })
    except Exception as exc:
        return _record_needs_human(store, repo, pr_number, head_sha, "fix_ci", labels, f"AO fix delegation failed: {exc}")


def build_pr_fix_prompt(*, repo: str, pr_number: int, pr_state: Dict[str, Any], reason: str, allowed_files: list[str]) -> str:
    return "\n".join([
        "You are a Hermes PR fix worker acting on a labeled GitHub pull request.",
        f"Repository: {repo}",
        f"PR: #{pr_number}",
        f"Branch: {pr_state.get('branch')}",
        f"Head SHA at delegation: {pr_state.get('head_sha')}",
        f"Reason: {reason}",
        "",
        "Scope:",
        "You may edit only the files already touched by this PR plus directly related tests/docs.",
        "If a correct fix requires any other file, stop and report needs_human.",
        "Allowed current PR files:",
        *[f"- {path}" for path in allowed_files[:200]],
        "",
        "Required workflow:",
        "1. Inspect the PR, failing checks, and review comments with gh.",
        "2. Make the smallest scoped fix on the existing PR branch.",
        "3. Run the relevant local verification.",
        "4. Commit and push to the same PR branch.",
        "5. Do not merge, release, publish, alter branch protection, or close the PR.",
    ])


def fetch_pr_changed_files(
    *,
    repo: str,
    pr_number: int,
    command_runner: Optional[Callable[..., Dict[str, Any]]] = None,
) -> list[str]:
    runner = command_runner or _run_command
    result = runner(["gh", "pr", "diff", str(int(pr_number)), "--repo", repo, "--name-only"], timeout=60)
    if int(result.get("exit_code") or 0) != 0:
        return []
    return [line.strip() for line in str(result.get("output") or "").splitlines() if _safe_relative_path(line.strip())]


def fetch_pr_labels(*, repo: str, pr_number: int) -> list[str]:
    try:
        payload = _github_get_json(f"{GITHUB_API_BASE}/repos/{repo}/issues/{int(pr_number)}/labels")
    except Exception:
        return []
    return _label_names(payload if isinstance(payload, list) else [])


def should_auto_merge(*, labels: set[str], readiness: Dict[str, Any], normalized: Dict[str, Any]) -> bool:
    if AUTO_MERGE_LABEL not in labels:
        return False
    if normalized.get("state") == "closed" or normalized.get("draft"):
        return False
    return bool(readiness.get("ready"))


def auto_merge_pr(
    *,
    store: DevGitHubPRAutomationStore,
    scm_store: DevSCMLifecycleStore,
    repo: str,
    pr_number: int,
    readiness: Dict[str, Any],
    labels: Iterable[str],
) -> Dict[str, Any]:
    try:
        approval_response = request_merge_approval(store=scm_store, readiness=readiness, requested_by="github-webhook:auto-merge")
        approval = scm_store.approve_merge_approval(
            approval_response["approval"]["approval_id"],
            approved_by="label:hermes:auto-merge",
            message="Auto-approved from trusted PR label after green gate snapshot.",
        )
        execution = execute_merge(
            store=scm_store,
            approval_id=approval["approval_id"],
            live_readiness=readiness,
            merge_method="squash",
        )
        status = "completed" if execution.get("merged") else "needs_human"
        warnings = [] if execution.get("merged") else [str(execution.get("reason") or "Merge refused.")]
        return store.record_run({
            "repo": repo,
            "pr_number": pr_number,
            "head_sha": readiness.get("head_sha"),
            "action": "merge",
            "status": status,
            "labels": sorted(set(labels)),
            "reason": "Auto-merge trusted label with green readiness.",
            "result": execution,
            "warnings": warnings,
        })
    except Exception as exc:
        return _record_needs_human(store, repo, pr_number, readiness.get("head_sha"), "merge", labels, f"Auto-merge failed: {exc}")


def should_auto_release(*, labels: set[str], normalized: Dict[str, Any]) -> bool:
    return (
        AUTO_RELEASE_LABEL in labels
        and normalized.get("repo") == "Felippen/Oryn"
        and normalized.get("event_type") == "pull_request"
        and normalized.get("action") == "closed"
        and bool(normalized.get("merged"))
    )


def auto_release_oryn_workspace(
    *,
    store: DevGitHubPRAutomationStore,
    repo: str,
    pr_number: int,
    labels: Iterable[str],
    release_dispatcher: Optional[Callable[..., Dict[str, Any]]] = None,
    command_runner: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    try:
        dispatcher = release_dispatcher or dispatch_oryn_workspace_release
        result = dispatcher(command_runner=command_runner)
        status = "completed" if result.get("ok") else "needs_human"
        return store.record_run({
            "repo": repo,
            "pr_number": pr_number,
            "head_sha": None,
            "action": "release",
            "status": status,
            "labels": sorted(set(labels)),
            "reason": "Auto-release trusted label after merge.",
            "result": result,
            "warnings": result.get("warnings") or [],
        })
    except Exception as exc:
        return _record_needs_human(store, repo, pr_number, None, "release", labels, f"Auto-release failed: {exc}")


def dispatch_oryn_workspace_release(*, command_runner: Optional[Callable[..., Dict[str, Any]]] = None) -> Dict[str, Any]:
    manifest = fetch_oryn_stable_manifest()
    version = next_patch_version(str(manifest.get("version") or "0.0.0"))
    build = int(manifest.get("build") or 0) + 1
    runner = command_runner or _run_command
    result = runner([
        "gh",
        "workflow",
        "run",
        "Publish Oryn Workspace Stable",
        "--repo",
        "Felippen/Oryn",
        "--ref",
        "main",
        "-f",
        f"version={version}",
        "-f",
        f"build={build}",
    ], timeout=60)
    ok = int(result.get("exit_code") or 0) == 0
    return {
        "ok": ok,
        "version": version,
        "build": build,
        "workflow": "Publish Oryn Workspace Stable",
        "command_result": result,
        "warnings": [] if ok else [str(result.get("output") or "Release workflow dispatch failed.")],
    }


def fetch_oryn_stable_manifest() -> Dict[str, Any]:
    token = _github_token()
    url = (
        f"{GITHUB_API_BASE}/repos/Felippen/Oryn/contents/"
        "apps/oryn-workspace/release/stable-manifest.json?ref=main"
    )
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "HermesDevGitHubPRAutomation",
        **({"Authorization": f"Bearer {token}"} if token else {}),
    }
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=8) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = base64.b64decode(str(payload.get("content") or "")).decode("utf-8")
    return json.loads(content)


def next_patch_version(version: str) -> str:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version.strip())
    if not match:
        raise ValueError(f"Unsupported version format: {version}")
    major, minor, patch = (int(part) for part in match.groups())
    return f"{major}.{minor}.{patch + 1}"


def fetch_github_review_gate(*, repo: str, pr_number: int) -> Dict[str, Any]:
    try:
        reviews = _github_get_json(f"{GITHUB_API_BASE}/repos/{repo}/pulls/{int(pr_number)}/reviews")
    except Exception as exc:
        return {"verdict": "unknown", "warnings": [f"GitHub reviews unavailable: {exc}"]}
    items = reviews if isinstance(reviews, list) else []
    copilot_reviews = [
        item for item in items
        if str(((item or {}).get("user") or {}).get("login") or "").lower() in COPILOT_LOGINS
    ]
    latest_by_user: Dict[str, str] = {}
    for item in items:
        login = str(((item or {}).get("user") or {}).get("login") or "").lower()
        state = str((item or {}).get("state") or "").upper()
        if login:
            latest_by_user[login] = state
    if any(state == "CHANGES_REQUESTED" for state in latest_by_user.values()):
        verdict = "changes_requested"
    elif copilot_reviews:
        verdict = "approved"
    else:
        verdict = "unknown"
    return {
        "object": "hermes.dev_code_review_run",
        "verdict": verdict,
        "status": "github",
        "findings": [],
        "summary": "GitHub review gate derived from PR reviews.",
        "evidence_refs": [f"github:{repo}#{pr_number}:reviews"],
        "warnings": [],
    }


def automation_summary(
    *,
    store: DevGitHubPRAutomationStore,
    scm_store: Optional[DevSCMLifecycleStore] = None,
    repo: Optional[str] = None,
    pr_number: Optional[int] = None,
    limit: int = 25,
) -> Dict[str, Any]:
    runs = store.list_runs(repo=repo, pr_number=pr_number, limit=limit) if repo or pr_number else store.latest_runs_by_pr(limit=limit)
    return {
        "ok": True,
        "object": "hermes.dev_pr_automation_summary",
        "data": runs,
        "total": len(runs),
    }


def run_manual_pr_automation_action(
    *,
    action: str,
    repo: str,
    pr_number: int,
    store: DevGitHubPRAutomationStore,
    scm_store: DevSCMLifecycleStore,
    router: Any = None,
    command_runner: Optional[Callable[..., Dict[str, Any]]] = None,
    release_dispatcher: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    pr_state = fetch_pr_state(repo=repo, pr_number=int(pr_number))
    labels = set(fetch_pr_labels(repo=repo, pr_number=int(pr_number)))
    if action == "request_copilot_review":
        run = request_copilot_review(
            store=store,
            repo=repo,
            pr_number=int(pr_number),
            head_sha=pr_state.get("head_sha"),
            labels=labels,
            command_runner=command_runner,
        )
    elif action in {"fix_ci", "fix_review_comments", "delegate_fix"}:
        run = delegate_pr_fix(
            store=store,
            repo=repo,
            pr_number=int(pr_number),
            pr_state=pr_state,
            labels=labels,
            reason="Manual PR automation fix delegation.",
            router=router,
            command_runner=command_runner,
        )
    elif action == "merge":
        review_gate = fetch_github_review_gate(repo=repo, pr_number=int(pr_number))
        readiness = scm_store.record_readiness(compose_merge_readiness(
            repo=repo,
            pr_number=int(pr_number),
            pr_state=pr_state,
            draft_status="approved_for_launch",
            verification={},
            code_review=review_gate,
        ))
        run = auto_merge_pr(
            store=store,
            scm_store=scm_store,
            repo=repo,
            pr_number=int(pr_number),
            readiness=readiness,
            labels={*labels, AUTO_MERGE_LABEL},
        )
    elif action == "release":
        run = auto_release_oryn_workspace(
            store=store,
            repo=repo,
            pr_number=int(pr_number),
            labels={*labels, AUTO_RELEASE_LABEL},
            release_dispatcher=release_dispatcher,
            command_runner=command_runner,
        )
    else:
        raise ValueError(f"Unsupported PR automation action: {action}")
    return {"ok": True, "object": "hermes.dev_pr_automation_action", "run": run}


def fetch_open_prs(
    *,
    repo: str,
    limit: int = 50,
    command_runner: Optional[Callable[..., Dict[str, Any]]] = None,
) -> tuple[list[Dict[str, Any]], list[str]]:
    return _fetch_pr_list(repo=repo, state="open", limit=limit, command_runner=command_runner)


def fetch_recent_merged_prs(
    *,
    repo: str,
    limit: int = 20,
    command_runner: Optional[Callable[..., Dict[str, Any]]] = None,
) -> tuple[list[Dict[str, Any]], list[str]]:
    return _fetch_pr_list(repo=repo, state="merged", limit=limit, command_runner=command_runner)


def project_id_for_repo(repo: str) -> str:
    mapping = dict(DEFAULT_PROJECT_IDS)
    raw = os.getenv("HERMES_DEV_PR_AUTOMATION_PROJECTS_JSON")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                mapping.update({str(key): str(value) for key, value in parsed.items()})
        except Exception:
            pass
    return mapping.get(repo, "OrynPlatform")


def _fetch_pr_list(
    *,
    repo: str,
    state: str,
    limit: int,
    command_runner: Optional[Callable[..., Dict[str, Any]]] = None,
) -> tuple[list[Dict[str, Any]], list[str]]:
    runner = command_runner or _run_command
    fields = (
        "number,isDraft,labels,headRefName,headRefOid,url,mergeable,mergeStateStatus,reviewDecision,statusCheckRollup"
        if state == "open"
        else "number,labels,mergedAt"
    )
    result = runner([
        "gh",
        "pr",
        "list",
        "--repo",
        repo,
        "--state",
        state,
        "--limit",
        str(max(1, min(int(limit or 50), 200))),
        "--json",
        fields,
    ], timeout=60)
    if int(result.get("exit_code") or 0) != 0:
        return [], [f"gh pr list failed for {repo}: {result.get('output') or 'unknown error'}"]
    try:
        payload = json.loads(str(result.get("output") or "[]"))
    except Exception as exc:
        return [], [f"gh pr list returned invalid JSON for {repo}: {exc}"]
    if not isinstance(payload, list):
        return [], [f"gh pr list returned non-list JSON for {repo}."]
    return [dict(item) for item in payload if isinstance(item, dict)], []


def _poll_should_request_copilot(
    *,
    store: DevGitHubPRAutomationStore,
    repo: str,
    pr_number: int,
    head_sha: Optional[str],
    labels: set[str],
    review_gate: Dict[str, Any],
    draft: bool,
) -> bool:
    if draft:
        return False
    if not labels.intersection({AUTO_FIX_LABEL, AUTO_MERGE_LABEL, AUTO_RELEASE_LABEL}):
        return False
    if str(review_gate.get("verdict") or "").lower() in {"approved", "changes_requested"}:
        return False
    return not store.has_action(
        repo=repo,
        pr_number=pr_number,
        action="request_copilot_review",
        head_sha=head_sha,
        statuses={"completed", "launched", "needs_human"},
    )


def _poll_should_auto_fix(
    *,
    store: DevGitHubPRAutomationStore,
    repo: str,
    pr_number: int,
    head_sha: Optional[str],
    labels: set[str],
    pr: Dict[str, Any],
    draft: bool,
) -> bool:
    if AUTO_FIX_LABEL not in labels or draft:
        return False
    if store.has_action(
        repo=repo,
        pr_number=pr_number,
        action="fix_ci",
        head_sha=head_sha,
        statuses={"launched"},
    ) or store.has_action(
        repo=repo,
        pr_number=pr_number,
        action="fix_review_comments",
        head_sha=head_sha,
        statuses={"launched"},
    ):
        return False
    if _poll_ci_failed(pr):
        return True
    review_decision = str(pr.get("reviewDecision") or "").upper()
    return review_decision in {"CHANGES_REQUESTED", "REVIEW_REQUIRED"}


def _poll_fix_reason(pr: Dict[str, Any]) -> str:
    if _poll_ci_failed(pr):
        return "CI reported a failing check/status."
    return "Review state needs follow-up."


def _poll_ci_failed(pr: Dict[str, Any]) -> bool:
    rollup = pr.get("statusCheckRollup")
    contexts = rollup if isinstance(rollup, list) else []
    for context in contexts:
        if not isinstance(context, dict):
            continue
        conclusion = str(context.get("conclusion") or "").upper()
        state = str(context.get("state") or context.get("status") or "").upper()
        if conclusion in {"FAILURE", "CANCELLED", "TIMED_OUT", "ACTION_REQUIRED"}:
            return True
        if state in {"FAILURE", "ERROR"}:
            return True
    return False


def _record_needs_human(
    store: DevGitHubPRAutomationStore,
    repo: str,
    pr_number: int,
    head_sha: Optional[str],
    action: str,
    labels: Iterable[str],
    reason: str,
) -> Dict[str, Any]:
    return store.record_run({
        "repo": repo,
        "pr_number": pr_number,
        "head_sha": head_sha,
        "action": action,
        "status": "needs_human",
        "labels": sorted(set(labels)),
        "reason": reason,
        "result": {},
        "warnings": [reason],
    })


def _event_failed(normalized: Dict[str, Any]) -> bool:
    conclusion = str(normalized.get("conclusion") or normalized.get("status") or "").lower()
    if conclusion in {"success", "neutral", "skipped"}:
        return False
    if conclusion in {"failure", "timed_out", "cancelled", "action_required", "error"}:
        return True
    return False


def _label_names(labels: Any) -> list[str]:
    names: list[str] = []
    for item in labels or []:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
        else:
            name = str(item or "").strip()
        if name:
            names.append(name)
    return names


def _safe_relative_path(path: str) -> bool:
    if not path or path.startswith("/") or ".." in Path(path).parts:
        return False
    return True


def _max_fix_attempts() -> int:
    try:
        return max(0, min(int(os.getenv("HERMES_DEV_PR_AUTOMATION_MAX_FIX_ATTEMPTS", "2")), 10))
    except Exception:
        return 2


def _run_command(command: list[str], *, timeout: int = 60) -> Dict[str, Any]:
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
    output = (result.stdout or result.stderr or "").strip()
    return {
        "command": " ".join(command),
        "exit_code": result.returncode,
        "output": output[:4000],
    }


def _github_get_json(url: str) -> Any:
    token = _github_token()
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "HermesDevGitHubPRAutomation",
        **({"Authorization": f"Bearer {token}"} if token else {}),
    }
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=8) as response:
        return json.loads(response.read().decode("utf-8"))


def _github_token() -> str:
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if token:
        return token.strip()
    try:
        result = subprocess.run(["gh", "auth", "token"], check=False, capture_output=True, text=True, timeout=3)
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _webhook_from_row(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    item = dict(row)
    item["object"] = "hermes.dev_github_webhook_event"
    item["accepted"] = bool(item.get("accepted"))
    for key, default in (("result", {}), ("payload", {}), ("warnings", [])):
        item[key] = _json_value(item.get(key), default)
    return item


def _run_from_row(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    item = dict(row)
    item["object"] = "hermes.dev_pr_automation_run"
    for key, default in (("labels", []), ("result", {}), ("warnings", [])):
        item[key] = _json_value(item.get(key), default)
    return item


def _json_value(raw: Any, default: Any) -> Any:
    try:
        return json.loads(raw or json.dumps(default))
    except Exception:
        return default


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
