#!/usr/bin/env python3
"""GitHub integration primitives for Hermes Code Mode (P1)."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from hermes_constants import get_hermes_home

GITHUB_API_BASE = "https://api.github.com"
GITHUB_API_VERSION = "2022-11-28"

_REDACT_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)(authorization\s*:\s*(?:bearer\s+)?)(\S+)"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(bearer\s+)\S+"), r"\1[REDACTED]"),
    (
        re.compile(
            r"(?i)((?:token|private_key|webhook_secret|access_token|refresh_token|client_secret)\s*[:=]\s*)\S+"
        ),
        r"\1[REDACTED]",
    ),
    (re.compile(r"gh[pousr]_[A-Za-z0-9_]{8,}"), "[REDACTED]"),
    (re.compile(r"\bHERMES_GITHUB_DEV_PAT\s*=\s*\S+"), "HERMES_GITHUB_DEV_PAT=[REDACTED]"),
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_value(name: str) -> str:
    value = os.getenv(name, "")
    if value:
        return value
    try:
        from hermes_cli.config import load_env

        return str(load_env().get(name, "") or "")
    except Exception:
        return ""


def redact_github_secrets(text: str) -> str:
    value = str(text or "")
    for pattern, replacement in _REDACT_RULES:
        value = pattern.sub(replacement, value)
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, separators=(",", ":"), sort_keys=True)


def _json_loads(value: Any, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


@dataclass(frozen=True)
class GitHubAppConfig:
    app_id: Optional[str]
    private_key_path: Optional[str]
    webhook_secret_configured: bool
    dev_pat_configured: bool
    allow_dev_pat: bool

    @property
    def app_configured(self) -> bool:
        return bool(self.app_id and self.private_key_path)

    @property
    def mode(self) -> str:
        if self.app_configured:
            return "github_app"
        if self.dev_pat_configured and self.allow_dev_pat:
            return "pat_dev"
        return "unconfigured"


class GitHubAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        rate_limit: Optional[dict[str, str]] = None,
        response: Optional[Any] = None,
    ) -> None:
        super().__init__(redact_github_secrets(message))
        self.message = redact_github_secrets(message)
        self.status_code = status_code
        self.rate_limit = rate_limit or {}
        self.response = response


class GitHubAPIClient:
    def __init__(
        self,
        token_provider: Callable[[], str],
        *,
        base_url: str = GITHUB_API_BASE,
        http_client: Optional[Any] = None,
    ) -> None:
        self._token_provider = token_provider
        self._base_url = base_url.rstrip("/")
        self._http = http_client or httpx.Client(timeout=30.0)

    @staticmethod
    def _rate_limit(headers: Any) -> dict[str, str]:
        return {
            "limit": str(headers.get("x-ratelimit-limit", "")),
            "remaining": str(headers.get("x-ratelimit-remaining", "")),
            "reset": str(headers.get("x-ratelimit-reset", "")),
            "resource": str(headers.get("x-ratelimit-resource", "")),
        }

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        url = path if path.startswith("http") else f"{self._base_url}/{path.lstrip('/')}"
        token = self._token_provider()
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
            "User-Agent": "Hermes-Agent-Code-Mode",
        }
        try:
            response = self._http.request(
                method.upper(),
                url,
                params=params,
                json=json_body,
                headers=headers,
            )
        except Exception as exc:
            raise GitHubAPIError(f"GitHub request failed: {exc}") from exc

        rate_limit = self._rate_limit(response.headers)
        if response.status_code >= 400:
            message = f"HTTP {response.status_code}"
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    message = str(payload.get("message") or message)
            except Exception:
                message = getattr(response, "text", message) or message
            raise GitHubAPIError(
                message,
                status_code=response.status_code,
                rate_limit=rate_limit,
                response=response,
            )

        if response.status_code == 204:
            data: Any = None
        else:
            try:
                data = response.json()
            except Exception:
                data = getattr(response, "text", "")
        return {"data": data, "status_code": response.status_code, "rate_limit": rate_limit}

    def list_paginated(
        self,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        limit: int = 100,
    ) -> list[Any]:
        page = 1
        per_page = min(100, max(1, int(limit)))
        results: list[Any] = []
        while len(results) < limit:
            query = dict(params or {})
            query["page"] = page
            query["per_page"] = per_page
            payload = self.request("GET", path, params=query)["data"]
            if isinstance(payload, dict) and isinstance(payload.get("repositories"), list):
                items = payload["repositories"]
            elif isinstance(payload, list):
                items = payload
            else:
                items = []
            if not items:
                break
            for item in items:
                if len(results) >= limit:
                    break
                results.append(item)
            if len(items) < per_page:
                break
            page += 1
        return results


class GitHubIntegrationStore:
    """Persistence layer over the main Hermes state.db."""

    _WRITE_MAX_RETRIES = 5

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or (get_hermes_home() / "state.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure schema migrations (including GitHub tables) are applied
        # through the canonical SessionDB path before direct sqlite access.
        try:
            from hermes_state import SessionDB

            SessionDB(db_path=self._db_path).close()
        except Exception:
            pass
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def _execute_write(self, fn):
        last_error = None
        for _ in range(self._WRITE_MAX_RETRIES):
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    result = fn(self._conn)
                    self._conn.commit()
                    return result
                except BaseException:
                    self._conn.rollback()
                    raise
            except sqlite3.OperationalError as exc:
                last_error = exc
                if "locked" not in str(exc).lower():
                    raise
                time.sleep(0.05)
        if last_error:
            raise last_error

    @staticmethod
    def _row(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
        if not row:
            return None
        result = dict(row)
        for key in ("permissions_json", "events_json", "labels_json", "assignees_json", "milestone_json"):
            if key in result:
                result[key.removesuffix("_json")] = _json_loads(
                    result.pop(key),
                    [] if key.endswith("s_json") else {},
                )
        for key in ("private", "archived", "disabled", "mergeable", "draft", "protected"):
            if key in result and result[key] is not None:
                result[key] = bool(result[key])
        return result

    def upsert_installation(self, installation: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        installation_id = int(installation["installation_id"])

        def _do(conn):
            conn.execute(
                """
                INSERT INTO github_app_installations
                    (id, installation_id, account_login, account_type, app_id, permissions_json,
                     events_json, status, created_at, updated_at, last_synced_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(installation_id) DO UPDATE SET
                    account_login=excluded.account_login,
                    account_type=excluded.account_type,
                    app_id=excluded.app_id,
                    permissions_json=excluded.permissions_json,
                    events_json=excluded.events_json,
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    last_synced_at=excluded.last_synced_at
                """,
                (
                    str(uuid.uuid4()),
                    installation_id,
                    installation.get("account_login"),
                    installation.get("account_type"),
                    str(installation.get("app_id") or ""),
                    _json_dumps(installation.get("permissions") or {}),
                    _json_dumps(installation.get("events") or []),
                    installation.get("status") or "active",
                    now,
                    now,
                    now,
                ),
            )

        self._execute_write(_do)
        return self.get_installation(installation_id) or {}

    def get_installation(self, installation_id: int) -> Optional[dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM github_app_installations WHERE installation_id = ?",
            (int(installation_id),),
        ).fetchone()
        return self._row(row)

    def list_installations(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM github_app_installations ORDER BY updated_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [self._row(row) or {} for row in rows]

    def upsert_repository(self, repo: dict[str, Any], *, installation_id: Optional[int] = None) -> dict[str, Any]:
        now = time.time()
        owner = repo.get("owner", {}).get("login") if isinstance(repo.get("owner"), dict) else repo.get("owner")
        full_name = repo.get("full_name") or f"{owner}/{repo.get('name')}"
        if "/" in str(full_name):
            owner, name = str(full_name).split("/", 1)
        else:
            name = repo.get("name")

        def _do(conn):
            conn.execute(
                """
                INSERT INTO github_repositories
                    (id, installation_id, github_repo_id, owner, name, full_name, default_branch, private,
                     html_url, clone_url, ssh_url, archived, disabled, pushed_at, created_at, updated_at, last_synced_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(full_name) DO UPDATE SET
                    installation_id=excluded.installation_id,
                    github_repo_id=excluded.github_repo_id,
                    owner=excluded.owner,
                    name=excluded.name,
                    default_branch=excluded.default_branch,
                    private=excluded.private,
                    html_url=excluded.html_url,
                    clone_url=excluded.clone_url,
                    ssh_url=excluded.ssh_url,
                    archived=excluded.archived,
                    disabled=excluded.disabled,
                    pushed_at=excluded.pushed_at,
                    updated_at=excluded.updated_at,
                    last_synced_at=excluded.last_synced_at
                """,
                (
                    str(uuid.uuid4()),
                    installation_id,
                    repo.get("id") or repo.get("github_repo_id"),
                    owner,
                    name,
                    full_name,
                    repo.get("default_branch"),
                    int(bool(repo.get("private"))),
                    repo.get("html_url"),
                    repo.get("clone_url"),
                    repo.get("ssh_url"),
                    int(bool(repo.get("archived"))),
                    int(bool(repo.get("disabled"))),
                    repo.get("pushed_at"),
                    now,
                    now,
                    now,
                ),
            )

        self._execute_write(_do)
        return self.get_repository(str(owner), str(name)) or {}

    def list_repositories(
        self,
        *,
        owner: Optional[str] = None,
        q: Optional[str] = None,
        installation_id: Optional[int] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if owner:
            clauses.append("owner = ?")
            params.append(owner)
        if q:
            clauses.append("(full_name LIKE ? OR name LIKE ?)")
            params.extend([f"%{q}%", f"%{q}%"])
        if installation_id is not None:
            clauses.append("installation_id = ?")
            params.append(int(installation_id))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        rows = self._conn.execute(
            f"SELECT * FROM github_repositories {where} ORDER BY full_name ASC LIMIT ?",
            tuple(params),
        ).fetchall()
        return [self._row(row) or {} for row in rows]

    def get_repository(self, owner: str, repo: str) -> Optional[dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM github_repositories WHERE full_name = ?",
            (f"{owner}/{repo}",),
        ).fetchone()
        return self._row(row)

    def upsert_issue(self, repo_full_name: str, issue: dict[str, Any]) -> dict[str, Any]:
        now = time.time()

        def _do(conn):
            conn.execute(
                """
                INSERT INTO github_issues
                    (id, repo_full_name, github_issue_id, number, title, state, author_login, labels_json,
                     assignees_json, milestone_json, html_url, created_at, updated_at, closed_at, last_synced_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo_full_name, number) DO UPDATE SET
                    github_issue_id=excluded.github_issue_id,
                    title=excluded.title,
                    state=excluded.state,
                    author_login=excluded.author_login,
                    labels_json=excluded.labels_json,
                    assignees_json=excluded.assignees_json,
                    milestone_json=excluded.milestone_json,
                    html_url=excluded.html_url,
                    updated_at=excluded.updated_at,
                    closed_at=excluded.closed_at,
                    last_synced_at=excluded.last_synced_at
                """,
                (
                    str(uuid.uuid4()),
                    repo_full_name,
                    issue.get("id") or issue.get("github_issue_id"),
                    issue.get("number"),
                    issue.get("title"),
                    issue.get("state"),
                    (issue.get("user") or {}).get("login") if isinstance(issue.get("user"), dict) else issue.get("author_login"),
                    _json_dumps(issue.get("labels") or []),
                    _json_dumps(issue.get("assignees") or []),
                    _json_dumps(issue.get("milestone") or {}),
                    issue.get("html_url"),
                    issue.get("created_at") or now,
                    issue.get("updated_at") or now,
                    issue.get("closed_at"),
                    now,
                ),
            )

        self._execute_write(_do)
        row = self._conn.execute(
            "SELECT * FROM github_issues WHERE repo_full_name = ? AND number = ?",
            (repo_full_name, issue.get("number")),
        ).fetchone()
        return self._row(row) or {}

    def list_issues(self, repo_full_name: str, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM github_issues WHERE repo_full_name = ? ORDER BY updated_at DESC LIMIT ?",
            (repo_full_name, int(limit)),
        ).fetchall()
        return [self._row(row) or {} for row in rows]

    def upsert_pull_request(self, repo_full_name: str, pr: dict[str, Any]) -> dict[str, Any]:
        now = time.time()

        def _do(conn):
            conn.execute(
                """
                INSERT INTO github_pull_requests
                    (id, repo_full_name, github_pr_id, number, title, state, author_login, base_branch, head_branch,
                     head_sha, mergeable, draft, html_url, created_at, updated_at, closed_at, merged_at, last_synced_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo_full_name, number) DO UPDATE SET
                    github_pr_id=excluded.github_pr_id,
                    title=excluded.title,
                    state=excluded.state,
                    author_login=excluded.author_login,
                    base_branch=excluded.base_branch,
                    head_branch=excluded.head_branch,
                    head_sha=excluded.head_sha,
                    mergeable=excluded.mergeable,
                    draft=excluded.draft,
                    html_url=excluded.html_url,
                    updated_at=excluded.updated_at,
                    closed_at=excluded.closed_at,
                    merged_at=excluded.merged_at,
                    last_synced_at=excluded.last_synced_at
                """,
                (
                    str(uuid.uuid4()),
                    repo_full_name,
                    pr.get("id") or pr.get("github_pr_id"),
                    pr.get("number"),
                    pr.get("title"),
                    pr.get("state"),
                    (pr.get("user") or {}).get("login") if isinstance(pr.get("user"), dict) else pr.get("author_login"),
                    (pr.get("base") or {}).get("ref") if isinstance(pr.get("base"), dict) else pr.get("base_branch"),
                    (pr.get("head") or {}).get("ref") if isinstance(pr.get("head"), dict) else pr.get("head_branch"),
                    (pr.get("head") or {}).get("sha") if isinstance(pr.get("head"), dict) else pr.get("head_sha"),
                    None if pr.get("mergeable") is None else int(bool(pr.get("mergeable"))),
                    int(bool(pr.get("draft"))),
                    pr.get("html_url"),
                    pr.get("created_at") or now,
                    pr.get("updated_at") or now,
                    pr.get("closed_at"),
                    pr.get("merged_at"),
                    now,
                ),
            )

        self._execute_write(_do)
        row = self._conn.execute(
            "SELECT * FROM github_pull_requests WHERE repo_full_name = ? AND number = ?",
            (repo_full_name, pr.get("number")),
        ).fetchone()
        return self._row(row) or {}

    def list_pull_requests(self, repo_full_name: str, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM github_pull_requests WHERE repo_full_name = ? ORDER BY updated_at DESC LIMIT ?",
            (repo_full_name, int(limit)),
        ).fetchall()
        return [self._row(row) or {} for row in rows]

    def upsert_branch(self, repo_full_name: str, branch: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        sha = (branch.get("commit") or {}).get("sha") if isinstance(branch.get("commit"), dict) else branch.get("sha")

        def _do(conn):
            conn.execute(
                """
                INSERT INTO github_branches
                    (id, repo_full_name, name, sha, protected, created_at, updated_at, last_synced_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo_full_name, name) DO UPDATE SET
                    sha=excluded.sha,
                    protected=excluded.protected,
                    updated_at=excluded.updated_at,
                    last_synced_at=excluded.last_synced_at
                """,
                (
                    str(uuid.uuid4()),
                    repo_full_name,
                    branch.get("name"),
                    sha,
                    int(bool(branch.get("protected"))),
                    now,
                    now,
                    now,
                ),
            )

        self._execute_write(_do)
        row = self._conn.execute(
            "SELECT * FROM github_branches WHERE repo_full_name = ? AND name = ?",
            (repo_full_name, branch.get("name")),
        ).fetchone()
        return self._row(row) or {}

    def list_branches(self, repo_full_name: str, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM github_branches WHERE repo_full_name = ? ORDER BY name ASC LIMIT ?",
            (repo_full_name, int(limit)),
        ).fetchall()
        return [self._row(row) or {} for row in rows]

    def get_delivery(self, delivery_id: str) -> Optional[dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM github_webhook_deliveries WHERE delivery_id = ?",
            (delivery_id,),
        ).fetchone()
        return self._row(row)

    def record_webhook_delivery(
        self,
        *,
        delivery_id: str,
        event: str,
        action: Optional[str],
        repo_full_name: Optional[str],
        sender_login: Optional[str],
        payload_hash: str,
        status: str,
        error: Optional[str] = None,
    ) -> dict[str, Any]:
        now = time.time()

        def _do(conn):
            conn.execute(
                """
                INSERT INTO github_webhook_deliveries
                    (id, delivery_id, event, action, repo_full_name, sender_login, payload_hash, status, error, received_at, processed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(delivery_id) DO NOTHING
                """,
                (
                    str(uuid.uuid4()),
                    delivery_id,
                    event,
                    action,
                    repo_full_name,
                    sender_login,
                    payload_hash,
                    status,
                    error,
                    now,
                    now if status not in {"received", "pending"} else None,
                ),
            )

        self._execute_write(_do)
        return self.get_delivery(delivery_id) or {}

    def update_delivery_status(self, delivery_id: str, status: str, error: Optional[str] = None) -> None:
        now = time.time()

        def _do(conn):
            conn.execute(
                """
                UPDATE github_webhook_deliveries
                SET status = ?, error = ?, processed_at = ?
                WHERE delivery_id = ?
                """,
                (status, error, now, delivery_id),
            )

        self._execute_write(_do)

    def create_chatops_command(
        self,
        *,
        delivery_id: Optional[str],
        repo_full_name: str,
        issue_number: Optional[int],
        pr_number: Optional[int],
        comment_id: Optional[int],
        sender_login: Optional[str],
        command: str,
        args: str = "",
        status: str = "pending",
        orchestrated_run_id: Optional[str] = None,
        code_session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        now = time.time()
        command_id = str(uuid.uuid4())

        def _do(conn):
            conn.execute(
                """
                INSERT INTO github_chatops_commands
                    (id, delivery_id, repo_full_name, issue_number, pr_number, comment_id, sender_login,
                     command, args, status, orchestrated_run_id, code_session_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    command_id,
                    delivery_id,
                    repo_full_name,
                    issue_number,
                    pr_number,
                    comment_id,
                    sender_login,
                    command,
                    args,
                    status,
                    orchestrated_run_id,
                    code_session_id,
                    now,
                    now,
                ),
            )

        self._execute_write(_do)
        return self.get_chatops_command(command_id) or {}

    def get_chatops_command(self, command_id: str) -> Optional[dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM github_chatops_commands WHERE id = ?",
            (command_id,),
        ).fetchone()
        return self._row(row)

    def update_chatops_command(self, command_id: str, **updates: Any) -> Optional[dict[str, Any]]:
        allowed = {"status", "orchestrated_run_id", "code_session_id", "args"}
        fields = {k: v for k, v in updates.items() if k in allowed}
        if not fields:
            return self.get_chatops_command(command_id)
        fields["updated_at"] = time.time()
        assignments = ", ".join(f"{name} = ?" for name in fields)
        values = list(fields.values()) + [command_id]

        def _do(conn):
            conn.execute(
                f"UPDATE github_chatops_commands SET {assignments} WHERE id = ?",
                tuple(values),
            )

        self._execute_write(_do)
        return self.get_chatops_command(command_id)

    def list_chatops_commands(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM github_chatops_commands ORDER BY created_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [self._row(row) or {} for row in rows]

    def create_status_report(self, report: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        report_id = str(uuid.uuid4())

        def _do(conn):
            conn.execute(
                """
                INSERT INTO github_status_reports
                    (id, repo_full_name, sha, context, status, conclusion, details_url, external_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report_id,
                    report.get("repo_full_name"),
                    report.get("sha"),
                    report.get("context"),
                    report.get("status"),
                    report.get("conclusion"),
                    report.get("details_url"),
                    report.get("external_id"),
                    now,
                    now,
                ),
            )

        self._execute_write(_do)
        row = self._conn.execute("SELECT * FROM github_status_reports WHERE id = ?", (report_id,)).fetchone()
        return self._row(row) or {}


class GitHubIntegrationService:
    """High-level GitHub integration facade."""

    _token_cache: dict[int, dict[str, Any]] = {}

    def __init__(self, db_path: Optional[Path] = None, *, http_client: Optional[Any] = None) -> None:
        self._db_path = db_path
        self._http_client = http_client

    def _store(self) -> GitHubIntegrationStore:
        return GitHubIntegrationStore(db_path=self._db_path)

    def config(self) -> GitHubAppConfig:
        return GitHubAppConfig(
            app_id=_env_value("HERMES_GITHUB_APP_ID").strip() or None,
            private_key_path=_env_value("HERMES_GITHUB_APP_PRIVATE_KEY_PATH").strip() or None,
            webhook_secret_configured=bool(_env_value("HERMES_GITHUB_WEBHOOK_SECRET").strip()),
            dev_pat_configured=bool(_env_value("HERMES_GITHUB_DEV_PAT").strip()),
            allow_dev_pat=_env_value("HERMES_GITHUB_ALLOW_DEV_PAT").strip() == "1",
        )

    def status(self) -> dict[str, Any]:
        config = self.config()
        store = self._store()
        try:
            installation_count = len(store.list_installations(limit=500))
            repository_count = len(store.list_repositories(limit=1000))
        finally:
            store.close()
        private_key_available = False
        if config.private_key_path:
            try:
                private_key_available = Path(config.private_key_path).expanduser().is_file()
            except Exception:
                private_key_available = False
        return {
            "configured": config.mode != "unconfigured",
            "mode": config.mode,
            "app_id_configured": bool(config.app_id),
            "private_key_configured": bool(config.private_key_path),
            "private_key_available": private_key_available,
            "webhook_secret_configured": config.webhook_secret_configured,
            "pat_dev_configured": config.dev_pat_configured and config.allow_dev_pat,
            "installations": installation_count,
            "repositories": repository_count,
        }

    def _make_app_jwt(self) -> str:
        config = self.config()
        if not config.app_configured:
            raise GitHubAPIError("GitHub App is not configured")
        try:
            import jwt
        except Exception as exc:
            raise GitHubAPIError("PyJWT is required for GitHub App authentication") from exc

        key_path = Path(str(config.private_key_path)).expanduser()
        try:
            private_key = key_path.read_text(encoding="utf-8")
        except Exception as exc:
            raise GitHubAPIError(f"Unable to read GitHub private key: {exc}") from exc

        now = int(time.time())
        payload = {"iat": now - 60, "exp": now + 540, "iss": str(config.app_id)}
        token = jwt.encode(payload, private_key, algorithm="RS256")
        return token if isinstance(token, str) else token.decode("utf-8")

    @staticmethod
    def _parse_expiry(expires_at: str) -> datetime:
        return datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))

    def _request_installation_token(self, installation_id: int) -> dict[str, Any]:
        client = GitHubAPIClient(self._make_app_jwt, http_client=self._http_client)
        payload = client.request("POST", f"/app/installations/{int(installation_id)}/access_tokens")["data"]
        data = payload if isinstance(payload, dict) else {}
        return {"token": data.get("token"), "expires_at": data.get("expires_at")}

    def get_installation_token(self, installation_id: int) -> str:
        config = self.config()
        if config.mode == "pat_dev":
            pat = _env_value("HERMES_GITHUB_DEV_PAT").strip()
            if not pat:
                raise GitHubAPIError("GitHub PAT fallback is enabled but token is missing")
            return pat

        cached = self._token_cache.get(int(installation_id))
        if cached:
            expires_at_dt = cached.get("expires_at_dt")
            if isinstance(expires_at_dt, datetime) and expires_at_dt > datetime.now(timezone.utc) + timedelta(seconds=60):
                return str(cached["token"])

        token_data = self._request_installation_token(int(installation_id))
        token = token_data.get("token")
        expires_at = token_data.get("expires_at")
        if not token or not expires_at:
            raise GitHubAPIError("GitHub installation token response is incomplete")
        self._token_cache[int(installation_id)] = {
            "token": token,
            "expires_at": expires_at,
            "expires_at_dt": self._parse_expiry(str(expires_at)),
        }
        return str(token)

    def api_client(self, installation_id: Optional[int] = None) -> GitHubAPIClient:
        config = self.config()
        if config.mode == "pat_dev":
            return GitHubAPIClient(lambda: _env_value("HERMES_GITHUB_DEV_PAT").strip(), http_client=self._http_client)

        if installation_id is None:
            store = self._store()
            try:
                installations = store.list_installations(limit=1)
            finally:
                store.close()
            if not installations:
                raise GitHubAPIError("No GitHub installation is registered")
            installation_id = int(installations[0]["installation_id"])

        return GitHubAPIClient(
            lambda: self.get_installation_token(int(installation_id)),
            http_client=self._http_client,
        )

    def post_issue_comment(
        self,
        *,
        repo_full_name: str,
        issue_number: int,
        body: str,
        installation_id: Optional[int] = None,
    ) -> dict[str, Any]:
        result = self.api_client(installation_id).request(
            "POST",
            f"/repos/{repo_full_name}/issues/{int(issue_number)}/comments",
            json_body={"body": body},
        )
        return result["data"] if isinstance(result["data"], dict) else {"result": result["data"]}

    def prepare_pull_request(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(uuid.uuid4()),
            "prepared_at": _utc_now_iso(),
            "repo_full_name": metadata.get("repo_full_name"),
            "title": metadata.get("title"),
            "head": metadata.get("head"),
            "base": metadata.get("base") or "main",
            "body": metadata.get("body") or "",
            "auto_push": False,
            "auto_merge": False,
        }
