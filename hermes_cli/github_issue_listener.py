"""Assignment-driven GitHub issue listener.

This module implements the small, conservative MVP for treating GitHub
issues as a Hermes messaging lane.  A scheduled poll can look for open issues
assigned to a Hermes machine user, keep a durable issue -> session mapping,
run/resume Hermes for the issue, and post the response back as a comment.

The listener deliberately keeps operational state in a local SQLite database
instead of editing issue bodies.  GitHub assignment is the routing signal;
local claims prevent duplicate pickup when a poll overlaps a still-running
job.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import psutil

from hermes_constants import get_hermes_home

WAITING_MARKER = "[HERMES_WAITING_FOR_RYAN]"
READY_TO_CLOSE_MARKER = "[HERMES_READY_TO_CLOSE]"
DEFAULT_WIP_COMMENT_BODY = "Hermes picked this up and is working on it now."
DEFAULT_STALE_AFTER_SECONDS = 60 * 60
DEFAULT_STATE_DB = get_hermes_home() / "github_issue_listener.db"


@dataclass(frozen=True)
class IssueRef:
    owner: str
    repo: str
    number: int

    @property
    def key(self) -> str:
        return f"{self.owner}/{self.repo}#{self.number}"


@dataclass
class IssueState:
    owner: str
    repo: str
    issue_number: int
    session_id: str | None = None
    status: str = "idle"
    current_run_id: str | None = None
    claimed_at: float | None = None
    last_comment_id_seen: int | None = None
    updated_at: float | None = None
    worker_pid: int | None = None
    worker_started_at: float | None = None
    log_path: str | None = None


class GitHubAPI(Protocol):
    def list_assigned_issues(self, owner: str, repo: str, assignee: str) -> list[dict[str, Any]]: ...
    def list_project_assigned_issues(self, project_owner: str, project_number: int, assignee: str) -> list[dict[str, Any]]: ...
    def get_issue(self, owner: str, repo: str, issue_number: int) -> dict[str, Any]: ...
    def list_comments(self, owner: str, repo: str, issue_number: int) -> list[dict[str, Any]]: ...
    def add_comment(self, owner: str, repo: str, issue_number: int, body: str) -> dict[str, Any]: ...
    def set_assignees(self, owner: str, repo: str, issue_number: int, assignees: list[str]) -> dict[str, Any]: ...
    def clear_assignees(self, owner: str, repo: str, issue_number: int, assignees: list[str]) -> dict[str, Any]: ...
    def set_project_status(
        self, project_owner: str, project_number: int, owner: str, repo: str, issue_number: int, status: str
    ) -> dict[str, Any]: ...
    def set_project_execution_mode(
        self, project_owner: str, project_number: int, owner: str, repo: str, issue_number: int, mode: str
    ) -> dict[str, Any]: ...


class HermesRunner(Protocol):
    def run_issue_turn(self, prompt: str, *, session_id: str | None) -> tuple[str, str]: ...


class ListenerStore:
    def __init__(self, path: Path | str = DEFAULT_STATE_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS issue_sessions (
                    owner TEXT NOT NULL,
                    repo TEXT NOT NULL,
                    issue_number INTEGER NOT NULL,
                    session_id TEXT,
                    status TEXT NOT NULL DEFAULT 'idle',
                    current_run_id TEXT,
                    claimed_at REAL,
                    last_comment_id_seen INTEGER,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (owner, repo, issue_number)
                )
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(issue_sessions)")}
            for name, definition in {
                "worker_pid": "INTEGER",
                "worker_started_at": "REAL",
                "log_path": "TEXT",
            }.items():
                if name not in columns:
                    conn.execute(f"ALTER TABLE issue_sessions ADD COLUMN {name} {definition}")
            conn.commit()

    def get(self, ref: IssueRef) -> IssueState | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM issue_sessions
                WHERE owner = ? AND repo = ? AND issue_number = ?
                """,
                (ref.owner, ref.repo, ref.number),
            ).fetchone()
        return self._row_to_state(row) if row else None

    def upsert(self, state: IssueState) -> None:
        state.updated_at = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO issue_sessions (
                    owner, repo, issue_number, session_id, status, current_run_id,
                    claimed_at, last_comment_id_seen, updated_at,
                    worker_pid, worker_started_at, log_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(owner, repo, issue_number) DO UPDATE SET
                    session_id = excluded.session_id,
                    status = excluded.status,
                    current_run_id = excluded.current_run_id,
                    claimed_at = excluded.claimed_at,
                    last_comment_id_seen = excluded.last_comment_id_seen,
                    updated_at = excluded.updated_at,
                    worker_pid = excluded.worker_pid,
                    worker_started_at = excluded.worker_started_at,
                    log_path = excluded.log_path
                """,
                (
                    state.owner,
                    state.repo,
                    state.issue_number,
                    state.session_id,
                    state.status,
                    state.current_run_id,
                    state.claimed_at,
                    state.last_comment_id_seen,
                    state.updated_at,
                    state.worker_pid,
                    state.worker_started_at,
                    state.log_path,
                ),
            )
            conn.commit()

    def try_claim(self, ref: IssueRef, *, run_id: str, stale_after_seconds: int) -> bool:
        """Atomically claim an issue unless a fresh running claim exists."""
        now = time.time()
        stale_before = now - stale_after_seconds
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT * FROM issue_sessions
                WHERE owner = ? AND repo = ? AND issue_number = ?
                """,
                (ref.owner, ref.repo, ref.number),
            ).fetchone()
            if row and row["status"] == "running" and (row["claimed_at"] or 0) > stale_before:
                conn.rollback()
                return False
            if row:
                conn.execute(
                    """
                    UPDATE issue_sessions
                    SET status = 'running', current_run_id = ?, claimed_at = ?, updated_at = ?,
                        worker_pid = NULL, worker_started_at = NULL, log_path = NULL
                    WHERE owner = ? AND repo = ? AND issue_number = ?
                    """,
                    (run_id, now, now, ref.owner, ref.repo, ref.number),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO issue_sessions (
                        owner, repo, issue_number, status, current_run_id, claimed_at, updated_at
                    ) VALUES (?, ?, ?, 'running', ?, ?, ?)
                    """,
                    (ref.owner, ref.repo, ref.number, run_id, now, now),
                )
            conn.commit()
            return True

    @staticmethod
    def _row_to_state(row: sqlite3.Row) -> IssueState:
        return IssueState(
            owner=row["owner"],
            repo=row["repo"],
            issue_number=row["issue_number"],
            session_id=row["session_id"],
            status=row["status"],
            current_run_id=row["current_run_id"],
            claimed_at=row["claimed_at"],
            last_comment_id_seen=row["last_comment_id_seen"],
            updated_at=row["updated_at"],
            worker_pid=row["worker_pid"],
            worker_started_at=row["worker_started_at"],
            log_path=row["log_path"],
        )


class GitHubRESTClient:
    def __init__(self, token: str | None = None, api_url: str = "https://api.github.com"):
        self.token = token or _load_github_token()
        if not self.token:
            raise RuntimeError("GitHub token not found; set GITHUB_PERSONAL_ACCESS_TOKEN, GITHUB_TOKEN, or authenticate gh")
        self.api_url = api_url.rstrip("/")

    def list_assigned_issues(self, owner: str, repo: str, assignee: str) -> list[dict[str, Any]]:
        query = f"repo:{owner}/{repo} is:issue is:open assignee:{assignee}"
        data = self._request("GET", f"/search/issues?q={urllib.parse.quote(query)}&per_page=50")
        return [item for item in data.get("items", []) if _issue_has_assignee(item, assignee)]

    def list_project_assigned_issues(self, project_owner: str, project_number: int, assignee: str) -> list[dict[str, Any]]:
        """List open Project V2 issue items currently assigned to ``assignee``.

        GitHub issue search can lag after assignment changes. The project board
        is Ryan's source of truth for routing, so this path queries Project V2
        directly and filters the issue content's current assignees.
        """
        items: list[dict[str, Any]] = []
        after: str | None = None
        while True:
            data = self._graphql(_PROJECT_ITEMS_QUERY, {"owner": project_owner, "number": project_number, "after": after})
            project = (data.get("organization") or {}).get("projectV2")
            if not project:
                raise RuntimeError(f"GitHub Project not found: {project_owner}/{project_number}")
            page = project["items"]
            for node in page.get("nodes") or []:
                content = (node or {}).get("content") or {}
                if content.get("__typename") != "Issue" or content.get("state") != "OPEN":
                    continue
                issue = _project_issue_to_rest_shape(content)
                if _issue_has_assignee(issue, assignee):
                    items.append(issue)
            info = page.get("pageInfo") or {}
            if not info.get("hasNextPage"):
                break
            after = info.get("endCursor")
        return items

    def get_issue(self, owner: str, repo: str, issue_number: int) -> dict[str, Any]:
        return dict(self._request("GET", f"/repos/{owner}/{repo}/issues/{issue_number}"))

    def list_comments(self, owner: str, repo: str, issue_number: int) -> list[dict[str, Any]]:
        return list(self._request("GET", f"/repos/{owner}/{repo}/issues/{issue_number}/comments?per_page=100"))

    def add_comment(self, owner: str, repo: str, issue_number: int, body: str) -> dict[str, Any]:
        return dict(self._request("POST", f"/repos/{owner}/{repo}/issues/{issue_number}/comments", {"body": body}))

    def set_assignees(self, owner: str, repo: str, issue_number: int, assignees: list[str]) -> dict[str, Any]:
        return dict(self._request("PATCH", f"/repos/{owner}/{repo}/issues/{issue_number}", {"assignees": assignees}))

    def clear_assignees(self, owner: str, repo: str, issue_number: int, assignees: list[str]) -> dict[str, Any]:
        if not assignees:
            return self.get_issue(owner, repo, issue_number)
        return dict(self._request("DELETE", f"/repos/{owner}/{repo}/issues/{issue_number}/assignees", {"assignees": assignees}))

    def set_project_execution_mode(
        self, project_owner: str, project_number: int, owner: str, repo: str, issue_number: int, mode: str
    ) -> dict[str, Any]:
        return self._set_project_single_select_field(
            project_owner, project_number, owner, repo, issue_number, "Execution mode", mode
        )

    def set_project_status(
        self, project_owner: str, project_number: int, owner: str, repo: str, issue_number: int, status: str
    ) -> dict[str, Any]:
        return self._set_project_single_select_field(project_owner, project_number, owner, repo, issue_number, "Status", status)

    def _set_project_single_select_field(
        self,
        project_owner: str,
        project_number: int,
        owner: str,
        repo: str,
        issue_number: int,
        field_name: str,
        option_name: str,
    ) -> dict[str, Any]:
        after: str | None = None
        project_id: str | None = None
        field_id: str | None = None
        option_id: str | None = None
        item_id: str | None = None

        while True:
            data = self._graphql(
                _PROJECT_ITEM_EXECUTION_MODE_QUERY,
                {"owner": project_owner, "number": project_number, "after": after},
            )
            project = (data.get("organization") or {}).get("projectV2")
            if not project:
                raise RuntimeError(f"GitHub Project not found: {project_owner}/{project_number}")
            project_id = project_id or project.get("id")
            if field_id is None:
                for field in ((project.get("fields") or {}).get("nodes") or []):
                    if (field or {}).get("name") != field_name:
                        continue
                    field_id = field.get("id")
                    for option in field.get("options") or []:
                        if str(option.get("name") or "").lower() == option_name.lower():
                            option_id = option.get("id")
                            break
                    break

            page = project["items"]
            for node in page.get("nodes") or []:
                content = (node or {}).get("content") or {}
                repository = content.get("repository") or {}
                repo_owner = repository.get("owner") or {}
                if (
                    content.get("__typename") == "Issue"
                    and content.get("number") == issue_number
                    and repository.get("name") == repo
                    and repo_owner.get("login") == owner
                ):
                    item_id = node.get("id")
                    break
            if item_id or not (page.get("pageInfo") or {}).get("hasNextPage"):
                break
            after = (page.get("pageInfo") or {}).get("endCursor")

        if not item_id:
            return {"updated": False, "reason": "project_item_not_found"}
        if not project_id or not field_id or not option_id:
            return {"updated": False, "reason": "single_select_field_or_option_not_found", "field": field_name}

        self._graphql(
            _PROJECT_UPDATE_SINGLE_SELECT_MUTATION,
            {
                "projectId": project_id,
                "itemId": item_id,
                "fieldId": field_id,
                "optionId": option_id,
            },
        )
        return {"updated": True, "field": field_name, "value": option_name, "item_id": item_id}

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.api_url}{path}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
                "User-Agent": "hermes-github-issue-listener",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API {method} {path} failed: {exc.code} {detail}") from exc

    def _graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        data = self._request("POST", "/graphql", {"query": query, "variables": variables})
        if data.get("errors"):
            raise RuntimeError(f"GitHub GraphQL failed: {data['errors']}")
        return dict(data.get("data") or {})


_PROJECT_ITEMS_QUERY = """
query HermesProjectAssignedIssues($owner: String!, $number: Int!, $after: String) {
  organization(login: $owner) {
    projectV2(number: $number) {
      items(first: 100, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes {
          content {
            __typename
            ... on Issue {
              number
              title
              body
              url
              state
              repository { name owner { login } }
              assignees(first: 20) { nodes { login } }
            }
          }
        }
      }
    }
  }
}
"""


_PROJECT_ITEM_EXECUTION_MODE_QUERY = """
query HermesProjectItemExecutionMode($owner: String!, $number: Int!, $after: String) {
  organization(login: $owner) {
    projectV2(number: $number) {
      id
      fields(first: 50) {
        nodes {
          ... on ProjectV2SingleSelectField {
            id
            name
            options { id name }
          }
        }
      }
      items(first: 100, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id
          content {
            __typename
            ... on Issue {
              number
              repository { name owner { login } }
            }
          }
        }
      }
    }
  }
}
"""


_PROJECT_UPDATE_SINGLE_SELECT_MUTATION = """
mutation HermesUpdateProjectSingleSelect($projectId: ID!, $itemId: ID!, $fieldId: ID!, $optionId: String!) {
  updateProjectV2ItemFieldValue(
    input: {
      projectId: $projectId
      itemId: $itemId
      fieldId: $fieldId
      value: { singleSelectOptionId: $optionId }
    }
  ) {
    projectV2Item { id }
  }
}
"""


def _project_issue_to_rest_shape(issue: dict[str, Any]) -> dict[str, Any]:
    repo = issue.get("repository") or {}
    repo_owner = repo.get("owner") or {}
    return {
        "number": issue.get("number"),
        "title": issue.get("title") or "",
        "body": issue.get("body") or "",
        "html_url": issue.get("url") or "",
        "repository": {"name": repo.get("name"), "owner": {"login": repo_owner.get("login")}},
        "assignees": [{"login": a.get("login")} for a in ((issue.get("assignees") or {}).get("nodes") or [])],
    }


def _issue_ref_from_issue(default_owner: str, default_repo: str, issue: dict[str, Any]) -> IssueRef:
    repo = issue.get("repository") or {}
    owner = ((repo.get("owner") or {}).get("login")) or default_owner
    name = repo.get("name") or default_repo
    return IssueRef(str(owner), str(name), int(issue["number"]))


def _issue_has_assignee(issue: dict[str, Any], assignee: str) -> bool:
    return any(((a or {}).get("login") == assignee) for a in (issue.get("assignees") or []))


class SubprocessHermesRunner:
    def __init__(
        self,
        hermes_bin: str = "hermes",
        *,
        source: str = "github_issue_listener",
        toolsets: str = "terminal,file,skills,web,vision",
    ):
        self.hermes_bin = hermes_bin
        self.source = source
        self.toolsets = toolsets

    def run_issue_turn(self, prompt: str, *, session_id: str | None) -> tuple[str, str]:
        before = time.time()
        cmd = [
            self.hermes_bin,
            "chat",
            "--quiet",
            "--source",
            self.source,
            "--toolsets",
            self.toolsets,
            "--query",
            prompt,
        ]
        if session_id:
            cmd.extend(["--resume", session_id])
        env = dict(os.environ)
        # Avoid recursive scheduling/notification behavior inside a listener run.
        env.setdefault("HERMES_GITHUB_ISSUE_LISTENER", "1")
        # Prefer the listener's GitHub token for gh/git operations inside the
        # child agent.  This keeps issue comments, PR creation, and routine
        # repo work on the same actor instead of falling back to the operator's
        # persisted gh login or MCP-server credentials.
        token = _load_github_token()
        if token:
            env.setdefault("GH_TOKEN", token)
            env.setdefault("GITHUB_TOKEN", token)
        proc = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=60 * 30)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"Hermes exited {proc.returncode}")
        new_session_id = session_id or _latest_session_id_after(self.source, before) or _latest_session_id_after(None, before) or ""
        response = _latest_assistant_message(new_session_id) if new_session_id else None
        return new_session_id, (response or proc.stdout.strip())


def _latest_assistant_message(session_id: str) -> str | None:
    db_path = get_hermes_home() / "state.db"
    if not db_path.exists():
        return None
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT content FROM messages
            WHERE session_id = ? AND role = 'assistant' AND content IS NOT NULL AND trim(content) != ''
            ORDER BY id DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
    return str(row[0]).strip() if row else None


def _latest_session_id_after(source: str | None, started_after: float) -> str | None:
    db_path = get_hermes_home() / "state.db"
    if not db_path.exists():
        return None
    with sqlite3.connect(db_path) as conn:
        if source:
            row = conn.execute(
                """
                SELECT id FROM sessions
                WHERE source = ? AND started_at >= ?
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (source, started_after - 5),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT id FROM sessions
                WHERE started_at >= ?
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (started_after - 5,),
            ).fetchone()
    return str(row[0]) if row else None


def _load_github_token() -> str | None:
    for name in ("GITHUB_PERSONAL_ACCESS_TOKEN", "GITHUB_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    try:
        proc = subprocess.run(["gh", "auth", "token"], text=True, capture_output=True, timeout=10)
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:
        pass
    return None


def build_issue_prompt(issue: dict[str, Any], comments: list[dict[str, Any]], *, continuation: bool) -> str:
    number = issue.get("number")
    title = issue.get("title") or ""
    body = issue.get("body") or ""
    url = issue.get("html_url") or ""
    recent_comments = comments[-5:]
    comment_text = "\n\n".join(
        f"Comment by {c.get('user', {}).get('login', 'unknown')} (id {c.get('id')}):\n{c.get('body') or ''}"
        for c in recent_comments
    )
    mode = "Continue the existing issue-bound Hermes session" if continuation else "Start work on this assigned GitHub issue"
    return f"""{mode}.

Issue: #{number} {title}
URL: {url}

Issue body:
{body}

Recent issue comments:
{comment_text or '(none)'}

Instructions:
- Treat this GitHub issue as the user-facing conversation lane.
- Work on the issue if it is safe and sufficiently specified.
- Return the concise issue-comment body as your final answer. Do not call GitHub issue comment tools yourself; the listener posts your final answer back to GitHub.
- Use GitHub through the terminal/`gh`/git path for PRs and repository writes. Do not use MCP GitHub tools for PR creation, because the listener must keep PRs, issue comments, and repo work on the same GitHub actor.
- If you need Ryan input, include the exact marker {WAITING_MARKER} in your final response and ask the next question.
- If you believe the issue is ready for Ryan acceptance/closure, include {READY_TO_CLOSE_MARKER} in your final response and ask for final confirmation before closure.
- Do not claim that a live listener/gateway restart has been enabled unless you actually verified it.
""".strip()


def strip_markers(text: str) -> tuple[str, str]:
    status = "idle"
    cleaned = text
    if WAITING_MARKER in cleaned:
        status = "waiting_for_ryan"
        cleaned = cleaned.replace(WAITING_MARKER, "").strip()
    elif READY_TO_CLOSE_MARKER in cleaned:
        status = "awaiting_close_approval"
        cleaned = cleaned.replace(READY_TO_CLOSE_MARKER, "").strip()
    return status, cleaned


def _pid_is_running(pid: int | None) -> bool:
    return bool(pid and pid > 0 and psutil.pid_exists(pid))


class BackgroundWorkerDispatcher:
    def __init__(
        self,
        *,
        state_db: str,
        hermes_bin: str = "hermes",
        owner: str = "ryanleeai",
        repo: str = "tasks",
        assignee: str = "wingboot",
        human_assignee: str | None = "seungjaeryanlee",
        project_owner: str | None = None,
        project_number: int | None = None,
        stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS,
    ):
        self.state_db = state_db
        self.hermes_bin = hermes_bin
        self.owner = owner
        self.repo = repo
        self.assignee = assignee
        self.human_assignee = human_assignee
        self.project_owner = project_owner
        self.project_number = project_number
        self.stale_after_seconds = stale_after_seconds

    def dispatch(self, ref: IssueRef, *, run_id: str) -> tuple[int, str]:
        run_dir = get_hermes_home() / "github-issue-listener" / "runs"
        run_dir.mkdir(parents=True, exist_ok=True)
        safe_run_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in run_id)
        log_path = run_dir / f"{safe_run_id}.log"
        cmd = [
            sys.executable,
            "-m",
            "hermes_cli.github_issue_listener",
            "run-issue",
            "--owner",
            ref.owner,
            "--repo",
            ref.repo,
            "--issue-number",
            str(ref.number),
            "--assignee",
            self.assignee,
            "--state-db",
            self.state_db,
            "--hermes-bin",
            self.hermes_bin,
            "--run-id",
            run_id,
            "--stale-after-seconds",
            str(self.stale_after_seconds),
        ]
        if self.human_assignee:
            cmd.extend(["--human-assignee", self.human_assignee])
        if self.project_owner:
            cmd.extend(["--project-owner", self.project_owner])
        if self.project_number is not None:
            cmd.extend(["--project-number", str(self.project_number)])
        env = dict(os.environ)
        token = _load_github_token()
        if token:
            env.setdefault("GH_TOKEN", token)
            env.setdefault("GITHUB_TOKEN", token)
        log_file = log_path.open("ab")
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            cwd=str(Path(__file__).resolve().parents[1]),
            start_new_session=True,
        )
        log_file.close()
        return int(proc.pid), str(log_path)


class GitHubIssueListener:
    def __init__(
        self,
        *,
        github: GitHubAPI,
        runner: HermesRunner,
        store: ListenerStore,
        owner: str,
        repo: str,
        assignee: str,
        human_assignee: str | None = None,
        project_owner: str | None = None,
        project_number: int | None = None,
        wip_comment_body: str | None = DEFAULT_WIP_COMMENT_BODY,
        stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS,
        dry_run: bool = False,
        dispatch_background: bool = False,
        worker_dispatcher: Any | None = None,
    ):
        self.github = github
        self.runner = runner
        self.store = store
        self.owner = owner
        self.repo = repo
        self.assignee = assignee
        self.human_assignee = human_assignee
        self.project_owner = project_owner
        self.project_number = project_number
        self.wip_comment_body = wip_comment_body
        self.stale_after_seconds = stale_after_seconds
        self.dry_run = dry_run
        self.dispatch_background = dispatch_background
        self.worker_dispatcher = worker_dispatcher

    def notify_work_started(self, ref: IssueRef, *, newest_comment_id: int | None) -> dict[str, Any]:
        """Publish early user-visible WIP signals immediately after claiming."""
        result: dict[str, Any] = {}
        if self.project_owner and self.project_number is not None:
            result["project_status"] = self.github.set_project_status(
                self.project_owner, self.project_number, ref.owner, ref.repo, ref.number, "In Progress"
            )
        if self.wip_comment_body:
            comment = self.github.add_comment(ref.owner, ref.repo, ref.number, self.wip_comment_body)
            result["wip_comment_id"] = comment.get("id")
            try:
                comment_id = int(comment.get("id") or 0) or None
            except (TypeError, ValueError):
                comment_id = None
            if comment_id:
                newest_comment_id = max(newest_comment_id or 0, comment_id)

        claimed = self.store.get(ref) or IssueState(ref.owner, ref.repo, ref.number)
        claimed.last_comment_id_seen = newest_comment_id
        self.store.upsert(claimed)
        result["newest_comment_id"] = newest_comment_id
        return result

    def run_claimed_issue(self, ref: IssueRef, *, run_id: str) -> dict[str, Any]:
        state = self.store.get(ref)
        if not state or state.current_run_id != run_id:
            return {"issue": ref.key, "action": "skipped_run_mismatch"}
        issue = self.github.get_issue(ref.owner, ref.repo, ref.number)
        comments = self.github.list_comments(ref.owner, ref.repo, ref.number)
        newest_comment_id = max((int(c.get("id") or 0) for c in comments), default=state.last_comment_id_seen or 0) or None
        continuation = bool(state.session_id)
        try:
            prompt = build_issue_prompt(issue, comments, continuation=continuation)
            session_id, response = self.runner.run_issue_turn(prompt, session_id=state.session_id)
            status, comment_body = strip_markers(response)
            if comment_body:
                added_comment = self.github.add_comment(ref.owner, ref.repo, ref.number, comment_body)
                try:
                    added_comment_id = int(added_comment.get("id") or 0) or None
                except (TypeError, ValueError):
                    added_comment_id = None
                if added_comment_id:
                    newest_comment_id = max(newest_comment_id or 0, added_comment_id)
            updated_issue = self.github.get_issue(ref.owner, ref.repo, ref.number)
            issue_closed = (updated_issue.get("state") or "").lower() == "closed"
            next_state = self.store.get(ref) or IssueState(ref.owner, ref.repo, ref.number)
            next_state.session_id = session_id or next_state.session_id
            next_state.status = "closed" if issue_closed else status
            next_state.current_run_id = None
            next_state.claimed_at = None
            next_state.worker_pid = None
            next_state.worker_started_at = None
            next_state.log_path = None
            next_state.last_comment_id_seen = newest_comment_id
            self.store.upsert(next_state)
            if issue_closed:
                assignees = [(a or {}).get("login") for a in (updated_issue.get("assignees") or [])]
                assignees = [a for a in assignees if a]
                if assignees:
                    self.github.clear_assignees(ref.owner, ref.repo, ref.number, assignees)
                if self.project_owner and self.project_number is not None:
                    self.github.set_project_execution_mode(
                        self.project_owner, self.project_number, ref.owner, ref.repo, ref.number, "automated"
                    )
            elif status in {"waiting_for_ryan", "awaiting_close_approval"} and self.human_assignee:
                self.github.set_assignees(ref.owner, ref.repo, ref.number, [self.human_assignee])
            return {"issue": ref.key, "action": "ran", "status": next_state.status, "session_id": next_state.session_id}
        except Exception as exc:
            failed = self.store.get(ref) or IssueState(ref.owner, ref.repo, ref.number)
            failed.status = "errored"
            failed.current_run_id = None
            failed.claimed_at = None
            failed.worker_pid = None
            failed.worker_started_at = None
            self.store.upsert(failed)
            return {"issue": ref.key, "action": "error", "error": str(exc)}

    def poll_once(self) -> dict[str, Any]:
        if self.project_owner and self.project_number is not None:
            issues = self.github.list_project_assigned_issues(self.project_owner, self.project_number, self.assignee)
        else:
            issues = self.github.list_assigned_issues(self.owner, self.repo, self.assignee)
        results: list[dict[str, Any]] = []
        for issue in issues:
            ref = _issue_ref_from_issue(self.owner, self.repo, issue)
            state = self.store.get(ref)
            comments = self.github.list_comments(ref.owner, ref.repo, ref.number)
            last_seen = state.last_comment_id_seen if state else None
            newest_comment_id = max((int(c.get("id") or 0) for c in comments), default=last_seen or 0) or None
            new_comments = [c for c in comments if last_seen is None or int(c.get("id") or 0) > last_seen]
            has_new_human_input = any((c.get("user") or {}).get("login") != self.assignee for c in new_comments)

            if state and state.status == "running":
                claimed_at = state.claimed_at or 0
                claimed_is_fresh = claimed_at > time.time() - self.stale_after_seconds
                if state.worker_pid:
                    if _pid_is_running(state.worker_pid):
                        results.append({"issue": ref.key, "action": "skipped_worker_alive", "pid": state.worker_pid})
                        continue
                    # A detached worker can die before it writes a log or clears
                    # the claim. Do not wait for the stale window in that case:
                    # clear only the dead-worker fields, then fall through and
                    # let try_claim create a fresh run.
                    state.status = "idle"
                    state.current_run_id = None
                    state.claimed_at = None
                    state.worker_pid = None
                    state.worker_started_at = None
                    state.log_path = None
                    self.store.upsert(state)
                elif claimed_is_fresh:
                    results.append({"issue": ref.key, "action": "skipped_running"})
                    continue

            if state and state.session_id and state.status in {"idle", "awaiting_close_approval"} and not has_new_human_input:
                results.append({"issue": ref.key, "action": "skipped_no_new_input"})
                continue

            # If the issue was waiting for Ryan and is visible here again, the
            # assignment back to the Hermes assignee is itself a continuation
            # signal.  No explicit @hermes command is required.
            run_id = f"{int(time.time())}-{ref.owner}-{ref.repo}-{ref.number}"
            if not self.store.try_claim(ref, run_id=run_id, stale_after_seconds=self.stale_after_seconds):
                results.append({"issue": ref.key, "action": "skipped_claimed"})
                continue

            if self.dry_run:
                claimed = self.store.get(ref) or IssueState(ref.owner, ref.repo, ref.number)
                claimed.status = "idle"
                claimed.current_run_id = None
                claimed.claimed_at = None
                claimed.last_comment_id_seen = newest_comment_id
                self.store.upsert(claimed)
                results.append({"issue": ref.key, "action": "dry_run_claimed"})
                continue

            start_notification = self.notify_work_started(ref, newest_comment_id=newest_comment_id)

            if self.dispatch_background:
                dispatcher = self.worker_dispatcher
                if dispatcher is None:
                    raise RuntimeError("dispatch_background=True requires a worker dispatcher")
                dispatch = dispatcher.dispatch if hasattr(dispatcher, "dispatch") else dispatcher
                try:
                    pid, log_path = dispatch(ref, run_id=run_id)
                except Exception as exc:
                    failed = self.store.get(ref) or IssueState(ref.owner, ref.repo, ref.number)
                    failed.status = "errored"
                    failed.current_run_id = None
                    failed.claimed_at = None
                    failed.worker_pid = None
                    failed.worker_started_at = None
                    self.store.upsert(failed)
                    results.append({"issue": ref.key, "action": "dispatch_error", "error": str(exc)})
                    continue
                dispatched = self.store.get(ref) or IssueState(ref.owner, ref.repo, ref.number)
                dispatched.worker_pid = pid
                dispatched.worker_started_at = time.time()
                dispatched.log_path = log_path
                self.store.upsert(dispatched)
                results.append({
                    "issue": ref.key,
                    "action": "dispatched",
                    "run_id": run_id,
                    "pid": pid,
                    "log_path": log_path,
                    "start_notification": start_notification,
                })
                continue

            run_result = self.run_claimed_issue(ref, run_id=run_id)
            run_result["start_notification"] = start_notification
            results.append(run_result)
        return {"checked": len(issues), "results": results}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Poll GitHub issues assigned to Hermes and run issue-bound Hermes sessions.")
    sub = parser.add_subparsers(dest="command")
    poll = sub.add_parser("poll", help="Run one polling pass")
    poll.add_argument("--owner", default="ryanleeai")
    poll.add_argument("--repo", default="tasks")
    poll.add_argument("--project-owner", default=None, help="GitHub org/user whose Project V2 board should be polled")
    poll.add_argument("--project-number", type=int, default=None, help="Project V2 number to poll instead of a single repo")
    poll.add_argument("--assignee", default="wingboot")
    poll.add_argument("--human-assignee", default="seungjaeryanlee")
    poll.add_argument("--state-db", default=str(DEFAULT_STATE_DB))
    poll.add_argument("--hermes-bin", default="hermes")
    poll.add_argument("--stale-after-seconds", type=int, default=DEFAULT_STALE_AFTER_SECONDS)
    poll.add_argument("--dry-run", action="store_true")
    poll.add_argument("--dispatch-background", action="store_true", help="Claim eligible issues, spawn detached workers, and exit quickly")
    poll.set_defaults(func=cmd_poll)

    run_issue = sub.add_parser("run-issue", help="Run one already-claimed issue worker")
    run_issue.add_argument("--owner", required=True)
    run_issue.add_argument("--repo", required=True)
    run_issue.add_argument("--issue-number", type=int, required=True)
    run_issue.add_argument("--run-id", required=True)
    run_issue.add_argument("--assignee", default="wingboot")
    run_issue.add_argument("--human-assignee", default="seungjaeryanlee")
    run_issue.add_argument("--project-owner", default=None)
    run_issue.add_argument("--project-number", type=int, default=None)
    run_issue.add_argument("--state-db", default=str(DEFAULT_STATE_DB))
    run_issue.add_argument("--hermes-bin", default="hermes")
    run_issue.add_argument("--stale-after-seconds", type=int, default=DEFAULT_STALE_AFTER_SECONDS)
    run_issue.set_defaults(func=cmd_run_issue)
    return parser


def cmd_poll(args: argparse.Namespace) -> int:
    dispatcher = None
    if args.dispatch_background and not args.dry_run:
        dispatcher = BackgroundWorkerDispatcher(
            state_db=args.state_db,
            hermes_bin=args.hermes_bin,
            owner=args.owner,
            repo=args.repo,
            assignee=args.assignee,
            human_assignee=args.human_assignee,
            project_owner=args.project_owner,
            project_number=args.project_number,
            stale_after_seconds=args.stale_after_seconds,
        )
    listener = GitHubIssueListener(
        github=GitHubRESTClient(),
        runner=SubprocessHermesRunner(args.hermes_bin),
        store=ListenerStore(args.state_db),
        owner=args.owner,
        repo=args.repo,
        assignee=args.assignee,
        human_assignee=args.human_assignee,
        project_owner=args.project_owner,
        project_number=args.project_number,
        stale_after_seconds=args.stale_after_seconds,
        dry_run=args.dry_run,
        dispatch_background=args.dispatch_background and not args.dry_run,
        worker_dispatcher=dispatcher,
    )
    print(json.dumps(listener.poll_once(), indent=2, sort_keys=True))
    return 0


def cmd_run_issue(args: argparse.Namespace) -> int:
    listener = GitHubIssueListener(
        github=GitHubRESTClient(),
        runner=SubprocessHermesRunner(args.hermes_bin),
        store=ListenerStore(args.state_db),
        owner=args.owner,
        repo=args.repo,
        assignee=args.assignee,
        human_assignee=args.human_assignee,
        project_owner=args.project_owner,
        project_number=args.project_number,
        stale_after_seconds=args.stale_after_seconds,
    )
    result = listener.run_claimed_issue(IssueRef(args.owner, args.repo, args.issue_number), run_id=args.run_id)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("action") != "error" else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
