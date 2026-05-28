"""GitHub issue resolution lane for Hermes Gateway."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import sqlite3
import time
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


logger = logging.getLogger(__name__)

DEFAULT_GUARDIAN_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_LOCAL_MODEL = "openai/qwen3-35b-uncensored"
DEFAULT_DECOMPOSE_MODEL = "qwen3-35b-uncensored"
DEFAULT_CLOUD_MODEL = "openrouter/deepseek/deepseek-v4-flash"
DEFAULT_KYBER_ENV = Path.home() / "kyberm0nk" / ".env"
DEFAULT_AIDER_BIN = Path.home() / "aider" / ".venv" / "bin" / "aider"
DEFAULT_STATE_DB = get_hermes_home() / "issue_resolution.db"
MASTER_PLAN_LABEL = "master-plan"
MASTER_PLAN_HEADING = "# Master Project Plan"


class AiderRole(str, Enum):
    """Supported Aider execution roles for issue resolution."""

    LOCAL_CODER = "local_coder"
    CLOUD_REVIEWER = "cloud_reviewer"


class IssueRunStatus(str, Enum):
    """Persisted issue run states."""

    QUEUED = "queued"
    RUNNING = "running"
    EXPANDED = "expanded"
    COMPLETED = "completed"
    FAILED = "failed"


class IssueRunType(str, Enum):
    """Issue run types handled by the lane."""

    ISSUE = "issue"
    MASTER = "master"
    SUB_ISSUE = "sub_issue"


@dataclass(frozen=True)
class IssueResolutionRequest:
    """Normalized request to resolve one GitHub issue."""

    repo: str
    issue_number: int
    workdir: Path
    branch: str | None = None


@dataclass(frozen=True)
class IssueMetadata:
    """GitHub issue metadata used to prompt the local coder."""

    number: int
    title: str
    body: str
    url: str
    labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class PullRequestMetadata:
    """GitHub pull request metadata emitted by the local coder lane."""

    number: int
    url: str
    head_ref_name: str
    head_ref_oid: str


@dataclass(frozen=True)
class EpicTask:
    """Atomic task generated from a master-plan issue."""

    title: str
    body: str


@dataclass(frozen=True)
class AiderInvocation:
    """Concrete Aider subprocess invocation."""

    command: list[str]
    env: dict[str, str]
    cwd: Path


@dataclass(frozen=True)
class IssueRun:
    """Persisted queued issue run."""

    id: int
    repo: str
    issue_number: int
    workdir: Path
    branch: str | None
    status: IssueRunStatus
    run_type: IssueRunType
    parent_run_id: int | None
    master_issue_number: int | None
    pr_number: int | None
    pr_url: str | None
    error: str | None


@dataclass(frozen=True)
class SubmitResult:
    """Result returned when a run is submitted to the queue."""

    run_id: int
    status: IssueRunStatus
    reused: bool


_RUN_NOTIFIERS: dict[int, Callable[[str], Awaitable[None]]] = {}
_DEFAULT_NOTIFY: Callable[[str], Awaitable[None]] | None = None
_QUEUE_WORKER_TASK: asyncio.Task | None = None
_QUEUE_GUARD: asyncio.Lock | None = None


async def run_issue_resolution(
    request: IssueResolutionRequest,
    *,
    notify: Callable[[str], Awaitable[None]],
) -> None:
    """Run one issue immediately without enqueueing it."""
    store = IssueStateStore()
    issue = await _load_issue(request.repo, request.issue_number)
    if is_master_issue(issue):
        run = store.enqueue_run(request, run_type=IssueRunType.MASTER)
        await _execute_master_issue(store, run, issue, notify)
        return
    run = store.enqueue_run(request, run_type=IssueRunType.ISSUE)
    await _execute_single_issue(store, run, issue, notify)


async def submit_issue_resolution(
    request: IssueResolutionRequest,
    *,
    notify: Callable[[str], Awaitable[None]],
) -> SubmitResult:
    """Persist a run and ensure the single-flight worker is active."""
    issue = await _load_issue(request.repo, request.issue_number)
    run_type = IssueRunType.MASTER if is_master_issue(issue) else IssueRunType.ISSUE
    store = IssueStateStore()
    existing = store.find_incomplete_run(request.repo, request.issue_number)
    run = existing or store.enqueue_run(request, run_type=run_type)
    _RUN_NOTIFIERS[run.id] = notify
    await notify(
        f"Hermes: Queued {'master epic' if run_type is IssueRunType.MASTER else 'issue'} "
        f"#{request.issue_number} as run #{run.id}."
    )
    await ensure_issue_queue_worker(notify=notify)
    return SubmitResult(run_id=run.id, status=run.status, reused=existing is not None)


async def resume_issue_resolution_queue(
    *,
    notify: Callable[[str], Awaitable[None]] | None = None,
) -> int:
    """Reset interrupted runs and resume the persistent queue after gateway startup."""
    store = IssueStateStore()
    resumed = store.reset_interrupted_runs()
    pending = store.count_pending_runs()
    if pending:
        if notify:
            await notify(f"Hermes: Resuming {pending} queued issue-resolution run(s).")
        await ensure_issue_queue_worker(notify=notify)
    return resumed + pending


async def ensure_issue_queue_worker(
    *,
    notify: Callable[[str], Awaitable[None]] | None = None,
) -> None:
    """Start the FIFO worker if it is not already running."""
    global _DEFAULT_NOTIFY, _QUEUE_GUARD, _QUEUE_WORKER_TASK
    if notify is not None:
        _DEFAULT_NOTIFY = notify
    if _QUEUE_GUARD is None:
        _QUEUE_GUARD = asyncio.Lock()
    async with _QUEUE_GUARD:
        if _QUEUE_WORKER_TASK and not _QUEUE_WORKER_TASK.done():
            return
        _QUEUE_WORKER_TASK = asyncio.create_task(_issue_queue_worker())


async def _issue_queue_worker() -> None:
    """Process queued issue runs in strict FIFO order."""
    store = IssueStateStore()
    while True:
        run = store.claim_next_run()
        if run is None:
            return
        notify = _RUN_NOTIFIERS.get(run.id) or _DEFAULT_NOTIFY or _noop_notify
        try:
            await _execute_run(store, run, notify)
        except Exception as exc:
            store.mark_failed(run.id, str(exc))
            await notify(f"Hermes: Issue run #{run.id} failed: {exc}")
            logger.exception("issue-resolution run %s failed", run.id)
        finally:
            _RUN_NOTIFIERS.pop(run.id, None)
            await _complete_ready_masters(store, notify)


async def _execute_run(
    store: "IssueStateStore",
    run: IssueRun,
    notify: Callable[[str], Awaitable[None]],
) -> None:
    issue = await _load_issue(run.repo, run.issue_number)
    if run.run_type is IssueRunType.MASTER or is_master_issue(issue):
        await _execute_master_issue(store, run, issue, notify)
        return
    await _execute_single_issue(store, run, issue, notify)


async def _execute_master_issue(
    store: "IssueStateStore",
    run: IssueRun,
    issue: IssueMetadata,
    notify: Callable[[str], Awaitable[None]],
) -> None:
    """Expand a master plan issue into queued sub-issues."""
    await notify(f"Hermes: Master Epic #{issue.number} detected; decomposing plan locally.")
    tasks = await decompose_master_plan(issue)
    if not tasks:
        raise RuntimeError("Guardian returned no atomic tasks for the master plan.")

    created = 0
    for position, task in enumerate(tasks, start=1):
        if store.subissue_exists(run.id, position):
            continue
        sub_issue = await _create_sub_issue(run.repo, issue, task, position)
        store.record_subissue(run.id, position, task, sub_issue)
        store.enqueue_run(
            IssueResolutionRequest(
                repo=run.repo,
                issue_number=sub_issue.number,
                workdir=run.workdir,
            ),
            run_type=IssueRunType.SUB_ISSUE,
            parent_run_id=run.id,
            master_issue_number=issue.number,
        )
        created += 1

    store.mark_expanded(run.id)
    await notify(
        f"Hermes: Master Epic #{issue.number} expanded into {created} new sub-issue(s)."
    )


async def _execute_single_issue(
    store: "IssueStateStore",
    run: IssueRun,
    issue: IssueMetadata,
    notify: Callable[[str], Awaitable[None]],
) -> None:
    """Run local coder, push branch, open PR, and trigger cloud reviewer."""
    default_branch = await _load_default_branch(run.repo)
    branch = run.branch or _issue_branch_name(issue)

    await notify(f"Hermes: Starting local coder for Issue #{issue.number}.")
    await _run(["git", "checkout", "-B", branch], cwd=run.workdir)
    local_prompt = _local_coder_prompt(run.repo, issue, branch)
    local_invocation = build_aider_invocation(AiderRole.LOCAL_CODER, run.workdir, local_prompt)
    await _run(local_invocation.command, cwd=local_invocation.cwd, env=local_invocation.env)

    await notify(f"Hermes: Local coder finished Issue #{issue.number}; pushing `{branch}`.")
    await _run(["git", "push", "-u", "origin", branch], cwd=run.workdir)

    pr = await _create_or_find_pr(run.repo, issue, branch, default_branch)
    store.record_pr(run.id, pr)
    await notify(f"Hermes: PR #{pr.number} created, triggering reviewer: {pr.url}")

    reviewer_prompt = _cloud_reviewer_prompt(run.repo, pr)
    reviewer_invocation = build_aider_invocation(AiderRole.CLOUD_REVIEWER, run.workdir, reviewer_prompt)
    reviewer_output = await _run(
        reviewer_invocation.command,
        cwd=reviewer_invocation.cwd,
        env=reviewer_invocation.env,
    )

    await _post_pr_feedback(run.repo, pr, _review_body(reviewer_output.stdout))
    store.mark_completed(run.id)
    await notify(f"Hermes: Cloud reviewer posted feedback on PR #{pr.number}.")


async def _complete_ready_masters(
    store: "IssueStateStore",
    notify: Callable[[str], Awaitable[None]],
) -> None:
    for run in store.expanded_masters_ready_to_complete():
        store.mark_completed(run.id)
        await notify(f"Hermes: Master Epic #{run.issue_number} completed; all sub-issues are finished.")


class IssueStateStore:
    """SQLite persistence for queued issue-resolution runs."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DEFAULT_STATE_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def enqueue_run(
        self,
        request: IssueResolutionRequest,
        *,
        run_type: IssueRunType,
        parent_run_id: int | None = None,
        master_issue_number: int | None = None,
    ) -> IssueRun:
        existing = self.find_incomplete_run(request.repo, request.issue_number)
        if existing:
            return existing
        now = _now()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO issue_runs (
                    repo, issue_number, workdir, branch, status, run_type,
                    parent_run_id, master_issue_number, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request.repo,
                    request.issue_number,
                    str(request.workdir),
                    request.branch,
                    IssueRunStatus.QUEUED.value,
                    run_type.value,
                    parent_run_id,
                    master_issue_number,
                    now,
                    now,
                ),
            )
            conn.commit()
            run_id = int(cur.lastrowid)
        return self.get_run(run_id)

    def find_incomplete_run(self, repo: str, issue_number: int) -> IssueRun | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM issue_runs
                WHERE repo = ? AND issue_number = ?
                  AND status IN (?, ?, ?)
                ORDER BY id ASC LIMIT 1
                """,
                (
                    repo,
                    issue_number,
                    IssueRunStatus.QUEUED.value,
                    IssueRunStatus.RUNNING.value,
                    IssueRunStatus.EXPANDED.value,
                ),
            ).fetchone()
        return self._row_to_run(row) if row else None

    def get_run(self, run_id: int) -> IssueRun:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM issue_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"Issue run {run_id} does not exist.")
        return self._row_to_run(row)

    def claim_next_run(self) -> IssueRun | None:
        now = _now()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM issue_runs
                WHERE status = ?
                ORDER BY id ASC LIMIT 1
                """,
                (IssueRunStatus.QUEUED.value,),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE issue_runs SET status = ?, updated_at = ? WHERE id = ?",
                (IssueRunStatus.RUNNING.value, now, int(row["id"])),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM issue_runs WHERE id = ?", (int(row["id"]),)).fetchone()
        return self._row_to_run(row)

    def reset_interrupted_runs(self) -> int:
        now = _now()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE issue_runs SET status = ?, updated_at = ? WHERE status = ?",
                (IssueRunStatus.QUEUED.value, now, IssueRunStatus.RUNNING.value),
            )
            conn.commit()
            return int(cur.rowcount or 0)

    def count_pending_runs(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM issue_runs WHERE status = ?",
                (IssueRunStatus.QUEUED.value,),
            ).fetchone()
        return int(row["count"] if row else 0)

    def mark_expanded(self, run_id: int) -> None:
        self._set_status(run_id, IssueRunStatus.EXPANDED)

    def mark_completed(self, run_id: int) -> None:
        self._set_status(run_id, IssueRunStatus.COMPLETED)

    def mark_failed(self, run_id: int, error: str) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE issue_runs SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                (IssueRunStatus.FAILED.value, error[:4000], now, run_id),
            )
            conn.commit()

    def record_pr(self, run_id: int, pr: PullRequestMetadata) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE issue_runs
                SET pr_number = ?, pr_url = ?, updated_at = ?
                WHERE id = ?
                """,
                (pr.number, pr.url, now, run_id),
            )
            conn.commit()

    def record_subissue(
        self,
        master_run_id: int,
        position: int,
        task: EpicTask,
        sub_issue: IssueMetadata,
    ) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO master_subissues (
                    master_run_id, position, sub_issue_number, title, body, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (master_run_id, position, sub_issue.number, task.title, task.body, now),
            )
            conn.commit()

    def subissue_exists(self, master_run_id: int, position: int) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM master_subissues
                WHERE master_run_id = ? AND position = ?
                """,
                (master_run_id, position),
            ).fetchone()
        return row is not None

    def list_child_runs(self, master_run_id: int) -> list[IssueRun]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM issue_runs WHERE parent_run_id = ? ORDER BY id ASC",
                (master_run_id,),
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def expanded_masters_ready_to_complete(self) -> list[IssueRun]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM issue_runs WHERE status = ? AND run_type = ? ORDER BY id ASC",
                (IssueRunStatus.EXPANDED.value, IssueRunType.MASTER.value),
            ).fetchall()
        ready: list[IssueRun] = []
        for row in rows:
            run = self._row_to_run(row)
            children = self.list_child_runs(run.id)
            if children and all(child.status is IssueRunStatus.COMPLETED for child in children):
                ready.append(run)
        return ready

    def _set_status(self, run_id: int, status: IssueRunStatus) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "UPDATE issue_runs SET status = ?, updated_at = ? WHERE id = ?",
                (status.value, now, run_id),
            )
            conn.commit()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS issue_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repo TEXT NOT NULL,
                    issue_number INTEGER NOT NULL,
                    workdir TEXT NOT NULL,
                    branch TEXT,
                    status TEXT NOT NULL,
                    run_type TEXT NOT NULL DEFAULT 'issue',
                    parent_run_id INTEGER,
                    master_issue_number INTEGER,
                    pr_number INTEGER,
                    pr_url TEXT,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_issue_runs_status_id
                    ON issue_runs(status, id);
                CREATE INDEX IF NOT EXISTS idx_issue_runs_repo_issue
                    ON issue_runs(repo, issue_number);
                CREATE INDEX IF NOT EXISTS idx_issue_runs_parent
                    ON issue_runs(parent_run_id);

                CREATE TABLE IF NOT EXISTS master_subissues (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    master_run_id INTEGER NOT NULL,
                    position INTEGER NOT NULL,
                    sub_issue_number INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(master_run_id, position)
                );
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _row_to_run(self, row: sqlite3.Row) -> IssueRun:
        return IssueRun(
            id=int(row["id"]),
            repo=str(row["repo"]),
            issue_number=int(row["issue_number"]),
            workdir=Path(str(row["workdir"])),
            branch=str(row["branch"]) if row["branch"] else None,
            status=IssueRunStatus(str(row["status"])),
            run_type=IssueRunType(str(row["run_type"])),
            parent_run_id=int(row["parent_run_id"]) if row["parent_run_id"] is not None else None,
            master_issue_number=int(row["master_issue_number"]) if row["master_issue_number"] is not None else None,
            pr_number=int(row["pr_number"]) if row["pr_number"] is not None else None,
            pr_url=str(row["pr_url"]) if row["pr_url"] else None,
            error=str(row["error"]) if row["error"] else None,
        )


def parse_issue_command_args(raw_args: str) -> IssueResolutionRequest:
    """Parse `/issue` arguments from Telegram or another gateway client."""
    tokens = shlex.split(raw_args or "")
    repo = ""
    issue_number: int | None = None
    workdir: Path | None = None
    branch: str | None = None

    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token in {"--repo", "-r"} and idx + 1 < len(tokens):
            repo = tokens[idx + 1]
            idx += 2
            continue
        if token in {"--issue", "-i"} and idx + 1 < len(tokens):
            issue_number = _parse_issue_number(tokens[idx + 1])
            idx += 2
            continue
        if token in {"--workdir", "-C"} and idx + 1 < len(tokens):
            workdir = Path(tokens[idx + 1]).expanduser()
            idx += 2
            continue
        if token in {"--branch", "-b"} and idx + 1 < len(tokens):
            branch = tokens[idx + 1]
            idx += 2
            continue
        if not repo:
            maybe_repo, maybe_issue = _parse_issue_url(token)
            if maybe_repo and maybe_issue is not None:
                repo = maybe_repo
                issue_number = maybe_issue
            else:
                repo = token
            idx += 1
            continue
        if issue_number is None:
            issue_number = _parse_issue_number(token)
            idx += 1
            continue
        raise ValueError(f"Unexpected argument: {token}")

    if not repo:
        repo = os.getenv("HERMES_ISSUE_REPO", "").strip()
    if issue_number is None:
        raise ValueError("Missing issue number.")
    if not _valid_repo(repo):
        raise ValueError("Repository must look like owner/name.")

    return IssueResolutionRequest(
        repo=repo,
        issue_number=issue_number,
        workdir=workdir or _default_workdir(repo),
        branch=branch,
    )


def github_issue_webhook_command(payload: dict[str, Any]) -> str | None:
    """Build a `/issue` command from a GitHub `issues` webhook payload."""
    issue = payload.get("issue") if isinstance(payload, dict) else None
    repository = payload.get("repository") if isinstance(payload, dict) else None
    if not isinstance(issue, dict) or not isinstance(repository, dict):
        return None
    if isinstance(issue.get("pull_request"), dict):
        return None
    repo = str(repository.get("full_name") or "").strip()
    number = issue.get("number")
    if not repo or not isinstance(number, int):
        return None
    return f"/issue --repo {shlex.quote(repo)} --issue {number}"


def is_master_issue(issue: IssueMetadata) -> bool:
    """Return True when an issue should be handled as a master epic."""
    labels = {label.strip().lower() for label in issue.labels}
    if MASTER_PLAN_LABEL in labels:
        return True
    return issue.body.lstrip().startswith(MASTER_PLAN_HEADING)


async def decompose_master_plan(
    issue: IssueMetadata,
    *,
    runtime_env: dict[str, str] | None = None,
) -> list[EpicTask]:
    """Ask Guardian to decompose a master plan into ordered atomic tasks."""
    env = _merged_runtime_env(runtime_env)
    content = await asyncio.to_thread(_guardian_decompose_request, issue, env)
    return parse_decomposition_response(content)


def parse_decomposition_response(content: str) -> list[EpicTask]:
    """Parse Guardian's JSON-only decomposition response."""
    raw = (content or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```$", "", raw).strip()
    match = re.search(r"(\{.*\}|\[.*\])", raw, flags=re.DOTALL)
    if match:
        raw = match.group(1)
    data = json.loads(raw)
    if isinstance(data, dict):
        items = data.get("tasks", [])
    else:
        items = data
    tasks: list[EpicTask] = []
    for idx, item in enumerate(items, start=1):
        if isinstance(item, str):
            title = item.strip()
            body = title
        elif isinstance(item, dict):
            title = str(item.get("title") or item.get("name") or f"Task {idx}").strip()
            body = str(item.get("body") or item.get("description") or title).strip()
        else:
            continue
        if title:
            tasks.append(EpicTask(title=title[:180], body=body))
    return tasks


def build_aider_invocation(
    role: AiderRole,
    cwd: Path,
    prompt: str,
    *,
    runtime_env: dict[str, str] | None = None,
) -> AiderInvocation:
    """Build the Aider subprocess command for a role."""
    env = _merged_runtime_env(runtime_env)
    aider_bin = Path(env.get("AIDER_BIN") or DEFAULT_AIDER_BIN).expanduser()
    if role is AiderRole.LOCAL_CODER:
        env["OPENAI_API_BASE"] = _normalize_guardian_base(
            env.get("GUARDIAN_BASE_URL") or DEFAULT_GUARDIAN_BASE_URL
        )
        env["OPENAI_API_KEY"] = _require_secret(env, "AIDER_GUARDIAN_API_KEY")
        model = (
            env.get("AIDER_LOCAL_MODEL")
            or env.get("AIDER_MODEL")
            or (f"openai/{env['DEFAULT_MODEL']}" if env.get("DEFAULT_MODEL") else DEFAULT_LOCAL_MODEL)
        )
        command = [
            str(aider_bin),
            "--model",
            model,
            "--yes",
            "--no-gitignore",
            "--message",
            prompt,
        ]
        return AiderInvocation(command=command, env=env, cwd=cwd)

    if role is AiderRole.CLOUD_REVIEWER:
        openrouter_key = _resolve_secret(env, "OPENROUTER_API_KEY", "OPENROUTER_API_KEY_FILE")
        if not openrouter_key:
            raise RuntimeError("OPENROUTER_API_KEY or OPENROUTER_API_KEY_FILE is required.")
        env["OPENROUTER_API_KEY"] = openrouter_key
        env["OPENAI_API_KEY"] = openrouter_key
        env.pop("OPENAI_API_BASE", None)
        env.pop("OPENAI_BASE_URL", None)
        model = env.get("AIDER_CLOUD_REVIEW_MODEL") or DEFAULT_CLOUD_MODEL
        command = [
            str(aider_bin),
            "--model",
            model,
            "--cache-prompts",
            "--no-auto-commits",
            "--yes",
            "--no-gitignore",
            "--message",
            prompt,
        ]
        return AiderInvocation(command=command, env=env, cwd=cwd)

    raise ValueError(f"Unsupported Aider role: {role}")


def _local_coder_prompt(repo: str, issue: IssueMetadata, branch: str) -> str:
    return (
        f"Create a new branch for Issue #{issue.number}, write the code to fix the issue, "
        "and push the branch to GitHub.\n\n"
        f"Repository: {repo}\n"
        f"Branch: {branch}\n"
        f"Issue URL: {issue.url}\n"
        f"Issue title: {issue.title}\n\n"
        f"Issue body:\n{issue.body or '(empty)'}\n"
    )


def _cloud_reviewer_prompt(repo: str, pr: PullRequestMetadata) -> str:
    return (
        "Review this new PR and create an inline comment thread directly inside "
        "the GitHub PR with feedback.\n\n"
        f"Repository: {repo}\n"
        f"PR: #{pr.number}\n"
        f"PR URL: {pr.url}\n\n"
        "Return concise feedback suitable for one GitHub inline review comment. "
        "Do not edit files or create commits."
    )


def _review_body(output: str) -> str:
    body = (output or "").strip()
    if not body:
        body = "Automated cloud reviewer completed without additional feedback."
    if len(body) > 6000:
        body = body[:6000] + "\n\n[truncated]"
    return f"Hermes Cloud Reviewer:\n\n{body}"


async def _load_issue(repo: str, issue_number: int) -> IssueMetadata:
    result = await _run([
        "gh",
        "issue",
        "view",
        str(issue_number),
        "--repo",
        repo,
        "--json",
        "number,title,body,url,labels",
    ])
    data = json.loads(result.stdout)
    labels = data.get("labels") or []
    label_names = tuple(
        str(item.get("name") or "") for item in labels if isinstance(item, dict) and item.get("name")
    )
    return IssueMetadata(
        number=int(data["number"]),
        title=str(data.get("title") or f"Issue #{issue_number}"),
        body=str(data.get("body") or ""),
        url=str(data.get("url") or ""),
        labels=label_names,
    )


async def _load_default_branch(repo: str) -> str:
    result = await _run(["gh", "repo", "view", repo, "--json", "defaultBranchRef"])
    data = json.loads(result.stdout)
    ref = data.get("defaultBranchRef") if isinstance(data, dict) else {}
    return str(ref.get("name") or "main") if isinstance(ref, dict) else "main"


async def _create_sub_issue(
    repo: str,
    master_issue: IssueMetadata,
    task: EpicTask,
    position: int,
) -> IssueMetadata:
    body = (
        f"Part of Master Issue #{master_issue.number}.\n\n"
        f"Master issue: {master_issue.url}\n\n"
        f"## Task {position}\n\n{task.body}"
    )
    result = await _run([
        "gh",
        "issue",
        "create",
        "--repo",
        repo,
        "--title",
        task.title,
        "--body",
        body,
    ])
    url = result.stdout.strip().splitlines()[-1].strip()
    match = re.search(r"/issues/(\d+)", url)
    if not match:
        raise RuntimeError(f"Could not parse issue number from gh output: {url}")
    return IssueMetadata(number=int(match.group(1)), title=task.title, body=body, url=url)


async def _create_or_find_pr(
    repo: str,
    issue: IssueMetadata,
    branch: str,
    base_branch: str,
) -> PullRequestMetadata:
    title = f"Fix #{issue.number}: {issue.title}"
    body = f"Closes #{issue.number}\n\nCreated by Hermes automated issue resolution skeleton."
    create = await _run(
        [
            "gh",
            "pr",
            "create",
            "--repo",
            repo,
            "--head",
            branch,
            "--base",
            base_branch,
            "--title",
            title,
            "--body",
            body,
        ],
        check=False,
    )
    if create.returncode not in {0, 1}:
        create.raise_for_status()

    result = await _run([
        "gh",
        "pr",
        "list",
        "--repo",
        repo,
        "--head",
        branch,
        "--json",
        "number,url,headRefName,headRefOid",
        "--limit",
        "1",
    ])
    rows = json.loads(result.stdout or "[]")
    if not rows:
        raise RuntimeError("PR creation completed but no PR was found for the branch.")
    row = rows[0]
    return PullRequestMetadata(
        number=int(row["number"]),
        url=str(row["url"]),
        head_ref_name=str(row["headRefName"]),
        head_ref_oid=str(row["headRefOid"]),
    )


async def _post_pr_feedback(repo: str, pr: PullRequestMetadata, body: str) -> None:
    diff = await _run(["gh", "pr", "diff", str(pr.number), "--repo", repo, "--patch"])
    anchor = _first_added_line(diff.stdout)
    if anchor is not None:
        path, line = anchor
        comment = await _run(
            [
                "gh",
                "api",
                f"repos/{repo}/pulls/{pr.number}/comments",
                "-f",
                f"body={body}",
                "-f",
                f"commit_id={pr.head_ref_oid}",
                "-f",
                f"path={path}",
                "-F",
                f"line={line}",
                "-f",
                "side=RIGHT",
            ],
            check=False,
        )
        if comment.returncode == 0:
            return

    await _run([
        "gh",
        "pr",
        "review",
        str(pr.number),
        "--repo",
        repo,
        "--comment",
        "--body",
        body,
    ])


def _guardian_decompose_request(issue: IssueMetadata, env: dict[str, str]) -> str:
    base_url = _normalize_guardian_base(env.get("GUARDIAN_BASE_URL") or DEFAULT_GUARDIAN_BASE_URL)
    api_key = (
        env.get("KYBERM0NK_GUARDIAN_API_KEY")
        or env.get("AIDER_GUARDIAN_API_KEY")
        or env.get("GUARDIAN_API_KEY")
        or env.get("OPENAI_API_KEY")
        or ""
    ).strip()
    if not api_key:
        raise RuntimeError("Guardian API key is required for master-plan decomposition.")
    model = env.get("HERMES_ISSUE_DECOMPOSE_MODEL") or env.get("DEFAULT_MODEL") or DEFAULT_DECOMPOSE_MODEL
    payload = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": 2500,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Decompose master project plans into atomic implementation issues. "
                    "Return only JSON shaped as {\"tasks\":[{\"title\":\"...\",\"body\":\"...\"}]}. "
                    "Each task must be independently implementable and ordered by dependency."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Issue #{issue.number}: {issue.title}\n\n"
                    f"{issue.body}\n\n"
                    "Create the smallest practical ordered list of technical tasks."
                ),
            },
        ],
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Guardian decomposition request failed: {exc}") from exc
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Guardian decomposition response had no choices.")
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if not content:
        raise RuntimeError("Guardian decomposition response was empty.")
    return str(content)


def _first_added_line(diff: str) -> tuple[str, int] | None:
    path: str | None = None
    new_line: int | None = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line.removeprefix("+++ b/")
            continue
        if line.startswith("@@"):
            match = re.search(r"\+(\d+)(?:,\d+)?", line)
            new_line = int(match.group(1)) if match else None
            continue
        if path is None or new_line is None:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            return path, new_line
        if line.startswith("-") and not line.startswith("---"):
            continue
        new_line += 1
    return None


@dataclass
class CompletedProcess:
    """Small subprocess result wrapper with a status helper."""

    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    def raise_for_status(self) -> None:
        if self.returncode != 0:
            rendered = shlex.join(self.command)
            detail = self.stderr.strip() or self.stdout.strip()
            raise RuntimeError(f"Command failed ({self.returncode}): {rendered}\n{detail}")


async def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> CompletedProcess:
    proc = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(cwd) if cwd else None,
        env={str(k): str(v) for k, v in (env or os.environ).items()},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    result = CompletedProcess(
        command=command,
        returncode=proc.returncode,
        stdout=stdout.decode(errors="replace"),
        stderr=stderr.decode(errors="replace"),
    )
    if check:
        result.raise_for_status()
    return result


def _issue_branch_name(issue: IssueMetadata) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", issue.title.lower()).strip("-")[:48]
    return f"issue/{issue.number}-{slug or 'fix'}"


def _parse_issue_url(value: str) -> tuple[str | None, int | None]:
    match = re.search(r"github\.com/([^/]+/[^/]+)/issues/(\d+)", value)
    if not match:
        return None, None
    return match.group(1), int(match.group(2))


def _parse_issue_number(value: str) -> int:
    cleaned = value.strip().removeprefix("#")
    if not cleaned.isdigit():
        raise ValueError("Issue number must be numeric.")
    return int(cleaned)


def _valid_repo(repo: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo or ""))


def _default_workdir(repo: str) -> Path:
    configured = os.getenv("HERMES_ISSUE_WORKDIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    path = Path.home() / repo.split("/", 1)[1]
    if path.exists():
        return path
    return Path.cwd()


def _merged_runtime_env(runtime_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if runtime_env is None else runtime_env)
    env_file = Path(env.get("KYBERM0NK_ENV", str(DEFAULT_KYBER_ENV))).expanduser()
    for key, value in _read_env_file(env_file).items():
        env.setdefault(key, value)
    _resolve_file_backed_secret(env, "OPENROUTER_API_KEY", "OPENROUTER_API_KEY_FILE")
    _resolve_file_backed_secret(env, "GITHUB_TOKEN", "GITHUB_TOKEN_FILE")
    if env.get("GITHUB_TOKEN") and not env.get("GH_TOKEN"):
        env["GH_TOKEN"] = env["GITHUB_TOKEN"]
    return env


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = value.strip().strip('"').strip("'")
    return values


def _resolve_file_backed_secret(env: dict[str, str], key_name: str, file_key_name: str) -> None:
    if env.get(key_name):
        return
    value = _read_secret_file(env.get(file_key_name, ""))
    if value:
        env[key_name] = value


def _resolve_secret(env: dict[str, str], key_name: str, file_key_name: str) -> str:
    value = env.get(key_name, "").strip()
    if value:
        return value
    return _read_secret_file(env.get(file_key_name, ""))


def _read_secret_file(path_value: str) -> str:
    if not path_value:
        return ""
    path = Path(path_value).expanduser()
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _require_secret(env: dict[str, str], key_name: str) -> str:
    value = env.get(key_name, "").strip()
    if not value:
        raise RuntimeError(f"{key_name} is required.")
    return value


def _normalize_guardian_base(value: str) -> str:
    return value.replace("host.docker.internal", "127.0.0.1").rstrip("/")


async def _noop_notify(_message: str) -> None:
    return None


def _now() -> float:
    return time.time()
