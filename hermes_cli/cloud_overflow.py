"""Prepare-only, READY-only cloud overflow planning.

This module deliberately stops before provider execution.  It is a bounded
control-plane primitive: it snapshots registered Kanban boards, filters only
explicit docs/research work, leases one deterministic candidate, and emits a
sanitized plan.  Provider adapters are explicit and mockable so a future,
separately-approved launcher can reuse the command/receipt contracts without
adding shell interpolation or implicit spend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence
from urllib.parse import urlparse


PROVIDER_ORDER = ("cursor-cloud", "claude-cloud", "codex-cloud")
EXPLICIT_WORK_CLASSES = frozenset({"docs", "documentation", "research"})
EXCLUDED_CLASSES = frozenset(
    {
        "credentials",
        "secrets",
        "money",
        "payments",
        "live-trading",
        "trading",
        "production-deploy",
        "deploy",
        "irreversible-data",
        "auth",
        "tenant-isolation",
        "provider-routing",
        "guardrail-mutation",
        "shared-writable-directory",
    }
)
_SAFE_ENV = frozenset({"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "CI"})
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_.:/-]{1,256}$")
_SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|secret|password|authorization|bearer|approval)"
)


class OverflowError(RuntimeError):
    """Base error for fail-closed overflow planning."""


class ProviderRefused(OverflowError):
    """Raised when a provider is not eligible under its safety gate."""


class CommentWriteError(OverflowError):
    """Raised when an audit receipt cannot be written."""


@dataclass(frozen=True)
class TaskSnapshot:
    id: str
    title: str
    body: str = ""
    status: str = "ready"
    assignee: Optional[str] = None
    claim_lock: Optional[str] = None
    skills: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    parents_satisfied: bool = True
    source_updated_at: Optional[int] = None
    workspace_kind: str = "worktree"
    workspace_path: Optional[str] = None
    branch_name: Optional[str] = None

    @classmethod
    def from_kanban_task(cls, task: Any) -> "TaskSnapshot":
        metadata = getattr(task, "metadata", None)
        if not isinstance(metadata, Mapping):
            metadata = {}
        labels = getattr(task, "labels", ())
        if isinstance(labels, str):
            labels = (labels,)
        skills = getattr(task, "skills", ()) or ()
        if isinstance(skills, str):
            skills = (skills,)
        return cls(
            id=str(task.id),
            title=str(task.title or ""),
            body=str(task.body or ""),
            status=str(task.status),
            assignee=getattr(task, "assignee", None),
            claim_lock=getattr(task, "claim_lock", None),
            skills=tuple(str(v) for v in skills if v),
            labels=tuple(str(v) for v in labels if v),
            metadata=dict(metadata),
            parents_satisfied=True,
            source_updated_at=getattr(task, "updated_at", None)
            or getattr(task, "started_at", None)
            or getattr(task, "created_at", None),
            workspace_kind=str(getattr(task, "workspace_kind", "worktree") or "worktree"),
            workspace_path=getattr(task, "workspace_path", None),
            branch_name=getattr(task, "branch_name", None),
        )


def source_revision(task: TaskSnapshot) -> str:
    """Hash immutable source fields; volatile claim/status is checked separately."""
    payload = {
        "id": task.id,
        "title": task.title,
        "body": task.body,
        "assignee": task.assignee,
        "skills": list(task.skills),
        "labels": list(task.labels),
        "metadata": task.metadata,
        "workspace_kind": task.workspace_kind,
        "workspace_path": task.workspace_path,
        "branch_name": task.branch_name,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BoardSnapshot:
    board: str
    running: int
    max_spawn: int
    tasks: tuple[TaskSnapshot, ...]
    reread: Optional[Callable[[str], Optional[TaskSnapshot]]] = None

    @property
    def saturated(self) -> bool:
        return self.running >= self.max_spawn


def _tokens(values: Iterable[Any]) -> set[str]:
    return {str(value).strip().lower().replace("_", "-") for value in values if value}


def _metadata_values(metadata: Mapping[str, Any], *keys: str) -> set[str]:
    values: list[Any] = []
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, (list, tuple, set)):
            values.extend(value)
        elif value is not None:
            values.append(value)
    return _tokens(values)


def classify_work(task: TaskSnapshot) -> tuple[Optional[str], Optional[str]]:
    """Classify from explicit metadata/labels/skills only; never title prose."""
    explicit = _tokens(task.skills) | _tokens(task.labels)
    explicit |= _metadata_values(task.metadata, "work_class", "type", "labels", "classes")
    excluded = explicit & EXCLUDED_CLASSES
    if excluded:
        return None, f"excluded:{sorted(excluded)[0]}"
    allowed = explicit & EXPLICIT_WORK_CLASSES
    if len(allowed) != 1:
        if not allowed:
            return None, "work_class_not_explicit"
        return None, "work_class_ambiguous"
    return next(iter(allowed)), None


def eligibility(task: TaskSnapshot) -> tuple[bool, str, Optional[str]]:
    if task.status != "ready":
        return False, "source_not_ready", None
    if task.claim_lock:
        return False, "source_claimed", None
    if not task.parents_satisfied:
        return False, "parents_not_satisfied", None
    work_class, reason = classify_work(task)
    if work_class is None:
        return False, reason or "ineligible", None
    return True, "eligible", work_class


def sanitize_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """Pass only non-secret process settings to a provider subprocess."""
    return {
        key: value
        for key, value in environ.items()
        if key in _SAFE_ENV and value and not _SECRET_RE.search(key)
    }


def _safe_prompt(task: TaskSnapshot) -> str:
    # The body is intentionally not forwarded: it may contain PII or secrets.
    return (
        f"ISO HOLD; draft PR only; task {task.id}; title: {task.title[:300]}. "
        "No merge, deploy, live trading, credential access, or schedule activation."
    )


def _parse_launch_output(stdout: str) -> tuple[Optional[str], Optional[str]]:
    try:
        value = json.loads(stdout)
    except (TypeError, ValueError):
        value = {}
    if isinstance(value, Mapping):
        session_id = value.get("session_id") or value.get("id")
        url = value.get("url") or value.get("session_url")
    else:
        session_id = url = None
    if session_id is None:
        match = re.search(r"(?:session[_ -]?id|task[_ -]?id)\s*[:=]\s*([A-Za-z0-9_.:/-]+)", stdout, re.I)
        session_id = match.group(1) if match else None
    if url is None:
        match = re.search(r"https?://[^\s\"']+", stdout)
        url = match.group(0) if match else None
    if session_id is not None and not _SESSION_ID_RE.fullmatch(str(session_id)):
        session_id = None
    if url is not None:
        parsed = urlparse(str(url))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            url = None
    return (str(session_id) if session_id else None, str(url) if url else None)


@dataclass(frozen=True)
class LaunchResult:
    session_id: Optional[str]
    url: Optional[str]
    argv: tuple[str, ...]


class CommandRunner(Protocol):
    def __call__(self, argv: Sequence[str], *, timeout: int, env: Mapping[str, str]) -> Any: ...


def _subprocess_runner(argv: Sequence[str], *, timeout: int, env: Mapping[str, str]) -> Any:
    return subprocess.run(
        list(argv),
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=dict(env),
        stdin=subprocess.DEVNULL,
    )


class ProviderAdapter:
    name: str

    def __init__(
        self,
        name: str,
        *,
        plan_authenticated: bool = False,
        isolated_checkout: Optional[str] = None,
        timeout: int = 300,
        runner: CommandRunner = _subprocess_runner,
    ) -> None:
        self.name = name
        self.plan_authenticated = plan_authenticated
        self.isolated_checkout = isolated_checkout
        self.timeout = timeout
        self.runner = runner

    @property
    def available(self) -> bool:
        return bool(self.plan_authenticated and self.isolated_checkout)

    def build_argv(self, task: TaskSnapshot) -> tuple[str, ...]:
        raise NotImplementedError

    def launch(self, task: TaskSnapshot) -> LaunchResult:
        if not self.available:
            raise ProviderRefused(f"{self.name} plan or isolated checkout unavailable")
        argv = self.build_argv(task)
        result = self.runner(argv, timeout=self.timeout, env=sanitize_environment(os.environ))
        session_id, url = _parse_launch_output(getattr(result, "stdout", "") or "")
        if not session_id and not url:
            raise OverflowError(f"{self.name} returned no session identity")
        return LaunchResult(session_id, url, argv)


class CursorCloudAdapter(ProviderAdapter):
    def build_argv(self, task: TaskSnapshot) -> tuple[str, ...]:
        return ("cursor", "cloud-agent", "run", "--repo", self.isolated_checkout or "", "--title", task.title[:300], "--prompt", _safe_prompt(task))


class ClaudeCloudAdapter(ProviderAdapter):
    def build_argv(self, task: TaskSnapshot) -> tuple[str, ...]:
        return ("claude", "--cloud", "--worktree", self.isolated_checkout or "", _safe_prompt(task))


class CodexCloudAdapter(ProviderAdapter):
    def __init__(self, *, env_id: Optional[str] = None, exact_spend_approval: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__("codex-cloud", **kwargs)
        self._env_id = env_id
        self._exact_spend_approval = exact_spend_approval

    @property
    def available(self) -> bool:
        # Values are held only in memory and never enter state, receipts, or logs.
        return bool(self.plan_authenticated and self.isolated_checkout and self._env_id and self._exact_spend_approval)

    def build_argv(self, task: TaskSnapshot) -> tuple[str, ...]:
        if not self._env_id or not self._exact_spend_approval:
            raise ProviderRefused("codex-cloud requires exact ENV_ID and exact spend approval")
        return ("codex", "cloud", "exec", "--env", self._env_id, _safe_prompt(task))


class OverflowState:
    """Single SQLite state store for leases, attempts, and sanitized receipts."""

    def __init__(self, path: Path | str, *, max_concurrency: int = 3) -> None:
        self.path = Path(path)
        self.max_concurrency = max(1, int(max_concurrency))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=5, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS overflow_state (
                    lease_key TEXT PRIMARY KEY,
                    board TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    task_updated_at INTEGER,
                    source_revision TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    external_session_id TEXT,
                    external_session_url TEXT,
                    status TEXT NOT NULL,
                    last_oracle TEXT NOT NULL,
                    veto TEXT,
                    next_node TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    next_attempt_at INTEGER
                );
                """
            )

    @staticmethod
    def idempotency_key(board: str, task_id: str, revision: str, provider: str) -> str:
        return f"{board}:{task_id}:{revision}:{provider}"

    def _active_count(self, conn: sqlite3.Connection) -> int:
        return int(conn.execute("SELECT COUNT(*) FROM overflow_state WHERE status IN ('planned','launched')").fetchone()[0])

    def acquire(self, *, board: str, task: TaskSnapshot, provider: str, now: Optional[int] = None) -> tuple[bool, str, str]:
        now = int(time.time() if now is None else now)
        revision = source_revision(task)
        key = self.idempotency_key(board, task.id, revision, provider)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("SELECT status FROM overflow_state WHERE lease_key = ?", (key,)).fetchone()
            if existing:
                conn.rollback()
                return False, "duplicate_lease", key
            if self._active_count(conn) >= self.max_concurrency:
                conn.rollback()
                return False, "global_concurrency_cap", key
            conn.execute(
                """INSERT INTO overflow_state
                (lease_key, board, task_id, task_updated_at, source_revision, provider,
                 attempt, idempotency_key, status, last_oracle, next_node, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, 'planned', 'lease_acquired', 'ROUTER', ?, ?)""",
                (key, board, task.id, task.source_updated_at, revision, provider, key, now, now),
            )
            conn.commit()
        return True, "lease_acquired", key

    def update(self, key: str, *, status: str, oracle: str, next_node: str, veto: Optional[str] = None, result: Optional[LaunchResult] = None, now: Optional[int] = None) -> None:
        now = int(time.time() if now is None else now)
        session_id = result.session_id if result else None
        url = result.url if result else None
        with self._connect() as conn:
            conn.execute(
                """UPDATE overflow_state SET status=?, last_oracle=?, next_node=?, veto=?,
                   external_session_id=?, external_session_url=?, updated_at=? WHERE lease_key=?""",
                (status, oracle, next_node, veto, session_id, url, now, key),
            )
            conn.commit()

    def get(self, key: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM overflow_state WHERE lease_key = ?", (key,)).fetchone()
        return dict(row) if row else None

    def record_failure(self, key: str, *, reason: str, attempt: int, now: Optional[int] = None) -> None:
        now = int(time.time() if now is None else now)
        next_at = now + exponential_backoff(attempt)
        self.update(key, status="failed", oracle="provider_failure", next_node="BLOCKED", veto=reason, now=now)
        with self._connect() as conn:
            conn.execute("UPDATE overflow_state SET next_attempt_at = ? WHERE lease_key = ?", (next_at, key))
            conn.commit()


@dataclass(frozen=True)
class TickResult:
    status: str
    board: Optional[str] = None
    task_id: Optional[str] = None
    provider: Optional[str] = None
    action: str = "no-op"
    reason: str = ""
    idempotency_key: Optional[str] = None
    receipt: Optional[dict[str, Any]] = None


def sanitize_receipt(*, provider: str, session_id: Optional[str], url: Optional[str], branch: Optional[str], workspace: Optional[str], status: str, idempotency_key: str) -> dict[str, Any]:
    """Return the only fields permitted in a source-card receipt."""
    clean: dict[str, Any] = {
        "provider": provider,
        "external_session_id": session_id if session_id and _SESSION_ID_RE.fullmatch(session_id) else None,
        "external_session_url": url if url and urlparse(url).scheme in {"http", "https"} and urlparse(url).netloc else None,
        "isolated_branch": str(branch or "")[:300],
        "isolated_workspace": str(workspace or "")[:500],
        "status": status,
        "idempotency_key": idempotency_key,
    }
    serialized = json.dumps(clean, sort_keys=True)
    if _SECRET_RE.search(serialized) or "prompt" in serialized.lower():
        raise OverflowError("receipt contains forbidden secret/prompt material")
    return clean


class CommentWriter(Protocol):
    def __call__(self, board: str, task_id: str, body: str) -> Any: ...


def run_tick(
    boards: Sequence[BoardSnapshot],
    *,
    state: OverflowState,
    adapters: Mapping[str, ProviderAdapter],
    comment_writer: Optional[CommentWriter] = None,
    fleet_paused: bool = False,
    kill_switch: bool = False,
    now: Optional[int] = None,
) -> TickResult:
    """Run one bounded prepare pass.  This function never invokes a provider."""
    if fleet_paused or kill_switch:
        return TickResult("blocked", action="no-op", reason="pause_or_kill_switch")
    for board in boards:
        if not board.saturated:
            continue
        for task in board.tasks:
            ok, reason, _ = eligibility(task)
            if not ok:
                continue
            if board.reread:
                current = board.reread(task.id)
                if current is None:
                    continue
                if source_revision(current) != source_revision(task) or current.status != "ready" or current.claim_lock or not current.parents_satisfied:
                    continue
            provider = next((name for name in PROVIDER_ORDER if name in adapters and adapters[name].available), None)
            if provider is None:
                continue
            acquired, lease_reason, key = state.acquire(board=board.board, task=task, provider=provider, now=now)
            if not acquired:
                return TickResult("no-op", board=board.board, task_id=task.id, provider=provider, reason=lease_reason, idempotency_key=key)
            # Prepare-only by construction: the provider adapter is injectable for
            # future approved launchers, but no adapter.launch call occurs here.
            state.update(key, status="planned", oracle="dry_run_no_provider_invocation", next_node="HUMAN_GATE")
            return TickResult(
                "planned",
                board=board.board,
                task_id=task.id,
                provider=provider,
                action="prepare-only",
                reason="eligible_saturated_ready_task",
                idempotency_key=key,
            )
    return TickResult("no-op", action="no-op", reason="no_eligible_candidate")


def load_fixture(path: Path | str) -> list[BoardSnapshot]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    boards: list[BoardSnapshot] = []
    for item in raw.get("boards", []):
        tasks = tuple(TaskSnapshot(**task) for task in item.get("tasks", []))
        boards.append(BoardSnapshot(str(item["board"]), int(item.get("running", 0)), int(item.get("max_spawn", 3)), tasks))
    return boards


def exponential_backoff(attempt: int, *, base_seconds: int = 30, cap_seconds: int = 3600) -> int:
    """Bound retry delay; attempt zero is the first immediate attempt."""
    return min(max(0, int(cap_seconds)), max(0, int(base_seconds)) * (2 ** max(0, int(attempt))))


def record_launch(
    *,
    state: OverflowState,
    lease_key: str,
    board: str,
    task: TaskSnapshot,
    adapter: ProviderAdapter,
    comment_writer: CommentWriter,
    approved: bool = False,
    now: Optional[int] = None,
) -> dict[str, Any]:
    """Approved-only launch/receipt seam; callers must opt in explicitly.

    The prepare CLI never calls this.  A future launcher must supply a separate
    approval decision and a real comment writer; a failed receipt comment marks
    the state unresolved instead of pretending the launch was audited.
    """
    if not approved:
        raise ProviderRefused("provider launch requires a separate explicit approval")
    result = adapter.launch(task)
    receipt = sanitize_receipt(
        provider=adapter.name,
        session_id=result.session_id,
        url=result.url,
        branch=task.branch_name,
        workspace=task.workspace_path,
        status="launched",
        idempotency_key=lease_key,
    )
    state.update(lease_key, status="launched", oracle="provider_identity_parsed", next_node="RECORDER", result=result, now=now)
    try:
        comment_writer(board, task.id, json.dumps(receipt, sort_keys=True))
    except Exception as exc:
        state.update(lease_key, status="unresolved", oracle="comment_write_failed", next_node="BLOCKED", veto="receipt_comment_failed", now=now)
        raise CommentWriteError("launch receipt comment failed; launch is unresolved") from exc
    state.update(lease_key, status="launched", oracle="receipt_comment_written", next_node="CHECKER", result=result, now=now)
    return receipt


def build_default_adapters() -> dict[str, ProviderAdapter]:
    # Defaults are unavailable: configuration/authentication is an explicit gate.
    return {
        "cursor-cloud": CursorCloudAdapter("cursor-cloud"),
        "claude-cloud": ClaudeCloudAdapter("claude-cloud"),
        "codex-cloud": CodexCloudAdapter(),
    }


def snapshot_registered_boards(*, max_spawn: int = 3, board: Optional[str] = None) -> list[BoardSnapshot]:
    """Read all registered boards through the existing Kanban APIs."""
    from hermes_cli import kanban_db as kb

    metas = kb.list_boards(include_archived=False)
    if board:
        metas = [meta for meta in metas if meta.get("slug") == board]
    snapshots: list[BoardSnapshot] = []
    for meta in metas:
        slug = str(meta.get("slug") or kb.DEFAULT_BOARD)
        with kb.connect_closing(board=slug) as conn:
            ready = kb.list_tasks(conn, status="ready", include_archived=False)
            tasks: list[TaskSnapshot] = []
            for task in ready:
                snapshot = TaskSnapshot.from_kanban_task(task)
                snapshot = TaskSnapshot(**{**asdict(snapshot), "parents_satisfied": kb._parents_satisfied(conn, task.id)})
                tasks.append(snapshot)
            running = kb.count_running_tasks(conn)
        def reread(task_id: str, slug: str = slug) -> Optional[TaskSnapshot]:
            with kb.connect_closing(board=slug) as fresh:
                task = kb.get_task(fresh, task_id)
                if task is None:
                    return None
                value = TaskSnapshot.from_kanban_task(task)
                return TaskSnapshot(**{**asdict(value), "parents_satisfied": kb._parents_satisfied(fresh, task_id)})
        snapshots.append(BoardSnapshot(slug, running, max_spawn, tuple(tasks), reread))
    return snapshots


def kanban_comment_writer(board: str, task_id: str, body: str) -> int:
    """Write an audit receipt through the native Kanban comment API."""
    from hermes_cli import kanban_db as kb

    with kb.connect_closing(board=board) as conn:
        return kb.add_comment(conn, task_id, "cloud-overflow", body)


def cloud_overflow_command(args: argparse.Namespace) -> int:
    """CLI entry point; only --dry-run fixture planning is exposed."""
    if not getattr(args, "fixture", None):
        print("cloud-overflow: --fixture is required for the prepare-only path", file=__import__("sys").stderr)
        return 2
    boards = load_fixture(args.fixture)
    state = OverflowState(args.state, max_concurrency=args.max_concurrency)
    adapters = {
        "cursor-cloud": CursorCloudAdapter("cursor-cloud", plan_authenticated=True, isolated_checkout="fixture"),
        "claude-cloud": ClaudeCloudAdapter("claude-cloud", plan_authenticated=True, isolated_checkout="fixture"),
        "codex-cloud": CodexCloudAdapter(plan_authenticated=False, isolated_checkout="fixture"),
    }
    result = run_tick(boards, state=state, adapters=adapters, fleet_paused=args.pause, kill_switch=args.kill_switch)
    payload = asdict(result)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(f"cloud-overflow: {result.status} action={result.action} board={result.board or '-'} task={result.task_id or '-'} provider={result.provider or '-'} reason={result.reason}")
    return 0


def build_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fixture", required=True, help="JSON fixture; required for the prepare-only CLI")
    parser.add_argument("--state", required=True, help="Isolated SQLite state-store path")
    parser.add_argument("--max-concurrency", type=int, default=3)
    parser.add_argument("--pause", action="store_true")
    parser.add_argument("--kill-switch", action="store_true")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Plan only; provider commands are never invoked")
    parser.add_argument("--json", action="store_true")
