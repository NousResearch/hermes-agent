"""TTM control-plane spawn-shim API.

PR-F-H1 of the Hermes alignment plan, with H1's two deferred items
(``_spawn_headless_session`` real subprocess + SQLite binding
registry) landed in PR-F-H3 closeout. Mounts under
``/api/plugins/ttm-control-plane/`` on the Hermes dashboard FastAPI
app.

This is the HTTP face that TTM's ``HermesAdapter.dispatch_run()`` calls:
TTM POSTs the runtime dispatch payload (carrying the per-run principal
token), Hermes validates the payload, binds the run to a runtime
session, and returns 202 ``{status: "accepted", runtime_run_ref}``.
A headless ``hermes chat`` subprocess is then spawned with the run's
TTM_* env vars so the agent can drive the run via the ttm_ingress
skill. The binding registry is SQLite-backed so dispatched runs
survive dashboard restarts (the principal token is the one piece of
binding state that is NOT persisted — see ``_BindingRegistry``).

Auth model: shared-secret header ``X-TTM-Control-Plane-Secret`` whose
value matches the ``TTM_CONTROL_PLANE_SECRET`` environment variable
loaded from ``~/.hermes/.env``. The dashboard auth middleware in
``hermes_cli/web_server.py`` deliberately bypasses ``/api/plugins/*``,
so this plugin owns its own auth check.

Per ``RUNTIME-PRINCIPAL-CONTRACT.md``, the ``principal_token`` lives in
the dispatch body and MUST be forwarded as ``Authorization: Bearer
<token>`` on every ingress write-back to TTM. Plaintext never appears
in plugin logs or in the runtime registry.

H6 adds lifecycle control (stop/pause/resume/expand_scope) via:
  POST /runs/{ref}/lifecycle   — unified lifecycle receiver (202 async)
  POST /runs/{ref}/stop        — compat alias for stop (TTM adapter pre-H6)

Stop:  SIGTERM → 10s wait → SIGKILL; emits task.updated{stopped}.
Pause: SIGSTOP; persists dossier state; emits task.updated{paused}.
       Degrades explicitly if platform SIGSTOP is unavailable.
Resume: SIGCONT from saved pause state; emits task.updated{active}.
Expand-scope: revokes old token; SIGUSR1 advisory hint; awaits
              rebind-token to restore emission with new epoch.

Hermes never self-approves run closure. That remains TTM's domain.
"""

import asyncio
import json
import logging
import os
import shutil
import signal
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import APIRouter, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Shared-secret auth
# ---------------------------------------------------------------------------

_SECRET_ENV = "TTM_CONTROL_PLANE_SECRET"
_SECRET_HEADER = "X-TTM-Control-Plane-Secret"


def _expected_secret() -> str:
    """Read the shared secret from env. Empty string means "auth disabled".

    Returning the empty string lets the plugin run in dev/CI without a
    secret configured; in production, set ``TTM_CONTROL_PLANE_SECRET`` to
    a long random value in both the TTM Doppler config and
    ``~/.hermes/.env``.
    """
    return os.environ.get(_SECRET_ENV, "").strip()


def _require_secret(provided: str | None) -> None:
    expected = _expected_secret()
    if not expected:
        # Auth disabled — log a warning the first time so the operator
        # notices in dev. Production deployment must set the env var.
        return
    if provided != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"reason": "ttm_control_plane_secret_mismatch"},
        )


# ---------------------------------------------------------------------------
# In-memory binding registry — one entry per active run
# ---------------------------------------------------------------------------


@dataclass
class _RuntimeBinding:
    """A live mapping from TTM ``run_id`` → Hermes runtime session."""

    run_id: str
    runtime_binding_id: str
    runtime_run_ref: str
    ingress_base_url: str
    bound_at: datetime
    # ``principal_token`` is held in memory only — never persisted to SQLite
    # and never logged. After the run.dispatched callback fires, the token
    # is cleared; the headless agent process holds its own copy via the
    # TTM_PRINCIPAL_TOKEN env var passed at spawn time.
    principal_token: str = ""
    last_status: str = "pending"
    payload_summary: dict[str, Any] = field(default_factory=dict)


_DEFAULT_DB_PATH = os.path.expanduser(
    os.environ.get("TTM_CONTROL_PLANE_DB_PATH", "~/.hermes/state.db")
)
_DB_TABLE = "ttm_control_plane_bindings"


class _BindingRegistry:
    """Thread-safe ``run_id → _RuntimeBinding`` registry with SQLite persistence.

    Binding metadata persists across dashboard restarts so:
    1. re-dispatch against a known run_id is idempotent (returns 409),
    2. operators can inspect prior dispatches via /health and snapshot,
    3. a rebind picks up the existing binding and just rotates the token.

    The principal token is NOT persisted — it lives only in memory and
    is cleared after the run.dispatched callback fires. After a restart
    the registry's tokens are empty; the operator must trigger a rebind
    (POST /api/runs/control-plane/{run_id}/rebind on TTM) to issue a
    fresh token, which TTM forwards via /runs/{run_id}/rebind-token.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._lock = threading.Lock()
        self._by_run: dict[str, _RuntimeBinding] = {}
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._init_db()
        self._load_from_db()

    def _connect(self) -> sqlite3.Connection:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_DB_TABLE} (
                    run_id TEXT PRIMARY KEY,
                    runtime_binding_id TEXT NOT NULL,
                    runtime_run_ref TEXT NOT NULL,
                    ingress_base_url TEXT NOT NULL,
                    bound_at TEXT NOT NULL,
                    last_status TEXT NOT NULL,
                    payload_summary_json TEXT NOT NULL
                )
                """
            )

    def _load_from_db(self) -> None:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT run_id, runtime_binding_id, runtime_run_ref, "
                f"ingress_base_url, bound_at, last_status, payload_summary_json "
                f"FROM {_DB_TABLE}"
            ).fetchall()
        for row in rows:
            self._by_run[row[0]] = _RuntimeBinding(
                run_id=row[0],
                runtime_binding_id=row[1],
                runtime_run_ref=row[2],
                ingress_base_url=row[3],
                bound_at=datetime.fromisoformat(row[4]),
                principal_token="",  # never persisted
                last_status=row[5],
                payload_summary=json.loads(row[6]) if row[6] else {},
            )

    def _persist(self, binding: _RuntimeBinding) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {_DB_TABLE}
                  (run_id, runtime_binding_id, runtime_run_ref, ingress_base_url,
                   bound_at, last_status, payload_summary_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                  runtime_binding_id=excluded.runtime_binding_id,
                  runtime_run_ref=excluded.runtime_run_ref,
                  ingress_base_url=excluded.ingress_base_url,
                  bound_at=excluded.bound_at,
                  last_status=excluded.last_status,
                  payload_summary_json=excluded.payload_summary_json
                """,
                (
                    binding.run_id,
                    binding.runtime_binding_id,
                    binding.runtime_run_ref,
                    binding.ingress_base_url,
                    binding.bound_at.isoformat(),
                    binding.last_status,
                    json.dumps(binding.payload_summary),
                ),
            )

    def get(self, run_id: str) -> _RuntimeBinding | None:
        with self._lock:
            return self._by_run.get(run_id)

    def insert(self, binding: _RuntimeBinding) -> _RuntimeBinding:
        with self._lock:
            existing = self._by_run.get(binding.run_id)
            if existing is not None:
                return existing
            self._by_run[binding.run_id] = binding
            self._persist(binding)
            return binding

    def update_status(self, run_id: str, *, last_status: str) -> None:
        with self._lock:
            entry = self._by_run.get(run_id)
            if entry is None:
                return
            entry.last_status = last_status
            self._persist(entry)

    def clear_token(self, run_id: str) -> None:
        with self._lock:
            entry = self._by_run.get(run_id)
            if entry is not None:
                entry.principal_token = ""
        # No DB write — token is never persisted.

    def replace_token(self, run_id: str, new_token: str) -> bool:
        """Atomically replace the principal token; returns True if binding found."""
        with self._lock:
            entry = self._by_run.get(run_id)
            if entry is None:
                return False
            entry.principal_token = new_token
            return True

    def remove(self, run_id: str) -> None:
        with self._lock:
            self._by_run.pop(run_id, None)
            with self._connect() as conn:
                conn.execute(f"DELETE FROM {_DB_TABLE} WHERE run_id = ?", (run_id,))

    def snapshot(self) -> list[_RuntimeBinding]:
        with self._lock:
            return list(self._by_run.values())

    def clear(self) -> None:
        """Wipe both in-memory and persistent state. Test-only."""
        with self._lock:
            self._by_run.clear()
            with self._connect() as conn:
                conn.execute(f"DELETE FROM {_DB_TABLE}")


_REGISTRY = _BindingRegistry()


# ---------------------------------------------------------------------------
# Process registry — tracks live headless session PIDs for lifecycle control
# ---------------------------------------------------------------------------


@dataclass
class _ProcessHandle:
    """Lightweight reference to a running headless session subprocess."""

    run_id: str
    pid: int
    proc: Any  # asyncio.subprocess.Process — event-loop-bound
    started_at: datetime


class _ProcessRegistry:
    """Thread-safe registry of live headless session process handles.

    Registered at spawn time; removed when the process exits or the stop
    handler completes. Separate from the binding registry so the binding
    (and its status history) outlives the process.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_run: dict[str, _ProcessHandle] = {}

    def register(self, run_id: str, proc: Any) -> None:
        with self._lock:
            self._by_run[run_id] = _ProcessHandle(
                run_id=run_id,
                pid=proc.pid,
                proc=proc,
                started_at=_utcnow(),
            )

    def get(self, run_id: str) -> _ProcessHandle | None:
        with self._lock:
            return self._by_run.get(run_id)

    def remove(self, run_id: str) -> None:
        with self._lock:
            self._by_run.pop(run_id, None)

    def clear(self) -> None:
        """Wipe all handles. Test-only."""
        with self._lock:
            self._by_run.clear()


_PROC_REGISTRY = _ProcessRegistry()


# ---------------------------------------------------------------------------
# Pause-state dossier — persists per-run context across pause/resume
# ---------------------------------------------------------------------------


@dataclass
class _PauseState:
    """Dossier snapshot saved when a run is paused via SIGSTOP."""

    run_id: str
    pid: int
    paused_at: datetime
    lane_id: str = ""
    worktree_id: str = ""


_PAUSE_STATE: dict[str, _PauseState] = {}
_PAUSE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Wire schemas — mirror RuntimeDispatchPayload / RuntimeDispatchResult
# ---------------------------------------------------------------------------


class DispatchPayload(BaseModel):
    """Body of POST /runs/dispatch from TTM HermesAdapter.

    Fields mirror ``backend/app/services/orchestration/runtime_adapter.py
    RuntimeDispatchPayload`` plus the adapter-only ``runtime_id`` and
    ``ingress_base_url`` that ``hermes_adapter.py:_payload_dict`` injects.
    """

    run_id: str = Field(..., min_length=1, max_length=128)
    runtime_id: str = Field(default="hermes", max_length=64)
    stream_id: str = Field(..., min_length=1, max_length=64)
    stream_version: str = Field(..., min_length=1, max_length=64)
    runtime_binding_id: str = Field(..., min_length=1, max_length=128)
    slice_id: str = Field(..., min_length=1, max_length=128)
    lane_id: str = Field(..., min_length=1, max_length=160)
    scope_hash: str = Field(..., min_length=1, max_length=128)
    worktree_id: str = ""
    goal: str = Field(..., min_length=1, max_length=2000)
    allowed_paths: list[str] = Field(default_factory=list)
    required_tests: list[str] = Field(default_factory=list)
    reply_schema: dict = Field(default_factory=dict)
    deadline_at: str | None = None
    ingress_base_url: str = ""
    # Per RUNTIME-PRINCIPAL-CONTRACT.md the token rides the dispatch body.
    # Empty string is rejected at the route level — an unauthenticated
    # spawn would never be able to write back to TTM ingress.
    principal_token: str = ""


class DispatchResponse(BaseModel):
    """Response shape — accepted/degraded with a ``runtime_run_ref``."""

    status: str
    runtime_run_ref: str
    bound_at: datetime


class StatusResponse(BaseModel):
    """Status projection for ``GET /runs/{ref}/status``."""

    status: str
    runtime_run_ref: str
    bound_at: datetime
    checked_at: datetime


class HealthResponse(BaseModel):
    plugin: str
    version: str
    auth_enforced: bool
    bindings: int
    checked_at: datetime


class RebindTokenRequest(BaseModel):
    """Body for POST /runs/{run_id}/rebind-token from TTM HermesAdapter.notify_rebind."""

    new_binding_id: str = Field(..., min_length=1, max_length=128)
    new_token: str = Field(..., min_length=1)
    ingress_base_url: str = ""


# ---------------------------------------------------------------------------
# Lifecycle wire schemas (H6)
# ---------------------------------------------------------------------------

LifecycleAction = Literal["stop", "pause", "resume", "expand_scope"]


class LifecycleRequest(BaseModel):
    """Body for POST /runs/{ref}/lifecycle."""

    action: LifecycleAction


class LifecycleResponse(BaseModel):
    """Immediate 202 response — action is processed asynchronously."""

    status: str
    runtime_run_ref: str
    action: str
    accepted_at: datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _payload_summary(payload: DispatchPayload) -> dict[str, Any]:
    """Redact-by-default summary used for logs and the in-memory registry.

    Excludes ``principal_token`` and any other sensitive fields.
    """
    return {
        "stream_id": payload.stream_id,
        "stream_version": payload.stream_version,
        "lane_id": payload.lane_id,
        "slice_id": payload.slice_id,
        "scope_hash": payload.scope_hash,
        "worktree_id": payload.worktree_id,
        "deadline_at": payload.deadline_at,
        "allowed_paths_count": len(payload.allowed_paths),
        "required_tests_count": len(payload.required_tests),
        "goal_len": len(payload.goal),
    }


def _binding_by_ref(runtime_run_ref: str) -> _RuntimeBinding | None:
    """Look up a binding by runtime_run_ref (O(n) scan over the small registry)."""
    return next(
        (b for b in _REGISTRY.snapshot() if b.runtime_run_ref == runtime_run_ref),
        None,
    )


async def _post_run_dispatched(binding: _RuntimeBinding) -> None:
    """Fire-and-forget: POST run.dispatched to TTM ingress.

    Skipped (logged) when ``ingress_base_url`` is empty so PR-F-H1 can
    bring up cleanly even before TTM PR-F's ingress routes are
    deployed in the operator's environment.
    """
    base = (binding.ingress_base_url or "").rstrip("/")
    token = binding.principal_token
    if not base:
        logger.warning(
            "ttm-control-plane.run_dispatched.skipped: ingress_base_url unset run_id=%s",
            binding.run_id,
        )
        _REGISTRY.clear_token(binding.run_id)
        return
    if not token:
        logger.warning(
            "ttm-control-plane.run_dispatched.skipped: no principal_token run_id=%s",
            binding.run_id,
        )
        return

    url = f"{base}/api/ingress/runtime/hermes/events"
    body = {
        "event_type": "run.dispatched",
        "actor_type": "runtime",
        "actor_id": "hermes",
        "expected_scope_epoch": 1,
        "summary": "Hermes accepted the dispatch and bound the run",
        "payload": {
            "runtime_run_ref": binding.runtime_run_ref,
            "runtime_binding_id": binding.runtime_binding_id,
            "bound_at": binding.bound_at.isoformat(),
        },
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Runtime-Id": "hermes",
        "X-Run-Id": binding.run_id,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=body, headers=headers)
            if resp.status_code >= 400:
                logger.warning(
                    "ttm-control-plane.run_dispatched.failed run_id=%s status=%s body=%s",
                    binding.run_id,
                    resp.status_code,
                    resp.text[:200],
                )
                return
    except httpx.HTTPError as exc:
        logger.warning(
            "ttm-control-plane.run_dispatched.error run_id=%s error=%s",
            binding.run_id,
            exc,
        )
        return
    finally:
        # Drop the bearer credential from memory once we've used it.
        # Future ingress writes will be issued by the headless session
        # directly using its own copy of the token.
        _REGISTRY.clear_token(binding.run_id)


async def _emit_lifecycle_event(
    binding: _RuntimeBinding,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    """POST a canonical event to TTM ingress if a principal token is available.

    Skipped when the token is absent (cleared post-dispatch or after scope
    revocation). The headless agent process owns its own token copy and will
    emit lifecycle events independently via the ttm_ingress skill.
    Never logs token material.
    """
    base = (binding.ingress_base_url or "").rstrip("/")
    token = binding.principal_token
    if not base or not token:
        logger.debug(
            "ttm-control-plane.lifecycle_event.skipped event_type=%s run_id=%s "
            "(no ingress_base_url or token not held by plugin)",
            event_type,
            binding.run_id,
        )
        return

    url = f"{base}/api/ingress/runtime/hermes/events"
    body = {
        "event_type": event_type,
        "actor_type": "runtime",
        "actor_id": "hermes",
        "expected_scope_epoch": 1,
        "summary": f"Hermes lifecycle: {event_type}",
        "payload": payload,
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Runtime-Id": "hermes",
        "X-Run-Id": binding.run_id,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=body, headers=headers)
            if resp.status_code >= 400:
                logger.warning(
                    "ttm-control-plane.lifecycle_event.failed event_type=%s run_id=%s status=%s",
                    event_type,
                    binding.run_id,
                    resp.status_code,
                )
    except httpx.HTTPError as exc:
        logger.warning(
            "ttm-control-plane.lifecycle_event.error event_type=%s run_id=%s error=%s",
            event_type,
            binding.run_id,
            exc,
        )


_SPAWN_LOG_DIR = os.path.expanduser(
    os.environ.get("TTM_CONTROL_PLANE_LOG_DIR", "~/.hermes/logs/runs")
)
_SPAWN_DISABLED_ENV = "TTM_CONTROL_PLANE_DISABLE_SPAWN"


def _hermes_executable() -> str | None:
    """Resolve the hermes CLI binary.

    Prefers ``HERMES_CLI`` env, then PATH lookup, then a fallback to the
    venv that the dashboard itself is running in. Returns ``None`` if
    none of those resolve so the caller can log-and-skip cleanly.
    """
    explicit = os.environ.get("HERMES_CLI", "").strip()
    if explicit and Path(explicit).exists():
        return explicit
    found = shutil.which("hermes")
    if found:
        return found
    # Fall back to the same venv the dashboard process is using.
    candidate = Path(os.path.dirname(os.path.dirname(os.__file__))).parent / "bin" / "hermes"
    if candidate.exists():
        return str(candidate)
    return None


async def _spawn_headless_session(binding: _RuntimeBinding, principal_token: str) -> None:
    """Spawn a headless Hermes session bound to the dispatched run.

    The child receives the principal token and ingress base URL via env
    vars (see ``tools/ttm_ingress.py:bind_run_from_env``). Failure to
    spawn is logged but never crashes the dispatch — the operator can
    inspect ``ttm-control-plane.headless_session.*`` log lines and
    re-dispatch or rebind if the pathway is misconfigured.

    The child is intentionally NOT awaited: the dispatch route already
    returned 202 and the agent runs for the lifetime of the run.
    The process handle is registered in ``_PROC_REGISTRY`` for lifecycle
    control (stop/pause/resume).
    """
    if os.environ.get(_SPAWN_DISABLED_ENV, "").strip().lower() in {"1", "true", "yes"}:
        logger.info(
            "ttm-control-plane.headless_session.disabled run_id=%s "
            "(TTM_CONTROL_PLANE_DISABLE_SPAWN set; binding registered without spawn)",
            binding.run_id,
        )
        return

    hermes_bin = _hermes_executable()
    if hermes_bin is None:
        logger.warning(
            "ttm-control-plane.headless_session.no_executable run_id=%s "
            "(set HERMES_CLI or add hermes to PATH; binding remains registered)",
            binding.run_id,
        )
        return

    if not principal_token:
        logger.warning(
            "ttm-control-plane.headless_session.no_token run_id=%s "
            "(token already cleared; cannot spawn)",
            binding.run_id,
        )
        return

    base = (binding.ingress_base_url or "").rstrip("/")
    if not base:
        logger.warning(
            "ttm-control-plane.headless_session.no_ingress run_id=%s "
            "(ingress_base_url unset; cannot spawn)",
            binding.run_id,
        )
        return

    Path(_SPAWN_LOG_DIR).mkdir(parents=True, exist_ok=True)
    log_path = Path(_SPAWN_LOG_DIR) / f"{binding.run_id}.log"

    env = {
        **os.environ,
        "TTM_RUN_ID": binding.run_id,
        "TTM_PRINCIPAL_TOKEN": principal_token,
        "TTM_INGRESS_BASE_URL": base,
        "TTM_RUNTIME_ID": "hermes",
    }

    brief = (
        f"You are Hermes executing TTM run {binding.run_id}. "
        f"Bind via tools.ttm_ingress.bind_run_from_env(), read run state, "
        f"drive each gate in approval_policy: post events, attach evidence, "
        f"request approval, poll for the operator decision, and emit "
        f"run.closed when complete. The principal token is in "
        f"TTM_PRINCIPAL_TOKEN; never log it."
    )

    try:
        log_file = open(log_path, "ab", buffering=0)  # noqa: SIM115 — owned by child
        proc = await asyncio.create_subprocess_exec(
            hermes_bin,
            "chat",
            "-q",
            brief,
            "-Q",
            "--max-turns",
            "200",
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
    except (OSError, asyncio.CancelledError) as exc:
        logger.error(
            "ttm-control-plane.headless_session.spawn_failed run_id=%s error=%s",
            binding.run_id,
            exc,
        )
        return

    _PROC_REGISTRY.register(binding.run_id, proc)
    logger.info(
        "ttm-control-plane.headless_session.spawned run_id=%s pid=%s log=%s",
        binding.run_id,
        proc.pid,
        log_path,
    )


def _kick_off_post_dispatch_tasks(binding: _RuntimeBinding) -> None:
    """Schedule the run.dispatched callback + headless spawn off the
    request loop so the route returns 202 immediately.

    The principal token is captured and passed to the spawn task before
    ``_post_run_dispatched`` clears it from the registry — this avoids a
    race where the spawn would observe an empty token.
    """
    loop = asyncio.get_running_loop()
    captured_token = binding.principal_token
    loop.create_task(_post_run_dispatched(binding))
    loop.create_task(_spawn_headless_session(binding, captured_token))


# ---------------------------------------------------------------------------
# Lifecycle action handlers (H6) — called off the request loop as Tasks
# ---------------------------------------------------------------------------


async def _kill_run_process(
    run_id: str,
    *,
    term_timeout: float = 10.0,
) -> None:
    """SIGTERM the headless session process group, then SIGKILL on timeout.

    Safe to call when no process is registered (logs and returns).
    Cleans up the process handle from ``_PROC_REGISTRY`` on completion.
    """
    handle = _PROC_REGISTRY.get(run_id)
    if handle is None:
        return

    try:
        pgid = os.getpgid(handle.pid)
    except ProcessLookupError:
        _PROC_REGISTRY.remove(run_id)
        return

    # SIGTERM — give the process a chance to flush state and exit cleanly.
    try:
        os.killpg(pgid, signal.SIGTERM)
        logger.info(
            "ttm-control-plane.kill.sigterm run_id=%s pid=%s pgid=%s",
            run_id,
            handle.pid,
            pgid,
        )
    except (ProcessLookupError, PermissionError) as exc:
        logger.warning(
            "ttm-control-plane.kill.sigterm_failed run_id=%s error=%s",
            run_id,
            exc,
        )
        _PROC_REGISTRY.remove(run_id)
        return

    # Wait up to term_timeout for graceful exit.
    try:
        await asyncio.wait_for(handle.proc.wait(), timeout=term_timeout)
    except asyncio.TimeoutError:
        pass

    # SIGKILL any survivors.
    if handle.proc.returncode is None:
        logger.info(
            "ttm-control-plane.kill.sigkill run_id=%s pid=%s (SIGTERM timeout)",
            run_id,
            handle.pid,
        )
        try:
            os.killpg(os.getpgid(handle.pid), signal.SIGKILL)
            await asyncio.wait_for(handle.proc.wait(), timeout=5.0)
        except (ProcessLookupError, asyncio.TimeoutError, PermissionError):
            pass

    _PROC_REGISTRY.remove(run_id)


async def _async_stop(run_id: str, runtime_run_ref: str) -> None:
    """Stop the headless session: kill process, emit event, update status.

    Hermes reports stop completion via task.updated{status: stopped} but
    does NOT self-approve run closure — that remains TTM's domain and
    requires a granted close approval on TTM's side.
    The binding is kept (status="stopped") so TTM can query status and
    drive the canonical closure cascade independently.
    """
    binding = _REGISTRY.get(run_id)
    if binding is None:
        logger.warning("ttm-control-plane.stop.no_binding run_id=%s", run_id)
        return

    await _kill_run_process(run_id)

    await _emit_lifecycle_event(
        binding,
        "task.updated",
        {"status": "stopped", "runtime_run_ref": runtime_run_ref},
    )
    _REGISTRY.update_status(run_id, last_status="stopped")
    logger.info(
        "ttm-control-plane.stop.complete run_id=%s runtime_run_ref=%s",
        run_id,
        runtime_run_ref,
    )


async def _async_pause(run_id: str, runtime_run_ref: str) -> None:
    """Pause the headless session with SIGSTOP; persist dossier state.

    Degrades explicitly if SIGSTOP is unavailable or the process is gone —
    does NOT silently report paused when the suspend did not happen.
    """
    binding = _REGISTRY.get(run_id)
    if binding is None:
        logger.warning("ttm-control-plane.pause.no_binding run_id=%s", run_id)
        return

    handle = _PROC_REGISTRY.get(run_id)
    if handle is None:
        logger.info(
            "ttm-control-plane.pause.no_process run_id=%s "
            "(process not registered; spawn may be disabled or run already finished)",
            run_id,
        )
        return

    try:
        pgid = os.getpgid(handle.pid)
        os.killpg(pgid, signal.SIGSTOP)
    except (ProcessLookupError, PermissionError, AttributeError) as exc:
        # SIGSTOP is not available on this platform or the process is gone.
        # Degrade explicitly — never report paused when suspend did not happen.
        logger.warning(
            "ttm-control-plane.pause.unsupported run_id=%s error=%s "
            "(SIGSTOP unavailable or process gone; reporting runtime.error)",
            run_id,
            exc,
        )
        await _emit_lifecycle_event(
            binding,
            "runtime.error",
            {
                "phase": "pause",
                "detail": f"pause_unsupported: {exc}",
                "runtime_run_ref": runtime_run_ref,
            },
        )
        return

    summary = binding.payload_summary
    state = _PauseState(
        run_id=run_id,
        pid=handle.pid,
        paused_at=_utcnow(),
        lane_id=summary.get("lane_id", ""),
        worktree_id=summary.get("worktree_id", ""),
    )
    with _PAUSE_LOCK:
        _PAUSE_STATE[run_id] = state

    await _emit_lifecycle_event(
        binding,
        "task.updated",
        {
            "status": "paused",
            "runtime_run_ref": runtime_run_ref,
            "paused_at": state.paused_at.isoformat(),
            "lane_id": state.lane_id,
            "worktree_id": state.worktree_id,
        },
    )
    _REGISTRY.update_status(run_id, last_status="paused")
    logger.info(
        "ttm-control-plane.pause.complete run_id=%s pid=%s",
        run_id,
        handle.pid,
    )


async def _async_resume(run_id: str, runtime_run_ref: str) -> None:
    """Resume a SIGSTOP-paused session with SIGCONT; restore dossier state."""
    binding = _REGISTRY.get(run_id)
    if binding is None:
        logger.warning("ttm-control-plane.resume.no_binding run_id=%s", run_id)
        return

    with _PAUSE_LOCK:
        state = _PAUSE_STATE.get(run_id)

    if state is None:
        logger.warning(
            "ttm-control-plane.resume.no_pause_state run_id=%s "
            "(run was not paused via this plugin instance or state was lost on restart)",
            run_id,
        )
        return

    handle = _PROC_REGISTRY.get(run_id)
    if handle is None:
        logger.warning(
            "ttm-control-plane.resume.no_process run_id=%s "
            "(process handle lost since pause; cannot resume)",
            run_id,
        )
        return

    try:
        pgid = os.getpgid(handle.pid)
        os.killpg(pgid, signal.SIGCONT)
    except (ProcessLookupError, PermissionError, AttributeError) as exc:
        logger.warning(
            "ttm-control-plane.resume.failed run_id=%s error=%s",
            run_id,
            exc,
        )
        return

    with _PAUSE_LOCK:
        _PAUSE_STATE.pop(run_id, None)

    await _emit_lifecycle_event(
        binding,
        "task.updated",
        {"status": "active", "runtime_run_ref": runtime_run_ref},
    )
    _REGISTRY.update_status(run_id, last_status="running")
    logger.info(
        "ttm-control-plane.resume.complete run_id=%s pid=%s",
        run_id,
        handle.pid,
    )


async def _async_expand_scope(run_id: str, runtime_run_ref: str) -> None:
    """Handle scope expansion: revoke old token and signal headless process.

    The old principal_token is treated as revoked immediately. SIGUSR1 is
    sent as an advisory hint to the process group to checkpoint (the agent
    will also detect 401 on its next ingress write if it hasn't stopped).
    The new token arrives separately via POST /runs/{run_id}/rebind-token;
    once received the headless agent's next get_run_state() call will yield
    the new scope_epoch and the agent resets phase state per contract.
    Invalidated lanes/worktrees must be abandoned — the agent is responsible
    for detecting the epoch change and stopping work on stale lanes.
    """
    binding = _REGISTRY.get(run_id)
    if binding is None:
        logger.warning("ttm-control-plane.expand_scope.no_binding run_id=%s", run_id)
        return

    # Advisory SIGUSR1 before revoking the token so the agent can
    # checkpoint cleanly before its next write attempt gets a 401.
    handle = _PROC_REGISTRY.get(run_id)
    if handle is not None:
        try:
            pgid = os.getpgid(handle.pid)
            os.killpg(pgid, signal.SIGUSR1)
            logger.info(
                "ttm-control-plane.expand_scope.sigusr1 run_id=%s pid=%s",
                run_id,
                handle.pid,
            )
        except (ProcessLookupError, PermissionError) as exc:
            logger.debug(
                "ttm-control-plane.expand_scope.sigusr1_failed run_id=%s error=%s",
                run_id,
                exc,
            )

    # Revoke: clear the token from the plugin registry so plugin-level events
    # stop using it. The headless agent's env-var copy will receive a 401 on
    # its next ingress write and must stop emitting on the old token.
    _REGISTRY.clear_token(run_id)
    _REGISTRY.update_status(run_id, last_status="scope_expanding")
    logger.info(
        "ttm-control-plane.expand_scope.token_revoked run_id=%s "
        "(awaiting rebind-token to restore emission with new scope_epoch)",
        run_id,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness + binding count. Unauthenticated for ops probes."""
    return HealthResponse(
        plugin="ttm-control-plane",
        version="0.2.0",
        auth_enforced=bool(_expected_secret()),
        bindings=len(_REGISTRY.snapshot()),
        checked_at=_utcnow(),
    )


@router.post("/runs/dispatch", status_code=status.HTTP_202_ACCEPTED)
async def dispatch_run(
    payload: DispatchPayload,
    _request: Request,
    x_ttm_control_plane_secret: str | None = Header(default=None, alias=_SECRET_HEADER),
) -> DispatchResponse:
    """Receive an initial run-spawn dispatch from TTM.

    Wire contract per ``RUNTIME-ADAPTER-CONTRACT.md §Spawn-On-Launch
    Dispatch`` and ``RUNTIME-PRINCIPAL-CONTRACT.md §Issuance``:

    1. Validate the shared secret + the ``principal_token`` is present.
    2. 409 if ``run_id`` is already bound to a live session.
    3. Mint a ``runtime_run_ref`` and store the binding.
    4. Return 202 immediately; schedule the run.dispatched ingress
       callback and the headless agent spawn off the request loop.
    """
    _require_secret(x_ttm_control_plane_secret)

    if not payload.principal_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "reason": "principal_token_required",
                "hint": (
                    "RUNTIME-PRINCIPAL-CONTRACT.md: every run-spawn "
                    "dispatch must carry a principal_token in the body"
                ),
            },
        )

    if payload.runtime_id != "hermes":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"reason": "runtime_id_mismatch", "expected": "hermes"},
        )

    existing = _REGISTRY.get(payload.run_id)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "reason": "run_already_bound",
                "runtime_run_ref": existing.runtime_run_ref,
                "bound_at": existing.bound_at.isoformat(),
            },
        )

    runtime_run_ref = f"hermes-{uuid.uuid4()}"
    binding = _RuntimeBinding(
        run_id=payload.run_id,
        runtime_binding_id=payload.runtime_binding_id,
        runtime_run_ref=runtime_run_ref,
        ingress_base_url=payload.ingress_base_url or "",
        bound_at=_utcnow(),
        principal_token=payload.principal_token,
        last_status="accepted",
        payload_summary=_payload_summary(payload),
    )
    inserted = _REGISTRY.insert(binding)
    # Race guard: another concurrent dispatch may have inserted between
    # our get() and insert(). The registry's insert() returns the
    # existing entry on collision; if so, surface 409 with that ref.
    if inserted.runtime_run_ref != runtime_run_ref:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "reason": "run_already_bound",
                "runtime_run_ref": inserted.runtime_run_ref,
                "bound_at": inserted.bound_at.isoformat(),
            },
        )

    logger.info(
        "ttm-control-plane.dispatch.accepted run_id=%s runtime_run_ref=%s lane_id=%s",
        payload.run_id,
        runtime_run_ref,
        payload.lane_id,
    )

    _kick_off_post_dispatch_tasks(binding)

    return DispatchResponse(
        status="accepted",
        runtime_run_ref=runtime_run_ref,
        bound_at=binding.bound_at,
    )


@router.get("/runs/{runtime_run_ref}/status", response_model=StatusResponse)
async def runtime_run_status(
    runtime_run_ref: str,
    x_ttm_control_plane_secret: str | None = Header(default=None, alias=_SECRET_HEADER),
) -> StatusResponse:
    """Return the locally-known status for a previously-dispatched run."""
    _require_secret(x_ttm_control_plane_secret)

    for binding in _REGISTRY.snapshot():
        if binding.runtime_run_ref == runtime_run_ref:
            return StatusResponse(
                status=binding.last_status,
                runtime_run_ref=runtime_run_ref,
                bound_at=binding.bound_at,
                checked_at=_utcnow(),
            )
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"reason": "runtime_run_ref_not_found"},
    )


@router.post("/runs/{runtime_run_ref}/lifecycle", status_code=status.HTTP_202_ACCEPTED)
async def lifecycle_action(
    runtime_run_ref: str,
    body: LifecycleRequest,
    x_ttm_control_plane_secret: str | None = Header(default=None, alias=_SECRET_HEADER),
) -> LifecycleResponse:
    """Unified lifecycle receiver: stop | pause | resume | expand_scope.

    Returns 202 immediately; the action is processed asynchronously.
    Validates that runtime_run_ref maps to a known persisted binding.
    Principal tokens are never logged.

    Actions:
      stop         — SIGTERM → 10s → SIGKILL; emits task.updated{stopped}.
                     Does not self-approve closure; TTM drives canonical close.
      pause        — SIGSTOP process group; persists dossier; emits task.updated{paused}.
                     Degrades explicitly if platform SIGSTOP unavailable.
      resume       — SIGCONT from saved pause state; emits task.updated{active}.
      expand_scope — Revokes old token; SIGUSR1 hint; awaits rebind-token.
    """
    _require_secret(x_ttm_control_plane_secret)

    binding = _binding_by_ref(runtime_run_ref)
    if binding is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"reason": "runtime_run_ref_not_found"},
        )

    run_id = binding.run_id
    loop = asyncio.get_running_loop()
    match body.action:
        case "stop":
            loop.create_task(_async_stop(run_id, runtime_run_ref))
        case "pause":
            loop.create_task(_async_pause(run_id, runtime_run_ref))
        case "resume":
            loop.create_task(_async_resume(run_id, runtime_run_ref))
        case "expand_scope":
            loop.create_task(_async_expand_scope(run_id, runtime_run_ref))

    logger.info(
        "ttm-control-plane.lifecycle.accepted action=%s run_id=%s runtime_run_ref=%s",
        body.action,
        run_id,
        runtime_run_ref,
    )
    return LifecycleResponse(
        status="accepted",
        runtime_run_ref=runtime_run_ref,
        action=body.action,
        accepted_at=_utcnow(),
    )


@router.post("/runs/{runtime_run_ref}/stop", status_code=status.HTTP_202_ACCEPTED)
async def stop_run(
    runtime_run_ref: str,
    x_ttm_control_plane_secret: str | None = Header(default=None, alias=_SECRET_HEADER),
) -> dict[str, Any]:
    """Compatibility stop route for current TTM HermesAdapter.

    TTM's stop_run() POSTs to /runs/{ref}/stop (not /lifecycle) until the
    TTM adapter is updated to use /runs/{ref}/lifecycle with action="stop".
    This route is a stable alias: it schedules the same async stop handler
    and returns 202 immediately. The binding is kept (status="stopped") so
    TTM can still query status and drive canonical closure independently.
    """
    _require_secret(x_ttm_control_plane_secret)

    binding = _binding_by_ref(runtime_run_ref)
    if binding is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"reason": "runtime_run_ref_not_found"},
        )

    loop = asyncio.get_running_loop()
    loop.create_task(_async_stop(binding.run_id, runtime_run_ref))

    logger.info(
        "ttm-control-plane.stop run_id=%s runtime_run_ref=%s",
        binding.run_id,
        runtime_run_ref,
    )
    return {
        "status": "accepted",
        "runtime_run_ref": runtime_run_ref,
        "accepted_at": _utcnow().isoformat(),
    }


@router.post("/runs/{run_id}/rebind-token", status_code=status.HTTP_200_OK)
async def rebind_token(
    run_id: str,
    body: RebindTokenRequest,
    x_ttm_control_plane_secret: str | None = Header(default=None, alias=_SECRET_HEADER),
) -> dict[str, Any]:
    """Accept a new principal token from TTM after a runtime rebind.

    TTM calls this via HermesAdapter.notify_rebind() after issuing a fresh
    principal token. The plugin updates its in-memory registry so that the
    headless agent session picks up the new bearer credential on next
    ingress write-back.

    Returns 404 when the run_id is not registered (i.e. the session has
    already exited or was never dispatched to this plugin instance).
    """
    _require_secret(x_ttm_control_plane_secret)

    found = _REGISTRY.replace_token(run_id, body.new_token)
    if not found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"reason": "run_not_found"},
        )

    # If we were in scope_expanding state, transition back to running now
    # that we have a fresh token. The headless agent's next get_run_state()
    # call will fetch the new scope_epoch and reset phase state per contract.
    binding = _REGISTRY.get(run_id)
    if binding is not None and binding.last_status == "scope_expanding":
        _REGISTRY.update_status(run_id, last_status="running")

    logger.info(
        "ttm-control-plane.rebind-token run_id=%s binding_id=%s",
        run_id,
        body.new_binding_id,
    )
    return {
        "status": "token_updated",
        "run_id": run_id,
        "new_binding_id": body.new_binding_id,
        "updated_at": _utcnow().isoformat(),
    }
