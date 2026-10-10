"""Guarded, atomic recovery for one exact Kanban task/PR lane.

This module is intentionally not wired into ordinary dispatcher scanning.  It
implements the one explicitly authorised same-card resume operation described by
``openspec/changes/kanban-same-lane-resume-repair`` while leaving the generic
active-PR guard untouched.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import os
import re
import select
import signal
import sqlite3
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Task

_AUTHORISED_TASK_ID = "t_dfa23a41"
_AUTHORISED_PR_URL = "https://github.com/NXE-ORG/nxe-helix-alpha/pull/155"
_AUTHORISED_PR_NUMBER = 155
_AUTHORISED_ASSIGNEE = "flynn"
_AUTHORISED_BASE = "dev"
_AUTHORISED_HEAD_REF = "wt/finance-pr153-producer-output-closure"
_AUTHORISED_HEAD_OID = "1b910d4ff9d1c9fe103da5df84d302ba81fc2cc1"
_AUTHORISED_WORKSPACE = "/home/hermes/nxe-helix-alpha/.worktrees/t_dfa23a41"
_AUTHORISED_UPSTREAM = "origin/wt/finance-pr153-producer-output-closure"
_AUTHORISED_RUN_ID = 1144
_AUTHORISED_OUTCOME = "changes_requested"
_PR_URL_RE = re.compile(r"https?://github\.com/[^/\s]+/[^/\s]+/pull/\d+", re.IGNORECASE)
_HOUSEKEEPING_EVENTS = {"claimed", "heartbeat", "reclaimed", "spawn_failed"}
_HOUSEKEEPING_OUTCOMES = {"reclaimed", "spawn_failed"}
_TERMINAL_RECEIPTS = {"spawned", "spawn_failed", "cleanup_failed"}
_PR_GET_PDEATHSIG = 2


@dataclass(frozen=True)
class SameLaneResumeRequest:
    task_id: str
    authorized_assignee: str
    expected_pr_url: str
    expected_pr_number: int
    expected_pr_base: str
    expected_pr_head_ref: str
    expected_pr_head_oid: str
    expected_workspace: str
    expected_upstream_ref: str
    required_substantive_run_id: int
    required_substantive_outcome: str
    resume_authorization_id: str


@dataclass(frozen=True)
class SameLaneResumeObservation:
    pr_url: str
    pr_number: int
    pr_state: str
    pr_base: str
    pr_head_ref: str
    pr_head_oid: str
    workspace: str
    workspace_registered: bool
    branch: str
    clean: bool
    local_oid: str
    upstream_ref: str
    upstream_oid: str


@dataclass(frozen=True)
class GatedChildIdentity:
    pid: int
    process_started_at: int
    process_group_id: int
    session_id: str
    ready_but_gated: bool = True


class GatedChild(Protocol):
    identity: GatedChildIdentity

    def release(self, *, task_id: str, run_id: int, authorization_id: str) -> bool:
        """Release only after publication; return after durable child acknowledgement."""
        raise NotImplementedError

    def terminate_and_confirm(self) -> bool:
        """Terminate the entire child process group/tree and prove extinction."""
        raise NotImplementedError


def _proc_rows() -> dict[int, tuple[int, int, int]]:
    """Return PID -> (PPID, process group, start time) from one /proc snapshot."""
    rows: dict[int, tuple[int, int, int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        with contextlib.suppress(OSError, ValueError, IndexError):
            raw = (entry / "stat").read_text(encoding="utf-8")
            # /proc/<pid>/stat field 2 (comm) is parenthesized and may contain
            # spaces or ')'.  The final ')' is therefore the only safe split.
            fields = raw.rsplit(")", 1)[1].strip().split()
            rows[int(entry.name)] = (int(fields[1]), int(fields[2]), int(fields[19]))
    return rows


def _descendant_identities(root_pid: int) -> set[tuple[int, int]]:
    rows = _proc_rows()
    children: dict[int, list[int]] = {}
    for pid, (ppid, _pgrp, _started_at) in rows.items():
        children.setdefault(ppid, []).append(pid)
    descendants: set[tuple[int, int]] = set()
    pending = list(children.get(root_pid, ()))
    while pending:
        pid = pending.pop()
        row = rows.get(pid)
        if row is None:
            continue
        descendants.add((pid, row[2]))
        pending.extend(children.get(pid, ()))
    return descendants


def _exact_process_is_live(pid: int, started_at: int) -> bool:
    row = _proc_rows().get(pid)
    return row is not None and row[2] == started_at


def _terminate_identity_tree(
    identity: GatedChildIdentity, *, process=None, descendants_forbidden: bool = False,
) -> bool:
    """Terminate the exact leader, its process group, and observed descendants."""
    leader_was_live = _exact_process_is_live(identity.pid, identity.process_started_at)
    if not leader_was_live and not descendants_forbidden:
        # Once a leader has disappeared, an escaped/reparented descendant cannot
        # be rediscovered from PID/PGID evidence. Fail closed unless the gated
        # protocol proves descendant creation was impossible in this phase.
        return False
    tracked: set[tuple[int, int]] = {(identity.pid, identity.process_started_at)}
    for sig, timeout in ((signal.SIGTERM, 3.0), (signal.SIGKILL, 3.0)):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # Capture descendants before signalling. Re-scan while waiting so a
            # descendant that calls setsid() cannot escape group cleanup.
            for pid, started_at in tuple(tracked):
                if _exact_process_is_live(pid, started_at):
                    tracked.update(_descendant_identities(pid))
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                os.killpg(identity.process_group_id, sig)
            for pid, started_at in tuple(tracked):
                if _exact_process_is_live(pid, started_at):
                    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                        os.kill(pid, sig)
            if process is not None:
                with contextlib.suppress(Exception):
                    process.poll()
            group_live = any(
                pgrp == identity.process_group_id
                for _pid, (_ppid, pgrp, _started_at) in _proc_rows().items()
            )
            if not group_live and not any(_exact_process_is_live(*item) for item in tracked):
                if process is not None:
                    with contextlib.suppress(Exception):
                        process.wait(timeout=0.1)
                return True
            time.sleep(0.05)
    group_live = any(
        pgrp == identity.process_group_id
        for _pid, (_ppid, pgrp, _started_at) in _proc_rows().items()
    )
    return not group_live and not any(_exact_process_is_live(*item) for item in tracked)


class LinuxGatedChild:
    """Controller half of the Linux gate/ack pipe pair."""

    def __init__(self, process, gate_fd: int, ack_fd: int, identity: GatedChildIdentity):
        self._process = process
        self._gate_fd = gate_fd
        self._ack_fd = ack_fd
        self.identity = identity

    def _read_ack(self, timeout: float) -> dict:
        ready, _, _ = select.select([self._ack_fd], [], [], timeout)
        if not ready:
            raise TimeoutError("gated child acknowledgement timed out")
        data = bytearray()
        while True:
            chunk = os.read(self._ack_fd, 1)
            if not chunk:
                raise RuntimeError("gated child exited before acknowledgement")
            if chunk == b"\n":
                value = json.loads(data.decode("utf-8"))
                if not isinstance(value, dict):
                    raise RuntimeError("invalid gated child acknowledgement")
                return value
            data.extend(chunk)
            if len(data) > 8192:
                raise RuntimeError("gated child acknowledgement too large")

    def release(self, *, task_id: str, run_id: int, authorization_id: str) -> bool:
        token = json.dumps(
            {
                "commit": True,
                "task_id": task_id,
                "run_id": run_id,
                "authorization_id": authorization_id,
            },
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        os.write(self._gate_fd, token)
        ack = self._read_ack(15.0)
        return ack == {"released": True}

    @staticmethod
    def _group_members(process_group_id: int) -> list[int]:
        return [
            pid for pid, (_ppid, pgrp, _started_at) in _proc_rows().items()
            if pgrp == process_group_id
        ]

    def terminate_and_confirm(self) -> bool:
        with contextlib.suppress(OSError):
            os.close(self._gate_fd)
        self._gate_fd = -1
        return _terminate_identity_tree(
            self.identity, process=self._process, descendants_forbidden=True
        )

    def close(self) -> None:
        for fd_name in ("_gate_fd", "_ack_fd"):
            fd = getattr(self, fd_name)
            if fd >= 0:
                with contextlib.suppress(OSError):
                    os.close(fd)
                setattr(self, fd_name, -1)


@dataclass(frozen=True)
class SameLaneResumeOutcome:
    disposition: str
    reason: Optional[str] = None
    task: Optional["Task"] = None
    run_id: Optional[int] = None


class SameLaneLaunchError(RuntimeError):
    """Launch failed with an explicit process-cleanup proof result."""

    def __init__(self, message: str, *, cleanup_confirmed: bool):
        super().__init__(message)
        self.cleanup_confirmed = cleanup_confirmed


@dataclass(frozen=True)
class SameLaneGuardSnapshot:
    task: "Task"
    observation: SameLaneResumeObservation
    substantive_run_id: int


def authorised_same_lane_request(authorization_id: str) -> SameLaneResumeRequest:
    """Build the only request authorised by the approved OpenSpec."""
    if not authorization_id or not authorization_id.strip():
        raise ValueError("resume_authorization_id is required")
    return SameLaneResumeRequest(
        task_id=_AUTHORISED_TASK_ID,
        authorized_assignee=_AUTHORISED_ASSIGNEE,
        expected_pr_url=_AUTHORISED_PR_URL,
        expected_pr_number=_AUTHORISED_PR_NUMBER,
        expected_pr_base=_AUTHORISED_BASE,
        expected_pr_head_ref=_AUTHORISED_HEAD_REF,
        expected_pr_head_oid=_AUTHORISED_HEAD_OID,
        expected_workspace=_AUTHORISED_WORKSPACE,
        expected_upstream_ref=_AUTHORISED_UPSTREAM,
        required_substantive_run_id=_AUTHORISED_RUN_ID,
        required_substantive_outcome=_AUTHORISED_OUTCOME,
        resume_authorization_id=authorization_id.strip(),
    )


def _request_is_authorised(request: SameLaneResumeRequest) -> bool:
    return request == authorised_same_lane_request(request.resume_authorization_id)


def _event_payload(row: sqlite3.Row) -> dict:
    raw = row["payload"]
    if not raw:
        return {}
    with contextlib.suppress(TypeError, ValueError):
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    return {}


def _latest_same_lane_receipt(
    conn: sqlite3.Connection, task_id: str,
) -> Optional[tuple[str, dict]]:
    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? "
        "AND kind IN ('resume_quarantined', 'spawned', 'spawn_failed', 'cleanup_failed') "
        "ORDER BY id DESC",
        (task_id,),
    ).fetchall()
    for row in rows:
        payload = _event_payload(row)
        if payload.get("same_lane_resume") and payload.get("resume_authorization_id"):
            return row["kind"], payload
    return None


def _authorization_receipt(
    conn: sqlite3.Connection, request: SameLaneResumeRequest,
) -> Optional[tuple[str, dict]]:
    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? "
        "AND kind IN ('resume_quarantined', 'spawned', 'spawn_failed', 'cleanup_failed') "
        "ORDER BY id DESC",
        (request.task_id,),
    ).fetchall()
    for row in rows:
        payload = _event_payload(row)
        if payload.get("resume_authorization_id") == request.resume_authorization_id:
            return row["kind"], payload
    return None


def _task_pr_urls(conn: sqlite3.Connection, task_id: str) -> set[str]:
    urls: set[str] = set()
    for row in conn.execute(
        "SELECT body AS text FROM task_comments WHERE task_id = ? UNION ALL "
        "SELECT payload AS text FROM task_events WHERE task_id = ?",
        (task_id, task_id),
    ).fetchall():
        urls.update(match.rstrip(".,);]") for match in _PR_URL_RE.findall(row["text"] or ""))
    task = conn.execute("SELECT body FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if task:
        urls.update(match.rstrip(".,);]") for match in _PR_URL_RE.findall(task["body"] or ""))
    return urls


def _run_has_worker_evidence(conn: sqlite3.Connection, run: sqlite3.Row) -> bool:
    if run["worker_pid"] is not None or run["summary"] or run["error"]:
        return True
    metadata = run["metadata"]
    if metadata and metadata not in ("{}", "null"):
        return True
    events = conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? AND run_id = ?",
        (run["task_id"], int(run["id"])),
    ).fetchall()
    kinds = {row["kind"] for row in events}
    if "spawned" in kinds or "resume_released" in kinds:
        return True
    return not kinds.issubset(_HOUSEKEEPING_EVENTS)


def _latest_substantive_run(conn: sqlite3.Connection, task_id: str) -> Optional[sqlite3.Row]:
    runs = conn.execute(
        "SELECT * FROM task_runs WHERE task_id = ? AND ended_at IS NOT NULL "
        "ORDER BY ended_at DESC, id DESC",
        (task_id,),
    ).fetchall()
    for run in runs:
        skip = (
            run["outcome"] in _HOUSEKEEPING_OUTCOMES
            and run["worker_pid"] is None
            and not _run_has_worker_evidence(conn, run)
        )
        if not skip:
            return run
    return None


def _has_required_review_event(
    conn: sqlite3.Connection, request: SameLaneResumeRequest,
) -> bool:
    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? AND run_id = ? "
        "ORDER BY id DESC",
        (request.task_id, request.required_substantive_run_id),
    ).fetchall()
    for row in rows:
        payload = _event_payload(row)
        if row["kind"] == "changes_requested":
            return True
        if payload.get("outcome") == "changes_requested" or payload.get("verdict") == "changes_requested":
            return True
    return False


def _validate_observation(
    request: SameLaneResumeRequest, observation: SameLaneResumeObservation,
) -> Optional[str]:
    expected = {
        "pr_url": request.expected_pr_url,
        "pr_number": request.expected_pr_number,
        "pr_state": "OPEN",
        "pr_base": request.expected_pr_base,
        "pr_head_ref": request.expected_pr_head_ref,
        "pr_head_oid": request.expected_pr_head_oid,
        "workspace": request.expected_workspace,
        "workspace_registered": True,
        "branch": request.expected_pr_head_ref,
        "clean": True,
        "local_oid": request.expected_pr_head_oid,
        "upstream_ref": request.expected_upstream_ref,
        "upstream_oid": request.expected_pr_head_oid,
    }
    for field, wanted in expected.items():
        observed = getattr(observation, field)
        if observed != wanted:
            return f"observation_{field}_mismatch"
    return None


def _validate_board_guards(
    conn: sqlite3.Connection,
    request: SameLaneResumeRequest,
    observation: SameLaneResumeObservation,
    *,
    expected_status: str,
) -> tuple[Optional[SameLaneGuardSnapshot], Optional[str]]:
    from hermes_cli import kanban_db as kb

    task = kb.get_task(conn, request.task_id)
    if task is None:
        return None, "task_missing"
    raw_task = conn.execute(
        "SELECT claim_lock, claim_expires, current_run_id, worker_pid, "
        "worker_started_at, session_id FROM tasks WHERE id = ?",
        (request.task_id,),
    ).fetchone()
    if task.status != expected_status:
        return None, "task_status_mismatch"
    if expected_status == "ready" and raw_task is not None and any(
        raw_task[column] is not None
        for column in (
            "claim_lock", "claim_expires", "current_run_id", "worker_pid",
            "worker_started_at", "session_id",
        )
    ):
        return None, "task_not_unclaimed"
    if expected_status == "blocked":
        if raw_task is not None and any(
            raw_task[column] is not None
            for column in (
                "claim_lock", "claim_expires", "current_run_id", "worker_pid",
                "worker_started_at", "session_id",
            )
        ):
            return None, "task_not_unclaimed"
        receipt = _authorization_receipt(conn, request)
        if receipt is None or receipt[0] != "resume_quarantined":
            return None, "quarantine_receipt_missing"
    if task.assignee != request.authorized_assignee:
        return None, "assignee_mismatch"
    if task.workspace_path != request.expected_workspace:
        return None, "workspace_metadata_mismatch"
    if task.branch_name != request.expected_pr_head_ref:
        return None, "branch_metadata_mismatch"

    competing = conn.execute(
        "SELECT id, status FROM tasks WHERE assignee = ? AND id != ? "
        "AND status NOT IN ('done', 'archived') ORDER BY id LIMIT 1",
        (request.authorized_assignee, request.task_id),
    ).fetchone()
    if competing is not None:
        return None, f"competing_lane:{competing['id']}:{competing['status']}"

    target_urls = _task_pr_urls(conn, request.task_id)
    if request.expected_pr_url not in target_urls:
        return None, "target_pr_ownership_missing"
    if any(url != request.expected_pr_url for url in target_urls):
        return None, "target_pr_ownership_conflict"
    owners = conn.execute(
        "SELECT id FROM tasks WHERE id != ? AND status NOT IN ('done', 'archived')",
        (request.task_id,),
    ).fetchall()
    if any(request.expected_pr_url in _task_pr_urls(conn, row["id"]) for row in owners):
        return None, "target_pr_owned_by_other_task"

    observation_reason = _validate_observation(request, observation)
    if observation_reason:
        return None, observation_reason

    substantive = _latest_substantive_run(conn, request.task_id)
    if substantive is None:
        return None, "substantive_run_missing"
    if int(substantive["id"]) != request.required_substantive_run_id:
        return None, "substantive_run_id_mismatch"
    if substantive["profile"] != "crash":
        return None, "substantive_run_profile_mismatch"
    if substantive["outcome"] != request.required_substantive_outcome:
        return None, "substantive_outcome_mismatch"
    if not _has_required_review_event(conn, request):
        return None, "substantive_review_event_missing"
    return SameLaneGuardSnapshot(task, observation, int(substantive["id"])), None


def _db_path(conn: sqlite3.Connection) -> Path:
    for _seq, name, filename in conn.execute("PRAGMA database_list"):
        if name == "main" and filename:
            return Path(filename)
    raise RuntimeError("same-lane resume requires a file-backed board")


class _ResumeFence:
    def __init__(self, conn: sqlite3.Connection, task_id: str, timeout_seconds: float = 10.0):
        digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:16]
        root = _db_path(conn).parent
        self.lock_path = root / f".same-lane-resume-{digest}.lock"
        self.metadata_path = root / f".same-lane-resume-{digest}.json"
        self.timeout_seconds = timeout_seconds
        self._handle = None

    def __enter__(self) -> "_ResumeFence":
        if not sys.platform.startswith("linux"):
            raise RuntimeError("same-lane resume requires Linux")
        import fcntl

        self._handle = self.lock_path.open("a+b")
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    self._handle.close()
                    self._handle = None
                    raise TimeoutError("same-lane resume fence is busy")
                time.sleep(0.05)

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        import fcntl

        if self._handle is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None

    def write_metadata(self, payload: dict) -> None:
        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        tmp = self.metadata_path.with_name(f"{self.metadata_path.name}.{os.getpid()}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, self.metadata_path)
        dir_fd = os.open(self.metadata_path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def clear_metadata(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            self.metadata_path.unlink()

    def read_metadata(self) -> Optional[dict]:
        with contextlib.suppress(OSError, ValueError, TypeError):
            value = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value
        return None


def _mark_cleanup_pending(fence: _ResumeFence) -> None:
    metadata = fence.read_metadata() or {}
    metadata.update({
        "phase": "cleanup_failed",
        "controller_pid": 0,
        "controller_started_at": 0,
    })
    fence.write_metadata(metadata)


def _process_identity_is_live(identity: GatedChildIdentity) -> bool:
    if identity.pid <= 0 or identity.process_started_at <= 0 or identity.process_group_id <= 0:
        return False
    return _exact_process_is_live(identity.pid, identity.process_started_at)


def _parent_death_signal_available() -> bool:
    value = ctypes.c_int()
    libc = ctypes.CDLL(None, use_errno=True)
    return libc.prctl(_PR_GET_PDEATHSIG, ctypes.byref(value), 0, 0, 0) == 0


def _observe_with_timeout(
    observer: Callable[[], SameLaneResumeObservation], timeout: float,
) -> SameLaneResumeObservation:
    result: list[SameLaneResumeObservation] = []
    error: list[BaseException] = []
    done = threading.Event()

    def run() -> None:
        try:
            result.append(observer())
        except BaseException as exc:  # propagate observer failures to the controller
            error.append(exc)
        finally:
            done.set()

    threading.Thread(target=run, name="same-lane-observer", daemon=True).start()
    if not done.wait(timeout):
        raise TimeoutError("observer timeout")
    if error:
        raise error[0]
    return result[0]


def _terminate_recorded_identity(
    identity: GatedChildIdentity, *, descendants_forbidden: bool = False,
) -> bool:
    """Kill and prove extinction of a crash-recovered process group/tree."""
    if not _process_identity_is_live(identity):
        group_empty = not LinuxGatedChild._group_members(identity.process_group_id)
        return descendants_forbidden and group_empty
    return _terminate_identity_tree(
        identity, descendants_forbidden=descendants_forbidden
    )


def _recover_quarantined_attempt(
    conn: sqlite3.Connection,
    request: SameLaneResumeRequest,
    fence: _ResumeFence,
) -> SameLaneResumeOutcome:
    """Reconcile a crash-surviving quarantine before any retry can launch."""
    from gateway.status import get_process_start_time

    metadata = fence.read_metadata()
    if metadata is None:
        return _compensate_unpublished(
            conn,
            request,
            phase="crash_recovery",
            error="resume fence metadata missing; cleanup cannot be proven",
            cleanup_confirmed=False,
        )
    owner_pid = int(metadata.get("controller_pid") or 0)
    owner_start = int(metadata.get("controller_started_at") or 0)
    if owner_pid > 0 and get_process_start_time(owner_pid) == owner_start:
        return SameLaneResumeOutcome("in_progress", "recorded_controller_is_live")
    pid = int(metadata.get("pid") or 0)
    if pid <= 0:
        if time.time() <= float(metadata.get("gate_deadline") or 0):
            return SameLaneResumeOutcome("in_progress", "pre_identity_gate_deadline_not_elapsed")
        outcome = _compensate_unpublished(
            conn,
            request,
            phase="crash_recovery",
            error="controller died before child identity publication",
            cleanup_confirmed=True,
        )
        fence.clear_metadata()
        return outcome
    identity = GatedChildIdentity(
        pid=pid,
        process_started_at=int(metadata.get("process_started_at") or 0),
        process_group_id=int(metadata.get("process_group_id") or 0),
        session_id=str(metadata.get("session_id") or ""),
    )
    cleanup_confirmed = not _process_identity_is_live(identity) and not LinuxGatedChild._group_members(
        identity.process_group_id
    )
    if not cleanup_confirmed:
        cleanup_confirmed = _terminate_recorded_identity(
            identity, descendants_forbidden=True
        )
    outcome = _compensate_unpublished(
        conn,
        request,
        phase="crash_recovery",
        error="reconciled controller crash during gated launch",
        cleanup_confirmed=cleanup_confirmed,
    )
    if cleanup_confirmed:
        fence.clear_metadata()
    else:
        _mark_cleanup_pending(fence)
    return outcome


def _validate_child_identity(
    conn: sqlite3.Connection, identity: GatedChildIdentity,
) -> Optional[str]:
    if not identity.ready_but_gated:
        return "child_not_gated"
    if not identity.session_id.strip():
        return "child_session_missing"
    if not _process_identity_is_live(identity):
        return "child_identity_not_live"
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        if os.getpgid(identity.pid) != identity.process_group_id:
            return "child_process_group_mismatch"
    duplicate = conn.execute(
        "SELECT id FROM tasks WHERE session_id = ? LIMIT 1",
        (identity.session_id,),
    ).fetchone()
    if duplicate is not None:
        return "child_session_duplicate"
    return None


def _quarantine(
    conn: sqlite3.Connection, request: SameLaneResumeRequest,
) -> bool:
    from hermes_cli import kanban_db as kb

    now = int(time.time())
    cur = conn.execute(
        "UPDATE tasks SET status = 'blocked', block_kind = 'resume_quarantined', "
        "last_failure_error = 'resume_quarantined' "
        "WHERE id = ? AND status = 'ready' AND claim_lock IS NULL "
        "AND current_run_id IS NULL AND worker_pid IS NULL AND worker_started_at IS NULL "
        "AND session_id IS NULL",
        (request.task_id,),
    )
    if cur.rowcount != 1:
        return False
    kb._append_event(
        conn,
        request.task_id,
        "resume_quarantined",
        {
            "same_lane_resume": True,
            "resume_authorization_id": request.resume_authorization_id,
            "at": now,
        },
        run_id=None,
    )
    return True


def _publish(
    conn: sqlite3.Connection,
    request: SameLaneResumeRequest,
    identity: GatedChildIdentity,
    *,
    ttl_seconds: Optional[int],
    publication_hook: Optional[Callable[[sqlite3.Connection, int], None]],
) -> int:
    from hermes_cli import kanban_db as kb

    now = int(time.time())
    lock = kb._claimer_id()
    expires = now + kb._resolve_claim_ttl_seconds(ttl_seconds)
    cur = conn.execute(
        "UPDATE tasks SET status = 'running', claim_lock = ?, claim_expires = ?, "
        "started_at = COALESCE(started_at, ?), worker_pid = ?, worker_started_at = ?, "
        "session_id = ?, block_kind = NULL, last_failure_error = NULL "
        "WHERE id = ? AND status = 'blocked' AND block_kind = 'resume_quarantined' "
        "AND claim_lock IS NULL AND current_run_id IS NULL AND worker_pid IS NULL",
        (
            lock,
            expires,
            now,
            identity.pid,
            identity.process_started_at,
            identity.session_id,
            request.task_id,
        ),
    )
    if cur.rowcount != 1:
        raise RuntimeError("quarantine publication CAS lost")
    task = kb.get_task(conn, request.task_id)
    run_cur = conn.execute(
        "INSERT INTO task_runs (task_id, profile, step_key, status, claim_lock, "
        "claim_expires, worker_pid, max_runtime_seconds, started_at) "
        "VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?)",
        (
            request.task_id,
            request.authorized_assignee,
            task.current_step_key if task else None,
            lock,
            expires,
            identity.pid,
            task.max_runtime_seconds if task else None,
            now,
        ),
    )
    if run_cur.lastrowid is None:
        raise RuntimeError("same-lane publication did not create a run id")
    run_id = int(run_cur.lastrowid)
    conn.execute("UPDATE tasks SET current_run_id = ? WHERE id = ?", (run_id, request.task_id))
    common = {
        "same_lane_resume": True,
        "resume_authorization_id": request.resume_authorization_id,
        "run_id": run_id,
    }
    kb._append_event(
        conn,
        request.task_id,
        "claimed",
        {**common, "lock": lock, "expires": expires},
        run_id=run_id,
    )
    kb._append_event(
        conn,
        request.task_id,
        "spawned",
        {
            **common,
            "pid": identity.pid,
            "started_at": identity.process_started_at,
            "process_group_id": identity.process_group_id,
            "session_id": identity.session_id,
        },
        run_id=run_id,
    )
    if publication_hook is not None:
        publication_hook(conn, run_id)
    return run_id


def _compensate_unpublished(
    conn: sqlite3.Connection,
    request: SameLaneResumeRequest,
    *,
    phase: str,
    error: str,
    cleanup_confirmed: bool,
) -> SameLaneResumeOutcome:
    from hermes_cli import kanban_db as kb

    with kb.write_txn(conn):
        if cleanup_confirmed:
            cur = conn.execute(
                "UPDATE tasks SET status = 'ready', claim_lock = NULL, claim_expires = NULL, "
                "current_run_id = NULL, worker_pid = NULL, worker_started_at = NULL, "
                "session_id = NULL, block_kind = NULL, last_failure_error = ? "
                "WHERE id = ? AND status = 'blocked' AND block_kind = 'resume_quarantined'",
                (f"same-lane {phase} failed: {error}"[:1000], request.task_id),
            )
            if cur.rowcount != 1:
                raise RuntimeError("quarantine compensation CAS lost")
            kind = "spawn_failed"
            disposition = "compensated"
        else:
            kind = "cleanup_failed"
            disposition = "cleanup_failed"
        kb._append_event(
            conn,
            request.task_id,
            kind,
            {
                "same_lane_resume": True,
                "resume_authorization_id": request.resume_authorization_id,
                "phase": phase,
                "error": error[:1000],
                "termination_confirmed": cleanup_confirmed,
            },
            run_id=None,
        )
    return SameLaneResumeOutcome(disposition, reason=error)


def _recover_published_release_failure(
    conn: sqlite3.Connection,
    request: SameLaneResumeRequest,
    run_id: int,
    *,
    error: str,
    cleanup_confirmed: bool,
) -> SameLaneResumeOutcome:
    from hermes_cli import kanban_db as kb

    with kb.write_txn(conn):
        if not cleanup_confirmed:
            kb._append_event(
                conn,
                request.task_id,
                "cleanup_failed",
                {
                    "same_lane_resume": True,
                    "resume_authorization_id": request.resume_authorization_id,
                    "run_id": run_id,
                    "phase": "gate_release",
                    "error": error[:1000],
                    "termination_confirmed": False,
                },
                run_id=run_id,
            )
            return SameLaneResumeOutcome("cleanup_failed", error, run_id=run_id)
        now = int(time.time())
        conn.execute(
            "UPDATE task_runs SET status = 'spawn_failed', outcome = 'spawn_failed', "
            "error = ?, ended_at = ?, claim_lock = NULL, claim_expires = NULL, worker_pid = NULL "
            "WHERE id = ? AND ended_at IS NULL",
            (error[:1000], now, run_id),
        )
        conn.execute(
            "UPDATE tasks SET status = 'ready', claim_lock = NULL, claim_expires = NULL, "
            "current_run_id = NULL, worker_pid = NULL, worker_started_at = NULL, "
            "session_id = NULL, last_failure_error = ? WHERE id = ? AND current_run_id = ?",
            (f"same-lane gate release failed: {error}"[:1000], request.task_id, run_id),
        )
        kb._append_event(
            conn,
            request.task_id,
            "spawn_failed",
            {
                "same_lane_resume": True,
                "resume_authorization_id": request.resume_authorization_id,
                "phase": "gate_release",
                "error": error[:1000],
                "termination_confirmed": True,
            },
            run_id=run_id,
        )
    return SameLaneResumeOutcome("compensated", error, run_id=run_id)


def _recover_published_attempt(
    conn: sqlite3.Connection,
    request: SameLaneResumeRequest,
    fence: _ResumeFence,
    receipt_payload: dict,
) -> SameLaneResumeOutcome:
    """Reconcile a controller crash after publication but before release ack."""
    from gateway.status import get_process_start_time
    from hermes_cli import kanban_db as kb

    run_id = int(receipt_payload.get("run_id") or 0)
    if run_id <= 0:
        return SameLaneResumeOutcome("cleanup_failed", "published receipt has no run id")
    released = conn.execute(
        "SELECT 1 FROM task_events WHERE task_id = ? AND run_id = ? "
        "AND kind = 'resume_released' LIMIT 1",
        (request.task_id, run_id),
    ).fetchone()
    if released is not None:
        fence.clear_metadata()
        return SameLaneResumeOutcome(
            "published", "authorization_receipt:resume_released",
            task=kb.get_task(conn, request.task_id), run_id=run_id,
        )

    metadata = fence.read_metadata() or {}
    owner_pid = int(metadata.get("controller_pid") or 0)
    owner_start = int(metadata.get("controller_started_at") or 0)
    if owner_pid > 0 and get_process_start_time(owner_pid) == owner_start:
        return SameLaneResumeOutcome("in_progress", "recorded_controller_is_live", run_id=run_id)

    row = conn.execute(
        "SELECT worker_pid, worker_started_at, session_id FROM tasks "
        "WHERE id = ? AND current_run_id = ?",
        (request.task_id, run_id),
    ).fetchone()
    identity = GatedChildIdentity(
        pid=int(metadata.get("pid") or (row["worker_pid"] if row else 0) or 0),
        process_started_at=int(
            metadata.get("process_started_at")
            or (row["worker_started_at"] if row else 0)
            or 0
        ),
        process_group_id=int(
            metadata.get("process_group_id")
            or receipt_payload.get("process_group_id")
            or 0
        ),
        session_id=str(metadata.get("session_id") or (row["session_id"] if row else "") or ""),
    )
    cleanup_confirmed = (
        identity.process_group_id > 0
        and not _process_identity_is_live(identity)
        and not LinuxGatedChild._group_members(identity.process_group_id)
    )
    if not cleanup_confirmed and identity.process_group_id > 0:
        cleanup_confirmed = _terminate_recorded_identity(
            identity, descendants_forbidden=True
        )
    outcome = _recover_published_release_failure(
        conn,
        request,
        run_id,
        error="reconciled controller crash after publication before release acknowledgement",
        cleanup_confirmed=cleanup_confirmed,
    )
    if cleanup_confirmed:
        fence.clear_metadata()
    else:
        _mark_cleanup_pending(fence)
    return outcome


def resume_same_lane(
    conn: sqlite3.Connection,
    request: SameLaneResumeRequest,
    *,
    observer: Callable[[], SameLaneResumeObservation],
    launcher: Optional[Callable[[SameLaneResumeRequest, "Task"], GatedChild]] = None,
    ttl_seconds: Optional[int] = None,
    publication_hook: Optional[Callable[[sqlite3.Connection, int], None]] = None,
    fence_timeout_seconds: float = 10.0,
    observer_timeout_seconds: float = 15.0,
    board: Optional[str] = None,
) -> SameLaneResumeOutcome:
    """Resume the exact authorised lane with gated launch and atomic publication.

    The task-scoped interprocess fence spans guard evaluation, quarantine,
    launch, publication, cleanup, and compensation.  The child cannot be
    released until the coherent task/run/claim/spawn transaction commits.
    """
    from gateway.status import get_process_start_time
    from hermes_cli import kanban_db as kb

    if not _request_is_authorised(request):
        return SameLaneResumeOutcome("refused", "request_not_authorised")
    if not sys.platform.startswith("linux"):
        return SameLaneResumeOutcome("refused", "unsupported_host")
    if not _parent_death_signal_available():
        return SameLaneResumeOutcome("refused", "parent_death_signal_unavailable")
    owner_start = get_process_start_time(os.getpid())
    if not owner_start:
        return SameLaneResumeOutcome("refused", "controller_identity_unavailable")

    child: Optional[GatedChild] = None
    publication_committed = False
    run_id: Optional[int] = None
    fence = _ResumeFence(conn, request.task_id, fence_timeout_seconds)
    try:
        with fence:
            receipt = _authorization_receipt(conn, request)
            if receipt is None:
                prior = _latest_same_lane_receipt(conn, request.task_id)
                if prior is not None:
                    prior_kind, prior_payload = prior
                    prior_auth = str(prior_payload.get("resume_authorization_id") or "")
                    if prior_auth != request.resume_authorization_id and prior_kind != "spawn_failed":
                        # Reconcile the prior attempt before a new authorization
                        # may replace any crash-surviving process identity.
                        prior_request = authorised_same_lane_request(prior_auth)
                        if prior_kind == "cleanup_failed" and prior_payload.get("run_id"):
                            return _recover_published_attempt(
                                conn, prior_request, fence, prior_payload
                            )
                        if prior_kind in {"resume_quarantined", "cleanup_failed"}:
                            return _recover_quarantined_attempt(conn, prior_request, fence)
                        if prior_kind == "spawned":
                            return _recover_published_attempt(
                                conn, prior_request, fence, prior_payload
                            )
            if receipt is not None:
                kind, payload = receipt
                if kind == "cleanup_failed" and payload.get("run_id"):
                    return _recover_published_attempt(conn, request, fence, payload)
                if kind in {"resume_quarantined", "cleanup_failed"}:
                    return _recover_quarantined_attempt(conn, request, fence)
                if kind == "spawned":
                    return _recover_published_attempt(conn, request, fence, payload)
                disposition = {
                    "spawn_failed": "compensated",
                }[kind]
                return SameLaneResumeOutcome(disposition, f"authorization_receipt:{kind}")

            from hermes_cli.kanban_db_lifecycle import normalize_ready_worker_start_residue

            normalization = normalize_ready_worker_start_residue(
                conn,
                request.task_id,
                resume_authorization_id=request.resume_authorization_id,
            )
            if normalization.disposition == "refused":
                return SameLaneResumeOutcome(
                    "refused", f"lifecycle_normalization_refused:{normalization.reason}"
                )

            try:
                observation = _observe_with_timeout(observer, observer_timeout_seconds)
            except Exception as exc:
                return SameLaneResumeOutcome("refused", f"observer_failed:{exc}")

            # Persist the fail-closed launch fence before quarantine becomes
            # durable.  A controller crash can therefore never leave a
            # quarantined card without enough phase/deadline evidence for the
            # next controller to reconcile it.
            nonce = uuid.uuid4().hex
            gate_deadline = int(time.time()) + 30
            fence.write_metadata({
                "resume_authorization_id": request.resume_authorization_id,
                "launch_nonce": nonce,
                "phase": "pre_quarantine",
                "controller_pid": os.getpid(),
                "controller_started_at": owner_start,
                "gate_deadline": gate_deadline,
            })
            with kb.write_txn(conn):
                snapshot, reason = _validate_board_guards(
                    conn, request, observation, expected_status="ready"
                )
                if not reason and not _quarantine(conn, request):
                    reason = "quarantine_cas_lost"
            if reason:
                fence.clear_metadata()
                return SameLaneResumeOutcome("refused", reason)
            assert snapshot is not None

            fence.write_metadata({
                "resume_authorization_id": request.resume_authorization_id,
                "launch_nonce": nonce,
                "phase": "launching",
                "controller_pid": os.getpid(),
                "controller_started_at": owner_start,
                "gate_deadline": gate_deadline,
            })

            try:
                if launcher is None:
                    from hermes_cli.kanban_db_dispatch import _launch_same_lane_worker
                    child = _launch_same_lane_worker(
                        request,
                        snapshot.task,
                        identity_path=str(fence.metadata_path),
                        launch_nonce=nonce,
                        deadline=gate_deadline,
                        controller_started_at=int(owner_start or 0),
                        board=board,
                    )
                else:
                    child = launcher(request, snapshot.task)
                identity_reason = _validate_child_identity(conn, child.identity)
                if identity_reason:
                    raise RuntimeError(identity_reason)
                fence.write_metadata({
                    "resume_authorization_id": request.resume_authorization_id,
                    "launch_nonce": nonce,
                    "phase": "gated",
                    "controller_pid": os.getpid(),
                    "controller_started_at": owner_start,
                    "gate_deadline": gate_deadline,
                    "pid": child.identity.pid,
                    "process_started_at": child.identity.process_started_at,
                    "process_group_id": child.identity.process_group_id,
                    "session_id": child.identity.session_id,
                })

                try:
                    second_observation = _observe_with_timeout(observer, observer_timeout_seconds)
                except Exception as exc:
                    raise RuntimeError(f"observer_revalidation_failed:{exc}") from exc
                with kb.write_txn(conn):
                    _snapshot, reason = _validate_board_guards(
                        conn, request, second_observation, expected_status="blocked"
                    )
                    if reason:
                        raise RuntimeError(f"guard_revalidation_failed:{reason}")
                    run_id = _publish(
                        conn,
                        request,
                        child.identity,
                        ttl_seconds=ttl_seconds,
                        publication_hook=publication_hook,
                    )
                publication_committed = True
                fence.write_metadata({
                    "resume_authorization_id": request.resume_authorization_id,
                    "launch_nonce": nonce,
                    "phase": "published",
                    "controller_pid": os.getpid(),
                    "controller_started_at": owner_start,
                    "gate_deadline": gate_deadline,
                    "run_id": run_id,
                    "pid": child.identity.pid,
                    "process_started_at": child.identity.process_started_at,
                    "process_group_id": child.identity.process_group_id,
                    "session_id": child.identity.session_id,
                })
                if not child.release(
                    task_id=request.task_id,
                    run_id=run_id,
                    authorization_id=request.resume_authorization_id,
                ):
                    raise RuntimeError("gate_release_not_acknowledged")
                released = conn.execute(
                    "SELECT 1 FROM task_events WHERE task_id = ? AND run_id = ? "
                    "AND kind = 'resume_released' LIMIT 1",
                    (request.task_id, run_id),
                ).fetchone()
                if released is None:
                    raise RuntimeError("child_acknowledged_without_resume_released")
                task = kb.get_task(conn, request.task_id)
                fence.clear_metadata()
                close_child = getattr(child, "close", None)
                if callable(close_child):
                    close_child()
                if task is not None:
                    kb._fire_task_hook("kanban_task_claimed", task, task.id, run_id)
                return SameLaneResumeOutcome("published", task=task, run_id=run_id)
            except Exception as exc:
                if publication_committed and run_id is not None:
                    released = conn.execute(
                        "SELECT 1 FROM task_events WHERE task_id = ? AND run_id = ? "
                        "AND kind = 'resume_released' LIMIT 1",
                        (request.task_id, run_id),
                    ).fetchone()
                    if released is not None:
                        # The child durably crossed the gate; it now belongs to
                        # ordinary worker lifecycle and must not be mistaken for
                        # an unpublished partial child after an ack race.
                        fence.clear_metadata()
                        close_child = getattr(child, "close", None)
                        if callable(close_child):
                            with contextlib.suppress(Exception):
                                close_child()
                        return SameLaneResumeOutcome(
                            "published",
                            "release_receipt_won_ack_race",
                            task=kb.get_task(conn, request.task_id),
                            run_id=run_id,
                        )
                cleanup_confirmed = (
                    bool(getattr(exc, "cleanup_confirmed", False))
                    if child is None else False
                )
                if child is not None:
                    with contextlib.suppress(Exception):
                        cleanup_confirmed = bool(child.terminate_and_confirm())
                    close_child = getattr(child, "close", None)
                    if callable(close_child):
                        with contextlib.suppress(Exception):
                            close_child()
                if publication_committed and run_id is not None:
                    outcome = _recover_published_release_failure(
                        conn,
                        request,
                        run_id,
                        error=str(exc),
                        cleanup_confirmed=cleanup_confirmed,
                    )
                else:
                    outcome = _compensate_unpublished(
                        conn,
                        request,
                        phase="publication" if child is not None else "launch",
                        error=str(exc),
                        cleanup_confirmed=cleanup_confirmed,
                    )
                if cleanup_confirmed:
                    fence.clear_metadata()
                else:
                    _mark_cleanup_pending(fence)
                return outcome
    except TimeoutError as exc:
        return SameLaneResumeOutcome("refused", f"lease_timeout:{exc}")
