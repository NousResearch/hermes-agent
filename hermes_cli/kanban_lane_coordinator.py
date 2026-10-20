"""Default-off reserve/spawn/bind coordinator for explicitly pinned workers.

This bounded integration deliberately does not resolve aliases, attest runtime
models, or authorize cross-provider routing. Those need separate provenance and
subscription transport evidence. An unresolved pin defers rather than bills an
alternative backend. Enable host-wide only after draining legacy dispatchers.
"""
from __future__ import annotations

import sqlite3
import uuid
from contextvars import ContextVar

current_settings = ContextVar("kanban_provider_lane_settings", default=None)
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from hermes_cli.kanban_lane_liveness import process_identity, reservation_alive, identity_alive
from hermes_cli.kanban_provider_lanes import Candidate, LaneLedger, load_router


def with_settings(raw, tick, *args, **kwargs):
    token = current_settings.set(raw)
    try:
        return tick(*args, **kwargs)
    finally:
        current_settings.reset(token)


class LaneDeferred(RuntimeError):
    """No worker started; retry without spending the task's failure budget."""


class LaneSpawnUncertain(RuntimeError):
    """A worker may exist; retain both reservation and board claim."""


@dataclass(frozen=True)
class LaneSettings:
    enabled: bool = False
    router_path: str = "~/.agents/model-monarchy/models.toml"
    minimum_free_bytes: int = 2 * 1024**3
    worker_headroom_bytes: int = 512 * 1024**2


def resolve_settings(raw=None) -> LaneSettings:
    if raw is None:
        from hermes_cli.config import load_config_readonly
        raw = (load_config_readonly() or {}).get("kanban", {}).get("provider_lanes", {})
    if not isinstance(raw, Mapping) or type(raw.get("enabled", False)) is not bool:
        raise LaneDeferred("invalid provider_lanes settings")
    if not raw.get("enabled", False):
        return LaneSettings()
    values = {}
    for key, default in (("minimum_free_bytes", 2 * 1024**3),
                         ("worker_headroom_bytes", 512 * 1024**2)):
        value = raw.get(key, default)
        if type(value) is not int or value <= 0:
            raise LaneDeferred(f"provider_lanes.{key} must be a positive integer")
        values[key] = value
    path = raw.get("router_path", LaneSettings.router_path)
    if not isinstance(path, str) or not path.strip():
        raise LaneDeferred("provider_lanes.router_path must be a path")
    return LaneSettings(enabled=True, router_path=path, **values)


def ledger_path() -> Path:
    from hermes_cli import kanban_db as kb
    return kb.kanban_home() / "kanban" / "provider-lanes.db"


def pinned_candidate(task, settings) -> Candidate:
    # Native anthropic OAuth can require paid extra usage. Only the explicit
    # CLI subscription provider may occupy the Claude lane; never substitute it.
    transports = {"openai-codex": "openai-codex",
                  "claude-subscription-directsdk-experimental": "anthropic"}
    lane = transports.get(task.provider_override)
    if not lane or not task.model_override:
        raise LaneDeferred("explicit compatible subscription provider/model pin required")
    try:
        routes = load_router(Path(settings.router_path).expanduser())
    except (OSError, ValueError) as exc:
        raise LaneDeferred("router unavailable or invalid") from exc
    # Exact pins also protect gate/release work in otherwise ordinary profiles.
    # No role/title heuristic is allowed to weaken an explicit pin.
    if not any(c.provider == lane and c.model == task.model_override
               for candidates in routes.values() for c in candidates):
        raise LaneDeferred("exact provider/model pin absent from router")
    return Candidate(lane, task.model_override, task.reasoning_effort)


def board_path(conn) -> str:
    rows = conn.execute("PRAGMA database_list").fetchall()
    path = next((row[2] for row in rows if row[1] == "main"), "")
    if not path:
        raise LaneDeferred("file-backed board required for crash recovery")
    return str(Path(path).resolve())


def legacy_workers_present(ledger, current_board, task, run) -> bool:
    """Unknown/untracked active workers prevent activation, across all boards.

    Inspect closed runs too: a terminal card does not prove physical exit.
    Do not import requested config as observed runtime provenance.
    """
    from hermes_cli import kanban_db as kb
    tracked = {(r["board"], r["task"], r["run"]) for r in ledger.snapshot()}
    tracked.add((current_board, task, run))
    paths = {Path(current_board), kb.kanban_home() / "kanban.db"}
    paths.update(kb.boards_root().glob("*/kanban.db"))
    for path in paths:
        if not path.exists():
            continue
        try:
            with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5) as conn:
                rows = conn.execute("SELECT id, task_id, worker_pid, worker_started_at, ended_at FROM task_runs").fetchall()
            for rid, tid, pid, fingerprint, ended in rows:
                if (str(path.resolve()), tid, rid) in tracked:
                    continue
                if not pid:
                    if ended is None:
                        return True
                    continue
                try:
                    alive = identity_alive(process_identity(pid, fingerprint))
                except ValueError:
                    alive = None
                if alive is not False:
                    return True
        except sqlite3.Error as exc:
            raise LaneDeferred("cannot reconcile existing board workers") from exc
    return False


def spawn_reserved(conn, task, workspace, board, spawn, settings):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as dispatch
    candidate = pinned_candidate(task, settings)
    path = board_path(conn)
    if task.current_run_id is None:
        raise LaneDeferred("claimed run required")
    owner = uuid.uuid4().hex
    ledger = LaneLedger(ledger_path())
    try:
        def sample():
            # Called inside the ledger write transaction: sibling lane admissions
            # cannot race this scan or spend the same conservative headroom.
            if legacy_workers_present(ledger, path, task.id, task.current_run_id):
                return None
            available = dispatch._system_memory_sample().get("mem_available_kib")
            return available * 1024 if type(available) is int and available >= 0 else None

        reservation = ledger.reserve(board=path, task=task.id, run=task.current_run_id,
                                     candidates=(candidate,), owner=owner, alive=reservation_alive,
                                     memory_sample=sample,
                                     minimum_free_bytes=settings.minimum_free_bytes,
                                     worker_headroom_bytes=settings.worker_headroom_bytes)
        if reservation is None:
            raise LaneDeferred("lane capacity, memory, or legacy-worker admission deferred")
        token, requested = reservation
        try:
            with kb.write_txn(conn):
                kb._append_event(conn, task.id, "lane_reserved", {
                    "token": token, "requested_provider": requested.provider,
                    "requested_model": requested.model, "transport": task.provider_override,
                }, run_id=task.current_run_id)
        except BaseException:
            ledger.release(token, owner)
            raise
        # Once invoked, even a raising adapter might have created a child.
        # Never cancel that pending reservation based only on an exception.
        try:
            pid = spawn(task, workspace, board)
            if type(pid) is not int or pid <= 0:
                raise ValueError("spawn did not return a physical worker PID")
            dispatch._set_worker_pid(conn, task.id, pid)
            row = conn.execute("SELECT worker_started_at FROM task_runs WHERE id = ?",
                               (task.current_run_id,)).fetchone()
            worker = process_identity(pid, row[0] if row else None)
            if not ledger.bind_worker(token, owner, worker):
                raise ValueError("reservation binding refused")
            return pid
        except Exception as exc:
            raise LaneSpawnUncertain("spawn/bind outcome uncertain; reservation retained") from exc
    finally:
        ledger.close()
