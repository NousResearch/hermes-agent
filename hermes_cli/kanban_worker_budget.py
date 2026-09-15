"""Opt-in, machine-readable worker turn/phase budget + handoff telemetry (#111303).

Two halves of one schema:

* **Producer** — ``snapshot()`` normalizes a live worker's turn/phase budget into a
  versioned, secret-safe dict that ``emit()`` appends to ``task_events`` as a
  ``worker_budget`` event (run-scoped). Called from the agent loop at an opt-in
  checkpoint (``agent.activity_tracking._touch_activity``, rate-limited) and once at
  turn end (``agent.turn_finalizer.finalize_turn``).
* **Reader** — ``report_for()`` merges the newest recorded snapshot with the
  authoritative ``task_runs``/``task_events`` state so a running *and* a completed
  worker answer the same normalized field set, and ``_cmd_worker_budget`` prints it
  (``hermes kanban worker-budget <task>``).

Contract rules:

* Unavailable telemetry is ``None`` (JSON ``null``) — never ``0``. The ``unavailable``
  list names every null leaf so consumers don't have to guess.
* Field construction is a whitelist: counters, enums, ids and durations only. No
  prompts, transcripts, tool arguments, credentials or free-text error bodies are ever
  read off the agent, so the surface is secret-safe by construction.
* Key order is fixed and each snapshot is a plain JSON object, so serialization is
  deterministic for a given input set.
* Telemetry is OFF by default: enable with ``HERMES_KANBAN_WORKER_BUDGET_TELEMETRY=1``
  (explicit ``0``/``false`` wins) or ``kanban.worker_budget_telemetry: true`` in config.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from typing import Any, Optional

logger = logging.getLogger("hermes_cli.kanban_worker_budget")

SCHEMA = "hermes.kanban.worker_budget/1"
SCHEMA_VERSION = 1
EVENT_KIND = "worker_budget"

ENV_OPT_IN = "HERMES_KANBAN_WORKER_BUDGET_TELEMETRY"
CONFIG_SECTION = "kanban"
CONFIG_KEY = "worker_budget_telemetry"

PHASE_TURN = "turn"
PHASE_HANDOFF = "handoff"
PHASE_TERMINAL = "terminal"

# The runtime always withholds ONE turn from the normal loop: at exhaustion the model
# still gets a toolless summary/verification call (``_handle_max_iterations`` via
# ``agent.turn_finalizer._resolve_budget_fallback``), and a worker that armed its grace
# call spends it the same way (``agent.turn_iteration_prep.begin_iteration``). That
# slot is the reserved verification/handoff capacity: it can never be used for
# productive work, so it is reported separately from ``remaining``.
HANDOFF_RESERVE_TURNS = 1

# Checkpoint cadence for mid-turn snapshots (the terminal snapshot is never throttled).
CHECKPOINT_MIN_INTERVAL_SECONDS = 60.0

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}

# Cost statuses whose amount is meaningless; ``included`` is a real (zero) value.
_COST_UNAVAILABLE_STATUSES = {"", "unknown", "unavailable"}

_last_checkpoint_emit = 0.0


# --- Small normalizers (never coerce a non-number into 0) ---

def _int_or_none(value: Any) -> Optional[int]:
    """``int`` for real numeric values, ``None`` otherwise (bools excluded)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _float_or_none(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _int_str_or_none(value: Any) -> Optional[int]:
    """``int`` for a numeric value or numeric string (env vars); ``None`` otherwise."""
    if isinstance(value, str):
        stripped = value.strip()
        return int(stripped) if stripped.lstrip("-").isdigit() else None
    return _int_or_none(value)


def _bool_or_none(value: Any) -> Optional[bool]:
    return bool(value) if isinstance(value, bool) else None


def _str_or_none(value: Any, *, limit: int = 120) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return value[:limit] if value else None


def _ratio_or_none(value: Any) -> Optional[float]:
    """The configured warning ratio, kept as a float (0/1/absent -> None)."""
    ratio = _float_or_none(value)
    return ratio if ratio is not None and 0 < ratio < 1 else None


# --- Opt-in gate ---

def telemetry_enabled() -> bool:
    """True when worker budget telemetry is opted in (env wins over config)."""
    raw = (os.environ.get(ENV_OPT_IN) or "").strip().lower()
    if raw in _FALSY:
        return False
    if raw in _TRUTHY:
        return True
    try:
        from hermes_cli.config import load_config

        return bool((load_config() or {}).get(CONFIG_SECTION, {}).get(CONFIG_KEY, False))
    except Exception:
        return False


def _identity_from_env(
    *, task_id: Optional[str], run_id: Optional[int], profile: Optional[str], board: Optional[str]
) -> dict:
    def _env(name: str) -> Optional[str]:
        return _str_or_none((os.environ.get(name) or "").strip())

    return {
        "task_id": task_id or _env("HERMES_KANBAN_TASK"),
        "run_id": run_id if run_id is not None else _int_str_or_none(_env("HERMES_KANBAN_RUN_ID")),
        "profile": profile or _env("HERMES_PROFILE"),
        "board": board or _env("HERMES_KANBAN_BOARD"),
    }


# --- Producer ---

def _budget_view(agent: Any) -> dict:
    """Turn budget from the agent-local :class:`~agent.iteration_budget.IterationBudget`."""
    budget = getattr(agent, "iteration_budget", None)
    total = _int_or_none(getattr(budget, "max_total", None))
    used = _int_or_none(getattr(budget, "used", None))
    remaining = _int_or_none(getattr(budget, "remaining", None))
    reserved = HANDOFF_RESERVE_TURNS if budget is not None else None
    usable = (
        max(0, remaining - reserved)
        if remaining is not None and reserved is not None
        else None
    )
    return {
        "total": total,
        "used": used,
        "remaining": remaining,
        "reserved_handoff": reserved,
        "usable_remaining": usable,
    }


def _transcript_counts(messages: Any) -> tuple[Optional[int], Optional[int]]:
    """``(turns that called a tool, turns that ended without one)`` in the live transcript.

    Both are read straight off the transcript instead of subtracting budget counters, so
    the numbers stay correct across turns (a goal-loop worker resets its iteration budget
    every turn but keeps one transcript).
    """
    if messages is None:
        return (None, None)
    try:
        from agent.message_content import flatten_message_text
    except Exception:  # pragma: no cover - defensive (import cycle only)
        flatten_message_text = lambda content: str(content or "")  # noqa: E731

    tool_turns = 0
    no_action_turns = 0
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        if msg.get("tool_calls"):
            tool_turns += 1
        elif str(flatten_message_text(msg.get("content")) or "").strip():
            no_action_turns += 1
    return (tool_turns, no_action_turns)


def _provider_rate_limited(agent: Any) -> Optional[bool]:
    """True while the agent sits in a provider rate-limit cooldown; None when untracked."""
    until = getattr(agent, "_rate_limited_until", None)
    if isinstance(until, bool) or not isinstance(until, (int, float)):
        return None
    return until > time.monotonic()


def _subagent_views() -> Optional[list[dict]]:
    """Live delegated children with their own turn budgets; None when unavailable.

    Reuses the shared subagent registry (``tools.delegate_tool_registry``) and each
    child's own ``get_activity_summary()`` — never the child's transcript or goal text.
    """
    try:
        from tools import delegate_tool_registry as registry
    except Exception:
        return None
    try:
        with registry._active_subagents_lock:
            records = [dict(r) for r in registry._active_subagents.values()]
    except Exception:  # pragma: no cover - defensive
        return None

    views: list[dict] = []
    for record in records:
        child = record.get("agent")
        summary: dict = {}
        getter = getattr(child, "get_activity_summary", None)
        if callable(getter):
            try:
                summary = getter() or {}
            except Exception:  # pragma: no cover - defensive
                summary = {}
        total = _int_or_none(summary.get("max_iterations"))
        used = _int_or_none(summary.get("api_call_count"))
        views.append({
            "subagent_id": _str_or_none(record.get("subagent_id")),
            "parent_id": _str_or_none(record.get("parent_id")),
            "depth": _int_or_none(record.get("depth")),
            "status": _str_or_none(record.get("status")),
            "tool_count": _int_or_none(record.get("tool_count")),
            "turns": {
                "total": total,
                "used": used,
                "remaining": (max(0, total - used) if total is not None and used is not None else None),
            },
        })
    # Stable order regardless of registry iteration order.
    views.sort(key=lambda entry: (entry["subagent_id"] or "", entry["depth"] if entry["depth"] is not None else -1))
    return views


def snapshot(
    agent: Any,
    *,
    phase: str,
    exit_reason: Optional[str] = None,
    messages: Any = None,
    now: Optional[int] = None,
    task_id: Optional[str] = None,
    run_id: Optional[int] = None,
    profile: Optional[str] = None,
    board: Optional[str] = None,
) -> dict:
    """Normalize a live worker's budget/phase state into the versioned wire record.

    Every value is either a real reading or ``None``; nothing is read off the agent that
    could carry prompt, credential or tool content.
    """
    ts = int(now if now is not None else time.time())
    tool_turns, no_action_turns = _transcript_counts(messages)
    cost_status = _str_or_none(getattr(agent, "session_cost_status", None))
    cost_usd = (
        None
        if (cost_status or "").strip().lower() in _COST_UNAVAILABLE_STATUSES
        else _float_or_none(getattr(agent, "session_estimated_cost_usd", None))
    )
    reason = _str_or_none(exit_reason) if phase != PHASE_TURN else None
    warning_injected = getattr(agent, "_iteration_budget_warning_injected", None) is True

    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "recorded_at": ts,
        "source": "worker",
        "identity": _identity_from_env(
            task_id=task_id, run_id=run_id, profile=profile, board=board
        ),
        "phase": {"name": phase, "step_key": None, "role": None},
        "turns": _budget_view(agent),
        "counts": {
            "provider_attempts": _int_or_none(getattr(agent, "session_api_calls", None)),
            "tool_turns": tool_turns,
            "no_action_turns": no_action_turns,
            "provider_rate_limited": _provider_rate_limited(agent),
            "provider_fallback_active": _bool_or_none(getattr(agent, "_provider_fallback_active", None)),
            "fallback_index": _int_or_none(getattr(agent, "_fallback_index", None)),
        },
        "usage": {
            "input_tokens": _int_or_none(getattr(agent, "session_input_tokens", None)),
            "output_tokens": _int_or_none(getattr(agent, "session_output_tokens", None)),
            "cost_usd": cost_usd,
            "cost_status": cost_status,
        },
        "termination": {
            "reason": reason,
            "at": ts if reason else None,
            "warning_ratio": _ratio_or_none(getattr(agent, "budget_warning_ratio", None)),
            "warning_at": ts if warning_injected else None,
        },
        "subagents": _subagent_views(),
    }


def record(conn: sqlite3.Connection, snap: dict, *, run_id: Optional[int] = None) -> Optional[int]:
    """Append ``snap`` as a ``worker_budget`` event in the caller's transaction."""
    from hermes_cli import kanban_db as kb

    task_id = (snap.get("identity") or {}).get("task_id")
    if not task_id:
        return None
    rid = run_id if run_id is not None else (snap.get("identity") or {}).get("run_id")
    kb._append_event(conn, task_id, EVENT_KIND, snap, run_id=rid)
    return rid


def emit(
    agent: Any,
    *,
    phase: str,
    exit_reason: Optional[str] = None,
    messages: Any = None,
    now: Optional[int] = None,
    force: bool = False,
) -> Optional[dict]:
    """Best-effort: record one snapshot for the current Kanban worker; never raises.

    ``phase="turn"`` checkpoints are rate-limited by ``CHECKPOINT_MIN_INTERVAL_SECONDS``
    (pass ``force=True`` to bypass); terminal snapshots are always written. Returns the
    snapshot that was persisted, or ``None`` when it was skipped/failed.
    """
    global _last_checkpoint_emit

    if not telemetry_enabled():
        return None
    if phase == PHASE_TURN and not force:
        now_mono = time.monotonic()
        if (now_mono - _last_checkpoint_emit) < CHECKPOINT_MIN_INTERVAL_SECONDS:
            return None
        _last_checkpoint_emit = now_mono

    try:
        snap = snapshot(agent, phase=phase, exit_reason=exit_reason, messages=messages, now=now)
    except Exception:  # pragma: no cover - defensive
        logger.debug("worker-budget: snapshot failed", exc_info=True)
        return None
    if not (snap.get("identity") or {}).get("task_id"):
        return None  # not a Kanban worker — nothing to attribute the snapshot to

    try:
        from hermes_cli import kanban_db_connect as kbc

        with kbc.connect_closing() as conn:
            with kbc.write_txn(conn):
                record(conn, snap)
    except Exception:
        logger.debug("worker-budget: emit failed", exc_info=True)
        return None
    return snap


# --- Reader ---

def _load_snapshot(
    conn: sqlite3.Connection, task_id: str, run_id: Optional[int]
) -> Optional[dict]:
    """Newest ``worker_budget`` event for the run, else the task-scoped (run-less) one.

    A snapshot never migrates across runs: attempt N's record is not attempt N+1's. Events
    recorded without ``HERMES_KANBAN_RUN_ID`` (older spawns) are task-scoped by design.
    """
    def _query(where: str, params: tuple) -> Optional[dict]:
        row = conn.execute(
            f"SELECT payload FROM task_events WHERE task_id = ? AND kind = ? AND {where} "
            "ORDER BY id DESC LIMIT 1",
            (task_id, EVENT_KIND, *params),
        ).fetchone()
        if row is None or not row["payload"]:
            return None
        try:
            payload = json.loads(row["payload"])
        except (TypeError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    if run_id is not None:
        found = _query("run_id = ?", (run_id,))
        if found is not None:
            return found
    return _query("run_id IS NULL", ())


def _run_role(conn: sqlite3.Connection, run_id: Optional[int]) -> Optional[str]:
    """``reviewer``/``implementer`` from the run's ``claimed`` event ``source_status``."""
    if run_id is None:
        return None
    row = conn.execute(
        "SELECT payload FROM task_events WHERE run_id = ? AND kind = 'claimed' ORDER BY id DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    if row is None or not row["payload"]:
        return None
    try:
        payload = json.loads(row["payload"])
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or "source_status" not in payload:
        return None  # no source_status -> unknown, not "implementer"
    return "reviewer" if payload.get("source_status") == "review" else "implementer"


def _artifact_count(run: Any) -> Optional[int]:
    metadata = getattr(run, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    for key in ("artifacts", "changed_files"):
        value = metadata.get(key)
        if isinstance(value, list):
            return len(value)
    return None


def _attempts_view(runs: list, run_id: Optional[int], task: Any, conn: sqlite3.Connection, task_id: str) -> dict:
    """Retry/failure accounting from the authoritative run history."""
    ordered = sorted(runs, key=lambda r: r.id)
    index = next((i + 1 for i, r in enumerate(ordered) if r.id == run_id), None)
    outcomes = [(r.outcome or r.status or "") for r in ordered if r.ended_at]
    protocol_violations: Optional[int]
    try:
        from hermes_cli import kanban_db_dispatch as kbd

        protocol_violations = int(kbd._protocol_violation_streak(conn, task_id))
    except Exception:  # pragma: no cover - defensive
        protocol_violations = None
    return {
        "index": index,
        "total": len(ordered) if ordered else None,
        "consecutive_failures": _int_or_none(getattr(task, "consecutive_failures", None)),
        "protocol_violations": protocol_violations,
        "crashed": outcomes.count("crashed"),
        "timed_out": outcomes.count("timed_out"),
        "rate_limited": outcomes.count("rate_limited"),
    }


def _safe_current_board() -> Optional[str]:
    """Active board slug; None when unreadable (never raises into a report)."""
    try:
        from hermes_cli import kanban_db as kb

        return _str_or_none(kb.get_current_board())
    except Exception:  # pragma: no cover - defensive
        return None


def unavailable_paths(obj: Any, prefix: str = "") -> list[str]:
    """Dotted paths of every ``None`` leaf — the machine-checkable "not available" list."""
    out: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.extend(unavailable_paths(value, f"{prefix}{key}."))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            out.extend(unavailable_paths(value, f"{prefix}{i}."))
    elif obj is None and prefix:
        out.append(prefix.rstrip("."))
    return out


def report_for(
    conn: sqlite3.Connection, task_id: str, *, run_id: Optional[int] = None, now: Optional[int] = None
) -> Optional[dict]:
    """The normalized export for one worker run: snapshot ⊕ authoritative board state.

    Returns ``None`` when the task does not exist. A run that never recorded telemetry
    still returns the same shape — every worker-only field is ``None`` and named in
    ``unavailable``.
    """
    from hermes_cli import kanban_db as kb

    task = kb.get_task(conn, task_id)
    if task is None:
        return None
    ts = int(now if now is not None else time.time())
    run = kb.get_run(conn, run_id) if run_id is not None else kb.latest_run(conn, task_id)
    if run_id is not None and run is None:
        return None  # unknown run id — do not attribute another attempt's telemetry
    rid = run.id if run is not None else run_id
    runs = kb.list_runs(conn, task_id)
    snap = _load_snapshot(conn, task_id, rid)
    snap_identity = (snap or {}).get("identity") or {}
    snap_counts = (snap or {}).get("counts") or {}
    snap_usage = (snap or {}).get("usage") or {}
    snap_turns = (snap or {}).get("turns") or {}
    snap_phase = (snap or {}).get("phase") or {}
    snap_termination = (snap or {}).get("termination") or {}
    snap_recorded_at = (snap or {}).get("recorded_at")
    run_started = _int_or_none(getattr(run, "started_at", None))
    run_ended = _int_or_none(getattr(run, "ended_at", None))

    def _snap(key: str) -> Optional[Any]:
        return snap.get(key) if snap else None

    phase_name = (
        _str_or_none(snap_phase.get("name"))
        or (PHASE_TERMINAL if run_ended else PHASE_TURN if run is not None else None)
    )
    termination_reason = _str_or_none(snap_termination.get("reason")) or _str_or_none(
        getattr(run, "outcome", None)
    )

    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "generated_at": ts,
        "identity": {
            "task_id": task_id,
            # Snapshot identity wins when the recorded run is the one being reported.
            "run_id": rid if rid is not None else _int_or_none(snap_identity.get("run_id")),
            "profile": _str_or_none(getattr(run, "profile", None))
            or _str_or_none(snap_identity.get("profile"))
            or _str_or_none(getattr(task, "assignee", None)),
            "board": _str_or_none(snap_identity.get("board"))
            or _str_or_none(os.environ.get("HERMES_KANBAN_BOARD"))
            or _safe_current_board(),
            "session_id": _str_or_none(getattr(task, "session_id", None)),
            "parents": list(kb.parent_ids(conn, task_id)),
            "children": list(kb.child_ids(conn, task_id)),
        },
        "phase": {
            "name": phase_name,
            "step_key": _str_or_none(getattr(run, "step_key", None)),
            "role": _run_role(conn, rid),
            "recorded_at": _int_or_none(snap_recorded_at),
        },
        "turns": {
            "total": _int_or_none(snap_turns.get("total")),
            "used": _int_or_none(snap_turns.get("used")),
            "remaining": _int_or_none(snap_turns.get("remaining")),
            "reserved_handoff": _int_or_none(snap_turns.get("reserved_handoff")),
            "usable_remaining": _int_or_none(snap_turns.get("usable_remaining")),
        },
        "counts": {
            "provider_attempts": _int_or_none(snap_counts.get("provider_attempts")),
            "tool_turns": _int_or_none(snap_counts.get("tool_turns")),
            "no_action_turns": _int_or_none(snap_counts.get("no_action_turns")),
            "provider_rate_limited": _bool_or_none(snap_counts.get("provider_rate_limited")),
            "provider_fallback_active": _bool_or_none(snap_counts.get("provider_fallback_active")),
            "fallback_index": _int_or_none(snap_counts.get("fallback_index")),
        },
        "attempts": _attempts_view(runs, rid, task, conn, task_id),
        "time": {
            "started_at": run_started,
            "ended_at": run_ended,
            "elapsed_seconds": (
                max(0, (run_ended or ts) - run_started) if run_started is not None else None
            ),
            "max_runtime_seconds": _int_or_none(getattr(run, "max_runtime_seconds", None)),
            "last_heartbeat_at": _int_or_none(getattr(task, "last_heartbeat_at", None)),
            "worker_pid": _int_or_none(getattr(task, "worker_pid", None)),
        },
        "usage": {
            "input_tokens": _int_or_none(snap_usage.get("input_tokens")),
            "output_tokens": _int_or_none(snap_usage.get("output_tokens")),
            "cost_usd": _float_or_none(snap_usage.get("cost_usd")),
            "cost_status": _str_or_none(snap_usage.get("cost_status")),
        },
        "termination": {
            "reason": termination_reason,
            "at": _int_or_none(snap_termination.get("at")) or run_ended,
            "warning_ratio": _ratio_or_none(snap_termination.get("warning_ratio")),
            "warning_at": _int_or_none(snap_termination.get("warning_at")),
        },
        "handoff": {
            "available": bool(run is not None and (run.summary or run.metadata)),
            "run_status": _str_or_none(getattr(run, "status", None)),
            "outcome": _str_or_none(getattr(run, "outcome", None)),
            "summary_chars": len(run.summary) if run is not None and run.summary else None,
            "artifact_count": _artifact_count(run),
        },
        "subagents": _snap("subagents"),
        "source": {
            "kind": "worker" if snap else "store",
            "recorded_at": _int_or_none(snap_recorded_at),
        },
    }
    report["unavailable"] = sorted(unavailable_paths(report))
    return report


def collect(task_id: str, *, run_id: Optional[int] = None, board: Optional[str] = None) -> Optional[dict]:
    """``report_for`` against a fresh board connection (CLI/API entry point)."""
    from hermes_cli import kanban_db_connect as kbc

    with kbc.connect_closing(board=board) as conn:
        return report_for(conn, task_id, run_id=run_id)


def _cmd_worker_budget(args: Any) -> int:
    """``hermes kanban worker-budget <task_id> [--run N]`` — always emits the JSON record."""
    task_id = getattr(args, "task_id", None)
    if not task_id:
        print("kanban: worker-budget requires a task id", flush=True)
        return 2
    run_id = getattr(args, "run_id", None)
    board = getattr(args, "board", None)
    try:
        report = collect(task_id, run_id=run_id, board=board)
    except Exception as exc:
        print(f"kanban: worker-budget: {exc}", flush=True)
        return 1
    if report is None:
        detail = f" (run {run_id})" if run_id is not None else ""
        print(f"kanban: worker-budget: no such task or run {task_id!r}{detail}", flush=True)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


__all__ = [
    "SCHEMA",
    "SCHEMA_VERSION",
    "EVENT_KIND",
    "PHASE_HANDOFF",
    "PHASE_TERMINAL",
    "PHASE_TURN",
    "HANDOFF_RESERVE_TURNS",
    "collect",
    "emit",
    "record",
    "report_for",
    "snapshot",
    "telemetry_enabled",
    "unavailable_paths",
]
