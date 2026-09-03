"""Opt-in, lossless session rollover at a completed-turn boundary.

This module deliberately owns no prompt/history manipulation.  A completed
turn marks a durable request; the next inbound turn may atomically replace the
route with an empty child session.  The child carries only a lineage pointer
and session_search guidance, never a compaction or an LLM-produced summary.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Optional


END_REASON = "turn_boundary_rollover"
#: Distinct from ``END_REASON`` so a read-back can tell a core-armed drain
#: apart from a doctor-requested recovery of a stalled lifecycle.
RECOVERY_END_REASON = "turn_boundary_rollover_recovered"
RECOVERY_GUIDANCE = "Use session_search to recover earlier details if needed."
_PENDING_KEY = "_turn_boundary_rollover_pending"
_HANDOFF_KEY = "turn_boundary_handoff"
_LIFECYCLE_KEY = "turn_boundary_lifecycle"


@dataclass(frozen=True)
class RolloverPolicy:
    """Dynamic policy derived afresh from the active config and model window."""

    enabled: bool = False
    ratio: float = 0.0
    safety_margin_tokens: int = 0
    threshold_tokens: Optional[int] = None
    reserved_output_tokens: int = 0
    reserved_tool_result_tokens: int = 0
    reserved_checkpoint_tokens: int = 0
    closeout_iterations: int = 2

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "RolloverPolicy":
        raw = config if isinstance(config, Mapping) else {}
        enabled = raw.get("enabled") is True
        try:
            ratio = float(raw.get("ratio", 0.0))
        except (TypeError, ValueError):
            ratio = 0.0
        # A zero/invalid ratio is disabled rather than inheriting a fixed window.
        if not 0.0 < ratio < 1.0:
            enabled = False
        try:
            margin = max(0, int(raw.get("safety_margin_tokens", 0) or 0))
        except (TypeError, ValueError):
            margin = 0
        try:
            cap = int(raw.get("threshold_tokens"))
            if cap <= 0:
                cap = None
        except (TypeError, ValueError):
            cap = None
        def _nonnegative(name: str, default: int = 0) -> int:
            try:
                return max(0, int(raw.get(name, default) or 0))
            except (TypeError, ValueError):
                return default

        return cls(
            enabled, ratio, margin, cap,
            _nonnegative("reserved_output_tokens"),
            _nonnegative("reserved_tool_result_tokens"),
            _nonnegative("reserved_checkpoint_tokens"),
            max(1, _nonnegative("closeout_iterations", 2)),
        )

    def resolve(self, context_length: int, compression_threshold: int) -> Optional[int]:
        """Return a trigger strictly below the actual compression threshold.

        The caller supplies the *live* compressor threshold, which already
        includes model/provider-specific output reserve and fallback changes.
        There is intentionally no fallback context-length constant here.
        """
        if not self.enabled:
            return None
        try:
            window = int(context_length)
            compression = int(compression_threshold)
        except (TypeError, ValueError):
            return None
        if window <= 0 or compression <= 1:
            return None
        # ``threshold_tokens`` is retained for config-file compatibility and
        # observability only. A fixed count must never decide rollover across
        # models with different context windows.
        candidate = int(window * self.ratio)
        # Strictly less than compression even when the configured headroom is 0.
        ceiling = compression - max(self.safety_margin_tokens, 1)
        if ceiling <= 0:
            return None
        return min(candidate, ceiling) if candidate > 0 else None


class TurnBoundaryRollover:
    """SessionDB persistence seam shared by CLI, TUI, and gateway owners."""

    def __init__(self, db: Any) -> None:
        self._db = db

    def mark_pending(
        self, session_id: str, *, threshold_tokens: int,
        lifecycle: Mapping[str, Any] | None = None,
    ) -> bool:
        """Persist a request after a response has fully committed."""
        if not session_id or threshold_tokens <= 0:
            return False
        row = self._db.get_session(session_id)
        if not row or row.get("ended_at") is not None:
            return False
        if _model_config(row).get(_PENDING_KEY):
            return True
        # Do not read/replace model_config: /model, /reasoning and /yolo can
        # mutate independent keys between the read above and this write.
        patch = getattr(self._db, "patch_session_model_config", None)
        if not callable(patch):
            return False
        lifecycle_data = dict(lifecycle or {})
        checkpoint = lifecycle_data.get("checkpoint")
        lifecycle_data["checkpoint"] = _checkpoint_with_repo_evidence(
            checkpoint if isinstance(checkpoint, Mapping) else {}, row
        )
        patch_data: dict[str, Any] = {
            _PENDING_KEY: {"threshold_tokens": int(threshold_tokens)},
        }
        patch_data[_LIFECYCLE_KEY] = lifecycle_data
        patch(session_id, patch_data)
        return True

    def request_recovery(
        self, session_id: str, *, idempotency_key: str
    ) -> Optional[str]:
        """Arm exactly one doctor-driven rollover for a stalled session.

        Returns ``"armed"`` for the request that actually armed the pending
        marker, ``"already_armed"`` when any request (this key or another) is
        already pending, and ``None`` when the session cannot be recovered.
        The caller is responsible for having verified the safety conditions;
        this method only guarantees at-most-one continuation per session.
        """
        if not session_id or not idempotency_key:
            return None
        row = self._db.get_session(session_id)
        if not row or row.get("ended_at") is not None:
            return None
        if _model_config(row).get(_PENDING_KEY):
            return "already_armed"
        patch = getattr(self._db, "patch_session_model_config", None)
        if not callable(patch):
            return None
        patch(session_id, {_PENDING_KEY: {
            "threshold_tokens": 0,
            "recovery_key": str(idempotency_key),
            "requested_at": time.time(),
        }})
        # Re-read: a concurrent doctor/core arm resolves to exactly one owner.
        pending = _model_config(self._db.get_session(session_id) or {}).get(_PENDING_KEY)
        if not isinstance(pending, Mapping):
            return None
        return "armed" if pending.get("recovery_key") == str(idempotency_key) else "already_armed"

    def adopt_at_turn_boundary(
        self,
        session_id: str,
        *,
        active_work: bool,
        turn_lease_holder: str | None = None,
    ) -> Optional[str]:
        """Atomically close a pending parent and create one empty child.

        ``active_work`` is supplied by the lifecycle owner.  The database
        transition itself is idempotent: concurrent arrivals can observe only
        one successful replacement because the pending parent is consumed in
        the same SQLite write transaction that inserts the child.
        """
        if active_work or not session_id:
            return None
        child_id = uuid.uuid4().hex

        def _write(conn: Any) -> Optional[str]:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if row is None or row["ended_at"] is not None:
                return None
            parent = dict(row)
            config = _model_config(parent)
            pending = config.pop(_PENDING_KEY, None)
            if not pending:
                return None
            recovery_key = (
                str(pending.get("recovery_key") or "")
                if isinstance(pending, Mapping) else ""
            )
            # A child is a fresh runtime, not a fresh identity. Preserve every
            # parent runtime/model setting except the consumed rollover marker.
            child_config = dict(config)
            child_config[_HANDOFF_KEY] = {
                "previous_session_id": session_id,
                "recovery": RECOVERY_GUIDANCE,
            }
            if recovery_key:
                # Binds the continuation to the exact recovery request so a
                # doctor read-back can prove which arm produced this child.
                child_config[_HANDOFF_KEY]["recovery_key"] = recovery_key
            # Keep routing identity, source, cwd, profile, model and role
            # namespace intact.  The transcript is intentionally not copied.
            conn.execute(
                """INSERT INTO sessions (
                    id, source, user_id, session_key, chat_id, chat_type, thread_id,
                    display_name, origin_json, model, model_config, system_prompt,
                    parent_session_id, cwd, profile_name, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%s','now'))""",
                (child_id, parent.get("source"), parent.get("user_id"),
                 parent.get("session_key"), parent.get("chat_id"), parent.get("chat_type"),
                 parent.get("thread_id"), parent.get("display_name"), parent.get("origin_json"),
                 parent.get("model"), json.dumps(child_config, separators=(",", ":")),
                 parent.get("system_prompt"), session_id, parent.get("cwd"),
                 parent.get("profile_name")),
            )
            if turn_lease_holder:
                # A rollover is a new lease domain. Move ownership inside this
                # transaction so alias-key arrivals cannot run in the gap
                # between adoption and the child's inbound persistence.
                moved = conn.execute(
                    "UPDATE session_turn_leases SET conversation_id = ? "
                    "WHERE conversation_id = ? AND holder = ?",
                    (child_id, session_id, turn_lease_holder),
                ).rowcount
                if moved != 1:
                    raise RuntimeError("turn-boundary rollover lease was not owned")
            changed = conn.execute(
                "UPDATE sessions SET ended_at = strftime('%s','now'), end_reason = ? "
                "WHERE id = ? AND ended_at IS NULL",
                (RECOVERY_END_REASON if recovery_key else END_REASON, session_id),
            ).rowcount
            if changed != 1:
                raise RuntimeError("turn-boundary rollover parent changed during adoption")
            return child_id

        return self._db._execute_write(_write)

    def _set_model_config(self, session_id: str, config: dict[str, Any]) -> None:
        def _write(conn: Any) -> None:
            conn.execute(
                "UPDATE sessions SET model_config = ? WHERE id = ? AND ended_at IS NULL",
                (json.dumps(config, separators=(",", ":")), session_id),
            )
        self._db._execute_write(_write)


def _model_config(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("model_config")
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _checkpoint_with_repo_evidence(checkpoint: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    """Capture bounded work evidence without inspecting secrets or arbitrary cwd."""
    payload = dict(checkpoint)
    cwd = str(payload.get("worktree") or row.get("cwd") or "").strip()
    payload["worktree"] = cwd or None
    for key in ("changed_files", "verification_evidence", "pending_workers", "result_pointers", "external_effects", "external_readback"):
        payload.setdefault(key, [])
    payload.setdefault("branch", None)
    payload.setdefault("head", None)
    worktree = Path(cwd) if cwd else None
    if worktree is None or not worktree.is_dir() or not (worktree / ".git").exists():
        return payload
    try:
        def git(*args: str) -> str:
            return subprocess.run(
                ["git", *args], cwd=worktree, check=True, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=2,
            ).stdout.strip()
        payload["branch"] = git("branch", "--show-current") or None
        payload["head"] = git("rev-parse", "HEAD") or None
        if not payload["changed_files"]:
            payload["changed_files"] = [line for line in git("status", "--porcelain").splitlines() if line]
    except (OSError, subprocess.SubprocessError):
        pass
    return payload


def allows_new_work(agent: Any) -> bool:
    """Fail closed for new tools/delegations once a parent is draining."""
    db = getattr(agent, "_session_db", None)
    session_id = str(getattr(agent, "session_id", "") or "")
    if db is None or not session_id:
        return True
    try:
        state = _model_config(db.get_session(session_id) or {}).get(_LIFECYCLE_KEY)
    except Exception:
        return False
    return not isinstance(state, Mapping) or state.get("state") != "draining"


def allows_new_delegation(agent: Any) -> bool:
    """Dedicated pre-dispatch delegation admission gate while draining."""
    return allows_new_work(agent)


def _publish_lifecycle_status(
    db: Any, session_id: str, status: Mapping[str, Any]
) -> None:
    """Write the recovery-visible lifecycle fields without arming anything.

    Doctor recovery decisions read these fields, so they must exist on every
    completed turn — not only on the turn that happens to arm a rollover.
    """
    patch = getattr(db, "patch_session_model_config", None)
    if callable(patch):
        patch(session_id, {_LIFECYCLE_KEY: dict(status)})


def _lifecycle_status(
    db: Any, session_id: str, decision: Any, *, made_progress: bool,
    in_flight_workers: int, active_tool_call: bool, api_calls: int,
    cache_read_tokens: int, compactions: int, now: float,
) -> dict[str, Any]:
    """Merge a new decision with the session's persisted lifecycle history."""
    try:
        previous = _model_config(db.get_session(session_id) or {}).get(_LIFECYCLE_KEY)
    except Exception:
        previous = None
    previous = dict(previous) if isinstance(previous, Mapping) else {}

    state = decision.state.value
    # A state's entry time only moves when the state itself changes; a stall
    # is measured from when the session ENTERED the state, not from the last
    # status write (which happens on every completed turn).
    if previous.get("state") == state:
        state_entered_at = float(previous.get("state_entered_at") or now)
    else:
        state_entered_at = now
    # "Meaningful progress" is real model work this turn. A completed turn
    # that made zero API calls must not refresh the stall clock.
    last_progress_at = (
        now if made_progress else float(previous.get("last_progress_at") or now)
    )
    return {
        "state": state,
        "state_entered_at": state_entered_at,
        "last_progress_at": last_progress_at,
        "updated_at": now,
        "context_utilization": decision.context_utilization,
        "remaining_context_tokens": decision.remaining_context_tokens,
        "remaining_iterations": decision.remaining_iterations,
        "reserved_headroom_tokens": decision.reserved_headroom_tokens,
        "in_flight_workers": int(in_flight_workers),
        "active_tool_call": bool(active_tool_call),
        # Observability only — never a lifecycle decision input.
        "api_calls": int(api_calls),
        "cache_read_tokens": int(cache_read_tokens),
        "compactions": int(compactions),
    }


def mark_completed_turn(agent: Any, result: Mapping[str, Any]) -> bool:
    """Re-read policy and live model budget after a durably completed response."""
    if not isinstance(result, Mapping) or result.get("failed") or result.get("interrupted"):
        return False
    # A response is eligible only after its final durable transcript write
    # succeeded. finalize_turn reports that exact failure in cleanup_errors.
    # Partial/compression recovery results are not completed boundaries.
    if (
        result.get("completed") is not True
        or result.get("partial")
        or result.get("compression_exhausted")
        or any(str(error).startswith("persist_session:") for error in result.get("cleanup_errors", ()) or ())
    ):
        return False
    db = getattr(agent, "_session_db", None)
    session_id = str(getattr(agent, "session_id", "") or "")
    compressor = getattr(agent, "context_compressor", None)
    if db is None or not session_id or compressor is None:
        return False
    try:
        from hermes_cli.config import load_config
        config = load_config() or {}
        policy = RolloverPolicy.from_config(config.get("session_rollover"))
        trigger = policy.resolve(
            getattr(compressor, "context_length", 0),
            getattr(compressor, "threshold_tokens", 0),
        )
        used = int(getattr(compressor, "last_prompt_tokens", 0) or 0)
        from agent.session_lifecycle import LifecycleBudget, LifecycleState, evaluate_lifecycle

        api_calls = int(result.get("api_calls", 0) or 0)
        cache_read_tokens = int(result.get("cache_read_tokens", 0) or 0)
        compactions = int(result.get("compactions", 0) or 0)
        in_flight_workers = len(getattr(agent, "_active_children", ()) or ())
        active_tool_call = bool(getattr(agent, "_executing_tools", False))
        decision = evaluate_lifecycle(LifecycleBudget(
            context_window_tokens=int(getattr(compressor, "context_length", 0) or 0),
            prompt_tokens=used,
            reserved_output_tokens=policy.reserved_output_tokens,
            reserved_tool_result_tokens=policy.reserved_tool_result_tokens,
            reserved_checkpoint_tokens=policy.reserved_checkpoint_tokens,
            max_iterations=int(getattr(agent, "max_iterations", sys.maxsize) or sys.maxsize),
            iterations_used=api_calls,
            api_calls=api_calls,
            cache_read_tokens=cache_read_tokens,
            compactions=compactions,
            in_flight_workers=in_flight_workers,
            closeout_iterations=policy.closeout_iterations,
        ))
        lifecycle = _lifecycle_status(
            db, session_id, decision,
            made_progress=api_calls > 0,
            in_flight_workers=in_flight_workers,
            active_tool_call=active_tool_call,
            api_calls=api_calls,
            cache_read_tokens=cache_read_tokens,
            compactions=compactions,
            now=time.time(),
        )
        if trigger is None or (
            decision.state is not LifecycleState.DRAINING and used < trigger
        ):
            # Status stays observable even when this turn arms nothing: doctor
            # stall detection depends on it existing for healthy sessions too.
            _publish_lifecycle_status(db, session_id, lifecycle)
            return False
        lifecycle["checkpoint"] = {
            "goal": str(getattr(agent, "current_goal", "") or ""),
            "cwd": str(getattr(agent, "cwd", "") or ""),
            "api_calls": api_calls,
            "in_flight_workers": in_flight_workers,
        }
        return TurnBoundaryRollover(db).mark_pending(
            session_id, threshold_tokens=trigger, lifecycle=lifecycle,
        )
    except Exception:
        return False


def adopt_agent_at_turn_boundary(
    agent: Any,
    *,
    active_work: bool = False,
    turn_lease_holder: str | None = None,
) -> Optional[str]:
    """Adopt a pending child before loading the next inbound user turn."""
    db = getattr(agent, "_session_db", None)
    old_id = str(getattr(agent, "session_id", "") or "")
    if db is None or not old_id:
        return None
    try:
        child_id = TurnBoundaryRollover(db).adopt_at_turn_boundary(
            old_id,
            active_work=active_work,
            turn_lease_holder=turn_lease_holder,
        )
    except Exception:
        return None
    if not child_id:
        return None
    previous = list(getattr(agent, "_session_messages", []) or [])
    agent.session_id = child_id
    agent._session_messages = []
    reset = getattr(agent, "reset_session_state", None)
    if callable(reset):
        reset(previous_messages=previous, old_session_id=old_id, carry_over_context=False)
    return child_id


def consume_handoff_note(agent: Any) -> str:
    """Consume the child's recovery pointer for one API request only.

    It is deliberately not a transcript message or api_content sidecar: the
    next user turn remains exactly the user's inbound content on disk.
    """
    db = getattr(agent, "_session_db", None)
    session_id = str(getattr(agent, "session_id", "") or "")
    if db is None or not session_id:
        return ""
    row = db.get_session(session_id)
    config = _model_config(row or {})
    handoff = config.get(_HANDOFF_KEY)
    if not isinstance(handoff, Mapping):
        return ""
    previous = str(handoff.get("previous_session_id") or "").strip()
    recovery = str(handoff.get("recovery") or "").strip()
    if not previous or not recovery:
        return ""
    patch = getattr(db, "patch_session_model_config", None)
    if not callable(patch):
        return ""
    patch(session_id, {_HANDOFF_KEY: None})
    return f"[Fresh session handoff: previous session {previous}. {recovery}]"