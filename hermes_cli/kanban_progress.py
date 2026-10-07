"""Forward-progress tracking for kanban workers.

``tasks.last_heartbeat_at`` answers "is the worker process making API traffic?".
It cannot answer "is it getting anywhere?", because the auto-heartbeat bridge in
``tools/kanban_tools`` mirrors *every* ``_touch_activity`` tick onto the board and
``_touch_activity`` fires on provider retries, stream reconnects and repeated
model calls (#31752). A worker stuck in a call loop keeps its heartbeat perfectly
fresh while making no forward progress, so every existing reaper
(``release_stale_claims``, ``detect_stale_running``) ignores it until the
wall-clock bounds bite (1 h heartbeat gap, 4 h stale timeout).

This module adds the missing signal: a *tool-call signature*. A step counts as
progress when the signature differs from the immediately preceding one; the same
tool with the same arguments in a row bumps ``progress_repeat_count`` instead of
refreshing ``last_progress_at``. A signature is a short hash of the tool name and
canonically-serialised arguments, so it is stable across processes and languages
without storing the (potentially huge or sensitive) arguments themselves.

Everything here is pure and dependency-free: the tracker runs inside the worker
process, the classifier runs board-side (dashboard/CLI), and neither raises into
the agent loop.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

# A running worker whose signature has not changed for this long, while its
# heartbeat stays fresh, is making no forward progress. 15 min is well under the
# 1 h heartbeat-gap floor and the 4 h stale timeout, and comfortably above the
# slowest single legitimate tool call (a build, a large download).
DEFAULT_PROGRESS_STALL_SECONDS = 15 * 60

# N consecutive identical tool-call signatures (same tool, same arguments) is
# the classic logic-loop shape. Deliberately high: a healthy worker can call
# ``terminal`` twice on the same command (retry after a transient failure)
# without being a loop.
DEFAULT_PROGRESS_LOOP_REPEAT_LIMIT = 8

# A worker with no heartbeat at all for this long is treating as gone, not
# merely stalled. Matches the claim-heartbeat backstop in ``kanban_db``.
DEFAULT_PROGRESS_ZOMBIE_SECONDS = 60 * 60

# The board column only needs a stable short id, not a recoverable value.
_SIGNATURE_CHARS = 12


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return max(minimum, int(raw))
    except (TypeError, ValueError):
        return default


def progress_stall_seconds() -> int:
    return _env_int("HERMES_KANBAN_PROGRESS_STALL_SECONDS", DEFAULT_PROGRESS_STALL_SECONDS)


def progress_loop_repeat_limit() -> int:
    return _env_int("HERMES_KANBAN_PROGRESS_LOOP_REPEAT_LIMIT", DEFAULT_PROGRESS_LOOP_REPEAT_LIMIT)


def progress_zombie_seconds() -> int:
    return _env_int("HERMES_KANBAN_PROGRESS_ZOMBIE_SECONDS", DEFAULT_PROGRESS_ZOMBIE_SECONDS)


def tool_signature(tool_name: str, args: Optional[Mapping[str, Any]]) -> str:
    """Short, stable signature for one tool call: ``<name>:<sha1(args)[:12]>``.

    Arguments are canonicalised with sorted keys and compact separators so two
    semantically identical calls hash the same regardless of key order; anything
    unsupported by ``json`` falls back to ``str`` rather than raising (the
    tracker must never break the agent loop).
    """
    name = (tool_name or "").strip() or "?"
    try:
        blob = json.dumps(args or {}, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        blob = repr(args)
    digest = hashlib.sha1(blob.encode("utf-8", "replace")).hexdigest()[:_SIGNATURE_CHARS]
    return f"{name}:{digest}"


@dataclass
class ProgressSnapshot:
    """The board-writable slice of a worker's progress state."""

    last_progress_at: Optional[int]
    progress_repeat_count: int
    tool_calls_total: int
    progress_signature: Optional[str]

    def as_heartbeat_payload(self) -> dict:
        return {
            "last_progress_at": self.last_progress_at,
            "progress_repeat_count": self.progress_repeat_count,
            "tool_calls_total": self.tool_calls_total,
        }


class ProgressTracker:
    """In-process tool-call progress for one dispatcher-spawned worker.

    ``note`` is O(1) and must be called once per tool call (the agent's tool
    executor does). Nothing here touches the board; ``heartbeat_current_worker_from_env``
    reads :meth:`snapshot` on its 60 s cadence.
    """

    def __init__(self, *, now: Optional[float] = None) -> None:
        self.last_progress_at: Optional[int] = int(now if now is not None else time.time())
        self.progress_repeat_count: int = 0
        self.tool_calls_total: int = 0
        self.progress_signature: Optional[str] = None
        self.distinct_signatures: int = 0

    def note(self, tool_name: str, args: Optional[Mapping[str, Any]] = None) -> str:
        """Record one tool call; returns its signature. Never raises."""
        try:
            sig = tool_signature(tool_name, args)
            now = int(time.time())
            self.tool_calls_total += 1
            if sig == self.progress_signature:
                self.progress_repeat_count += 1
            else:
                self.progress_signature = sig
                self.progress_repeat_count = 1
                self.distinct_signatures += 1
                self.last_progress_at = now
            return sig
        except Exception:
            # A tracker bug must not kill the worker; treat it as no-op.
            return ""

    def snapshot(self) -> ProgressSnapshot:
        return ProgressSnapshot(
            last_progress_at=self.last_progress_at,
            progress_repeat_count=self.progress_repeat_count,
            tool_calls_total=self.tool_calls_total,
            progress_signature=self.progress_signature,
        )


# --- Board-side classification ------------------------------------------------

WORKING = "working"
STALLED = "stalled"
LOOPING = "looping"
ZOMBIE = "zombie"
UNKNOWN = "unknown"

SEVERITY = {WORKING: 0, STALLED: 1, LOOPING: 1, ZOMBIE: 2, UNKNOWN: 0}


@dataclass
class Liveness:
    """Board-side verdict for one running worker, with the numbers behind it."""

    state: str
    reason: str
    heartbeat_age_s: Optional[int]
    progress_age_s: Optional[int]
    repeat_count: Optional[int]

    def as_dict(self) -> dict:
        return {
            "state": self.state,
            "reason": self.reason,
            "heartbeat_age_s": self.heartbeat_age_s,
            "progress_age_s": self.progress_age_s,
            "repeat_count": self.repeat_count,
        }


def _opt_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def classify_liveness(
    *,
    now: Optional[int] = None,
    status: Optional[str] = None,
    pid_alive: Optional[bool] = None,
    started_at: Optional[int] = None,
    last_heartbeat_at: Optional[int] = None,
    last_progress_at: Optional[int] = None,
    progress_repeat_count: Optional[int] = None,
    stall_seconds: Optional[int] = None,
    loop_repeat_limit: Optional[int] = None,
    zombie_seconds: Optional[int] = None,
) -> Liveness:
    """Classify a running worker as working / stalled / looping / zombie.

    Order matters: a dead process (or a heartbeat older than the zombie
    threshold) is a zombie regardless of what its last signature looked like;
    a tight repeat loop outranks a plain stall because it names the cause.

    ``progress_age_s`` is None when the row carries no progress column at all
    (a worker or board that predates the signal) — that is reported as
    ``unknown``, never guessed as a stall. ``started_at`` is accepted for call
    sites that have it but is not used to infer progress.
    """
    now_i = int(now if now is not None else time.time())
    hb = _opt_int(last_heartbeat_at)
    prog = _opt_int(last_progress_at)
    repeats = _opt_int(progress_repeat_count) or 0

    hb_age = (now_i - hb) if hb is not None else None
    prog_age = (now_i - prog) if prog is not None else None

    stall = int(stall_seconds if stall_seconds is not None else progress_stall_seconds())
    loop_limit = int(loop_repeat_limit if loop_repeat_limit is not None else progress_loop_repeat_limit())
    zombie = int(zombie_seconds if zombie_seconds is not None else progress_zombie_seconds())

    if status is not None and status != "running":
        return Liveness(UNKNOWN, f"task status is {status}", hb_age, prog_age, repeats)

    if pid_alive is False:
        return Liveness(ZOMBIE, "worker process is gone", hb_age, prog_age, repeats)

    if hb_age is not None and hb_age > zombie:
        return Liveness(
            ZOMBIE,
            f"no heartbeat for {hb_age}s (>{zombie}s)",
            hb_age, prog_age, repeats,
        )

    # No progress column on the row at all: a worker (or board) that predates the
    # signal. Absence is not evidence of a stall — say so rather than guess.
    if prog is None:
        return Liveness(UNKNOWN, "no forward-progress signal recorded", hb_age, None, repeats)

    if repeats >= loop_limit:
        return Liveness(
            LOOPING,
            f"same tool call repeated {repeats}x (>= {loop_limit})",
            hb_age, prog_age, repeats,
        )

    if prog_age is not None and prog_age > stall:
        return Liveness(
            STALLED,
            f"no new tool call for {prog_age}s (>{stall}s) while heartbeat stays fresh",
            hb_age, prog_age, repeats,
        )

    return Liveness(WORKING, "recent distinct tool activity", hb_age, prog_age, repeats)
