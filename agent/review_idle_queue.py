"""Idle deferral for background reviews on the managed local runtime.

The post-turn review fork replays the whole conversation on the review
runtime. On a cloud provider that costs seconds and runs concurrently
with whatever the user does next. When the review runtime IS the managed
llama-server, the same fork monopolizes the GPU the user's next prompt
needs, for minutes — and the next live turn cancels it, so an active
session tends to pay the decode cost AND lose the learning.

This module keeps the decision to learn exactly where it was (turn end,
nudge intervals, full-strength model, full transcript) and moves only
the execution moment: reviews bound for the managed local endpoint are
queued and dispatched when the machine is quiet. Everything else runs
immediately, as before.

Policy (auxiliary.background_review.defer):
    auto  (default) — defer exactly when the resolved review runtime
                      targets the managed local server.
    never           — old behavior everywhere.
Explicit /refine (focus set) never defers: an explicit ask runs now,
matching its bypass of the enabled gate.

Queue semantics:
- One slot per session, newest snapshot wins. A review replays the whole
  conversation, so a newer snapshot strictly supersedes an older one —
  coalescing is deduplication, not loss.
- Preempted (cancelled-by-live-turn) reviews are requeued by the spawn
  wrapper observing the run token's cancel flag, not killed-and-forgotten.
- Aged-out events (defer_max_age_s, default 30 min) dispatch regardless
  of idleness — deferral may delay learning, never lose it.
- In-memory, best-effort: dropped on process exit, the same durability
  contract the immediate daemon-thread fork always had.

Idle truth comes from the supervisor's /slots (machine-level: it sees
every client of the managed server, including other Hermes profiles) and
must hold for a settle window so a review is not launched into the gap
between two quick prompts. Local in-process turn liveness is tracked via
note_turn_started/note_turn_finished from run_conversation.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Sustained-quiet window before dispatch. Long enough that "typed two
# prompts back to back" does not look idle; short enough that walking
# away for coffee runs the queue.
_IDLE_SETTLE_S = 15.0
# Poll cadence while the queue is non-empty. The thread parks when empty.
_POLL_INTERVAL_S = 5.0
# Age at which a queued review dispatches regardless of idleness.
_MAX_AGE_DEFAULT_S = 30.0 * 60.0


def defer_mode(task_cfg: Optional[Dict[str, Any]]) -> str:
    """'auto' (default) or 'never' from auxiliary.background_review.defer."""
    raw = str((task_cfg or {}).get("defer", "auto")).strip().lower()
    return raw if raw in ("auto", "never") else "auto"


def defer_max_age_s(task_cfg: Optional[Dict[str, Any]]) -> float:
    raw = (task_cfg or {}).get("defer_max_age_s", _MAX_AGE_DEFAULT_S)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _MAX_AGE_DEFAULT_S
    return value if value > 0 else _MAX_AGE_DEFAULT_S


def review_targets_managed_local(agent: Any,
                                 task_cfg: Optional[Dict[str, Any]]) -> bool:
    """Would this review fork decode on the llama-server WE manage?

    Resolves the review runtime the same way the fork itself will and
    exact-matches its netloc against the supervisor state file — the
    matcher that cannot false-positive on external local servers. Any
    failure reads False: immediate spawn is always the safe default.

    Order matters: the netloc probe (one TTL-cached state-file read)
    runs FIRST, so machines with no managed server — every cloud-only
    install — return False without resolving the review runtime at all.
    This wrapper runs on the turn's tail; runtime resolution belongs on
    that path only when a managed server actually exists.
    """
    try:
        from agent.auxiliary_client import (
            _is_managed_local_endpoint,
            _managed_local_netloc,
        )

        if not _managed_local_netloc():
            return False
        from agent.background_review import _resolve_review_runtime

        runtime = _resolve_review_runtime(agent, task_cfg)
        return _is_managed_local_endpoint(runtime.get("base_url"))
    except Exception:  # noqa: BLE001
        return False


class _PendingReview:
    __slots__ = ("agent", "kwargs", "enqueued_at", "session_key")

    def __init__(self, agent: Any, session_key: str, kwargs: Dict[str, Any]):
        self.agent = agent
        self.session_key = session_key
        self.kwargs = kwargs
        self.enqueued_at = time.monotonic()


class ReviewIdleQueue:
    """Session-coalescing queue + idle-gated dispatcher thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: Dict[str, _PendingReview] = {}
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._live_turns = 0
        self._quiet_since: Optional[float] = None
        # Test seams — replaced by unit tests, never in production.
        self._now: Callable[[], float] = time.monotonic
        self._server_idle: Callable[[], bool] = _managed_server_idle

    # ── turn liveness (this process) ────────────────────────────

    def note_turn_started(self) -> None:
        with self._lock:
            self._live_turns += 1
            self._quiet_since = None

    def note_turn_finished(self) -> None:
        with self._lock:
            self._live_turns = max(0, self._live_turns - 1)
            if self._live_turns == 0:
                self._quiet_since = self._now()
        self._wake.set()

    # ── queue ────────────────────────────────────────────────────

    def enqueue(self, agent: Any, session_key: str,
                kwargs: Dict[str, Any]) -> None:
        """Add (or replace — newest snapshot wins) a session's pending review."""
        with self._lock:
            existing = self._pending.get(session_key)
            item = _PendingReview(agent, session_key, kwargs)
            # Stamp through the queue's clock (test seam); keep the ORIGINAL
            # enqueue time on coalesce so a busy session cannot push its
            # review's age-out forever.
            item.enqueued_at = (existing.enqueued_at if existing is not None
                                else self._now())
            self._pending[session_key] = item
        self._ensure_thread()
        self._wake.set()
        logger.info("Background review deferred (session=%s, queued=%d)",
                    session_key[-12:], len(self._pending))

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    # ── dispatcher ───────────────────────────────────────────────

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, daemon=True, name="bg-review-idle-queue")
                self._thread.start()

    def _quiet_for(self) -> float:
        """Seconds this process has been turn-free (0 while a turn runs)."""
        with self._lock:
            if self._live_turns > 0 or self._quiet_since is None:
                return 0.0
            return self._now() - self._quiet_since

    def _pop_dispatchable(self) -> Optional[_PendingReview]:
        """Oldest aged-out item, else any item once quiet+idle hold."""
        with self._lock:
            if not self._pending:
                return None
            items = sorted(self._pending.values(),
                           key=lambda p: p.enqueued_at)
            aged = [p for p in items
                    if self._now() - p.enqueued_at
                    >= defer_max_age_s(p.kwargs.get("task_cfg"))]
            candidate = aged[0] if aged else None
        if candidate is None:
            if self._quiet_for() < _IDLE_SETTLE_S:
                return None
            if not self._server_idle():
                return None
            with self._lock:
                if not self._pending:
                    return None
                candidate = min(self._pending.values(),
                                key=lambda p: p.enqueued_at)
        with self._lock:
            return self._pending.pop(candidate.session_key, None)

    def _run(self) -> None:
        while True:
            self._wake.wait()
            with self._lock:
                if not self._pending:
                    self._wake.clear()
                    continue
            item = None
            try:
                item = self._pop_dispatchable()
                if item is not None:
                    if not self._still_enabled(item):
                        logger.info(
                            "Deferred background review dropped: reviews "
                            "were disabled while it was queued (session=%s)",
                            item.session_key[-12:])
                        continue
                    logger.info(
                        "Dispatching deferred background review "
                        "(session=%s, waited=%.0fs, queued=%d)",
                        item.session_key[-12:],
                        self._now() - item.enqueued_at,
                        self.pending_count())
                    item.agent._spawn_background_review_now(**item.kwargs)
            except Exception:  # noqa: BLE001 — dispatcher must survive anything
                logger.warning("Deferred review dispatch failed",
                               exc_info=True)
            if item is None:
                time.sleep(_POLL_INTERVAL_S)

    @staticmethod
    def _still_enabled(item: _PendingReview) -> bool:
        """Re-check the enabled gate at DISPATCH time.

        The entry wrapper gates at enqueue time, but minutes may pass in
        the queue — a user who sets background_review.enabled: false while
        a review waits means it, and the dispatch must not resurrect it.
        Fail-open like the gate itself (a broken config never silently
        disables reviews)."""
        try:
            from agent.background_review import load_background_review_settings

            enabled, _ = load_background_review_settings()
            return enabled
        except Exception:  # noqa: BLE001
            return True


def _managed_server_idle() -> bool:
    """Machine-level idle: no processing slot on any loaded model of the
    managed router. Unreachable/no state file reads idle (nothing to
    contend with). One /models + one /slots call per loaded model."""
    try:
        from hermes_cli.local_runtime.supervisor import state_path

        state = json.loads(state_path().read_text(encoding="utf-8"))
        base = str(state.get("base_url", "")).rsplit("/v1", 1)[0]
        key = str(state.get("api_key", ""))
        if not base:
            return True
        headers = {"Authorization": f"Bearer {key}"}
        req = urllib.request.Request(f"{base}/models", headers=headers)
        with urllib.request.urlopen(req, timeout=3) as r:
            models = json.loads(r.read())
        loaded = [m["id"] for m in models.get("data", [])
                  if (m.get("status") or {}).get("value") in ("loaded", "ready")]
        from urllib.parse import quote

        for mid in loaded:
            req = urllib.request.Request(f"{base}/slots?model={quote(mid)}",
                                         headers=headers)
            with urllib.request.urlopen(req, timeout=3) as r:
                slots = json.loads(r.read())
            if any(s.get("is_processing") for s in slots
                   if isinstance(s, dict)):
                return False
        return True
    except Exception:  # noqa: BLE001
        return True


# Module singleton — one queue per process, like the load-progress watcher.
QUEUE = ReviewIdleQueue()
