"""Idle deferral for background reviews on the managed local runtime.

On the managed llama-server the post-turn review fork monopolizes the GPU the next prompt
needs and the next live turn cancels it (decode cost paid, learning lost). Reviews bound for
the managed endpoint are therefore queued and dispatched when the machine is quiet
(``auxiliary.background_review.defer``: ``auto`` = exactly that case, ``never`` = old behavior;
explicit /refine never defers). One slot per session, newest snapshot wins (a review replays
the whole conversation, so coalescing is dedup, not loss); aged-out items (defer_max_age_s,
default 30 min) dispatch regardless of idleness; in-memory best-effort like the immediate
fork. Idle truth is the supervisor's /slots held for a settle window.

Deferral must never become an unbounded memory owner: queued entries hold the snapshot as
frozen JSON bytes (detached from the live transcript's object graph) and the parent agent
weakly, under a process-wide byte/session budget that evicts the oldest best-effort reviews.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_IDLE_SETTLE_S = 15.0  # quiet window: two back-to-back prompts must not look idle, a coffee break must
_POLL_INTERVAL_S = 5.0  # poll cadence while non-empty; the thread parks when empty
_MAX_AGE_DEFAULT_S = 30.0 * 60.0  # dispatch regardless of idleness past this age
# aggregate caps on what the queue may retain (best-effort reviews must not outrank foreground memory)
_SNAPSHOT_BUDGET_BYTES = 8 * 1024 * 1024
_SESSION_BUDGET = 16


def defer_mode(task_cfg: Optional[Dict[str, Any]]) -> str:
    """'auto' (default) or 'never' from auxiliary.background_review.defer."""
    raw = str((task_cfg or {}).get("defer", "auto")).strip().lower()
    return raw if raw in ("auto", "never") else "auto"


def defer_max_age_s(task_cfg: Optional[Dict[str, Any]]) -> float:
    try:
        value = float((task_cfg or {}).get("defer_max_age_s", _MAX_AGE_DEFAULT_S))
    except (TypeError, ValueError):
        return _MAX_AGE_DEFAULT_S
    return value if value > 0 else _MAX_AGE_DEFAULT_S


def review_targets_managed_local(agent: Any, task_cfg: Optional[Dict[str, Any]]) -> bool:
    """Would this review fork decode on the llama-server WE manage? Exact netloc match against the
    supervisor state file; any failure reads False (immediate spawn is the safe default). The cheap
    TTL-cached netloc probe runs FIRST so cloud-only installs skip runtime resolution on the turn's tail."""
    try:
        from agent.auxiliary_client import _is_managed_local_endpoint, _managed_local_netloc

        if not _managed_local_netloc():
            return False
        from agent.background_review import _resolve_review_runtime

        runtime = _resolve_review_runtime(agent, task_cfg)
        return _is_managed_local_endpoint(runtime.get("base_url"))
    except Exception:  # noqa: BLE001
        return False


def _dead_ref(_agent: Any = None) -> Any:
    """Stand-in for objects that cannot be weak-referenced: the review drops at dispatch."""
    return None


def _freeze_snapshot(messages_snapshot: Any) -> Optional[bytes]:
    """UTF-8 JSON bytes for the cloned transcript, or None when it is not serializable.

    Freezing detaches the queue from the transcript's nested object graph and large immutable
    strings while preserving exact message data for reconstruction at dispatch.
    """
    try:
        return json.dumps(
            messages_snapshot, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class _PendingReview:
    agent_ref: Callable[[], Any]
    session_key: str
    frozen_snapshot: bytes
    # dispatch arguments minus messages_snapshot (frozen separately)
    kwargs: Dict[str, Any]
    snapshot_bytes: int
    enqueued_at: float


class ReviewIdleQueue:
    """Session-coalescing queue + idle-gated dispatcher thread."""

    def __init__(
        self,
        snapshot_budget_bytes: int = _SNAPSHOT_BUDGET_BYTES,
        session_budget: int = _SESSION_BUDGET,
    ) -> None:
        self._lock = threading.Lock()
        self._pending: Dict[str, _PendingReview] = {}
        self._snapshot_bytes = 0
        self._snapshot_budget_bytes = snapshot_budget_bytes
        self._session_budget = session_budget
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._live_turns = 0
        self._quiet_since: Optional[float] = None
        # Test seams — replaced by unit tests, never in production.
        self._now: Callable[[], float] = time.monotonic
        self._server_idle: Callable[[], bool] = _managed_server_idle

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

    def enqueue(self, agent: Any, session_key: str, kwargs: Dict[str, Any]) -> None:
        """Add (or replace — newest snapshot wins) a session's pending review, keeping the ORIGINAL
        enqueue time on coalesce so a busy session cannot push its age-out forever.

        The snapshot is frozen into JSON bytes and the parent held weakly, so a pending entry pins
        neither the live transcript graph nor an otherwise-collected AIAgent. Entries that cannot
        be frozen, or whose snapshot alone exceeds the whole queue budget, are rejected — background
        review is best-effort and must not outrank foreground memory ownership.
        """
        rest = dict(kwargs)
        frozen = _freeze_snapshot(rest.pop("messages_snapshot", None))
        if frozen is None:
            logger.warning(
                "Deferred background review rejected: snapshot is not JSON-serializable (session=%s)",
                session_key[-12:],
            )
            return
        if len(frozen) > self._snapshot_budget_bytes:
            logger.warning(
                "Deferred background review rejected: snapshot %d bytes exceeds queue budget %d (session=%s)",
                len(frozen),
                self._snapshot_budget_bytes,
                session_key[-12:],
            )
            return
        try:
            agent_ref = weakref.ref(agent)
        except TypeError:
            agent_ref = _dead_ref
        with self._lock:
            existing = self._pending.get(session_key)
            enqueued_at = existing.enqueued_at if existing is not None else self._now()
            if existing is not None:
                self._snapshot_bytes -= existing.snapshot_bytes
            self._pending[session_key] = _PendingReview(
                agent_ref, session_key, frozen, rest, len(frozen), enqueued_at
            )
            self._snapshot_bytes += len(frozen)
            self._evict_over_budget_locked()
        self._ensure_thread()
        self._wake.set()
        logger.info("Background review deferred (session=%s, queued=%d)", session_key[-12:], len(self._pending))

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def pending_bytes(self) -> int:
        with self._lock:
            return self._snapshot_bytes

    def _evict_over_budget_locked(self) -> None:
        """Oldest best-effort reviews go first once the queue exceeds its byte/session budget."""
        while len(self._pending) > self._session_budget:
            self._evict_oldest_locked("pending sessions over budget")
        while self._snapshot_bytes > self._snapshot_budget_bytes and self._pending:
            self._evict_oldest_locked("snapshot bytes over budget")

    def _evict_oldest_locked(self, reason: str) -> None:
        oldest = min(
            self._pending.values(), key=lambda p: (p.enqueued_at, p.session_key)
        )
        del self._pending[oldest.session_key]
        self._snapshot_bytes -= oldest.snapshot_bytes
        logger.warning(
            "Deferred background review evicted: %s (session=%s)",
            reason,
            oldest.session_key[-12:],
        )

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, daemon=True, name="bg-review-idle-queue")
                self._thread.start()

    def _quiet_for(self) -> float:
        """Seconds this process has been turn-free (0 while a turn runs)."""
        with self._lock:
            if self._live_turns > 0 or self._quiet_since is None:
                return 0.0
            return self._now() - self._quiet_since

    def _pop_dispatchable(self) -> Optional[_PendingReview]:
        """Oldest aged-out item, else the oldest item once quiet+idle hold."""
        with self._lock:
            if not self._pending:
                return None
            now = self._now()
            aged = [p for p in self._pending.values()
                    if now - p.enqueued_at >= defer_max_age_s(p.kwargs.get("task_cfg"))]
            candidate = min(aged, key=lambda p: p.enqueued_at) if aged else None
        if candidate is None and (self._quiet_for() < _IDLE_SETTLE_S or not self._server_idle()):
            return None
        with self._lock:
            if candidate is None:
                if not self._pending:
                    return None
                candidate = min(self._pending.values(), key=lambda p: p.enqueued_at)
            popped = self._pending.pop(candidate.session_key, None)
            if popped is not None:
                self._snapshot_bytes -= popped.snapshot_bytes
            return popped

    def _dispatch(self, item: _PendingReview) -> None:
        """Run one popped review: enabled gate, live-parent check, snapshot thaw, spawn.

        A dead parent or an unreadable frozen snapshot drops the best-effort review instead of
        raising — the dispatcher thread must survive anything its entries hand it.
        """
        if not self._still_enabled(item):
            logger.info(
                "Deferred background review dropped: reviews were disabled while it was queued (session=%s)",
                item.session_key[-12:],
            )
            return
        agent = item.agent_ref()
        if agent is None:
            logger.info(
                "Deferred background review dropped: parent agent is gone (session=%s)",
                item.session_key[-12:],
            )
            return
        try:
            kwargs = dict(item.kwargs)
            kwargs["messages_snapshot"] = json.loads(
                item.frozen_snapshot.decode("utf-8")
            )
        except (UnicodeDecodeError, ValueError):
            logger.warning(
                "Deferred background review dropped: frozen snapshot unreadable (session=%s)",
                item.session_key[-12:],
            )
            return
        logger.info(
            "Dispatching deferred background review (session=%s, waited=%.0fs, queued=%d)",
            item.session_key[-12:],
            self._now() - item.enqueued_at,
            self.pending_count(),
        )
        agent._spawn_background_review_now(**kwargs)

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
                    self._dispatch(item)
            except Exception:  # noqa: BLE001 — dispatcher must survive anything
                logger.warning("Deferred review dispatch failed", exc_info=True)
            if item is None:
                time.sleep(_POLL_INTERVAL_S)

    @staticmethod
    def _still_enabled(item: _PendingReview) -> bool:
        """Re-check the enabled gate at DISPATCH time (disabling reviews while queued must stick). Fail-open."""
        try:
            from agent.background_review import load_background_review_settings

            return load_background_review_settings()[0]
        except Exception:  # noqa: BLE001
            return True


def _managed_server_idle() -> bool:
    """No processing slot on any loaded model of the managed router; unreachable/no state file reads idle."""
    try:
        from hermes_cli.local_runtime.supervisor import state_path
        from urllib.parse import quote

        state = json.loads(state_path().read_text(encoding="utf-8"))
        base = str(state.get("base_url", "")).rsplit("/v1", 1)[0]
        headers = {"Authorization": f"Bearer {state.get('api_key', '')}"}
        if not base:
            return True

        def _get(path: str) -> Any:
            with urllib.request.urlopen(urllib.request.Request(f"{base}{path}", headers=headers), timeout=3) as r:
                return json.loads(r.read())

        loaded = [m["id"] for m in _get("/models").get("data", [])
                  if (m.get("status") or {}).get("value") in ("loaded", "ready")]
        return not any(
            s.get("is_processing") for mid in loaded for s in _get(f"/slots?model={quote(mid)}") if isinstance(s, dict)
        )
    except Exception:  # noqa: BLE001
        return True


# Module singleton — one queue per process, like the load-progress watcher.
QUEUE = ReviewIdleQueue()


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import List  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
