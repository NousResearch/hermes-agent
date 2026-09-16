"""Idle deferral for background reviews on the managed local runtime.

On the managed llama-server the post-turn review fork monopolizes the GPU the next prompt
needs and the next live turn cancels it (decode cost paid, learning lost). Reviews bound for
the managed endpoint are therefore queued and dispatched when the machine is quiet
(``auxiliary.background_review.defer``: ``auto`` = exactly that case, ``never`` = old behavior;
explicit /refine never defers). One slot per session, newest snapshot wins (a review replays
the whole conversation, so coalescing is dedup, not loss); aged-out items (defer_max_age_s,
default 30 min) dispatch regardless of idleness; in-memory best-effort like the immediate
fork. Idle truth is the supervisor's /slots held for a settle window.
"""

from __future__ import annotations

import contextvars
import json
import logging
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, Optional

from agent.review_admission import REASON_DEFERRED, REASON_DISABLED_WHILE_QUEUED, owner_tag

logger = logging.getLogger(__name__)

_IDLE_SETTLE_S = 15.0  # quiet window: two back-to-back prompts must not look idle, a coffee break must
_POLL_INTERVAL_S = 5.0  # poll cadence while non-empty; the thread parks when empty
_MAX_AGE_DEFAULT_S = 30.0 * 60.0  # dispatch regardless of idleness past this age


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


@dataclass(slots=True)
class _PendingReview:
    agent: Any
    session_key: Hashable
    kwargs: Dict[str, Any]
    enqueued_at: float
    context: contextvars.Context


def _owner_log_tag(session_key: Hashable) -> str:
    """Deterministic redacted label for a profile/session queue owner."""
    if isinstance(session_key, tuple) and len(session_key) >= 2:
        return owner_tag(session_key[0], session_key[1])
    return owner_tag("", session_key)


class ReviewIdleQueue:
    """Session-coalescing queue + idle-gated dispatcher thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: Dict[Hashable, _PendingReview] = {}
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

    def enqueue(
        self,
        agent: Any,
        session_key: Hashable,
        kwargs: Dict[str, Any],
        *,
        replace_existing: bool = True,
        reason: str = REASON_DEFERRED,
    ) -> None:
        """Add (or replace — newest snapshot wins) a session's pending review, keeping the ORIGINAL
        enqueue time on coalesce so a busy session cannot push its age-out forever. A retry uses
        ``replace_existing=False`` so it cannot overwrite a newer snapshot queued while dispatching,
        and names its own ``reason`` so the line does not read as a fresh deferral."""
        dispatch_context = contextvars.copy_context()
        with self._lock:
            existing = self._pending.get(session_key)
            if existing is not None and not replace_existing:
                return
            enqueued_at = existing.enqueued_at if existing is not None else self._now()
            self._pending[session_key] = _PendingReview(
                agent, session_key, kwargs, enqueued_at, dispatch_context
            )
        self._ensure_thread()
        self._wake.set()
        logger.info(
            "Background review deferred (owner=%s, reason=%s, queued=%d)",
            _owner_log_tag(session_key), reason, len(self._pending),
        )

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

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
                    self._dispatch_item(item)
            except Exception:  # noqa: BLE001 — dispatcher must survive anything
                logger.warning("Deferred review dispatch failed", exc_info=True)
            if item is None:
                time.sleep(_POLL_INTERVAL_S)

    def _dispatch_item(self, item: _PendingReview) -> None:
        """Dispatch one popped item; separated so aged/preempted interleavings are deterministic."""
        item.context.run(self._dispatch_item_in_context, item)

    def _dispatch_item_in_context(self, item: _PendingReview) -> None:
        """Re-check policy and spawn inside the profile Context captured at enqueue time."""
        if not self._still_enabled(item):
            logger.info(
                "Deferred background review dropped (owner=%s, reason=%s)",
                _owner_log_tag(item.session_key), REASON_DISABLED_WHILE_QUEUED,
            )
            return
        logger.info(
            "Dispatching deferred background review (owner=%s, waited=%.0fs, queued=%d)",
            _owner_log_tag(item.session_key), self._now() - item.enqueued_at, self.pending_count(),
        )
        dispatch_kwargs = dict(item.kwargs)
        dispatch_kwargs["_idle_queue_origin"] = True
        item.agent._spawn_background_review_now(**dispatch_kwargs)

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

        state = json.loads(state_path().read_text(encoding="utf-8-sig"))
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
