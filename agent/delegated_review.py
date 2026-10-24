"""Skills review owned by a durable child session, independent of its retired runtime.

The turn only captures an eligible snapshot. Teardown prepares a separate review fork
before closing the child, and never attaches that fork to the child's close fan-out.
Only an active/queued review has an in-memory owner; completion drops the owner.
"""

from __future__ import annotations

import json
import contextvars
import logging
import threading
from pathlib import Path
from typing import Any

from agent import background_review as review

logger = logging.getLogger(__name__)
_owners_lock = threading.Lock()
_owners: dict[str, _ReviewOwner] = {}


def _key(session_id: str, db_path: Any = None) -> str:
    from hermes_constants import get_hermes_home
    path = Path(db_path) if db_path is not None else get_hermes_home() / "state.db"
    return json.dumps([str(path.expanduser().resolve()), session_id])


def queue_delegated_review(child: Any, messages: list[dict]) -> None:
    """Capture an eligible skills nudge; do not race the child's imminent close()."""
    enabled, task_cfg = review.load_background_review_settings()
    if not enabled:
        return
    from agent.turn_finalizer import _clone_background_review_messages
    from agent.secret_scope import current_secret_scope, current_secret_scope_home, set_secret_scope
    from hermes_constants import get_hermes_home, set_hermes_home_override
    # Teardown can happen in a Future callback after the original profile scope has
    # exited. Carry only profile authority, not live-turn DB/callback/agent context.
    context = contextvars.Context()
    home = get_hermes_home()
    context.run(set_hermes_home_override, home)
    secrets = current_secret_scope()
    if secrets is not None:
        context.run(set_secret_scope, dict(secrets), profile_home=current_secret_scope_home() or str(home))
    child._delegate_pending_review = (_clone_background_review_messages(messages), task_cfg, context)


def cancel_delegated_review(session_id: str, *, db_path: Any = None) -> None:
    """Fence a queued review or preempt its request before resuming/closing the child."""
    key = _key(session_id, db_path)
    with _owners_lock:
        owner = _owners.get(key)
    if owner is None:
        return
    from agent.review_idle_queue import QUEUE
    # Fence before dequeue: a dispatcher may already hold the queued item.
    owner._discard_background_review()
    QUEUE.cancel(key)


def finish_delegated_review(child: Any) -> None:
    """Transfer the pending snapshot to a standalone fork BEFORE ordinary child close.

    No original child object, database handle, or child callback is retained by the
    worker. The shared review engine still enforces its iteration/input-token budgets.
    """
    pending = child.__dict__.pop("_delegate_pending_review", None)
    if pending is None:
        return
    snapshot, task_cfg, context = pending
    context.run(_prepare_review, child, snapshot, task_cfg)


def _prepare_review(child: Any, snapshot: list[dict], task_cfg: dict) -> None:
    if not review._parent_can_emit_tool_calls(child) and not review._resolve_review_runtime(child, task_cfg).get("routed"):
        return
    db_path = getattr(getattr(child, "_session_db", None), "db_path", None)
    stable_id = str(getattr(child, "_delegate_session_id", None) or child.session_id)
    cancel_delegated_review(stable_id, db_path=db_path)
    fork, _runtime, routed = contextvars.copy_context().run(
        review.build_cache_parity_fork, child, task_cfg, max_iterations=review._REVIEW_MAX_ITERATIONS)
    fork._delegate_skills_only_review = True
    fork._memory_enabled = fork._user_profile_enabled = False
    owner = _ReviewOwner(_key(stable_id, db_path), child.session_id, db_path, fork, snapshot, task_cfg, routed)
    with _owners_lock:
        _owners[owner.key] = owner
    try:
        from agent.review_idle_queue import QUEUE, defer_mode, review_targets_managed_local
        if defer_mode(task_cfg) != "never" and review_targets_managed_local(fork, task_cfg):
            QUEUE.enqueue(owner, owner.key, {"task_cfg": task_cfg})
        else:
            owner._spawn_background_review_now()
    except BaseException:
        owner._discard_background_review()
        raise


class _ReviewOwner:
    """Only review resources; deliberately no reference or bound method of the child."""

    def __init__(self, key, session_id, db_path, fork, snapshot, task_cfg, routed):
        self.key, self.session_id, self.db_path = key, session_id, db_path
        self.snapshot, self.task_cfg, self.routed = snapshot, task_cfg, routed
        self.state = review._ReviewForkState(review_agent=fork)
        self._background_review_lock = threading.Lock()
        self._background_review_run = None
        self.run = review.prepare_background_review_run(self)
        self._start_lock = threading.Lock()
        self._started = False
        self._released = False

    def _spawn_background_review_now(self, **_kwargs) -> None:
        from tools.thread_context import propagate_context_to_thread
        with self._start_lock:
            if self._released or self._started:
                return
            if self.run.cancel_requested.is_set():
                self._release()
                return
            self._started = True
            try:
                threading.Thread(target=propagate_context_to_thread(self._work),
                                 daemon=True, name="delegated-skill-review").start()
            except BaseException:
                self._started = False
                self._release()
                raise

    def _discard_background_review(self) -> None:
        with self._start_lock:
            if self._released:
                return
            if not self._started:
                self.run.cancel()
                self._release()
                return
        review.cancel_background_review_for_live_turn(
            self, message="delegated session activated or closed",
            tool_reason="delegated skill review superseded")

    def _release(self) -> None:
        """Called by the queued owner or worker, never concurrently with its request."""
        self._released = True
        review.finish_background_review_run(self, self.run)
        review._track_review_fork(self, self.state.review_agent, register=False)
        if self.state.review_agent is not None:
            review._release_fork_clients(self.state.review_agent)
            self.state.review_agent = None
        self.snapshot = []
        self.state.review_messages = []
        with _owners_lock:
            if _owners.get(self.key) is self:
                _owners.pop(self.key)

    def _work(self) -> None:
        from tools.skill_manager_guards import background_review_write_scope
        review._set_thread_approval_callback(review._bg_review_auto_deny)
        try:
            with review.thread_scoped_silence(), background_review_write_scope(self.run.cancel_requested):
                review._execute_review_fork(
                    self, self.snapshot, review._SKILL_REVIEW_PROMPT, self.task_cfg, self.run,
                    self.state, routed=self.routed, review_memory=False)
            actions = review.summarize_background_review_actions(self.state.review_messages, self.snapshot)
            review._log_review_completion(self.state.review_usage, review._classify_review_result(actions))
        except Exception:
            logger.warning("Delegated skills review failed for %s", self.session_id, exc_info=True)
        finally:
            self._record_usage()
            self._release()
            review._set_thread_approval_callback(None)

    def _record_usage(self) -> None:
        if self.db_path is None or not self.state.review_usage:
            return
        from hermes_state_registry import acquire, release
        db = None
        try:
            db = acquire(self.db_path)
            self._session_db = db
            review._record_review_usage_to_parent(self, self.state.review_usage)
        except Exception:
            logger.debug("Delegated review usage recording failed", exc_info=True)
        finally:
            self._session_db = None
            if db is not None:
                release(db)
