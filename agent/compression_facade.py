"""Host-side ``AIAgent._compress_context`` wrapper.

Publishes the commit fence ``hard_interrupt()`` reads, runs the compressor on a snapshot under the progress
timeout, mirrors ``_DB_PERSISTED_MARKER`` stamps back onto the live lists and rebinds the session context.
Extracted from ``run_agent.py``; every method resolves through ``AIAgent``'s MRO unchanged.
"""

import contextlib
import logging
import copy
import threading

from agent.session_activity import ActivityProvenance

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("run_agent")


def _timeout_fallback_prompt(agent, system_message: str) -> str:
    """Cached prompt, else a fresh build, else the raw ``system_message`` (never raises).
    Resolved lazily by the timeout wrapper: an eager rebuild would raise before compress_context runs when
    ``_cached_system_prompt`` is unset and the builder fails."""
    if cached := getattr(agent, "_cached_system_prompt", None):
        return cached
    try:
        return agent._build_system_prompt(system_message)
    except Exception:
        logger.debug("compress_context timeout fallback prompt rebuild failed; using raw system_message", exc_info=True)
        return system_message or ""


def _report_compression_timeout(
    agent, *, idle: float, waited: float, since_progress: float, total_ceiling: float, total_exhausted: bool,
    progress_observed: bool,
) -> None:
    """Host-side timeout bookkeeping: log, activity stamp, cooldown ladder, user warning."""
    from agent.conversation_compression import mark_context_compression_timed_out
    mark_context_compression_timed_out(agent)
    if total_exhausted:
        logger.warning(
            "Context compression reached its total ceiling after %.1fs (progress observed=%s); continuing without compression",
            waited, progress_observed,
        )
    else:
        logger.warning(
            "Context compression made no progress for %.1fs (total wait %.1fs, ceiling %.1fs); continuing without compression",
            since_progress, waited, total_ceiling,
        )
    touch = getattr(agent, "_touch_activity", None)
    if callable(touch):
        try:
            touch("context compression timed out", provenance=ActivityProvenance.AGENT_COMPRESSION_TIMEOUT)
        except Exception:
            logger.debug("compress_context timeout activity touch failed", exc_info=True)
    # Same timeout cooldown ladder as summary-LLM timeouts: avoid re-burning the full idle budget every turn.
    record = getattr(getattr(agent, "context_compressor", None), "record_timeout_failure", None)
    if callable(record):
        try:
            if total_exhausted:
                record("host compress_context total ceiling exhausted", failure_kind="ceiling_exhausted")
            else:
                record("host compress_context timeout (no summary progress)", failure_kind="stalled")
        except Exception:
            logger.debug("failed to record compress_context timeout cooldown", exc_info=True)
    emit = getattr(agent, "_emit_warning", None)
    if not callable(emit):
        return
    if total_exhausted:
        progress = " after summary output was observed" if progress_observed else ""
        emit(
            "⚠ Context compression reached its total ceiling "
            f"after {waited:.1f}s{progress}. No messages were "
            "dropped — continuing without compression. Run /compress to retry or /new for a clean session."
        )
    else:
        emit(
            f"⚠ Context compression timed out after {idle:.1f}s with no output from the summary "
            "model. No messages were dropped — continuing without compression. Run /compress to retry, /new "
            "for a clean session, or check auxiliary.compression."
        )


def _warn_commit_overrun(agent, waited: float, ceiling: float) -> None:
    """Commit-phase ceiling breach: the SessionDB mutation must complete, so only surface it."""
    emit = getattr(agent, "_emit_warning", None)
    if callable(emit):
        emit(
            f"⚠ Context compression commit is taking unusually long ({waited:.0f}s, ceiling {ceiling:.0f}s). "
            "Waiting for it to finish safely — if this persists, check SessionDB health (disk / lock contention)."
        )


def _sync_persisted_markers(target_messages, source_messages) -> None:
    """Mirror ``_DB_PERSISTED_MARKER`` stamps from the worker's snapshot onto a live list.
    Matched by scoped identity; timestamp-less repeated content is ambiguous, so every scoped match is
    stamped. Imported UNCONDITIONALLY: a silent fallback literal would split the stamping key from the flush's
    and resurrect the duplicate-row bug."""
    from agent.context_compressor import _DB_PERSISTED_MARKER
    from agent.conversation_compression import _stamp_scoped_twins
    if not isinstance(target_messages, list) or not isinstance(source_messages, list):
        return
    for source_message in source_messages:
        if isinstance(source_message, dict) and source_message.get(_DB_PERSISTED_MARKER):
            _stamp_scoped_twins(target_messages, source_message)


def _run_under_progress_timeout(
    agent, run, messages, system_message, *, active_fence, fence_registration_lock, idle_timeout, total_ceiling
):
    """Run ``run(fence, target_messages=snapshot)`` on the pool under the progress-aware timeout.
    The pooled worker must NEVER share the caller's live transcript — a late engine after a host timeout could
    rewrite it. It deep-snapshots on the worker and publishes only via an ADMITTED commit; a no-op/abort
    returns the snapshot unchanged, so the ORIGINAL list is handed back to keep identity semantics."""
    from agent.conversation_compression import CompressionCommitFence, run_compress_context_with_progress_timeout

    def _snapshot_worker(fence=None):
        # #76354 review F3: the pooled worker must NEVER share the caller's live transcript. Plugin/legacy
        # context engines are allowed to mutate their input list in place; after a host timeout the worker
        # stays alive, so a shared list would let a late engine rewrite the live conversation (roles,
        # ordering, persisted content) behind the caller's back. Deep-snapshot here, on the worker thread,
        # so the caller's list object is never touched by pooled code. Results are published to
        # caller-visible state only via the returned value of an ADMITTED commit (the host discards results
        # on timeout/cancel); durable SessionDB mutation is already gated behind the commit fence inside
        # compress_context.
        snapshot = copy.deepcopy(messages)
        result_msgs, result_prompt = run(fence, target_messages=snapshot)
        return (messages if result_msgs is snapshot else result_msgs), result_prompt

    timeout_cause = {"total_exhausted": False, "progress_observed": False}

    def _on_timeout_cause(total_exhausted, progress_observed):
        timeout_cause.update(total_exhausted=total_exhausted, progress_observed=progress_observed)

    def _on_timeout(idle, waited, since_progress):
        _report_compression_timeout(
            agent, idle=idle, waited=waited, since_progress=since_progress, total_ceiling=total_ceiling, **timeout_cause
        )

    def _publish_new_fence():
        # The stall-fallback retry needs a fence the aborted attempt cannot veto; publish
        # it on the slot hard_interrupt() reads. The caller's finally restores its fence.
        retry_fence = CompressionCommitFence()
        with fence_registration_lock:
            agent._active_compression_commit_fence = retry_fence
        return retry_fence

    return run_compress_context_with_progress_timeout(
        worker=_snapshot_worker, messages=messages,
        system_prompt_fallback=lambda: _timeout_fallback_prompt(agent, system_message),
        idle_timeout_seconds=idle_timeout, total_ceiling_seconds=total_ceiling, on_timeout=_on_timeout,
        on_timeout_cause=_on_timeout_cause,
        on_commit_overrun=lambda waited, ceiling: _warn_commit_overrun(agent, waited, ceiling), fence=active_fence,
        telemetry_agent=agent, new_fence=_publish_new_fence,
    )


def _mirror_result_onto_live_lists(agent, result, messages, *, direct_path: bool) -> None:
    """Mirror persisted-marker stamps from the result list onto the live list(s)."""
    if not (isinstance(result, tuple) and result and isinstance(result[0], list)):
        return
    result_messages = result[0]
    # Direct-path callers bypass the snapshot worker but still need the post-publish mirror.
    if direct_path or result_messages is not messages:
        _sync_persisted_markers(messages, result_messages)
    session_messages = getattr(agent, "_session_messages", None)
    if isinstance(session_messages, list) and session_messages is not messages:
        # Durable-parent adoption can leave `_session_messages` on the pre-adoption list.
        _sync_persisted_markers(session_messages, result_messages)


def _rebind_caller_session_context(agent) -> None:
    """Propagate a rotated session id to the CALLER's thread/ContextVar (idempotent otherwise).
    The worker thread rotated hermes_logging's thread-local id; post-compression tools must resolve
    HERMES_SESSION_ID to the child id."""
    with contextlib.suppress(Exception):
        from hermes_logging import set_session_context
        set_session_context(agent.session_id)
    try:
        from gateway.session_context import set_current_session_id
        if agent.session_id:
            set_current_session_id(agent.session_id)
    except Exception:
        logger.debug("post-compression session ContextVar rebind failed", exc_info=True)


class CompressionFacadeMixin:
    """``_compress_context`` (see module docstring)."""

    def _compress_context(
        self, messages: list, system_message: str, *, approx_tokens: int = None, task_id: str = "default",
        focus_topic: str = None, force: bool = False, bypass_cooldown: bool = False,
        defer_context_engine_notification: bool = False, commit_fence=None,
    ) -> tuple:
        """Forwarder — see ``agent.conversation_compression.compress_context``.
        ``force=True`` (manual /compress) bypasses the summary-failure cooldown; ``bypass_cooldown=True``
        (provider-proven overflow recovery) runs one real attempt while the cooldown stays armed.

        ``force=True`` is passed by the manual ``/compress`` slash command so users can bypass the
        summary-failure cooldown after an auto-compress abort. Auto-compress callers use the default
        ``force=False``. See #100661.
        """
        # Per-attempt timeout signal for turn-start preflight and in-loop consumers: a stalled
        # compression must not be mistaken for a structural no-op. Thread-local + per-agent lock.
        # A stalled compression must not be mistaken for a structural no-op and followed by the oversized
        # provider request it was meant to prevent. The typed helper upgrades the simple attribute to
        # thread-local state guarded by a per-agent lock so overlapping automatic/manual entrypoints cannot
        # clobber each other's outcome (#98741).
        from agent.conversation_compression import (
            CompressionCommitFence, compress_context, reset_context_compression_timeout_outcome,
            resolve_context_compression_timeouts,
        )
        reset_context_compression_timeout_outcome(self)
        from agent.portal_tags import (
            get_affinity_scope, get_conversation_context, reset_affinity_scope, reset_conversation_context,
            set_affinity_scope, set_conversation_context,
        )
        from agent.prompt_cache_scope import declared_conversation_scope_safe
        # Out-of-turn compaction (/compact, gateway /compress, partial head compression) runs outside
        # run_conversation's ambient scope; publish the root as a fallback so the summarizer's call carries
        # the conversation tag. No-op for in-turn callers. Same for the ROUTING scope when declared.
        token = None
        if get_conversation_context() is None:
            root = self._conversation_root_id()
            if root:
                token = set_conversation_context(root)
        # Initialized alongside `token`: the turn-lease timeout/interrupt early returns leave the try block
        # before set_affinity_scope() runs, and the finally reads this name unconditionally
        # (UnboundLocalError otherwise — the 4 red cross-process lease tests on PR #97158).
        affinity_token = None
        if get_affinity_scope() is None:
            declared = declared_conversation_scope_safe(self)
            if declared:
                affinity_token = set_affinity_scope(declared)
        # Every compression has a fence; hard_interrupt() uses this exact instance to serialize cancel
        # admission against begin_commit(). Publication is serialized so overlapping automatic/manual
        # entrypoints cannot replace the fence of the attempt currently committing.
        active_fence = commit_fence or CompressionCommitFence()
        fence_registration_lock = vars(self).setdefault("_compression_commit_fence_lock", threading.RLock())
        with fence_registration_lock:
            missing_fence = object()
            previous_fence = vars(self).get("_active_compression_commit_fence", missing_fence)
            self._active_compression_commit_fence = active_fence
        try:

            def _run(fence=None, target_messages=None):
                return compress_context(
                    self, target_messages if target_messages is not None else messages, system_message,
                    approx_tokens=approx_tokens, task_id=task_id, focus_topic=focus_topic, force=force,
                    bypass_cooldown=bypass_cooldown,
                    defer_context_engine_notification=(defer_context_engine_notification), commit_fence=fence,
                )

            # Callers that already own a progress-aware wait (gateway session
            # hygiene) pass commit_fence and must not be double-wrapped.
            direct_path = commit_fence is not None
            if not direct_path:
                idle_timeout, total_ceiling = resolve_context_compression_timeouts()
                direct_path = idle_timeout <= 0
            if direct_path:
                result = _run(active_fence)
            else:
                result = _run_under_progress_timeout(
                    self, _run, messages, system_message,
                    active_fence=active_fence, fence_registration_lock=fence_registration_lock,
                    idle_timeout=idle_timeout, total_ceiling=total_ceiling,
                )
            _mirror_result_onto_live_lists(self, result, messages, direct_path=direct_path)
            _rebind_caller_session_context(self)
            return result
        finally:
            with fence_registration_lock:
                if previous_fence is missing_fence:
                    vars(self).pop("_active_compression_commit_fence", None)
                else:
                    self._active_compression_commit_fence = previous_fence
            # Restore whatever the caller had, so a compaction never leaks its tag into the surrounding scope.
            if token is not None:
                reset_conversation_context(token)
            if affinity_token is not None:
                reset_affinity_scope(affinity_token)
