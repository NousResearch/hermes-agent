"""Bounded current-query recall; background completion never publishes recall state."""

from __future__ import annotations

import logging
import math
import threading
import time

from plugins.memory.honcho.client import spawn_context_thread

logger = logging.getLogger("plugins.memory.honcho")


def prefetch_sync(provider, query: str) -> str:
    timeout = provider._config.timeout if provider._config else None
    budget = timeout if timeout is not None and math.isfinite(timeout) and timeout > 0 else 5.0
    deadline = time.monotonic() + budget
    if provider._is_trivial_prompt(query):
        return ""
    # No lock wait can extend the request budget. Keep the slot occupied even after
    # a caller times out, until the actual worker (including any HTTP call) exits.
    if not provider._recall_sync_lock.acquire(blocking=False):
        return ""
    cancelled = threading.Event()
    try:
        worker = provider._recall_sync_thread
        if worker is not None and worker.is_alive():
            return ""
        generation = provider._recall_generation = object()
        session, turn = provider._session_key, provider._turn_count
        if not provider._session_ready():
            provider._start_session_init_background(blocking=False)
            if provider._init_thread is not None:
                provider._init_thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if not provider._session_ready():
                return provider._pop_auth_notice()
        manager = provider._manager
        if (generation is not provider._recall_generation or session != provider._session_key
                or turn != provider._turn_count or time.monotonic() >= deadline):
            return ""

        context_due = (
            not (provider._injection_frequency == "first-turn" and turn > 1)
            and turn - provider._last_context_turn >= provider._context_cadence
        )
        dialectic_due = turn - provider._last_dialectic_turn >= provider._effective_cadence()
        if not context_due and not dialectic_due:
            return ""
        # Resolve all mutable settings before starting the worker. The captured
        # manager/session remain its only I/O owner, even if the provider is reused.
        levels = [provider._resolve_pass_level(i, query=query) for i in range(provider._dialectic_depth)]
        rewriter = provider._query_rewriter if provider._query_rewrite_enabled else None
        holder = {}

        def expired() -> bool:
            return (cancelled.is_set() or generation is not provider._recall_generation
                    or time.monotonic() >= deadline)

        def retrieve() -> None:
            try:
                if expired():
                    return
                base = provider._format_first_turn_context(manager.get_prefetch_context(
                    session, query, current_query_only=True,
                ) or {}) if context_due else ""
                if expired():
                    return
                dialectic = ""
                if dialectic_due:
                    rewritten = ""
                    if rewriter:
                        try:
                            rewritten = rewriter(query).strip()
                        except Exception as exc:
                            logger.debug("Honcho query rewriter failed: %s", exc)
                    results = []
                    for i, level in enumerate(levels):
                        if expired():
                            return
                        if results and provider._signal_sufficient(results[-1]):
                            break
                        prompt = (provider._build_dialectic_prompt(i, results, not base) if results else
                                  rewritten or "Recall evidence relevant to the user's current request.")
                        prompt = f"Current user request:\n{query}\n\n{prompt}"
                        result = manager.dialectic_query(session, prompt, reasoning_level=level,
                                                        peer="user", raise_errors=True)
                        if result and result.strip():
                            results.append(result)
                    dialectic = results[-1] if results else ""
                holder["result"] = (base, dialectic)
            except Exception as exc:
                logger.debug("Honcho synchronous recall failed: %s", exc)

        worker = spawn_context_thread(retrieve, name="honcho-recall-sync")
        provider._recall_sync_thread = worker
        worker.start()
        worker.join(timeout=max(0.0, deadline - time.monotonic()))
        if (worker.is_alive() or time.monotonic() >= deadline or "result" not in holder
                or generation is not provider._recall_generation
                or provider._manager is not manager or provider._session_key != session
                or provider._turn_count != turn):
            return ""
        base, dialectic = holder["result"]
        if context_due:
            provider._last_context_turn = turn
        if dialectic_due:
            provider._last_dialectic_turn = turn
            if dialectic:
                provider._dialectic_empty_streak = 0
            else:
                provider._dialectic_empty_streak += 1
        parts = [provider._pop_auth_notice(), base, dialectic]
        return provider._truncate_to_budget("\n\n".join(part for part in parts if part and part.strip()))
    finally:
        # Thread.join may return before a coarse host clock crosses the deadline.
        # Record caller abandonment explicitly before a late HTTP call can resume.
        cancelled.set()
        provider._recall_sync_lock.release()
