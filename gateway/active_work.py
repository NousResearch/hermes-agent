"""Re-persist ``active_agents`` when a NON-chat work source changes.

Why this module exists
----------------------
``GatewayRunner._active_work_count()`` is an aggregate over four sources:
chat turns (``_running_agents``), cron jobs, API-server runs, and deferred
agent workers.  ``_persist_active_agents()`` — the only thing that writes
that aggregate into ``gateway_state.json`` — was called from the CHAT-turn
boundary alone (claim in ``_handle_message``, release in
``_release_running_agent_state``).

So the file recorded whatever the other three sources happened to be worth
at the instant a chat turn ended, and nothing ever corrected it:

    06:20:22.040  an in-process cron job starts
    06:20:56.726  turn ends -> _persist_active_agents() -> writes active_agents=1
                  (0 chat slots + 1 in-flight cron job)
    06:21:07.847  cron job finishes -> NOTHING re-persists
    ...           active_agents stays 1 for the next 3.5 hours

That inflated count is not cosmetic: it feeds ``/api/status``,
``gateway_busy``, the restart-drain decision and external idle/health
watchers, all of which then believe the gateway is busy forever.

The fix is an EDGE, not a heartbeat: each non-chat source calls
:func:`notify_active_work_changed` when its own in-flight set moves, exactly
like the chat path already does at claim/release.  A periodic re-write was
explicitly rejected — it would mask a genuinely lost decrement instead of
fixing the boundary that lost it.

Deliberately no imports of ``gateway.run`` at module scope: callers live in
``cron/scheduler.py`` (which also runs in bare ``hermes cron run`` processes)
and in adapters.  The runner is resolved through ``sys.modules`` so this is a
true no-op — not even an import — outside a live gateway process.
"""

from __future__ import annotations

import sys

__all__ = ["notify_active_work_changed"]


def notify_active_work_changed() -> None:
    """Persist the live active-work aggregate, if a gateway runner exists.

    Safe to call from any thread and from processes that have no gateway at
    all (CLI cron runs, tests, the standalone kanban daemon): resolving the
    runner is a ``sys.modules`` lookup plus a weakref call, and every failure
    mode degrades to a no-op.  ``_persist_active_agents`` itself is
    best-effort and never raises.
    """
    try:
        run_module = sys.modules.get("gateway.run")
        if run_module is None:
            # No gateway in this process — nothing owns gateway_state.json.
            return
        ref = getattr(run_module, "_gateway_runner_ref", None)
        runner = ref() if callable(ref) else None
        if runner is None:
            return
        persist = getattr(runner, "_persist_active_agents", None)
        if callable(persist):
            persist()
    except Exception:
        # Status persistence is diagnostic; it must never disrupt the work
        # whose completion triggered it.
        pass
