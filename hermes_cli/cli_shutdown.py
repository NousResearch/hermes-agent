"""Classic-CLI shutdown and one-shot finalize helpers: process session-id sync, deferred agent startup, exit watchdog, cleanup steps, session-finalize notifications and the terminal input-mode reset.

Split out of ``cli.py``; ``cli`` re-exports every public name and moved bodies late-bind
cli-level names through ``from cli import ...`` at call time so facade monkeypatch seams hold.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from contextlib import suppress

# Log-record parity with the origin module.
logger = logging.getLogger("cli")


def _cli():
    """Late import of the ``cli`` facade: mutable CLI module state (and its test seams) lives there."""
    import cli

    return cli


def _sync_process_session_id(session_id: str) -> None:
    """Keep process-local session-id consumers aligned after CLI switches."""
    from gateway.session_context import set_current_session_id

    set_current_session_id(session_id)


def _flush_logging_and_stdio() -> None:
    """Best-effort ``logging.shutdown()`` + stdout/stderr flush before ``os._exit``."""
    with suppress(Exception):
        logging.shutdown()
    for _stream in (sys.stdout, sys.stderr):
        with suppress(Exception):
            _stream.flush()


def _float_env(name: str, default: float) -> float:
    """``float(os.getenv(name))``, or ``default`` when unset/unparseable."""
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _exit_watchdog_timeout() -> float:
    """``HERMES_EXIT_WATCHDOG_S`` as a float (default 30; ``0`` disables)."""
    from cli import _float_env
    return _float_env("HERMES_EXIT_WATCHDOG_S", 30.0)


def _arm_exit_watchdog(timeout_s: float | None = None, *, from_signal: bool = False) -> None:
    """Daemon timer that ``os._exit(0)``s after ``timeout_s`` once shutdown has begun.

    Backstop for a cleanup step wedged on network I/O and for interpreter teardown
    blocked joining non-daemon threads (ThreadPoolExecutor's atexit join). The daemon
    timer survives ``Py_FinalizeEx``'s joins. ``HERMES_EXIT_WATCHDOG_S=0`` disables.

    1. 2. Interpreter teardown blocked joining non-daemon threads — stdlib ``ThreadPoolExecutor`` workers
    are joined unconditionally by ``concurrent.futures``' atexit hook even after ``shutdown(wait=False)``,
    so one tool thread wedged on a socket held the process open forever (#27563 class).
    """
    from cli import _exit_watchdog_timeout, _flush_logging_and_stdio
    if timeout_s is None:
        timeout_s = _exit_watchdog_timeout()
    if timeout_s <= 0:
        return
    # Never under pytest: a delayed os._exit(0) would silently kill the test worker.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return

    def _watchdog():
        time.sleep(timeout_s)
        # The signal-armed watchdog yields to cleanup's own timer once cleanup is running.
        if from_signal and _cli()._cleanup_in_progress:
            return

        try:
            logger.warning(
                "Exit watchdog fired after %.0fs — forcing process exit "
                "(a cleanup step or non-daemon thread is wedged).",
                timeout_s,
            )
        except Exception:
            pass
        _flush_logging_and_stdio()
        os._exit(0)

    with suppress(Exception):  # never block shutdown on watchdog setup
        threading.Thread(target=_watchdog, daemon=True, name="exit-watchdog").start()


def _shutdown_agent_memory_provider(agent) -> None:
    """Memory-provider shutdown (on_session_end + shutdown_all) at the real session boundary."""
    if not (agent and hasattr(agent, 'shutdown_memory_provider')):
        return
    # A /new shortly before exit leaves an LLM-bound boundary task queued; shutdown_all()'s
    # ~5s drain would cancel it, so give it a bounded head start (watchdog is the backstop).
    _mm = getattr(agent, '_memory_manager', None)
    if _mm is not None and hasattr(_mm, 'flush_pending'):
        with suppress(Exception):
            _mm.flush_pending(timeout=10)
    # Forward the agent's transcript so on_session_end hooks see the real conversation;
    # no-arg fallback for stubs / partially-initialised agents.
    _session_msgs = getattr(agent, '_session_messages', None)
    _sid = getattr(agent, "session_id", None) or "<unknown>"
    # ``_session_messages`` is set on ``AIAgent.__init__`` and refreshed every turn via
    # ``_persist_session``. Fall back to no-arg on test stubs / partially-initialised agents where the
    # attribute is missing. See #15165.
    if isinstance(_session_msgs, list):
        logger.info("CLI cleanup calling memory shutdown for session %s with %d message(s)", _sid, len(_session_msgs))
        agent.shutdown_memory_provider(_session_msgs)
    else:
        logger.info("CLI cleanup calling memory shutdown for session %s without session message list", _sid)
        agent.shutdown_memory_provider()


def _stop_cli_wake_word() -> None:
    from tools.wake_word import stop_listening
    if _cli()._cli_wake_owner is not None:
        stop_listening(owner=_cli()._cli_wake_owner)


def _interrupt_async_delegations() -> None:
    from tools.async_delegation import interrupt_all
    interrupt_all(reason="CLI shutdown")


def _shutdown_mcp_servers() -> None:
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    shutdown_mcp_servers()


def _shutdown_cached_aux_clients() -> None:
    # Otherwise AsyncHttpxClientWrapper.__del__ fires on a closed loop ("Press ENTER to continue...").
    from agent.auxiliary_client import shutdown_cached_clients
    shutdown_cached_clients()


# Ordered teardown steps (attribute names, resolved at call time so tests can patch them)
# and the exception class each swallows.
_CLEANUP_STEPS = (
    ("_stop_cli_wake_word", Exception), ("_cleanup_all_terminals", Exception),
    ("_interrupt_async_delegations", Exception), ("_cleanup_all_browsers", Exception),
    ("_shutdown_mcp_servers", BaseException), ("_shutdown_cached_aux_clients", Exception),
)


def _should_emit_cleanup_session_finalize(session_id: str | None) -> bool:
    # A handed-off session is owned by the gateway process — never finalize it here.
    # The CLI must not finalize it on exit — that sets end_reason on a row the gateway reopened and is
    # actively writing to, causing the handoff leg to vanish from session history (#88234).
    if session_id is not None and session_id in _cli()._handed_off_session_ids:
        return False
    if not _cli()._single_query_finalize_attempted_session_ids:
        return True
    if session_id is None:
        return False
    return session_id not in _cli()._single_query_finalize_attempted_session_ids


def _notify_session_finalize(*, session_id: str | None, platform: str = "cli", reason: str = "shutdown") -> None:
    with suppress(Exception):
        from hermes_cli.lifecycle import finalize_session
        finalize_session(session_id=session_id, platform=platform, reason=reason)


def _oneshot_agent_and_session(cli):
    """``(agent, session_id)`` for a one-shot run; the agent's id wins over the CLI's."""
    agent = getattr(cli, "agent", None)
    return agent, getattr(agent, "session_id", None) or getattr(cli, "session_id", None)


def _invoke_interrupted_session_end(agent, session_id, reason: str, **extra) -> None:
    """Best-effort ``on_session_end`` hook for a turn cut short (never raises)."""
    with suppress(Exception):
        from hermes_cli.lifecycle import invoke_hook as _invoke_hook
        _invoke_hook(
            "on_session_end", session_id=session_id, completed=False, interrupted=True,
            model=getattr(agent, "model", None), platform=getattr(agent, "platform", None) or "cli",
            reason=reason, **extra,
        )


def _emit_interrupted_session_end(cli, *, reason: str = "keyboard_interrupt") -> None:
    """Best-effort on_session_end hook for interrupted non-interactive runs."""
    from cli import _invoke_interrupted_session_end, _oneshot_agent_and_session
    agent, session_id = _oneshot_agent_and_session(cli)
    if agent is None:
        return

    with suppress(Exception):
        agent.interrupt(reason.replace("_", " "), stop_kind="user_stop")

    if session_id in _cli()._handed_off_session_ids:  # gateway owns the lifecycle now
        return
    if session_id:
        with suppress(Exception):
            cli.session_id = session_id

    _invoke_interrupted_session_end(
        agent, session_id, reason,
        task_id=getattr(agent, "_current_task_id", "") or "",
        turn_id=getattr(agent, "_current_turn_id", "") or "",
        api_request_id=getattr(agent, "_current_api_request_id", "") or "",
    )


def _notify_single_query_session_finalize(cli, *, reason: str = "shutdown") -> None:
    from cli import _notify_session_finalize, _oneshot_agent_and_session
    agent, session_id = _oneshot_agent_and_session(cli)
    if session_id in _cli()._single_query_finalize_attempted_session_ids:
        return
    if session_id in _cli()._handed_off_session_ids:  # gateway owns the lifecycle now
        return

    try:
        _notify_session_finalize(session_id=session_id, platform=getattr(agent, "platform", None) or "cli", reason=reason)
    finally:
        _cli()._single_query_finalize_attempted_session_ids.add(session_id)


def _flush_one_shot_session_store(cli) -> None:
    """Durably flush + finalize the one-shot session row before exit (idempotent, best-effort).

    One-shot runs get a single turn, so nothing retries a transiently-failed transcript
    flush, closes the session row, or drains token deltas the kanban ``os._exit(0)``
    path skips. Handed-off sessions are left alone.

    - a turn whose in-loop ``_flush_messages_to_session_db`` failed under write-lock contention (e.g. a busy
    multiplex gateway sharing state.db) was silently lost — the reply reached stdout and agent.log but the
    resumed session's stored history never changed (#88583); - the resumed/created titled session row was
    left dangling open (``ended_at``/``end_reason`` NULL) on every one-shot exit; - queued async
    token-accounting deltas relied on interpreter-exit hooks, which the kanban SIGTERM path's
    ``os._exit(0)`` skips entirely.
    Idempotent and best-effort: ``_persist_session`` dedupes via the per-message ``_DB_PERSISTED_MARKER``
    stamps (already-written turns are not re-written) and ``end_session`` no-ops on an already-ended row.
    See #88234.
    """
    from cli import _oneshot_agent_and_session
    agent, session_id = _oneshot_agent_and_session(cli)
    if agent is None or not session_id or session_id in _cli()._handed_off_session_ids:
        return
    if getattr(agent, "_persist_disabled", False):
        return
    # Passing cli.conversation_history keeps resumed messages identity-skipped even when
    # the failed flush never stamped them.
    try:
        msgs = getattr(agent, "_session_messages", None)
        if isinstance(msgs, list) and msgs and hasattr(agent, "_persist_session"):
            agent._persist_session(msgs, getattr(cli, "conversation_history", None))
    except Exception:
        logger.debug("one-shot final session persist retry failed", exc_info=True)
    db = getattr(agent, "_session_db", None) or getattr(cli, "_session_db", None)
    if db is None:
        return
    try:
        db.flush_token_counts()
    except Exception:
        logger.debug("one-shot token-count drain failed", exc_info=True)
    try:
        db.end_session(session_id, "cli_close")
    except Exception:
        logger.debug("one-shot end_session failed", exc_info=True)


def _wait_for_oneshot_background_completions(cli) -> None:
    """Bounded linger for notify_on_complete background processes (children write to our pipes).

    Waits on the whole registry: a one-shot process hosts one agent, and task_id
    filtering would skip processes registered before the session id settled.

    Skipped when the quiet -Q notify-resume loop already consumed the run's linger
    budget: it calls wait_for_pending_completions with a shared deadline, so a
    re-wait here would double-block on the same stuck notify_on_complete child.

    See #90879.
    """
    from cli import _oneshot_agent_and_session
    from tools.process_registry import process_registry

    if getattr(cli, "_quiet_notify_linger_done", False):
        return
    _agent, task_id = _oneshot_agent_and_session(cli)
    result = process_registry.wait_for_pending_completions(None)
    if result.get("waited"):
        logger.info(
            "One-shot exit linger for session %s: completed=%s timed_out=%s",
            task_id or "<unknown>",
            result.get("completed"),
            result.get("timed_out"),
        )


def _finalize_single_query(cli) -> None:
    """Close one-shot CLI resources before releasing the active session lease."""
    from cli import _flush_one_shot_session_store, _notify_single_query_session_finalize, _run_cleanup, _wait_for_oneshot_background_completions
    try:
        # Order matters: linger for spawned background work BEFORE any teardown (the
        # parent owns those children's stdout pipes); then the durable flush, since
        # memory-provider shutdown inside _run_cleanup can issue aux-LLM calls and
        # nothing after it may fail in a way that loses the turn.
        for step, what in (
            (_wait_for_oneshot_background_completions, "background completion wait"),
            (_flush_one_shot_session_store, "session store flush"),
        ):
            try:
                step(cli)
            except Exception:
                logger.debug("one-shot %s failed", what, exc_info=True)
        _notify_single_query_session_finalize(cli)
        _run_cleanup(notify_session_finalize=False)
    finally:
        cli._release_active_session()
