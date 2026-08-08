"""Gateway CLI entry points (god-file slice 31 of #54962).

Extracted from ``gateway/run.py``: the thin entry-point functions that sat at
the end of the 26.8k-line module.

- ``main`` — the ``python -m gateway.run`` CLI entry point.
- ``_exit_after_graceful_shutdown`` — the wedge-proof ``os._exit`` backstop
  that every graceful exit path funnels through (#53107).

``start_gateway`` itself remains in ``gateway/run.py``: it wires the whole
``GatewayRunner`` lifecycle and calls the housekeeping daemons
(``_start_gateway_housekeeping``, ``_start_cron_ticker``,
``_run_planned_stop_watcher``) that a sibling PR extracts into
``gateway/housekeeping.py`` in parallel. ``main`` reaches ``start_gateway``
through a lazy call-time import — ``gateway.run`` is always fully loaded before
``main`` runs, and a module-level import here would create a circular import.
"""

import asyncio
import os
import sys

from gateway.config import GatewayConfig


def main():
    """CLI entry point for the gateway."""
    # start_gateway is imported lazily at call time: gateway.run is always
    # fully loaded by the time main() runs (it is invoked from gateway.run's
    # ``__main__`` block, or after importing gateway.run), and a module-level
    # import here would create a circular import between gateway.entry and
    # gateway.run.
    from gateway.run import start_gateway

    # Force UTF-8 stdio on Windows — gateway logs and startup banner would
    # otherwise UnicodeEncodeError on cp1252 consoles.  No-op on POSIX.
    try:
        from hermes_cli.stdio import configure_windows_stdio
        configure_windows_stdio()
    except Exception:
        pass

    import argparse

    parser = argparse.ArgumentParser(description="Hermes Gateway - Multi-platform messaging")
    parser.add_argument("--config", "-c", help="Path to gateway config file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    config = None
    if args.config:
        import yaml
        with open(args.config, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            config = GatewayConfig.from_dict(data)

    # start_gateway() performs the full graceful teardown (adapters
    # disconnected, sessions saved + flushed, SQLite closed, cron/MCP stopped,
    # PID file + runtime lock released) before it returns OR raises SystemExit
    # with an explicit code. Force-exit afterwards so a wedged non-daemon worker
    # thread (e.g. a ThreadPoolExecutor tool/LLM call blocked with no timeout)
    # cannot block interpreter finalization (Py_FinalizeEx joins all non-daemon
    # threads, incl. concurrent.futures' _python_exit) and strand the gateway
    # half-shut down with the supervisor unable to restart it (#53107).
    #
    # SystemExit is caught explicitly: start_gateway raises it on the
    # clean-fatal-config (#51228), planned-restart, and service-restart paths,
    # all of which complete teardown first. Routing those codes through the
    # same os._exit backstop means EVERY exit path is wedge-proof, not just the
    # boolean-return ones.
    try:
        success = asyncio.run(start_gateway(config))
        exit_code = 0 if success else 1
    except SystemExit as e:
        # e.code may be None (→ 0), an int, or a str (→ 1, like CPython).
        if e.code is None:
            exit_code = 0
        elif isinstance(e.code, int):
            exit_code = e.code
        else:
            exit_code = 1
    _exit_after_graceful_shutdown(exit_code)


def _exit_after_graceful_shutdown(exit_code: int) -> None:
    """Flush stdio, release the PID file + runtime lock, then hard-exit.

    Graceful teardown is already complete by the time this runs, so there is
    nothing left that needs a clean interpreter shutdown. We deliberately use
    ``os._exit`` (not ``sys.exit``): ``sys.exit`` raises ``SystemExit``, which
    triggers ``Py_FinalizeEx`` → ``wait_for_thread_shutdown`` and joins every
    non-daemon thread — exactly the hang (#53107) a wedged tool-worker causes.

    ``os._exit`` bypasses ``atexit`` handlers, so we cannot rely on the
    ``atexit``-registered ``remove_pid_file`` / ``release_gateway_runtime_lock``
    (registered in ``start_gateway``) to run. The full-shutdown path releases
    both explicitly in ``_stop_impl``, but the EARLY exit paths —
    clean-fatal-config (#51228) and startup-aborted-before-running — raise
    ``SystemExit`` right after ``runner.start()`` without going through
    ``_stop_impl``, so on those paths ``atexit`` was the only thing releasing
    them. Now that those paths are routed through this backstop (#53107),
    release both here explicitly. Both calls are idempotent —
    ``remove_pid_file`` only unlinks a PID file that belongs to this process,
    and ``release_gateway_runtime_lock`` no-ops when the lock is already
    released — so this is a no-op on the normal shutdown path and the actual
    cleanup on the early-exit paths.

    Logging IS drained here: the rotating file handlers are driven by an
    async ``QueueListener`` on a dedicated thread (see
    ``hermes_logging._register_queued_handler``), so records emitted right
    before shutdown may still be sitting in the in-memory queue. ``os._exit``
    below bypasses ``atexit``, so the ``atexit``-registered listener drain
    never runs on this path — we drain explicitly (bounded, via
    ``drain_log_queue``) or lose the last log lines (including the shutdown
    reason on the early-exit paths). Stdio is flushed too.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    # Release PID + runtime lock BEFORE the log drain: the drain is bounded but
    # could still take up to its timeout on a wedged disk, and these locks must
    # never be stranded. os._exit skips atexit, and the early SystemExit exit
    # paths never run _stop_impl, so release here (idempotent).
    try:
        from gateway.status import remove_pid_file, release_gateway_runtime_lock
        remove_pid_file()
        release_gateway_runtime_lock()
    except Exception:
        pass
    # Mark this life cleanly exited in the lifecycle sentinel (NS-608). This
    # is the single funnel every graceful exit passes through, so the next
    # boot's unclean-death detector only fires for genuine SIGKILL/OOM/VM
    # deaths. Ownership-guarded internally: a --replace old life won't
    # clobber the replacement's freshly claimed "running" sentinel.
    try:
        from gateway.lifecycle_ledger import mark_exited
        mark_exited(exit_code, reason="graceful_shutdown")
    except Exception:
        pass
    # Drain the async log queue: os._exit bypasses atexit, so the listener's
    # atexit drain won't fire. Use drain_log_queue() (bounded, no restart), NOT
    # flush_log_queue(): if the listener is wedged on the rotation lock — the
    # exact failure this async-logging change survives — an unbounded stop()
    # join would re-freeze the shutdown. drain_log_queue() no-ops when logging
    # never initialized a queue (very early aborts), so this is always safe.
    try:
        from hermes_logging import drain_log_queue
        drain_log_queue(timeout=1.0)
    except Exception:
        pass
    os._exit(exit_code)
