"""hermes workflow daemon — runs CronPoller + KanbanDispatcher.

Start via:
    hermes workflow daemon --interval 60

Or supervised:
    systemctl --user start hermes-workflow-dispatcher   (Linux)
    launchctl load ~/Library/LaunchAgents/ai.hermes.workflow-dispatcher.plist  (macOS)

IMPORTANT: If no supervisor is installed this process does NOT auto-restart
on crash. Use foreground mode (hermes workflow daemon) for development; use
a supervisor for production.

Signal handling:
    SIGINT / SIGTERM → clean shutdown (tasks cancelled, daemon exits 0)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from typing import Any

log = logging.getLogger("workflow.daemon")


async def _main(args: Any) -> int:
    from ._shared import get_engine  # noqa: PLC0415
    from engine.cron.poller import CronPoller  # noqa: PLC0415
    from engine.dispatcher.kanban import KanbanDispatcher  # noqa: PLC0415

    engine = get_engine()
    poller = CronPoller(engine, poll_interval_s=args.interval)
    dispatcher = KanbanDispatcher(engine)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _handle_stop() -> None:
        log.info("workflow daemon: shutdown signal received")
        stop.set()

    # Signal handlers — guarded for Windows where add_signal_handler is absent
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_stop)
        except (NotImplementedError, OSError):
            # Windows / some embedded loops — fall back to signal.signal
            signal.signal(sig, lambda *_: _handle_stop())

    log.info(
        "workflow daemon started (interval=%.1fs, pid=%d)",
        args.interval,
        __import__("os").getpid(),
    )

    tasks = [
        asyncio.create_task(poller.run_forever(), name="wf-cron-poller"),
        asyncio.create_task(dispatcher.run_forever(), name="wf-kanban-dispatcher"),
    ]

    await stop.wait()

    log.info("workflow daemon: cancelling tasks")
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # Write/clear pidfile if requested
    if args.pidfile:
        try:
            import os  # noqa: PLC0415
            os.unlink(args.pidfile)
        except OSError:
            pass

    log.info("workflow daemon: clean exit")
    return 0


def _setup(sub: argparse.ArgumentParser) -> None:
    """Configure the argparse subparser for `hermes workflow daemon`."""
    sub.add_argument(
        "--interval",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="Poll interval in seconds for cron poller and kanban dispatcher (default: 60).",
    )
    sub.add_argument(
        "--pidfile",
        default=None,
        metavar="PATH",
        help="Write PID to this file on start; remove on clean exit.",
    )

    def _run(ns: argparse.Namespace) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        )
        # Write pidfile before entering loop
        if ns.pidfile:
            import os  # noqa: PLC0415
            with open(ns.pidfile, "w") as f:
                f.write(str(os.getpid()))
        sys.exit(asyncio.run(_main(ns)))

    sub.set_defaults(func=_run)
