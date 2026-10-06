"""Gateway housekeeping chores that run once per served profile (``profile_scoped_chore``).

Each reads only its own profile's home and config, which the wrapper binds per profile."""

import logging

logger = logging.getLogger("gateway.run")


def _housekeeping_curator() -> None:
    """maybe_run_curator() is gated by config.interval_hours (7 days default); this is the poll."""
    from agent.curator import maybe_run_curator
    maybe_run_curator(idle_for_seconds=float("inf"), on_summary=lambda msg: logger.info("curator: %s", msg))


def _housekeeping_bot_desktop_idle() -> None:
    """Stop this profile's Bot Desktop once it has been idle past ``bot_desktop.idle_stop_minutes``.

    ``runtime.stop_if_idle`` holds the rules (never while a human holds the screen). The ``hermes serve``
    lease watcher runs only after a client connects and visits only the launch home plus profiles a
    client has addressed since that process started, so a screen auto-started by a cron or messaging
    turn for a profile nobody opened in Desktop never stopped (~150 MB idle, ~700 MB with a page open).
    A Desktop-owned backend covers the profiles no gateway owns (``hermes_cli/desktop_idle_screens.py``).
    Screens are host-level files and processes, so this also stops idle screens started by ``hermes
    serve`` or the CLI; ``stop()`` is lock-guarded."""
    from tools.bot_desktop import runtime as _bd_runtime
    _bd_runtime.stop_if_idle()
