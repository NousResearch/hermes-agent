"""Compatibility shim — the real implementation is ``hermes_startup_watchdog``.

The startup-liveness watchdog (OOF-298) must be armable *before* the
``gateway`` package is imported: ``gateway/__init__`` eagerly pulls in the
config/session/delivery graph, and an import-time deadlock is squarely inside
the watchdog's coverage mandate. The implementation therefore lives at the
repository top level as a stdlib-only module.

This shim keeps the intuitive ``gateway.startup_watchdog`` import path
working for code that runs after the package is loaded (the disarm site in
``gateway.run``, tests, operators poking at a REPL).
"""

from hermes_startup_watchdog import (  # noqa: F401
    DEFAULT_STARTUP_WATCHDOG_TIMEOUT_S,
    ENV_STARTUP_WATCHDOG,
    ENV_STARTUP_WATCHDOG_TIMEOUT_S,
    SERVICE_RESTART_EXIT_CODE,
    StartupWatchdogHandle,
    arm_startup_watchdog,
    disarm_startup_watchdog,
    get_startup_watchdog_dump_path,
    kick_startup_watchdog,
    report_startup_progress,
    resolve_startup_watchdog_timeout,
    startup_watchdog_disabled,
)

__all__ = [
    "DEFAULT_STARTUP_WATCHDOG_TIMEOUT_S",
    "ENV_STARTUP_WATCHDOG",
    "ENV_STARTUP_WATCHDOG_TIMEOUT_S",
    "SERVICE_RESTART_EXIT_CODE",
    "StartupWatchdogHandle",
    "arm_startup_watchdog",
    "disarm_startup_watchdog",
    "get_startup_watchdog_dump_path",
    "kick_startup_watchdog",
    "report_startup_progress",
    "resolve_startup_watchdog_timeout",
    "startup_watchdog_disabled",
]
