"""PLUGIN-COMPAT stub (revert-scheduled; see COMPAT_MANIFEST.md).

``gateway.startup_watchdog`` was folded into ``gateway.shutdown_watchdog`` in the Sep 2026 decomposition. This stub keeps the
old import path alive for external plugins only; internal code must import ``gateway.shutdown_watchdog``.
"""
from gateway.shutdown_watchdog import *  # noqa: F401,F403
