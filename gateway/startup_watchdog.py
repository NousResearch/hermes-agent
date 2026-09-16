"""PLUGIN-COMPAT stub (revert-scheduled; see COMPAT_MANIFEST.md).

``gateway.startup_watchdog`` keeps the historical startup-watchdog import path alive for
external plugins. The implementation lives in the stdlib-only ``hermes_startup_watchdog``
module; do not alias this path to the shutdown watchdog.
"""
from hermes_startup_watchdog import *  # noqa: F401,F403
