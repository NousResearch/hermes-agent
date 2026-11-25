"""Load the stop path's lazily imported modules at boot, before an upgrade can replace their files.

A running gateway outlives an in-place upgrade (a package manager or a checkout move replaces the files
under it). A function-level import that first runs during the stop would load the newer file against
the older modules already in memory, and the stop could fail on a name this process lacks (#159067).
"""
from __future__ import annotations

import importlib
import logging

logger = logging.getLogger("gateway.run")

# Repo modules the stop and restart paths import inside functions. Boot imports them, so a stop reads
# them from memory: after an in-place upgrade the files on disk may import names this process lacks.
# tests/gateway/test_stop_path_preload.py fails when a stop-path import is missing here. The last row
# is one hop further: modules those helpers import lazily themselves on a stop (delegation interrupt,
# storage health, browser cleanup, the tail's MCP close).
STOP_PATH_MODULES = (
    "cron.jobs", "cron.scheduler",
    "gateway.config_loader", "gateway.drain_control", "gateway.host_rendezvous", "gateway.lifecycle_ledger",
    "gateway.restart_loop_guard", "gateway.restart_notice", "gateway.restart_watcher_env",
    "gateway.restart_when_idle", "gateway.run_api", "gateway.session_acp_lifecycle",
    "gateway.session_hosted_service", "gateway.session_llm_oneshot", "gateway.shutdown_flush", "gateway.status",
    "gateway.stuck_loop_counts",
    "hermes_cli._launchers", "hermes_cli.nous_auth_keepalive", "hermes_cli.observability.shared_metrics_process",
    "persistence_assembly",
    "tools.async_delegation", "tools.browser_session_lifecycle", "tools.environments.base",
    "tools.environments.local", "tools.process_registry", "tools.terminal_tool_lifecycle",
    "tools.delegate_tool_resumable", "persistence.sqlite.services", "tools.browser_session",
    "tools.browser_session_cloud", "tools.browser_launcher", "tui_gateway.launch_profile_policy",
)


def preload_stop_path_modules() -> None:
    """Import every stop-path module while the files on disk still match the running code."""
    for name in STOP_PATH_MODULES:
        try:
            importlib.import_module(name)
        except Exception:  # health: allow BLE001 -- boot goes on; the stop imports it again then
            logger.warning("Stop-path module %s did not import at boot; a stop after an in-place upgrade may "
                           "read it from disk", name, exc_info=True)
