"""PLUGIN-COMPAT stub (revert-scheduled; see COMPAT_MANIFEST.md).

``hermes_cli.observability.relay_runtime`` was folded into ``agent.relay_runtime`` in the Sep 2026 decomposition. This stub keeps the
old import path alive for external plugins only; internal code must import ``agent.relay_runtime``.
"""
from agent.relay_runtime import *  # noqa: F401,F403
