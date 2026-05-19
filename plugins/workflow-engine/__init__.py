"""
workflow-engine plugin — Phase 1 skeleton.

Exposes `register(host)` for the Hermes plugin loader.
"""

from __future__ import annotations


def register(host) -> None:  # noqa: ANN001
    """Register the workflow-engine plugin with the Hermes dashboard host."""
    from plugins.workflow_engine.dashboard.plugin_api import router  # noqa: PLC0415

    host.include_router(router, prefix="/api/plugins/workflow-engine")
