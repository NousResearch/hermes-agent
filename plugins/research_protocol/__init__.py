"""Opt-in Hermes Research Protocol plugin.

PR2 registers exactly three fail-closed planner tools. Runtime authority remains
unavailable until an operator supplies an absolute artifact root. Context reads
and approval writes additionally require distinct
``RESEARCH_PROTOCOL_READER_DATABASE_URL`` and
``RESEARCH_PROTOCOL_WRITER_DATABASE_URL`` authorities, respectively.
"""

from __future__ import annotations

import logging
from typing import Any

from .planner_tools import (
    check_planner_approvals,
    check_planner_artifacts,
    check_planner_context,
    configure_planner_runtime,
    plan_approval_request,
    plan_artifact_write,
    plan_context_read,
)
from .runtime import RuntimeConfigurationError, build_runtime_bundle
from .schemas import (
    PLAN_APPROVAL_REQUEST_SCHEMA,
    PLAN_ARTIFACT_WRITE_SCHEMA,
    PLAN_CONTEXT_READ_SCHEMA,
)

logger = logging.getLogger(__name__)

_TOOLS = (
    (
        "plan_context_read",
        PLAN_CONTEXT_READ_SCHEMA,
        plan_context_read,
        check_planner_context,
    ),
    (
        "plan_artifact_write",
        PLAN_ARTIFACT_WRITE_SCHEMA,
        plan_artifact_write,
        check_planner_artifacts,
    ),
    (
        "plan_approval_request",
        PLAN_APPROVAL_REQUEST_SCHEMA,
        plan_approval_request,
        check_planner_approvals,
    ),
)


def _plugin_config() -> dict[str, Any]:
    from hermes_cli.config import load_config

    config = load_config() or {}
    plugins = config.get("plugins") or {}
    entries = plugins.get("entries") or {}
    entry = entries.get("research-protocol") or {}
    if not isinstance(entry, dict):
        raise RuntimeConfigurationError(
            "plugins.entries.research-protocol must be a mapping"
        )
    return entry


def _configure_from_local_state() -> None:
    try:
        bundle = build_runtime_bundle(_plugin_config())
    except (RuntimeConfigurationError, OSError, ValueError):
        configure_planner_runtime(None)
        logger.info(
            "Research Protocol planner runtime is unavailable; "
            "explicit local configuration is incomplete or unsafe"
        )
        return
    configure_planner_runtime(
        bundle.handlers,
        artifact_available=bundle.artifact_available,
        context_available=bundle.context_available,
        approval_available=bundle.approval_available,
    )


def register(ctx) -> None:
    """Register exactly the bounded planner tool surface."""
    _configure_from_local_state()
    for name, schema, handler, check_fn in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="planner",
            schema=schema,
            handler=handler,
            check_fn=check_fn,
            is_async=True,
        )
