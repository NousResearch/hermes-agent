"""Executive Integration Layer — Hermes-side decision layer.

Connects the Hermes conversation flow to Executive v2. Decides
whether a user message should be routed to:

  * CHAT     — pure conversation
  * TOOL     — single instrumental action
  * EXECUTIVE — multi-step, requires Executive v2
  * CLARIFY  — insufficient information; ask the user
  * REJECT   — action prohibited; refuse

This layer does NOT execute Executive v2. It only prepares an
``ExecutiveLaunchRequest`` for operator approval.

Cardinal rules (HARD, never overridden):
  * NO LLM invocation
  * NO provider call (openai, anthropic, litellm, etc.)
  * NO worker invocation (delegate_task, kanban.create, etc.)
  * NO subprocess (subprocess.run, os.system, os.popen)
  * NO network access (no requests, urllib, httpx, aiohttp)
  * NO Kanban DB mutation
  * NO GBrain / Obsidian / NotebookLM calls
  * NO gateway restart
  * NO R7 / Hermes artifact modification
  * NO self-improvement activation
  * NO DB writes (stateless)
  * NO new tables
  * NO commit / push / PR
"""

from __future__ import annotations

from .types import (
    RouteKind,
    LaunchStatus,
    SummaryKind,
    ObjectiveGatewayDecision,
    ExecutiveLaunchRequest,
    ExecutiveUserSummary,
    ExecutiveIntegrationMetrics,
)

from .objective_gateway import (
    ObjectiveGateway,
)

from .router import (
    ExecutiveIntegrationRouter,
)

from .launcher import (
    ExecutiveLauncher,
    LAUNCH_FINGERPRINT_KEYS,
)

from .result_adapter import (
    ExecutiveResultAdapter,
)

from .metrics import (
    ExecutiveIntegrationMetricsCollector,
)

from .wiring import (
    maybe_route_with_executive_integration,
    _build_eil_response,
    _get_available_tool_names,
)

__all__ = [
    # types
    "RouteKind",
    "LaunchStatus",
    "SummaryKind",
    "ObjectiveGatewayDecision",
    "ExecutiveLaunchRequest",
    "ExecutiveUserSummary",
    "ExecutiveIntegrationMetrics",
    # components
    "ObjectiveGateway",
    "ExecutiveIntegrationRouter",
    "ExecutiveLauncher",
    "ExecutiveResultAdapter",
    "ExecutiveIntegrationMetricsCollector",
    "LAUNCH_FINGERPRINT_KEYS",
    # conversation-loop wiring
    "maybe_route_with_executive_integration",
    "_build_eil_response",
    "_get_available_tool_names",
]
