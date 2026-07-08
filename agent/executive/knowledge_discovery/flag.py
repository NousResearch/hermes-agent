"""Default-OFF flag resolver for Knowledge Discovery (B1 wiring).

Mirrors the pattern of agent.executive.flag.resolve_v2_enabled:
1. Per-instance attribute on ``agent`` (highest priority)
2. Environment variable ``HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED``
3. Default: False

This module is PURE: no I/O, no global state mutation, no file writes.
The function is safe to call from hot paths (e.g. the
ObjectiveEngine.discover_evidence_pack gate).

The engine is **off by default**. Even with the env var set, the wiring
remains a no-op unless an EvidencePackEngine is explicitly injected into
the ObjectiveEngine (via the ``evidence_engine=`` constructor kwarg). This
matches the design in
``~/.hermes/reports/HERMES_B1_INTEGRATION_DESIGN_READONLY/interface_contracts.md``
section 7.
"""

from __future__ import annotations

import os
from typing import Any

_ENV_VAR = "HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED"
_TRUTHY = {"1", "true", "yes", "on"}


def resolve_knowledge_discovery_enabled(agent: Any | None = None) -> bool:
    """Resolve whether B1 knowledge discovery is enabled.

    Resolution order (any truthy -> enabled):

    1. ``agent._executive_knowledge_discovery_enabled`` (per-instance,
       highest priority — used by tests to enable in-process).
    2. ``HERMES_EXECUTIVE_KNOWLEDGE_DISCOVERY_ENABLED`` env var
       (truthy: ``1``, ``true``, ``yes``, ``on``; case-insensitive).
    3. Default: ``False``.

    Returns ``False`` on any unexpected input (never raises).
    """
    try:
        if agent is not None:
            flag = getattr(agent, "_executive_knowledge_discovery_enabled", None)
            if flag:
                return True
    except Exception:
        pass
    try:
        env = os.environ.get(_ENV_VAR, "")
        if env and env.strip().lower() in _TRUTHY:
            return True
    except Exception:
        pass
    return False
