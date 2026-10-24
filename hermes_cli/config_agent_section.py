"""Agent-section tolerance and normalization, moved verbatim from ``hermes_cli.config``
so that file keeps going down toward its code-health line cap (moved code keeps its
cap; see root AGENTS.md). Callers import these names from :mod:`hermes_cli.config`,
which re-exports them, so existing import paths are unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _tolerant_agent_section(config: dict[str, Any]) -> dict[str, Any]:
    """``config['agent']`` as a fresh dict, dropping a malformed spelling (#144067).

    A non-mapping ``agent:`` value (``fast``, ``true``, ``90``, ``[max_turns]``)
    used to die inside ``dict(config.get('agent') or {})`` with a raw
    ValueError/TypeError, taking down every config-loading command — including
    ``hermes config show`` and ``hermes status``, the two a user needs to fix
    the file. Ignore the section with a warning naming the key, like the other
    malformed sections; the bad spelling is dropped so a save normalizes it away."""
    value = config.get("agent")
    if isinstance(value, dict):
        return dict(value)
    if value is not None:
        logger.warning(
            "Ignoring config.yaml section 'agent': expected a mapping, got %s (%r); "
            "config commands keep working so the file can be fixed",
            type(value).__name__, value,
        )
        config.pop("agent", None)
    return {}


def _normalize_max_turns_config(config: dict[str, Any]) -> dict[str, Any]:
    """Move legacy root-level ``max_turns`` under ``agent``; the schema default is injected only
    when the user set max_turns somewhere (so save_config can otherwise omit it)."""
    config = dict(config)
    agent_config = _tolerant_agent_section(config)
    if "max_turns" in config and "max_turns" not in agent_config:
        agent_config["max_turns"] = config["max_turns"]
    if agent_config or "agent" in config:  # a sparse save must not grow an `agent: {}` section
        config["agent"] = agent_config
    config.pop("max_turns", None)
    return config
