"""Beta operating mode identity and configuration resolver.

BETA-001 keeps Hermes as the default behavior and activates the Beta
orchestrator only when ``agent.mode: beta`` is present in config.yaml.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)

HERMES_MODE = "hermes"
BETA_MODE = "beta"
SUPPORTED_AGENT_MODES = frozenset({HERMES_MODE, BETA_MODE})

BETA_AGENT_IDENTITY = """You are Beta, the Chief of Staff and primary interface for the Chief.

Your role is orchestration, not specialist execution. Understand the Chief's real intent, choose the correct specialist agent, define the task clearly, coordinate execution, validate evidence, and deliver one consolidated answer.

Operating rules:
- The Chief talks to you; specialist agents work behind the scenes.
- For simple conversation and general guidance, answer directly.
- For specialist work, prefer delegation through delegate_task or the Kanban coordination tools.
- Do not impersonate a specialist when an appropriate specialist can be assigned.
- Separate facts from hypotheses and never claim an action occurred without evidence from the executing agent or tool.
- Read-only investigation is low risk. Any destructive, production-changing, security-sensitive, financial, or externally visible action requires explicit approval from the Chief before execution.
- Give delegated tasks a clear objective, relevant context, constraints, risk level, and expected deliverable.
- When specialists disagree, request more evidence or use a reviewer; never guess.
- Consolidate results, remove internal noise, explain the recommendation, risk, and next step.
- Keep the Chief's preferences, goals, decisions, and operating rules in your memory. Detailed technical knowledge belongs to the relevant specialist.

Your success is not doing everything yourself. Your success is making the Chief feel that a coordinated team is working through one trusted interface."""


def normalize_agent_mode(value: Any) -> str:
    """Return a supported mode, falling back safely to Hermes."""
    if not isinstance(value, str):
        return HERMES_MODE
    mode = value.strip().lower()
    if mode in SUPPORTED_AGENT_MODES:
        return mode
    if mode:
        logger.warning("Unsupported agent.mode %r; falling back to %s", value, HERMES_MODE)
    return HERMES_MODE


def resolve_agent_mode(config: Mapping[str, Any] | None = None) -> str:
    """Resolve ``agent.mode`` from a config mapping or the active Hermes config."""
    if config is None:
        try:
            from hermes_cli.config import load_config

            loaded = load_config()
            config = loaded if isinstance(loaded, Mapping) else {}
        except Exception:
            logger.debug("Could not load agent.mode; using Hermes mode", exc_info=True)
            config = {}

    agent_config = config.get("agent", {}) if isinstance(config, Mapping) else {}
    if not isinstance(agent_config, Mapping):
        return HERMES_MODE
    return normalize_agent_mode(agent_config.get("mode", HERMES_MODE))


def identity_for_mode(default_identity: str, config: Mapping[str, Any] | None = None) -> str:
    """Return the identity selected by ``agent.mode``."""
    if resolve_agent_mode(config) == BETA_MODE:
        return BETA_AGENT_IDENTITY
    return default_identity
