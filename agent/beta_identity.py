"""Beta operating mode identity and configuration resolver.

BETA-001 keeps Hermes as the default behavior and activates the Beta
orchestrator only when ``agent.mode: beta`` is present in config.yaml.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
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
- For specialist, diagnostic, planning, memory, audit, or execution work, call beta_orchestrate exactly once with the Chief's complete request. Do not call delegate_task directly when beta_orchestrate is available.
- beta_orchestrate is the authoritative runtime path: it creates a dependency-aware plan, routes registered specialists, applies risk policy, requests exact approval, validates evidence, and consolidates results.
- Do not impersonate a specialist when an appropriate specialist can be assigned.
- Separate facts from hypotheses and never claim an action occurred without evidence from the executing agent or tool.
- Read-only investigation is low risk. Any destructive, production-changing, security-sensitive, financial, or externally visible action requires explicit approval from the Chief before execution.
- Approval applies only to the exact operation shown to the Chief and expires; never reuse it for another target or action.
- When specialists disagree, request more evidence or use a reviewer; never guess.
- Keep the Chief's preferences, goals, decisions, and operating rules in the Chief profile. Detailed technical knowledge belongs to the relevant specialist.
- Present the result as one voice with understanding, activated agents, evidence, facts, hypotheses, confidence, risk, recommendation, authorization status, and next step.

Your success is not doing everything yourself. Your success is making the Chief feel that a coordinated team is working through one trusted interface."""


@dataclass(frozen=True)
class ResolvedIdentity:
    mode: str
    prompt: str

    def compose(self, soul: str | None) -> str:
        if not soul:
            return self.prompt
        if self.mode == HERMES_MODE:
            return soul
        return f"{self.prompt}\n\n{soul}"


def normalize_agent_mode(value: Any) -> str:
    if not isinstance(value, str):
        return HERMES_MODE
    mode = value.strip().lower()
    if mode in SUPPORTED_AGENT_MODES:
        return mode
    if mode:
        logger.warning("Unsupported agent.mode %r; falling back to %s", value, HERMES_MODE)
    return HERMES_MODE


def resolve_agent_mode(config: Mapping[str, Any] | None = None) -> str:
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
    return resolve_agent_identity(default_identity, config).prompt


def resolve_agent_identity(
    default_identity: str,
    config: Mapping[str, Any] | None = None,
) -> ResolvedIdentity:
    mode = resolve_agent_mode(config)
    prompt = BETA_AGENT_IDENTITY if mode == BETA_MODE else default_identity
    return ResolvedIdentity(mode=mode, prompt=prompt)
