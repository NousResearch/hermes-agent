"""
Self Evolution Plugin
=====================

Agent self-optimization and continuous evolution system.

Architecture:
  - Telemetry: collects tool/session data via hooks
  - Quality Scorer: evaluates session outcomes
  - Dream Engine: nightly reflection at 1:00
  - Evolution Proposer: generates improvement proposals
  - Feishu Notifier: pushes proposals at 19:00 for user approval
  - Evolution Executor: applies approved changes with rollback support
  - Strategy Injector: injects learned hints into sessions

Design references from Claude Code:
  - conversation-analyzer (hookify): dream analysis pattern
  - Ralph Wiggum: iterative evolution with rollback
  - learning-output-style: session-start strategy injection
  - rule_engine (hookify): conditional strategy matching
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register(ctx) -> None:
    """Plugin entry point — called by Hermes PluginManager.

    Registers:
      - 3 hooks: post_tool_call, on_session_end, pre_llm_call
      - 3 slash commands: /evolve, /reflect, /evolution_status
    """
    from self_evolution.db import init_db
    init_db()

    from self_evolution.hooks import register_all as register_hooks
    register_hooks(ctx)

    logger.info("self_evolution plugin loaded: 3 hooks, telemetry active")
