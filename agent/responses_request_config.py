"""Responses-only request controls read from the shared agent config section."""
from __future__ import annotations

import logging
from typing import Any

from agent.codex_proxy_replay_window import proxy_replay_turns_from_config

logger = logging.getLogger(__name__)


def apply_responses_request_config(agent: Any, section: dict[str, Any]) -> None:
    """Parse Responses verbosity and the proxy-only encrypted replay window."""
    verbosity = str(section.get("text_verbosity") or "").strip().lower()
    if verbosity and verbosity not in {"low", "medium", "high"}:
        logger.warning("Unknown agent.text_verbosity %r; expected low, medium or high — ignoring", verbosity)
        verbosity = ""
    agent.text_verbosity = verbosity or None
    agent.codex_proxy_replay_turns = proxy_replay_turns_from_config(section.get("codex_proxy_replay_turns"))
