"""Turn-end policy hooks (``pre_turn_end``).

Fires once per turn when the agent is about to deliver a final text answer
(no tool calls), regardless of whether files were edited. Complements
``pre_verify`` (which only fires after code edits) with a general-purpose
continuation gate for policy checks that apply to every turn — for example,
detecting responses that promise execution ('voy a X', 'I will run Y') but
emit no tool call.

The hook is opt-in: it only fires when at least one ``pre_turn_end`` callback
is registered. Without a registered hook, this module is a no-op and adds
zero cost to the turn loop.
"""

from __future__ import annotations

from typing import Any, Optional

from utils import is_truthy_value

DEFAULT_MAX_TURN_END_NUDGES = 2


def max_turn_end_nudges(config: Optional[dict[str, Any]] = None) -> int:
    """Bound on consecutive ``pre_turn_end`` continue directives per turn (>= 0).

    Default 2 (more conservative than ``pre_verify``'s 3): mode-error-style
    checks are typically binary — either the response emitted a tool call or
    it didn't — so a third nudge rarely changes the outcome.
    """
    try:
        raw = _agent_cfg(config).get("max_turn_end_nudges")
        if raw is None:
            return DEFAULT_MAX_TURN_END_NUDGES
        return max(0, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_MAX_TURN_END_NUDGES


def turn_end_hook_enabled(config: Optional[dict[str, Any]] = None) -> bool:
    """Opt-out via ``agent.pre_turn_end_hook: false``. Default enabled."""
    val = _agent_cfg(config).get("pre_turn_end_hook", True)
    return is_truthy_value(val, default=True)


def _agent_cfg(config: Optional[dict[str, Any]]) -> dict[str, Any]:
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:
            config = {}
    agent_cfg = config.get("agent") if isinstance(config, dict) else None
    return agent_cfg if isinstance(agent_cfg, dict) else {}


__all__ = [
    "DEFAULT_MAX_TURN_END_NUDGES",
    "max_turn_end_nudges",
    "turn_end_hook_enabled",
]
