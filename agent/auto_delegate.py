#!/usr/bin/env python3
"""
Auto-delegate (dispatcher mode) — opt-in ``delegation.auto_delegate`` support.

When enabled, every user turn is handed to a child agent via the existing
``delegate_task`` machinery before the main loop starts. The parent only
receives the child's summary; the parent conversation never accumulates the
child's intermediate tool calls or reasoning.

Design constraints (see AGENTS.md):

- **Prompt caching is sacred.** This module never touches the system prompt,
  never injects a synthetic user message, and never mutates the message list
  mid-loop. The turn is handed off *before* the main loop begins, exactly like
  the existing ``api_mode == "codex_app_server"`` early return in
  ``agent/conversation_loop.py``. The parent's ``messages`` list only gains a
  real user message + one assistant summary message (normal role alternation).
- **Core narrow waist.** No new model tool is added — ``delegate_task`` is
  already a core tool. The footprint is one boolean in the existing
  ``delegation`` config block (default ``False``), documented in
  ``cli-config.yaml.example``.

Failure is always graceful: if delegation is disabled, misconfigured, or the
child errors out, the turn falls through to the normal loop — dispatcher mode
never hard-fails a turn.

v0.21.3 adaptation notes: the turn loop was refactored around the ``_LoopState``
dataclass (``agent/conversation_loop.py``), so the handoff takes the same four
values the codex app-server branch takes (``user_message`` / ``messages`` /
``effective_task_id``) as explicit keyword arguments instead of loop locals. The
returned dict is shaped like ``agent/turn_finalizer.py::finalize_turn``'s result
(plus ``auto_delegated`` / ``task_id``) so the CLI / gateway / TUI persist and
render the turn exactly as a normal terminal turn.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Parent-side message for the child's context: the child has no access to the
# parent's transcript, so the goal must be self-contained.
_DISPATCHER_CONTEXT = (
    "Dispatcher mode: this turn was auto-delegated by the parent agent "
    "(delegation.auto_delegate). Complete the goal as a self-contained task "
    "and return a concise final summary. The parent has no visibility into "
    "your intermediate steps."
)


def _load_delegation_config() -> dict:
    """Read the ``delegation`` config block via the shared persistent loader.

    Same resolution path as ``tools/delegate_tool.py::_load_config``:
    prefers ``load_config_readonly()`` (honors the active HERMES_HOME/profile)
    and falls back to the legacy ``cli.CLI_CONFIG`` loader. Read-only — do not
    mutate the returned dict.
    """
    try:
        from hermes_cli.config import load_config_readonly

        full = load_config_readonly()
        cfg = full.get("delegation") or {}
        if isinstance(cfg, dict):
            return cfg
    except Exception:
        logger.debug("auto_delegate: shared config loader unavailable", exc_info=True)
    try:
        from cli import CLI_CONFIG

        cfg = CLI_CONFIG.get("delegation") or {}
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        logger.debug("auto_delegate: legacy config loader unavailable", exc_info=True)
        return {}


def auto_delegate_enabled() -> bool:
    """Return whether dispatcher mode is on (``delegation.auto_delegate``).

    Defaults to ``False`` — the behavior is strictly opt-in.
    """
    try:
        from utils import is_truthy_value

        return is_truthy_value(_load_delegation_config().get("auto_delegate", False))
    except Exception:
        return False


def _extract_first_summary(payload: Any) -> Optional[str]:
    """Pull the first child summary out of a ``delegate_task`` JSON payload.

    ``delegate_task`` returns ``{"results": [{"status": "ok", "summary": ...,
    "task_index": ...}], "total_duration_seconds": ...}`` (unchanged in
    v0.21.3). Returns ``None`` when there is no usable summary (error/empty
    results).
    """
    if not isinstance(payload, dict):
        return None
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return None
    first = results[0]
    if not isinstance(first, dict):
        return None
    status = str(first.get("status") or "").lower()
    summary = first.get("summary")
    if status != "ok" or not isinstance(summary, str) or not summary.strip():
        return None
    return summary.strip()


def _shape_terminal_result(
    agent: Any, final_response: str, messages: List[Dict[str, Any]], effective_task_id: str
) -> Dict[str, Any]:
    """Build a turn result mirroring ``turn_finalizer.finalize_turn``'s keys.

    The v0.21.3 terminal result carries more keys than the pre-refactor one
    (``turn_exit_reason``, usage/cost counters, ``model``/``provider``/
    ``base_url``, ``session_id``, ...). Every key is read with a ``getattr``
    fallback so a partially-initialized agent (tests, alternate hosts) cannot
    raise here. ``auto_delegated`` marks the turn so callers/telemetry can tell
    a handoff from a normal turn.
    """
    try:
        from agent.turn_finalizer import _SESSION_COST_KEYS, _SESSION_TOKEN_KEYS, _last_turn_reasoning
    except Exception:  # pragma: no cover - defensive; imports are cheap and stable
        _SESSION_TOKEN_KEYS = ()
        _SESSION_COST_KEYS = ()
        _last_turn_reasoning = None

    last_reasoning = None
    if _last_turn_reasoning is not None:
        try:
            last_reasoning = _last_turn_reasoning(messages)
        except Exception:
            last_reasoning = None

    result: Dict[str, Any] = {
        "final_response": final_response,
        "last_reasoning": last_reasoning,
        "messages": messages,
        # The handoff counts as the parent's one unit of work for the turn
        # (children bill their own session) — same convention as the codex
        # app-server handoff / pre-refactor dispatcher mode.
        "api_calls": 1,
        "completed": True,
        "turn_exit_reason": "auto_delegated",
        "failed": False,
        "partial": False,
        "interrupted": False,
        "response_transformed": False,
        "pre_transform_response": "",
        "response_previewed": getattr(agent, "_response_was_previewed", False),
        "model": getattr(agent, "model", None),
        "provider": getattr(agent, "provider", None),
        "base_url": getattr(agent, "base_url", None),
        **{key: getattr(agent, f"session_{key}", 0) for key in _SESSION_TOKEN_KEYS},
        **{key: getattr(agent, f"session_{key}", None) for key in _SESSION_COST_KEYS},
        "session_id": getattr(agent, "session_id", None),
        # Dispatcher-mode provenance (extra keys; normal-turn consumers ignore them).
        "auto_delegated": True,
        "task_id": effective_task_id,
    }
    return result


def try_auto_delegate(
    agent: Any,
    user_message: Any = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    effective_task_id: str = "",
) -> Optional[Dict[str, Any]]:
    """Hand the turn to a child agent when dispatcher mode is enabled.

    Returns a terminal turn-result dict (shaped like a normal
    ``run_conversation`` result, plus ``auto_delegated``/``task_id``) on
    success, or ``None`` to fall through to the normal loop. ``None`` is
    returned whenever the mode is off, the turn is not delegatable (empty
    message, no agent context, already inside a delegated child), or delegation
    failed — dispatcher mode never hard-fails a turn.

    Called from ``_run_conversation_turn`` right after the
    ``codex_app_server`` turn-handoff branch, passing the ``_LoopState`` slots
    (``s.user_message`` / ``s.messages`` / ``s.effective_task_id``).
    """
    if not auto_delegate_enabled():
        return None
    if agent is None or not isinstance(user_message, str) or not user_message.strip():
        return None
    # Never re-delegate from inside a delegated child: a nested handoff would
    # burn budget re-dispatching the same goal (the depth cap would reject the
    # spawn and log a warning on every child turn, and the children's own
    # config is inherited). Dispatcher mode is a top-level-turn feature.
    if getattr(agent, "_delegate_depth", 0):
        logger.debug("auto_delegate: skipping inside a delegated child agent")
        return None

    # Deferred import: tools.delegate_tool imports the whole tool registry;
    # conversation_loop already triggers it, but keeping this module
    # import-light avoids circular-import surprises for unit tests.
    try:
        from tools.delegate_tool import delegate_task

        result_str = delegate_task(
            goal=user_message.strip(),
            context=_DISPATCHER_CONTEXT,
            parent_agent=agent,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning(
            "auto_delegate: delegation failed, falling back to normal loop: %s",
            exc,
        )
        return None

    if not isinstance(result_str, str):
        logger.warning("auto_delegate: unexpected delegate_task return type %s", type(result_str))
        return None

    try:
        payload = json.loads(result_str)
    except Exception:
        logger.warning("auto_delegate: unparseable delegate_task result; falling back")
        return None

    final_response = _extract_first_summary(payload)
    if not final_response:
        logger.warning("auto_delegate: no usable summary; falling back to normal loop")
        return None

    # Shape the result exactly like a normal terminal turn so callers
    # (CLI/gateway/TUI) persist and render it identically. The parent's
    # messages list gains only the user message (already present) plus this
    # one assistant summary — normal role alternation, cache-safe. Never
    # mutate the input list in place: append to a copy.
    result_messages = list(messages) if isinstance(messages, list) else []
    result_messages.append({"role": "assistant", "content": final_response})
    return _shape_terminal_result(agent, final_response, result_messages, effective_task_id)
