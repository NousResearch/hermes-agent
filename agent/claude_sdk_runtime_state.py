"""Per-turn state of the claude-agent-sdk runtime, threaded through its phases.

``run_claude_agent_sdk_turn`` creates one ``_SdkTurnState`` per call and hands
it to the phase helpers in the ``claude_sdk_runtime_*`` siblings, which read
and update it in place; the facade assembles the result from it at the end.
The shape follows ``agent/conversation_loop.py::_LoopState``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class _SdkTurnState:
    """Every local the SDK turn threads through its phase helpers.

    Inputs (set by the facade before any phase runs): ``user_input`` (the
    coerced prompt), ``original_user_message`` (for the external memory sync),
    ``messages`` (the live transcript the projected rows are appended to) and
    ``messages_before_attempt`` (a deep copy taken AFTER the attempt-effects
    reset, so ``effects.mutated`` compares against the pre-attempt transcript).

    Per-turn wiring: ``on_interim_assistant`` / ``on_tool_iteration`` are the
    visibility callbacks refreshed every turn by ``_refresh_turn_visibility``.

    Outcome, filled by the phases in order: ``turn``, ``resumed`` and
    ``turn_session_cwd`` by ``_run_sdk_attempts``; ``effects``,
    ``failover_reason`` and ``user_interrupted`` by
    ``_reconcile_turn_outcome``; ``usage_result`` and ``should_review_skills``
    by ``_account_turn``.

    ``turn_session_cwd`` is the workspace the SDK session that produced this
    turn was actually created in, sampled before ``run_turn``. The resume id is
    persisted BOUND to it, so a later turn in a different workspace declines
    the binding instead of resuming a foreign session.
    """

    user_input: Any
    original_user_message: Any
    messages: List[Dict[str, Any]]
    messages_before_attempt: List[Dict[str, Any]]
    on_interim_assistant: Any = None
    on_tool_iteration: Any = None
    turn: Any = None
    resumed: bool = False
    turn_session_cwd: Any = None
    effects: Any = None
    failover_reason: Any = None
    user_interrupted: bool = False
    usage_result: Dict[str, Any] = field(default_factory=dict)
    should_review_skills: bool = False
