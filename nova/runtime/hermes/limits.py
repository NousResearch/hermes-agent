"""What each NOVA limit actually does **on this runtime**.

Enforcement is a claim about a specific runtime, so it is declared by the adapter rather
than by the platform. Every entry's ``verified_at`` names the call site read to confirm
the claim; the audit behind them is
``docs/platform/BUDGET_ENFORCEMENT_AUDIT.md``.

An entry may not claim an enforcing class without a verified call site — a test asserts it.
"""

from __future__ import annotations

from typing import Mapping, Optional

from nova.policy.limits import (
    HARD_BOUNDARY,
    HARD_PREEMPTIVE,
    OBSERVED_ONLY,
    RECORDED_ONLY,
    SOFT_ADVISORY,
    LimitFact,
)

#: The register of every limit NOVA exposes. Each entry's ``verified_at`` names the call
#: site read to confirm the claim; an entry without one is not allowed to be enforcing.
LIMIT_FACTS: tuple[LimitFact, ...] = (
    LimitFact(
        key="max_turns",
        enforcement=HARD_PREEMPTIVE,
        summary="Caps model iterations for one agent run. The strongest bound available.",
        compiles_to="agent.max_turns",
        verified_at="cli.py:_init_turn_limits -> conversation_loop.py:1514 loop condition",
    ),
    LimitFact(
        key="max_concurrent_tasks",
        enforcement=HARD_PREEMPTIVE,
        summary="Caps how many tasks this agent runs at once.",
        compiles_to="kanban.max_in_progress_per_profile",
        verified_at="gateway/kanban_watchers_dispatcher.py:116",
    ),
    LimitFact(
        key="max_tool_calls_per_run",
        enforcement=HARD_BOUNDARY,
        summary=(
            "Refuses further tool calls in one run once the count is reached. Stops a "
            "runaway tool loop, which is what runaway spend looks like in practice."
        ),
        compiles_to="(NOVA policy plugin)",
        verified_at="hermes_cli/plugins_dispatch.py:_HOOK_TIMEOUT_FAIL_CLOSED_HOOKS",
    ),
    LimitFact(
        key="delegation.max_concurrent_children",
        enforcement=HARD_PREEMPTIVE,
        summary="Caps parallel subagents. Each child consumes tokens independently.",
        compiles_to="delegation.max_concurrent_children",
        verified_at="tools/delegate_tool.py:469",
    ),
    LimitFact(
        key="delegation.max_depth",
        enforcement=HARD_PREEMPTIVE,
        summary="Caps delegation tree depth. Each level multiplies cost.",
        compiles_to="delegation.max_spawn_depth",
        verified_at="tools/delegate_tool.py:188",
    ),
    LimitFact(
        key="delegation.max_child_turns",
        enforcement=HARD_PREEMPTIVE,
        summary="Caps model iterations per subagent.",
        compiles_to="delegation.max_iterations",
        verified_at="tools/delegate_tool.py:451",
    ),
    LimitFact(
        key="delegation.child_timeout_seconds",
        enforcement=HARD_PREEMPTIVE,
        summary="Wall-clock cap per subagent. Floor of 30s; 0 disables.",
        compiles_to="delegation.child_timeout_seconds",
        verified_at="tools/delegate_tool_config.py:_get_child_timeout",
    ),
    LimitFact(
        key="delegation.orchestrator_enabled",
        enforcement=HARD_PREEMPTIVE,
        summary="When false, every subagent is a leaf and cannot spawn further children.",
        compiles_to="delegation.orchestrator_enabled",
        verified_at="tools/delegate_tool.py:189",
    ),
    LimitFact(
        key="soft_wrapup_after_seconds",
        enforcement=SOFT_ADVISORY,
        summary=(
            "Asks the agent to wrap up after this long. NOT a limit: it injects a message "
            "and nothing terminates if the agent keeps going."
        ),
        compiles_to="agent.run_budget_seconds",
        verified_at="agent/conversation_loop.py:119 _maybe_inject_run_budget_wrapup",
    ),
    # Both of these were RECORDED_ONLY on the grounds that NOVA "does not create tasks".
    # Phase 5 made it create them, so the grounds stopped being true and the classification
    # understated what NOVA enforces. The error was in the safe direction and was still an
    # error: /budget renders these words to customers.
    #
    # The qualification is real and stays: they apply to work NOVA submits. A task a human
    # creates with `hermes kanban create` carries whatever that command was given.
    LimitFact(
        key="max_task_runtime_seconds",
        enforcement=HARD_PREEMPTIVE,
        summary=(
            "On work NOVA submits: the dispatcher SIGTERMs, waits, then SIGKILLs a worker "
            "past this cap and re-queues the task. Applied from the agent's spec, or from "
            "an objective step when that states a narrower one."
        ),
        compiles_to="tasks.max_runtime_seconds (on submitted work)",
        verified_at="hermes_cli/kanban_db_dispatch.py:enforce_max_runtime",
    ),
    LimitFact(
        key="max_retries",
        enforcement=HARD_PREEMPTIVE,
        summary=(
            "On work NOVA submits: the consecutive-failure breaker trips on the Nth "
            "failure and blocks the task rather than respawning forever."
        ),
        compiles_to="tasks.max_retries (on submitted work)",
        verified_at="hermes_cli/kanban_db.py DEFAULT_FAILURE_LIMIT / tasks.max_retries",
    ),
    LimitFact(
        key="tokens",
        enforcement=OBSERVED_ONLY,
        summary=(
            "Token counts are measured and reported. They are NOT a ceiling: no plugin can "
            "veto a model call, and the figures lag behind a background writer."
        ),
        compiles_to="",
        verified_at="agent/turn_api_request.py:44 returns None; hermes_state_usage.py:1",
    ),
    LimitFact(
        key="estimated_cost_usd",
        enforcement=OBSERVED_ONLY,
        summary=(
            "The runtime's own cost estimate, reported as-is. NOT a spending limit and not "
            "an invoice — reconcile against the provider's billing."
        ),
        compiles_to="",
        verified_at="hermes_state_schema.py:68 session_model_usage",
    ),
)

_BY_KEY: Mapping[str, LimitFact] = {fact.key: fact for fact in LIMIT_FACTS}


def limit_fact(key: str) -> Optional[LimitFact]:
    return _BY_KEY.get(key)


def enforcement_of(key: str) -> str:
    """This runtime's enforcement class for one limit.

    An unknown key defaults to the weakest class: a new control must be registered here
    before it can claim to stop anything.
    """
    fact = _BY_KEY.get(key)
    return fact.enforcement if fact else OBSERVED_ONLY
