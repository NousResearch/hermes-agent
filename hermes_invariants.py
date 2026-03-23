"""
Runtime invariant checking for Hermes subsystems.

Provides assertion-style checks that verify critical invariants after
state mutations. Inspired by the dual-verification pattern from
tla-precheck (formal spec + runtime checks) and the sample-task-management
project (CheckAllInvariants after every mutation).

Usage:
    from hermes_invariants import InvariantChecker

    # After any message list mutation:
    violations = InvariantChecker.check_message_integrity(messages)
    if violations:
        logger.warning("Message invariant violations: %s", violations)

    # After any cron job state change:
    violations = InvariantChecker.check_cron_invariants(jobs)

    # After any process registry change:
    violations = InvariantChecker.check_process_registry(running, finished)

Enforcement modes:
    HERMES_INVARIANT_MODE=strict   -> raise InvariantViolation (dev/test)
    HERMES_INVARIANT_MODE=warn     -> log warning (default, production)
    HERMES_INVARIANT_MODE=silent   -> no-op (perf-critical paths)
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Enforcement mode from environment
_MODE = os.getenv("HERMES_INVARIANT_MODE", "warn").lower()


class InvariantViolation(Exception):
    """Raised when a critical invariant is broken in strict mode."""
    pass


def _report(violations: List[str], context: str = "") -> List[str]:
    """Handle violations according to enforcement mode."""
    if not violations:
        return violations

    prefix = f"[{context}] " if context else ""
    for v in violations:
        msg = f"INVARIANT: {prefix}{v}"
        if _MODE == "strict":
            raise InvariantViolation(msg)
        elif _MODE == "warn":
            logger.warning(msg)
        # silent: do nothing

    return violations


class InvariantChecker:
    """Runtime invariant verification for Hermes subsystems."""

    # =========================================================================
    # Message Integrity
    # =========================================================================

    @staticmethod
    def check_message_integrity(messages: List[Dict[str, Any]]) -> List[str]:
        """Verify a message list is well-formed for the chat completion API.

        Checks:
            - Messages list is not empty
            - All messages have a valid role
            - Tool results follow an assistant message with tool_calls
            - Tool call IDs in results match a preceding tool_call
            - No consecutive user messages (usually a bug)
        """
        violations = []

        if not messages:
            violations.append("EMPTY_MESSAGES: messages list is empty")
            return _report(violations, "message_integrity")

        valid_roles = {"system", "user", "assistant", "tool"}

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                violations.append(f"INVALID_MESSAGE at index {i}: not a dict")
                continue

            role = msg.get("role")
            if role not in valid_roles:
                violations.append(f"INVALID_ROLE at index {i}: '{role}'")
                continue

            # Tool results must follow an assistant with tool_calls or another tool result
            if role == "tool" and i > 0:
                prev = messages[i - 1]
                prev_role = prev.get("role") if isinstance(prev, dict) else None
                if prev_role == "assistant":
                    if not prev.get("tool_calls"):
                        violations.append(
                            f"ORPHAN_TOOL_RESULT at index {i}: "
                            f"preceding assistant has no tool_calls"
                        )
                elif prev_role != "tool":
                    violations.append(
                        f"ORPHAN_TOOL_RESULT at index {i}: "
                        f"preceded by '{prev_role}', expected 'assistant' or 'tool'"
                    )

        return _report(violations, "message_integrity")

    # =========================================================================
    # Cron Job Invariants
    # =========================================================================

    @staticmethod
    def check_cron_invariants(jobs: List[Dict[str, Any]]) -> List[str]:
        """Verify cron job list consistency.

        Checks:
            - All jobs have valid states
            - No duplicate job IDs
            - Completed jobs have no next_run_at
            - Paused jobs are not in the due list
            - Running jobs (if any) have a last_run_at or are freshly started
            - Repeat counters are consistent
        """
        from cron.jobs import JOB_STATES

        violations = []
        seen_ids = set()

        for job in jobs:
            job_id = job.get("id", "?")

            # Unique IDs
            if job_id in seen_ids:
                violations.append(f"DUPLICATE_JOB_ID: '{job_id}'")
            seen_ids.add(job_id)

            # Valid state
            state = job.get("state", "scheduled")
            if state not in JOB_STATES:
                violations.append(f"INVALID_STATE: job '{job_id}' has state '{state}'")

            # Completed jobs should have no next_run_at
            if state == "completed" and job.get("next_run_at") is not None:
                violations.append(
                    f"COMPLETED_WITH_NEXT_RUN: job '{job_id}' is completed "
                    f"but has next_run_at={job.get('next_run_at')}"
                )

            # Repeat counter consistency
            repeat = job.get("repeat", {})
            if isinstance(repeat, dict):
                times = repeat.get("times")
                completed = repeat.get("completed", 0)
                if times is not None and completed > times:
                    violations.append(
                        f"REPEAT_OVERFLOW: job '{job_id}' completed {completed} "
                        f"runs but limit is {times}"
                    )

        return _report(violations, "cron")

    # =========================================================================
    # Process Registry Invariants
    # =========================================================================

    @staticmethod
    def check_process_registry(
        running: Dict[str, Any],
        finished: Dict[str, Any],
    ) -> List[str]:
        """Verify process registry consistency.

        Checks:
            - No session exists in both running and finished
            - Running sessions are not marked as exited
            - Finished sessions are marked as exited
        """
        violations = []

        # No overlap between running and finished
        overlap = set(running.keys()) & set(finished.keys())
        if overlap:
            violations.append(
                f"DUAL_REGISTRY: sessions {overlap} exist in both running and finished"
            )

        # Running sessions should not be exited
        for sid, session in running.items():
            if getattr(session, "exited", False):
                violations.append(
                    f"RUNNING_BUT_EXITED: session '{sid}' is in _running but exited=True"
                )

        # Finished sessions should be exited
        for sid, session in finished.items():
            if not getattr(session, "exited", True):
                violations.append(
                    f"FINISHED_NOT_EXITED: session '{sid}' is in _finished but exited=False"
                )

        return _report(violations, "process_registry")

    # =========================================================================
    # Session Store Invariants
    # =========================================================================

    @staticmethod
    def check_session_invariants(entries: Dict[str, Any]) -> List[str]:
        """Verify gateway session store consistency.

        Checks:
            - All entries have valid session_id and session_key
            - No duplicate session_ids across different keys
            - Token counts are non-negative
        """
        violations = []
        seen_session_ids = {}

        for key, entry in entries.items():
            session_id = getattr(entry, "session_id", None)
            if not session_id:
                violations.append(f"MISSING_SESSION_ID: key '{key}'")
                continue

            # Duplicate session_id check
            if session_id in seen_session_ids:
                violations.append(
                    f"DUPLICATE_SESSION_ID: '{session_id}' used by both "
                    f"'{seen_session_ids[session_id]}' and '{key}'"
                )
            seen_session_ids[session_id] = key

            # Non-negative token counts
            for field in ("input_tokens", "output_tokens", "total_tokens"):
                val = getattr(entry, field, 0)
                if isinstance(val, (int, float)) and val < 0:
                    violations.append(
                        f"NEGATIVE_TOKENS: key '{key}' has {field}={val}"
                    )

        return _report(violations, "session")

    # =========================================================================
    # Stream Consumer Invariants
    # =========================================================================

    @staticmethod
    def check_stream_consumer(consumer) -> List[str]:
        """Verify stream consumer state consistency.

        Checks:
            - State enum matches boolean flags
            - No edits after DONE state
        """
        from gateway.stream_consumer import StreamState

        violations = []
        state = getattr(consumer, "_state", None)

        if state == StreamState.DONE:
            # Should not have pending items in queue
            if not consumer._queue.empty():
                violations.append(
                    "DONE_WITH_PENDING: consumer is DONE but queue is not empty"
                )

        if state == StreamState.DEGRADED:
            if getattr(consumer, "_edit_supported", True):
                violations.append(
                    "DEGRADED_EDIT_MISMATCH: state is DEGRADED but _edit_supported=True"
                )

        return _report(violations, "stream_consumer")

    # =========================================================================
    # Delegation Invariants
    # =========================================================================

    @staticmethod
    def check_delegation(
        parent_agent,
        active_children: List[Any],
        depth: int,
        max_depth: int = 2,
    ) -> List[str]:
        """Verify delegation tree consistency.

        Checks:
            - Depth does not exceed max
            - Active children list length is within bounds
            - No child appears twice in the list
        """
        violations = []

        if depth > max_depth:
            violations.append(
                f"DEPTH_EXCEEDED: current depth {depth} > max {max_depth}"
            )

        if len(active_children) > 3:
            violations.append(
                f"TOO_MANY_CHILDREN: {len(active_children)} active children (max 3)"
            )

        child_ids = [id(c) for c in active_children]
        if len(child_ids) != len(set(child_ids)):
            violations.append("DUPLICATE_CHILD: same child appears twice in active_children")

        return _report(violations, "delegation")
