"""
In-process autonomy state for AUTONOMOUS_INITIATION_PATH_V1.

This module holds the only mutable runtime state owned by the autonomy
initiation path. It is intentionally:

  * module-level (not persisted to disk)
  * thread-unsafe by design (Hermes runs a single worker per process)
  * reset by ``reset()`` between tests and at the end of every canary

The state is intentionally NOT stored in config.yaml, .env, profiles, or
any other persistent location. Activation is ephemeral and scope-limited.
"""

from __future__ import annotations

import threading
import time
from typing import Optional


# Lock for the rare case two threads call into the initiator in the same
# process (e.g. during teardown). Single-process workers do not need this
# in normal operation but the tests fire concurrent attempts.
_LOCK = threading.Lock()


# In-process flag. False by default (autonomy off). When the canary is
# authorized, a human-supervised test harness or the dispatcher sets this
# to True. The state is reset by reset() at the end of every canary.
_ENABLED: bool = False


# When set to True, no autonomous initiation can succeed. The kill switch
# is read before any policy gate. The dispatcher, the operator, or the
# SUCCESS_EVALUATOR may set this.
_KILL_SWITCH: bool = False


# The id of the currently active autonomous objective (if any). The
# concurrency budget is enforced via this slot: while non-None, no other
# autonomous objective may be admitted.
_ACTIVE_OBJECTIVE_ID: Optional[str] = None


# The task id of the currently active autonomous run (the task created by
# the initiator). Used by tests and by post-run accounting to find the
# task that the autonomous admission created.
_ACTIVE_TASK_ID: Optional[str] = None


# When the current active objective was admitted. Set by attempt_autonomous
# _initiation on success, cleared by reset().
_ACTIVE_OBJECTIVE_INITIATED_AT: Optional[float] = None


# Policy version this state was configured for. Defaults to "1.0.0".
# attempt_autonomous_initiation compares the spec's policy_version to this.
_POLICY_VERSION: str = "1.0.0"


# Default scope: which profile the autonomous tasks target. Defaults to the
# profile of the current process (resolved at enable time, not stored).
_PROFILE: Optional[str] = None


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def enable(*, policy_version: str = "1.0.0", profile: Optional[str] = None) -> None:
    """Activate the autonomy initiation path. Ephemeral and in-process only.

    Idempotent: calling enable() multiple times is safe; the most recent
    policy_version and profile win. Does not touch config, profile, .env,
    or any persistent file.
    """
    global _ENABLED, _KILL_SWITCH, _POLICY_VERSION, _PROFILE
    with _LOCK:
        _ENABLED = True
        _KILL_SWITCH = False
        _POLICY_VERSION = policy_version
        _PROFILE = profile


def disable() -> None:
    """Deactivate the autonomy initiation path and clear all slots.

    Calling disable() also clears the kill switch, the active objective
    id, the active task id, and the initiated-at timestamp. Used at the
    end of every canary and between tests.
    """
    global _ENABLED, _KILL_SWITCH, _ACTIVE_OBJECTIVE_ID, _ACTIVE_TASK_ID
    global _ACTIVE_OBJECTIVE_INITIATED_AT, _POLICY_VERSION, _PROFILE
    with _LOCK:
        _ENABLED = False
        _KILL_SWITCH = False
        _ACTIVE_OBJECTIVE_ID = None
        _ACTIVE_TASK_ID = None
        _ACTIVE_OBJECTIVE_INITIATED_AT = None


def fire_kill_switch() -> None:
    """Set the kill switch. Subsequent attempts return PAUSED_NEEDS_HUMAN.

    Does not change _ENABLED. Tests and operators can fire the kill switch
    while autonomy is enabled, simulating an emergency stop.
    """
    global _KILL_SWITCH
    with _LOCK:
        _KILL_SWITCH = True


def clear_kill_switch() -> None:
    """Clear the kill switch. Used by tests; not exposed to operators."""
    global _KILL_SWITCH
    with _LOCK:
        _KILL_SWITCH = False


def is_enabled() -> bool:
    """Return True if autonomy has been enabled (and not disabled since)."""
    with _LOCK:
        return _ENABLED


def is_kill_switch_active() -> bool:
    """Return True if the kill switch has been fired (and not cleared)."""
    with _LOCK:
        return _KILL_SWITCH


def get_active_objective_id() -> Optional[str]:
    """Return the id of the currently active autonomous objective, if any."""
    with _LOCK:
        return _ACTIVE_OBJECTIVE_ID


def get_active_task_id() -> Optional[str]:
    """Return the id of the task created by the most recent successful
    attempt, if any. Used by tests to confirm exactly-one semantics."""
    with _LOCK:
        return _ACTIVE_TASK_ID


def get_active_initiated_at() -> Optional[float]:
    """Return the unix timestamp the current active objective was admitted."""
    with _LOCK:
        return _ACTIVE_OBJECTIVE_INITIATED_AT


def get_policy_version() -> str:
    """Return the policy version this state was configured for."""
    with _LOCK:
        return _POLICY_VERSION


def get_profile() -> Optional[str]:
    """Return the profile this autonomy state is scoped to, if any."""
    with _LOCK:
        return _PROFILE


def reserve_active(objective_id: str, task_id: str) -> None:
    """Internal: called by the initiator when an objective is admitted.

    Sets the active objective id and task id. The initiator must already
    have passed all policy gates before calling this.
    """
    global _ACTIVE_OBJECTIVE_ID, _ACTIVE_TASK_ID, _ACTIVE_OBJECTIVE_INITIATED_AT
    with _LOCK:
        _ACTIVE_OBJECTIVE_ID = objective_id
        _ACTIVE_TASK_ID = task_id
        _ACTIVE_OBJECTIVE_INITIATED_AT = time.time()


def clear_active() -> None:
    """Internal: called by the initiator when the active run finishes.

    Does NOT change _ENABLED or _KILL_SWITCH. The operator decides whether
    to disable the autonomy state or leave it enabled for the next run.
    """
    global _ACTIVE_OBJECTIVE_ID, _ACTIVE_TASK_ID, _ACTIVE_OBJECTIVE_INITIATED_AT
    with _LOCK:
        _ACTIVE_OBJECTIVE_ID = None
        _ACTIVE_TASK_ID = None
        _ACTIVE_OBJECTIVE_INITIATED_AT = None


def reset() -> None:
    """Reset all state. Intended for use in tests between cases."""
    disable()
    clear_kill_switch()
    clear_active()
