"""Poller-argv streak: a poller re-invoked with identical args must form a streak even when
its result changes (that is what polling looks like), while non-poller tools keep the original
(tool, args, result) semantics.

Regression for the 2026-11-09 measurement: one session spent 96 of 297 turns re-running
`gh pr checks`, ~22M billed tokens, because the result changed on every poll (CI progressing)
so the identical-call streak never reached its threshold.
"""

from __future__ import annotations

import pytest

from agent.tool_guardrails import (
    POLLER_ARGV_PREFIXES,
    STALL_GUARD_IDENTICAL_CALL_THRESHOLD,
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    _is_poller_args,
)


@pytest.mark.parametrize(
    "command",
    [
        "gh pr checks 123",
        "gh pr checks 123 --watch",
        "gh run view 999",
        "gh run list --limit 5",
        "gh pr view 42",
        "kubectl get pods",
        "docker ps -a",
        "gcloud builds describe abc",
    ],
)
def test_poller_argv_recognised(command: str) -> None:
    assert _is_poller_args({"command": command}) is True


@pytest.mark.parametrize(
    "args",
    [
        {"command": "git status -sb"},
        {"command": "ls -la"},
        {"path": "/tmp/x"},
        {},
        None,
    ],
)
def test_non_poller_not_recognised(args) -> None:
    assert _is_poller_args(args) is False


def _fire_call(controller: ToolCallGuardrailController, args, results) -> int | None:
    """Return the 1-based call number that first raised a notice, else None."""
    for index, result in enumerate(results, 1):
        obs = controller.observe_call("terminal", args, result, tool_call_id=f"t{index}")
        if obs.notice:
            return index
    return None


def test_poller_streak_fires_despite_changing_results() -> None:
    """The regression case: same args, different result every time, must still fire."""
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig())
    args = {"command": "gh pr checks 123"}
    results = ["pending: 2 checks", "pending: 1 check", "in_progress", "pass: 5 checks", "pass"]
    fired = _fire_call(controller, args, results)
    assert fired == STALL_GUARD_IDENTICAL_CALL_THRESHOLD


def test_non_poller_changing_results_does_not_fire() -> None:
    """Unchanged behaviour: a non-poller with a changing result is not a streak."""
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig())
    args = {"command": "git status -sb"}
    results = ["M a.py", "M b.py", "?? c.txt", "M d.py", "M e.py"]
    assert _fire_call(controller, args, results) is None


def test_non_poller_identical_result_still_fires() -> None:
    """Unchanged behaviour: the original identical-result streak still works."""
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig())
    args = {"command": "git status -sb"}
    fired = _fire_call(controller, args, ["M a.py"] * 5)
    assert fired == STALL_GUARD_IDENTICAL_CALL_THRESHOLD


def test_poller_prefix_list_is_non_empty() -> None:
    """Guard the constant against an accidental empty tuple silently disabling the fix."""
    assert POLLER_ARGV_PREFIXES
    assert all(isinstance(prefix, str) and prefix for prefix in POLLER_ARGV_PREFIXES)
