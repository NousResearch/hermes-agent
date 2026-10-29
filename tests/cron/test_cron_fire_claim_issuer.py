"""A cron fire-claim loss is a system abort, not a user stop (#84207).

``_run_agent_with_watchdog`` aborts the turn when the one-shot run's fire claim is lost —
another actor took ownership. It interrupts the agent WITH A MESSAGE, so without an explicit
``tool_reason`` the producer is booked as ``_REASON_NEW_MESSAGE`` and the turn's exit reason
says ``interrupted_by_user``: the user never asked for the turn to stop.

The neighbouring watchdog abort in the same module already names its producer
(``tool_reason="cron inactivity watchdog"``); this pins the claim-loss abort doing the same.
"""

from __future__ import annotations

import threading

import pytest

from agent.interrupt_control import interrupt_issuer
from cron.scheduler import _run_agent_with_watchdog


class _Agent:
    """Records the interrupt the watchdog delivers.

    ``interrupt`` mirrors the real signature — ``tool_reason`` is keyword-only there, so the
    double must not widen it: a positional producer would pass this test and raise in the field.
    """

    def __init__(self) -> None:
        self.interrupt_message = None
        self._tool_interrupt_reason = None
        self._interrupt_requested = False

    def interrupt(self, message=None, *, tool_reason=None):
        self.interrupt_message = message
        self._interrupt_requested = True
        self._tool_interrupt_reason = tool_reason
        return True

    def run_conversation(self, prompt, task_id=None):
        return {"final_response": "", "response_len": 0}

    def get_activity_summary(self):
        return {"seconds_since_activity": 0.0}


def _oneshot_job() -> dict:
    return {
        "schedule": {"kind": "once"},
        "run_claim": {"by": "owner-1"},
        "execution_id": "exec-1",
    }


def test_fire_claim_loss_names_the_claim_loss_as_the_issuer():
    """The abort must attribute itself, so the turn does not read as ``interrupted_by_user``."""
    agent = _Agent()
    cancel_event = threading.Event()
    cancel_event.set()  # the claim was taken over: the watchdog must abort the turn

    with pytest.raises(RuntimeError, match="lost its durable fire claim ownership"):
        _run_agent_with_watchdog(
            agent, "do the thing", _oneshot_job(), "job-1", "nightly", "task-1", cancel_event,
        )

    assert agent._interrupt_requested, "the claim-loss abort must interrupt the agent"
    assert "fire claim ownership was lost" in (agent.interrupt_message or "")
    # The producer is named, so the exit reason is not ``interrupted_by_user``.
    assert interrupt_issuer(agent) == "cron_fire_claim_lost"
