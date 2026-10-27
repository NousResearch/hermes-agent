"""Regression tests for the task lifecycle gate (ADR-0081).

A background result must not re-activate a finished or cancelled task.  These
tests drive the REAL gateway code paths -- the delivery path, the ``/stop``
handler and the turn-start re-arm -- not a stand-in for them.

Covered (ADR-0081 regression plan):
  T1  stop while background work runs  -> the late result is not delivered
  T2  late result after the task ended  -> not delivered, but retrievable
  T3  two parallel tasks, one aborted   -> the other is still delivered
  T4  a new user instruction after /stop -> the new task starts normally
  T5  a job that fails after the abort   -> the failure stays retrievable
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform
from gateway.task_lifecycle import TASK_END_MARKER, task_lifecycles


SESSION = "agent:main:telegram:dm:12345"
OTHER_SESSION = "agent:main:telegram:dm:67890"


@pytest.fixture(autouse=True)
def _clean_registry():
    """Each test starts with an empty task registry."""
    task_lifecycles._by_session.clear()
    yield
    task_lifecycles._by_session.clear()


def _make_runner():
    """A GatewayRunner with just enough surface for the task hooks."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._running_agents = {}
    runner._pending_messages = {}
    return runner


def _make_delivery_runner():
    """A GatewayRunner that can run the real completion-delivery path."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    injected = []

    class _Adapter:
        async def handle_message(self, event, **kwargs):
            injected.append(event)
            event._gateway_accepted = True
            return True

    runner.adapters = {Platform.TELEGRAM: _Adapter()}
    runner._completion_delivery_lock = threading.Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048
    runner._injected = injected
    return runner


def _task_start(session_key: str = SESSION, index: int = -1) -> float:
    """When the task of a session began (``index`` -1 = the current one)."""
    tasks = task_lifecycles.tasks(session_key)
    assert tasks, "no task opened for this session"
    return tasks[index].started_at


def _completion_evt(session_key: str = SESSION, *, started_at=None, owner_task_id="turn-1", **overrides):
    """A background-process completion event as the gateway builds it.

    By default the result started DURING the session's current task -- that is
    how a real background process of that task looks.
    """
    if started_at is None:
        started_at = _task_start(session_key) + 1
    evt = {
        "type": "completion",
        "session_id": "proc_abc123",
        "session_key": session_key,
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "12345",
        "user_id": "12345",
        "user_name": "tester",
        # A real turn id, NOT an "sa-" id: the "sa-" prefix marks delegation
        # child processes, which are suppressed by default and would never
        # reach the delivery path at all.
        "owner_task_id": owner_task_id,
        "task_id": owner_task_id,
        "started_at": started_at,
        "exit_code": 0,
        "completion_reason": "exited",
    }
    evt.update(overrides)
    return evt


class _Claim:
    proceed = True
    early_result = None
    owner_session_id = ""
    proven_session_id = ""
    claim_id = None
    delegation_id = None
    defer = False


async def _preflight_stub(self, evt):
    return _Claim()


# ---------------------------------------------------------------------------
# T1 -- /stop while background work runs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_t1_stop_blocks_late_result_and_kills_task_processes():
    """After /stop the task is cancelled: the late result is refused and the
    task's own background processes are terminated."""
    runner = _make_runner()
    task_lifecycles.begin(SESSION, task_id="turn-1")

    killed = []

    class _FakeRegistry:
        def list_sessions(self, task_id=None, session_key=None, include_retained=False):
            return [SimpleNamespace(id="proc_1", exited=False)]

        def kill_process(self, session_id, *, source="process.kill", consume_output=True):
            killed.append((session_id, source))
            return {"status": "killed"}

    with patch("tools.process_registry.process_registry", _FakeRegistry()):
        runner._task_cancel(SESSION, reason="user_stop")

    assert task_lifecycles.state_of(SESSION) == "cancelled"
    # The task's own process was stopped through the existing ProcessRegistry.
    assert killed and killed[0][0] == "proc_1"

    # The late result must not start a new turn.
    assert runner._task_may_start_work(_completion_evt()) is False
    # ... and it stays retrievable in the register (ADR-0081 Anforderung 5).
    refused = task_lifecycles.refused_results(SESSION)
    assert len(refused) == 1
    assert refused[0]["process_id"] == "proc_abc123"


@pytest.mark.asyncio
async def test_t1_stop_keeps_the_session_open():
    """/stop ends the TASK, not the session: a new user turn works normally."""
    runner = _make_runner()
    task_lifecycles.begin(SESSION, task_id="turn-1")
    runner._task_cancel(SESSION, reason="user_stop")

    # A new user instruction re-arms the task under a NEW turn id.
    runner._task_begin_turn(SESSION, is_internal=False, task_id="turn-2")
    assert task_lifecycles.state_of(SESSION) == "active"
    # ... and the late result of the OLD task is still refused: it carries the
    # old turn id, so it belongs to the task that was closed, not to the new one.
    assert runner._task_may_start_work(
        _completion_evt(owner_task_id="turn-1"),
    ) is False
    # The refusal is recorded, so the result stays retrievable without a chat
    # notification (ADR-0081 Anforderung 5).
    assert task_lifecycles.refused_results(SESSION)


# ---------------------------------------------------------------------------
# T2 -- a late result after the task ended
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_t2_turn_end_without_live_work_drains_the_task():
    """A turn that ends with no work in flight ends the assignment: the next
    background result is refused, not delivered."""
    runner = _make_runner()
    task_lifecycles.begin(SESSION)

    with patch("gateway.task_lifecycle.session_has_live_work", return_value=False):
        runner._task_close_on_turn_end(SESSION, marker_found=False)

    assert task_lifecycles.state_of(SESSION) == "drained"
    assert runner._task_may_start_work(_completion_evt()) is False
    # The refusal is recorded, so it stays visible without a chat message.
    assert task_lifecycles.refused_results(SESSION)


@pytest.mark.asyncio
async def test_t2_turn_with_live_work_keeps_the_task_open():
    """A turn that leaves a background job running keeps the task active: that
    job's result must still be delivered."""
    runner = _make_runner()
    task_lifecycles.begin(SESSION)

    with patch("gateway.task_lifecycle.session_has_live_work", return_value=True):
        runner._task_close_on_turn_end(SESSION, marker_found=False)

    assert task_lifecycles.state_of(SESSION) == "active"
    assert runner._task_may_start_work(_completion_evt()) is True


@pytest.mark.asyncio
async def test_t2_model_marker_closes_the_task_even_with_live_work():
    """The model may report the end of the assignment explicitly."""
    runner = _make_runner()
    task_lifecycles.begin(SESSION)

    with patch("gateway.task_lifecycle.session_has_live_work", return_value=True):
        runner._task_close_on_turn_end(SESSION, marker_found=True)

    assert task_lifecycles.state_of(SESSION) == "drained"
    assert runner._task_may_start_work(_completion_evt()) is False


# ---------------------------------------------------------------------------
# T3 -- two parallel tasks, only one aborted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_t3_one_aborted_task_does_not_affect_the_other():
    """Aborting one task must not block a different task's result."""
    runner = _make_runner()
    task_lifecycles.begin(SESSION)
    task_lifecycles.begin(OTHER_SESSION)

    runner._task_cancel(SESSION, reason="user_stop")

    assert task_lifecycles.state_of(SESSION) == "cancelled"
    assert task_lifecycles.state_of(OTHER_SESSION) == "active"
    # The other task's result is still delivered -- no cross-effect.
    assert runner._task_may_start_work(_completion_evt(session_key=OTHER_SESSION)) is True
    # The aborted one is still refused.
    assert runner._task_may_start_work(_completion_evt(session_key=SESSION)) is False


# ---------------------------------------------------------------------------
# T4 -- a new user instruction after the abort
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_t4_new_user_instruction_re_arms_the_task():
    """After /stop the user writes again: the new task starts normally."""
    runner = _make_runner()
    task_lifecycles.begin(SESSION, task_id="turn-1")
    runner._task_cancel(SESSION, reason="user_stop")

    runner._task_begin_turn(SESSION, is_internal=False, task_id="turn-2")

    assert task_lifecycles.state_of(SESSION) == "active"
    # A result of the NEW task is delivered: it carries the new turn id.
    assert runner._task_may_start_work(
        _completion_evt(owner_task_id="turn-2"),
    ) is True
    # A result that starts AFTER the new task began is delivered into it, too.
    assert runner._task_may_start_work(
        _completion_evt(owner_task_id="turn-2", started_at=time.time() + 1),
    ) is True


@pytest.mark.asyncio
async def test_t4_internal_event_never_re_arms_the_task():
    """A background result must not be able to re-open the task it belongs to."""
    runner = _make_runner()
    task_lifecycles.begin(SESSION, task_id="turn-1")
    runner._task_cancel(SESSION, reason="user_stop")

    # An internal turn (the background result waking the session) does NOT re-arm.
    runner._task_begin_turn(SESSION, is_internal=True)

    assert task_lifecycles.state_of(SESSION) == "cancelled"
    assert runner._task_may_start_work(_completion_evt()) is False


# ---------------------------------------------------------------------------
# T5 -- a job that fails after the abort
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_t5_job_failure_after_abort_stays_retrievable():
    """A failing job after the abort is recorded, not delivered and not lost."""
    runner = _make_runner()
    task_lifecycles.begin(SESSION, task_id="turn-1")
    runner._task_cancel(SESSION, reason="user_stop")

    evt = _completion_evt(exit_code=1, completion_reason="failed_start")
    assert runner._task_may_start_work(evt) is False

    refused = task_lifecycles.refused_results(SESSION)
    assert len(refused) == 1
    assert refused[0]["exit_code"] == 1
    assert refused[0]["completion_reason"] == "failed_start"


# ---------------------------------------------------------------------------
# The gate itself: the delivery path must not inject a refused result
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gate_blocks_the_delivery_path_before_any_llm_call():
    """The refused result must not reach the adapter: no turn, no tokens."""
    from gateway.run import GatewayRunner

    runner = _make_delivery_runner()
    task_lifecycles.begin(SESSION)
    task_lifecycles.close(SESSION, "cancelled", reason="user_stop")

    with patch.object(GatewayRunner, "_preflight_completion_delivery", new=_preflight_stub):
        result = await runner._deliver_completion_notification_scoped(
            "text", _completion_evt(session_id="proc_refused"),
        )

    assert result is None
    assert runner._injected == []


@pytest.mark.asyncio
async def test_gate_lets_a_legitimate_result_through():
    """The gate must not block a result of a task that is still open."""
    from gateway.run import GatewayRunner

    runner = _make_delivery_runner()
    task_lifecycles.begin(SESSION)

    with patch.object(GatewayRunner, "_preflight_completion_delivery", new=_preflight_stub):
        result = await runner._deliver_completion_notification_scoped(
            "text", _completion_evt(session_id="proc_allowed"),
        )

    assert result is not None
    assert len(runner._injected) == 1


# ---------------------------------------------------------------------------
# The marker must not reach the user
# ---------------------------------------------------------------------------

def test_task_end_marker_is_detected():
    from gateway.task_lifecycle import agent_result_reports_task_end as _task_end_marker_in

    assert _task_end_marker_in({"final_response": f"Fertig. {TASK_END_MARKER}"}) is True
    assert _task_end_marker_in({"final_response": "Fertig."}) is False
    assert _task_end_marker_in(None) is False
    assert _task_end_marker_in({}) is False


# ---------------------------------------------------------------------------
# Marker-Randfaelle (ADR-0081): der Abschluss darf nicht vom Marker abhaengen
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("response, expect_marker", [
    ("Fertig.", False),
    ("", False),
    ("Fertig. [TASK_DONE]", True),
    ("[TASK_DONE] Fertig.", True),
    ("[TASK_DONE][TASK_DONE] Fertig.", True),
    ("Bitte [TASK_DONE] pruefen", True),
])
def test_marker_detection_is_exact_and_forgiving(response, expect_marker):
    """The marker is recognised where it is, and only where it is.

    ``agent_result_reports_task_end`` takes the agent RESULT (a dict), exactly as the
    turn-end hook calls it.
    """
    from gateway.task_lifecycle import agent_result_reports_task_end as _task_end_marker_in

    assert _task_end_marker_in({"final_response": response}) is expect_marker
    # A non-dict result (an error path) must not raise and must not report a marker.
    assert _task_end_marker_in(None) is False
    assert _task_end_marker_in({}) is False


def test_marker_is_removed_from_the_user_facing_text():
    """The marker is a report, not content: it must not reach the user."""
    from gateway.task_lifecycle import strip_task_end_marker

    clean, found = strip_task_end_marker("Fertig. [TASK_DONE]")
    assert found is True
    assert "[TASK_DONE]" not in clean
    assert "Fertig." in clean


def test_a_missing_marker_does_not_close_a_task_with_live_work():
    """Without the marker and with a job running, the task stays open."""
    runner = _make_runner()
    task_lifecycles.begin(SESSION, task_id="turn-1")

    with patch("gateway.task_lifecycle.session_has_live_work", return_value=True):
        runner._task_close_on_turn_end(SESSION, marker_found=False)

    assert task_lifecycles.state_of(SESSION) == "active"
