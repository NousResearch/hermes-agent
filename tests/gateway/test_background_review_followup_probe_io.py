"""The turn's follow-up probe reads the durable FIFO outside the admission lock.

``_followup_pending`` (``gateway/run_turn_runner.py``) answers review admission from the agent
thread, and ``_BackgroundReviewRun.begin_request`` samples it under the turn's
``FollowupAdmissionState`` lock. The gateway event loop takes that same per-session lock for
every accepted follow-up (``apply_followup_queue_mutation``), so a SQLite read under it
(``SessionAuthority.queued_followup_on_route`` -> ``has_queued_session_admission``) stalls the
loop for that session behind a cold per-thread connection, a WAL checkpoint or a busy store.
The durable sample runs with no lock held; the lock covers only the epoch compare and the
process-local slot/debounce/burst reads. A follow-up accepted while the sample ran moves the
epoch, so the compare sees it without a second read. Real GatewayRunner, adapter and
SessionAuthority over SQLite, as in ``test_background_review_followup_shared_fifo.py``; the
review is spawned through the production ``AIAgent._spawn_background_review`` path.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

import hermes_state_runtime
from agent import background_review
from run_agent import AIAgent
from tests.agent.test_background_review_compute_overlap import (
    _bare_agent,
    _config,
    _patch_config,
    clear_review_admission_state,  # autouse fixture: resets the review registry
    review_forks,  # fixture: captures every fork, runs the review inline
)
from tests.gateway.test_background_review_followup_admission import _TrackingRLock
from tests.gateway.test_background_review_followup_shared_fifo import _wire_probe
from tests.gateway.test_shared_busy_receive_loop import Adapter
from tests.waiting import async_wait_until


async def _owner(monkeypatch):
    """Real runner, adapter and authority over SQLite -> (runner, adapter, source, key)."""
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority

    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "fixture-user")
    runner = GatewayRunner(GatewayConfig())
    adapter = Adapter()
    adapter.gateway_runner = runner
    runner.adapters = {Platform.TELEGRAM: adapter}
    await initialize_session_authority(
        runner, profile_id="default", instance_id="owner"
    )
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(
        chat_id="busy-chat", chat_type="dm", user_id="fixture-user"
    )
    return runner, adapter, source, runner.session_store._generate_session_key(source)


def _observe_durable_reads(monkeypatch, observe):
    """Wrap the FIFO probe ``queued_followup_on_route`` runs; ``observe(session_id)`` is called
    on the probing thread before the real read."""
    real = hermes_state_runtime.has_queued_session_admission

    def observed(db, *, session_id):
        observe(session_id)
        return real(db, session_id=session_id)

    monkeypatch.setattr(hermes_state_runtime, "has_queued_session_admission", observed)


def _finish_turn(event):
    from gateway.session_results import execution_result

    execution_result.get()["result"] = {
        "final_response": "reply to " + event.text,
        "messages": [],
    }
    return "reply to " + event.text


@pytest.mark.asyncio
async def test_durable_fifo_probe_runs_outside_the_admission_lock(
    review_forks,  # health: allow F811 -- pytest injects the imported fixture by parameter name
    monkeypatch,
):
    """Every durable read the probe issues — from the pre-spawn gates and from
    ``begin_request`` under the production spawn path — sees the admission lock unheld."""
    _patch_config(monkeypatch, _config())
    runner, adapter, source, key = await _owner(monkeypatch)
    state = adapter.followup_admission_state(key)
    state.lock = tracking = _TrackingRLock()
    held_during_read: list[bool] = []
    _observe_durable_reads(
        monkeypatch,
        lambda _sid: held_during_read.append(tracking.held_by_current_thread()),
    )
    observed = {}

    async def turn(event, source, quick_key, run_generation):
        await runner._mark_durable_active_turn(event, quick_key)
        agent = _bare_agent("probe-io-session")
        _wire_probe(runner, adapter, source, quick_key, run_generation, agent=agent)
        observed["direct"] = await asyncio.to_thread(agent.followup_pending_callback)
        observed["reads_before_spawn"] = len(held_during_read)
        await asyncio.to_thread(
            AIAgent._spawn_background_review,
            agent,
            messages_snapshot=[{"role": "user", "content": "hello"}],
            review_memory=True,
        )
        return _finish_turn(event)

    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    from gateway.platforms.event import MessageEvent

    await adapter.handle_message(
        MessageEvent(text="one", source=source, message_id="m1")
    )
    await async_wait_until(
        lambda: "reply to one" in adapter.sent and not adapter._background_tasks,
        timeout=15,
    )
    assert observed["direct"] is False
    assert observed["reads_before_spawn"] == 1
    assert len(review_forks) == 1, "no follower was queued: the review must be admitted"
    # ``begin_request`` reached the durable probe too (the spawn path adds its own gates).
    assert len(held_during_read) > observed["reads_before_spawn"]
    assert held_during_read == [False] * len(held_during_read), (
        "the durable FIFO probe (SQLite) ran under the session's admission lock"
    )
    assert tracking.held_by_current_thread() is False


@pytest.mark.asyncio
async def test_followup_accepted_during_the_durable_sample_is_seen_without_a_second_read(
    monkeypatch,
):
    """A slot write that lands while the durable sample runs never waits behind it, and the
    epoch compare under the lock reports it as pending without re-reading SQLite."""
    runner, adapter, source, key = await _owner(monkeypatch)
    state = adapter.followup_admission_state(key)
    reads: list[str] = []
    landed: list[bool] = []

    def accept_followup_mid_sample(session_id):
        reads.append(session_id)
        if len(reads) > 1:
            return
        writer = threading.Thread(
            target=lambda: landed.append(
                adapter.apply_followup_queue_mutation(key, lambda: True)
            ),
            name="gateway-slot-write",
            daemon=True,
        )
        writer.start()
        writer.join(timeout=5.0)

    _observe_durable_reads(monkeypatch, accept_followup_mid_sample)
    observed = {}

    async def turn(event, source, quick_key, run_generation):
        await runner._mark_durable_active_turn(event, quick_key)
        agent = _wire_probe(runner, adapter, source, quick_key, run_generation)
        adapter.register_followup_review_cancel(quick_key, lambda: None)
        observed["epoch_before"] = state.epoch
        observed["pending"] = await asyncio.to_thread(agent.followup_pending_callback)
        observed["epoch_after"] = state.epoch
        return _finish_turn(event)

    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    from gateway.platforms.event import MessageEvent

    await adapter.handle_message(
        MessageEvent(text="one", source=source, message_id="m1")
    )
    await async_wait_until(
        lambda: "reply to one" in adapter.sent and not adapter._background_tasks,
        timeout=15,
    )
    assert landed == [True], "the slot write waited behind the durable sample"
    assert observed["epoch_after"] == observed["epoch_before"] + 1
    assert observed["pending"] is True, (
        "a follow-up accepted while the sample ran was not seen"
    )
    assert len(reads) == 1, "the stale sample was re-read under the lock"


def test_begin_request_takes_the_admission_sample_before_the_admission_lock():
    """The run samples the host probe with no lock held, then gates under the follow-up lock."""
    lock = _TrackingRLock()
    seen: list[tuple[str, bool]] = []

    def gate():
        seen.append(("gate", lock.held_by_current_thread()))

    run = background_review._BackgroundReviewRun(
        admission_gate=gate,
        admission_lock=lock,
        admission_sample=lambda: seen.append(("sample", lock.held_by_current_thread())),
    )

    assert run.begin_request(object()) is True
    assert seen == [("sample", False), ("gate", True)]


def test_failing_admission_sample_leaves_the_gate_to_fail_safe():
    """A sample hook that raises is not fatal to admission: the gate still runs (and samples on
    its own), so a broken store reads as the gate's own body-free refusal, never a crash."""
    gated: list[bool] = []

    def gate():
        gated.append(True)
        return "admission_probe_failed"

    def broken_sample():
        raise RuntimeError("store unavailable")

    run = background_review._BackgroundReviewRun(
        admission_gate=gate, admission_sample=broken_sample
    )

    assert run.begin_request(object()) is False
    assert gated == [True]
    assert run.refused_reason == "admission_probe_failed"
