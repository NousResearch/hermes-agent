"""G1 contracts for SIBLING/SELF auto-continue and deferred SELF restart."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

import gateway.deferred_restart as deferred_restart_module
import gateway.run as gateway_run
from gateway.auto_resume import (
    RESUME_KIND_SELF,
    RESUME_KIND_SIBLING,
    resume_kind_for_reason,
)
from gateway.config import GatewayConfig, Platform
from gateway.deferred_restart import (
    DeferredRestartCoordinator,
    DeferredRestartRequest,
    reconcile_deferred_restarts_at_boot,
    submit_deferred_restart,
)
from gateway.resume_requests import sweep_resume_requests
from gateway.run import _build_resume_pending_message
from gateway.session import SessionEntry, SessionSource, SessionStore
from tests.gateway.restart_test_helpers import make_restart_runner


def _source() -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="u1")


def _store(home: Path) -> tuple[SessionStore, SessionEntry]:
    store = SessionStore(sessions_dir=home / "sessions", config=GatewayConfig())
    return store, store.get_or_create_session(_source())


def _submit(home: Path, session_key: str, *, boot_id: str = "11:22", intent_ts: float = 10.0):
    return submit_deferred_restart(
        home,
        session_key=session_key,
        handoff="continue the migration",
        boot_id=boot_id,
        intent_ts=intent_ts,
        request_id="req-1",
    )


def test_t1_total_taxonomy_mapping() -> None:
    assert RESUME_KIND_SIBLING == "sibling"
    assert RESUME_KIND_SELF == "self"
    for reason in ("restart_interrupted", "restart_consumed_interrupted"):
        assert resume_kind_for_reason(reason) == RESUME_KIND_SELF
    for reason in (
        "shutdown_timeout",
        "restart_timeout",
        "reboot_interrupted",
        "manual_resume_request",
    ):
        assert resume_kind_for_reason(reason) == RESUME_KIND_SIBLING


def test_t2_t3_self_mark_metadata_roundtrips_synchronously(tmp_path: Path) -> None:
    store, entry = _store(tmp_path)
    assert store.mark_resume_pending(
        entry.session_key,
        "restart_interrupted",
        resume_kind=RESUME_KIND_SELF,
        resume_handoff="continue the migration",
        resume_request_id="req-1",
    )

    reloaded = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    reloaded._ensure_loaded()
    got = reloaded._entries[entry.session_key]
    assert got.resume_pending is True
    assert got.resume_kind == RESUME_KIND_SELF
    assert got.resume_handoff == "continue the migration"
    assert got.resume_request_id == "req-1"


def test_t3_self_note_names_kind_and_carries_handoff() -> None:
    note, surface_and_ask = _build_resume_pending_message(
        agent_history=[],
        message="",
        reason_phrase="a gateway restart",
        resume_kind=RESUME_KIND_SELF,
        resume_handoff="continue the migration",
    )
    assert "kind=self" in note
    assert "continue the migration" in note
    assert surface_and_ask is False


@pytest.mark.asyncio
async def test_t2_t8_release_arms_then_waits_for_real_delivery_ack(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner, adapter = make_restart_runner()
    runner.session_store, entry = _store(tmp_path)
    runner._running_agents[entry.session_key] = object()
    runner._running_agents["sibling-a"] = object()
    runner._running_agents["sibling-b"] = object()
    runner._gateway_loop = asyncio.get_running_loop()
    runner._current_boot_id = lambda: "11:22"
    runner._consume_restart_initiated_breadcrumb = (
        lambda session_key: session_key == entry.session_key
    )
    signals: list[int] = []
    runner.request_restart = lambda **_kwargs: signals.append(1) or True
    _submit(tmp_path, entry.session_key)

    assert runner._release_running_agent_state(entry.session_key) is True
    assert set(runner._running_agents) == {"sibling-a", "sibling-b"}
    assert signals == []
    assert entry.session_key in adapter._delivery_ack_callbacks
    adapter.acknowledge_response_delivery(entry.session_key)
    await asyncio.gather(*runner._background_tasks)

    assert signals == [1]
    durable = runner.session_store._entries[entry.session_key]
    assert durable.resume_kind == RESUME_KIND_SELF
    assert durable.resume_handoff == "continue the migration"


@pytest.mark.asyncio
async def test_t8_post_arm_callback_failure_keeps_task_ownership(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner, adapter = make_restart_runner()
    runner.session_store, entry = _store(tmp_path)
    runner._running_agents[entry.session_key] = object()
    runner._current_boot_id = lambda: "11:22"
    runner._consume_restart_initiated_breadcrumb = lambda session_key: bool(session_key)
    signals: list[int] = []
    runner.request_restart = lambda **_kwargs: signals.append(1) or True

    def _fail_callback_registration(*_args, **_kwargs) -> None:
        raise OSError("injected callback failure")

    monkeypatch.setattr(
        adapter,
        "register_delivery_ack_callback",
        _fail_callback_registration,
    )
    _submit(tmp_path, entry.session_key)

    assert runner._release_running_agent_state(entry.session_key) is True
    await asyncio.gather(*runner._background_tasks)

    assert signals == [1]
    assert "delivery state is UNKNOWN" in caplog.text
    durable = runner.session_store._entries[entry.session_key]
    assert durable.resume_kind == RESUME_KIND_SELF


def test_t2_delivery_ack_is_generation_owned_and_not_a_finally_callback() -> None:
    _runner, adapter = make_restart_runner()
    observed: list[str] = []
    adapter.register_delivery_ack_callback(
        "agent:main:telegram:dm:123",
        lambda: observed.append("ack"),
        generation=7,
    )

    adapter.acknowledge_response_delivery(
        "agent:main:telegram:dm:123", generation=6
    )
    adapter.acknowledge_response_delivery("agent:main:telegram:dm:123")
    assert observed == []
    assert adapter.pop_post_delivery_callback("agent:main:telegram:dm:123") is None

    adapter.acknowledge_response_delivery(
        "agent:main:telegram:dm:123", generation=7
    )
    assert observed == ["ack"]


def test_t2_ordinary_delivery_ack_without_request_retains_no_state() -> None:
    _runner, adapter = make_restart_runner()
    for generation in range(100):
        assert (
            adapter.acknowledge_response_delivery(
                f"agent:main:telegram:dm:{generation}",
                generation=generation,
            )
            is False
        )
    assert adapter._delivery_ack_callbacks == {}


def test_t8_repeated_self_restart_reserves_breadcrumb_for_release(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner, _adapter = make_restart_runner()
    runner.session_store, entry = _store(tmp_path)
    runner.session_store.mark_resume_pending(
        entry.session_key,
        "restart_consumed_interrupted",
        resume_kind=RESUME_KIND_SELF,
        resume_request_id="prior-request",
    )
    runner._current_boot_id = lambda: "11:22"
    _submit(tmp_path, entry.session_key)
    consumed: list[str] = []
    runner._consume_restart_initiated_breadcrumb = (
        lambda session_key: consumed.append(session_key) or True
    )

    runner._apply_post_turn_resume_gate(entry.session_key)

    assert consumed == []
    assert runner.session_store._entries[entry.session_key].resume_pending is False
    [request] = runner._get_deferred_restart_coordinator().scan()
    assert request.request_id == "req-1"
    assert request.state == "submitted"


def test_t9d_request_during_existing_drain_stays_submitted_for_next_boot(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner, adapter = make_restart_runner()
    runner.session_store, entry = _store(tmp_path)
    runner._running_agents[entry.session_key] = object()
    runner._draining = True
    request = _submit(tmp_path, entry.session_key)

    assert runner._release_running_agent_state(entry.session_key) is True
    assert DeferredRestartRequest.load(request.path).state == "submitted"
    assert adapter._delivery_ack_callbacks == {}
    assert getattr(runner, "_background_tasks", set()) == set()


@pytest.mark.asyncio
async def test_t9k_real_startup_sequence_reconciles_before_same_boot_schedule(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_RESUME_INTERRUPTED_TURNS", "auto")
    runner, _adapter = make_restart_runner()
    runner.session_store, entry = _store(tmp_path)
    runner._session_db = None
    runner._current_boot_id = lambda: "new:boot"
    runner._boot_started_at = 20.0
    runner._startup_restore_in_progress = True
    request = _submit(tmp_path, entry.session_key, boot_id="old:boot")
    DeferredRestartCoordinator(tmp_path, boot_id="old:boot").transition(
        request, "claimed"
    )

    assert await runner._restore_resume_pending_sessions_at_startup() == 1
    durable = runner.session_store._entries[entry.session_key]
    assert durable.resume_pending is True
    assert durable.resume_kind == RESUME_KIND_SELF
    assert durable.resume_request_id == "req-1"
    assert runner._startup_resume_modes[entry.session_key]["mode"] == "auto"
    assert entry.session_key in runner._resumed_this_boot
    assert not any((tmp_path / "gateway" / "resume_requests").iterdir())

    counts = json.loads((tmp_path / runner._STUCK_LOOP_FILE).read_text())
    assert counts[entry.session_key]["replay_request_ids"] == ["req-1"]
    assert len(counts[entry.session_key]["replay_marks"]) == 1


def test_t9_cross_host_reboot_uses_stable_wall_clock(
    tmp_path: Path, monkeypatch
) -> None:
    _store_obj, entry = _store(tmp_path)
    monkeypatch.setattr(deferred_restart_module.time, "time", lambda: 1_700_000_000.0)
    monkeypatch.setattr(deferred_restart_module.time, "monotonic", lambda: 50_000.0)
    request = submit_deferred_restart(
        tmp_path,
        session_key=entry.session_key,
        handoff="survive host reboot",
        boot_id="old:boot",
        request_id="req-host-reboot",
    )
    assert request.intent_ts == 1_700_000_000.0

    marked: list[str] = []
    assert reconcile_deferred_restarts_at_boot(
        tmp_path,
        current_boot_id="new:boot",
        boot_started_at=1_700_000_010.0,
        has_durable_mark=lambda req: False,
        record_replay=lambda req: None,
        mark_in_memory=lambda req: marked.append(req.request_id),
        flush_sessions=lambda: None,
        signal_restart=lambda: pytest.fail("cross-boot reconciliation signaled"),
    ) == 1
    assert marked == ["req-host-reboot"]


@pytest.mark.asyncio
async def test_t9k_prepare_calls_boot_reconciliation_even_in_prompt_mode(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_RESUME_INTERRUPTED_TURNS", "prompt")
    runner, _adapter = make_restart_runner()
    calls: list[int] = []
    runner._reconcile_deferred_restarts_at_boot = lambda: calls.append(1) or 0

    assert await runner._prepare_auto_resume_decisions() == 0
    assert calls == [1]


def test_t9k_failed_boot_reconciliation_remains_retryable(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner, _adapter = make_restart_runner()
    runner.session_store, _entry = _store(tmp_path)
    calls: list[int] = []

    def _flaky(*_args, **_kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("injected reconciliation failure")
        return 0

    monkeypatch.setattr(
        deferred_restart_module,
        "reconcile_deferred_restarts_at_boot",
        _flaky,
    )
    with pytest.raises(OSError, match="injected"):
        runner._reconcile_deferred_restarts_at_boot()
    assert getattr(runner, "_deferred_boot_reconciled", False) is False

    assert runner._reconcile_deferred_restarts_at_boot() == 0
    assert calls == [1, 1]
    assert runner._deferred_boot_reconciled is True


def test_t7a_plain_sweep_never_consumes_deferred_restart(tmp_path: Path) -> None:
    _store_obj, entry = _store(tmp_path)
    request = _submit(tmp_path, entry.session_key)

    assert sweep_resume_requests(tmp_path) == []
    assert request.path.exists()
    assert DeferredRestartRequest.load(request.path).state == "submitted"


@pytest.mark.parametrize(
    ("case_id", "request_boot", "session_key", "breadcrumb", "expected"),
    [
        ("T7a", "11:22", "agent:main:telegram:dm:123", None, "rejected"),
        ("T7b", "old:boot", "agent:main:telegram:dm:123", True, "rejected"),
        # The breadcrumb consumer is session-keyed; False here represents a
        # breadcrumb belonging to a different session.
        ("T7c", "11:22", "agent:main:telegram:dm:123", False, "rejected"),
        ("T7d", "11:22", "agent:main:telegram:dm:123", False, "rejected"),
        ("T7e", "11:22", "agent:main:telegram:dm:123", True, "armed"),
    ],
)
def test_t7_breadcrumb_validation_matrix(
    tmp_path: Path,
    case_id: str,
    request_boot: str,
    session_key: str,
    breadcrumb: bool | None,
    expected: str,
) -> None:
    request = submit_deferred_restart(
        tmp_path,
        session_key=session_key,
        handoff="h",
        boot_id=request_boot,
        intent_ts=10.0,
        request_id=case_id,
    )
    consumed: list[str] = []

    def consume(key: str) -> bool:
        consumed.append(key)
        return bool(breadcrumb)

    coordinator = DeferredRestartCoordinator(tmp_path, boot_id="11:22")
    result = coordinator.arm_for_session(
        "agent:main:telegram:dm:123",
        consume_breadcrumb=consume,
    )

    assert result == expected
    scanned = DeferredRestartRequest.load(next((tmp_path / "gateway" / "resume_requests").iterdir()))
    assert scanned.state == expected
    assert len(consumed) <= 1


@pytest.mark.asyncio
async def test_t8_delivery_barrier_and_repeated_schedule_are_one_shot(tmp_path: Path) -> None:
    _store_obj, entry = _store(tmp_path)
    _submit(tmp_path, entry.session_key)
    coordinator = DeferredRestartCoordinator(tmp_path, boot_id="11:22")
    assert coordinator.arm_for_session(entry.session_key, consume_breadcrumb=lambda _k: True) == "armed"

    delivered = asyncio.Event()
    signals: list[str] = []
    marks: list[str] = []
    task1 = coordinator.schedule_armed(
        entry.session_key,
        delivery_event=delivered,
        delivery_timeout=1.0,
        record_replay=lambda req: None,
        mark_self=lambda req: marks.append(req.request_id),
        signal_restart=lambda: signals.append("signal"),
    )
    task2 = coordinator.schedule_armed(
        entry.session_key,
        delivery_event=delivered,
        delivery_timeout=1.0,
        record_replay=lambda req: None,
        mark_self=lambda req: marks.append(req.request_id),
        signal_restart=lambda: signals.append("signal"),
    )
    assert task1 is task2
    await asyncio.sleep(0)
    assert signals == []
    delivered.set()
    await task1
    assert signals == ["signal"]
    assert marks == ["req-1"]


@pytest.mark.asyncio
async def test_t8_post_arm_scan_failure_keeps_task_ownership(tmp_path: Path) -> None:
    _store_obj, entry = _store(tmp_path)
    _submit(tmp_path, entry.session_key)
    coordinator = DeferredRestartCoordinator(tmp_path, boot_id="11:22")
    assert coordinator.arm_for_session(
        entry.session_key, consume_breadcrumb=lambda _k: True
    ) == "armed"
    coordinator.scan = lambda: []
    delivered = asyncio.Event()
    delivered.set()
    signals: list[int] = []

    outcome = await coordinator.schedule_armed(
        entry.session_key,
        delivery_event=delivered,
        delivery_timeout=1.0,
        record_replay=lambda req: None,
        mark_self=lambda req: None,
        signal_restart=lambda: signals.append(1),
    )
    assert outcome == "signaled"
    assert signals == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize("peer_count", [0, 1])
async def test_t8_post_rename_refresh_failure_keeps_exactly_one_signal(
    tmp_path: Path, monkeypatch, peer_count: int
) -> None:
    session_keys = ["agent:main:telegram:dm:123"]
    if peer_count:
        session_keys.append("agent:main:telegram:dm:456")
    for index, session_key in enumerate(session_keys, start=1):
        submit_deferred_restart(
            tmp_path,
            session_key=session_key,
            handoff=f"handoff-{index}",
            boot_id="11:22",
            intent_ts=10.0,
            request_id=f"req-{index}",
        )

    original_atomic_write = deferred_restart_module._atomic_write_json
    injected = False

    def _fail_first_armed_refresh(path: Path, payload: dict) -> None:
        nonlocal injected
        if not injected and path.name.endswith(".armed.json"):
            injected = True
            raise OSError("injected post-rename refresh failure")
        original_atomic_write(path, payload)

    monkeypatch.setattr(
        deferred_restart_module,
        "_atomic_write_json",
        _fail_first_armed_refresh,
    )
    coordinator = DeferredRestartCoordinator(tmp_path, boot_id="11:22")
    for session_key in session_keys:
        assert coordinator.arm_for_session(
            session_key, consume_breadcrumb=lambda _key: True
        ) == "armed"

    signals: list[int] = []
    tasks = []
    for session_key in session_keys:
        delivered = asyncio.Event()
        delivered.set()
        tasks.append(
            coordinator.schedule_armed(
                session_key,
                delivery_event=delivered,
                delivery_timeout=1.0,
                record_replay=lambda req: None,
                mark_self=lambda req: None,
                signal_restart=lambda: signals.append(1),
            )
        )

    outcomes = await asyncio.gather(*tasks)
    assert injected is True
    assert signals == [1]
    assert outcomes.count("signaled") == 1
    assert outcomes.count("coalesced") == peer_count


@pytest.mark.asyncio
async def test_stale_prior_boot_leader_is_recovered_by_epoch_not_pid(
    tmp_path: Path,
) -> None:
    _store_obj, entry = _store(tmp_path)
    _submit(tmp_path, entry.session_key, boot_id="new:boot")
    coordinator = DeferredRestartCoordinator(tmp_path, boot_id="new:boot")
    coordinator.arm_for_session(entry.session_key, consume_breadcrumb=lambda _k: True)
    coordinator.leader_dir.mkdir()
    (coordinator.leader_dir / "meta.json").write_text(
        json.dumps({"boot_id": "old:boot", "committed": False})
    )
    delivered = asyncio.Event()
    delivered.set()
    signals: list[int] = []
    await coordinator.schedule_armed(
        entry.session_key,
        delivery_event=delivered,
        delivery_timeout=1.0,
        record_replay=lambda req: None,
        mark_self=lambda req: None,
        signal_restart=lambda: signals.append(1),
    )
    assert signals == [1]


@pytest.mark.asyncio
async def test_stale_latch_delete_failure_yields_before_retry(
    tmp_path: Path, monkeypatch
) -> None:
    _store_obj, entry = _store(tmp_path)
    _submit(tmp_path, entry.session_key, boot_id="new:boot")
    coordinator = DeferredRestartCoordinator(tmp_path, boot_id="new:boot")
    coordinator.arm_for_session(entry.session_key, consume_breadcrumb=lambda _k: True)
    coordinator.leader_dir.mkdir()
    (coordinator.leader_dir / "meta.json").write_text(
        json.dumps({"boot_id": "old:boot", "committed": False})
    )
    delivered = asyncio.Event()
    delivered.set()
    attempts = 0
    observed_attempts: list[int] = []
    original_rmtree = deferred_restart_module.shutil.rmtree
    loop = asyncio.get_running_loop()

    def _sticky_rmtree(path, *args, **kwargs) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            loop.call_soon(lambda: observed_attempts.append(attempts))
        if attempts < 3:
            return
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(deferred_restart_module.shutil, "rmtree", _sticky_rmtree)
    signals: list[int] = []
    assert await coordinator.schedule_armed(
        entry.session_key,
        delivery_event=delivered,
        delivery_timeout=1.0,
        record_replay=lambda req: None,
        mark_self=lambda req: None,
        signal_restart=lambda: signals.append(1),
    ) == "signaled"
    assert observed_attempts == [1]
    assert signals == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "meta_text",
    [
        None,
        "{",
        "{}",
        json.dumps(
            {"boot_id": "11:22", "committed": True, "commit_ts": "invalid"}
        ),
    ],
)
async def test_unreadable_meta_never_false_coalesces(
    tmp_path: Path, meta_text: str | None
) -> None:
    _store_obj, entry = _store(tmp_path)
    _submit(tmp_path, entry.session_key)
    coordinator = DeferredRestartCoordinator(tmp_path, boot_id="11:22")
    coordinator.arm_for_session(entry.session_key, consume_breadcrumb=lambda _k: True)
    coordinator.leader_dir.mkdir()
    if meta_text is not None:
        (coordinator.leader_dir / "meta.json").write_text(meta_text)
    delivered = asyncio.Event()
    delivered.set()
    task = coordinator.schedule_armed(
        entry.session_key,
        delivery_event=delivered,
        delivery_timeout=1.0,
        record_replay=lambda req: None,
        mark_self=lambda req: pytest.fail("unreadable meta coalesced"),
        signal_restart=lambda: pytest.fail("unreadable meta signaled"),
    )
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(task), timeout=0.04)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    [request] = coordinator.scan()
    assert request.state == "armed"


@pytest.mark.asyncio
async def test_uncommitted_loser_uses_bounded_exponential_backoff(
    tmp_path: Path, monkeypatch
) -> None:
    _store_obj, entry = _store(tmp_path)
    _submit(tmp_path, entry.session_key)
    coordinator = DeferredRestartCoordinator(tmp_path, boot_id="11:22")
    coordinator.arm_for_session(entry.session_key, consume_breadcrumb=lambda _k: True)
    coordinator.leader_dir.mkdir()
    (coordinator.leader_dir / "meta.json").write_text(
        json.dumps({"boot_id": "11:22", "committed": False})
    )
    delivered = asyncio.Event()
    delivered.set()
    delays: list[float] = []

    class _StopPolling(Exception):
        pass

    async def _capture_sleep(delay: float) -> None:
        delays.append(delay)
        if len(delays) == 4:
            raise _StopPolling

    monkeypatch.setattr(deferred_restart_module.asyncio, "sleep", _capture_sleep)
    task = coordinator.schedule_armed(
        entry.session_key,
        delivery_event=delivered,
        delivery_timeout=1.0,
        record_replay=lambda req: None,
        mark_self=lambda req: pytest.fail("uncommitted loser coalesced"),
        signal_restart=lambda: pytest.fail("uncommitted loser signaled"),
    )
    with pytest.raises(_StopPolling):
        await task
    assert delays == [0.01, 0.02, 0.04, 0.08]


@pytest.mark.asyncio
async def test_waiter_cancellation_leaves_request_armed(tmp_path: Path) -> None:
    _store_obj, entry = _store(tmp_path)
    _submit(tmp_path, entry.session_key)
    coordinator = DeferredRestartCoordinator(tmp_path, boot_id="11:22")
    coordinator.arm_for_session(entry.session_key, consume_breadcrumb=lambda _k: True)
    task = coordinator.schedule_armed(
        entry.session_key,
        delivery_event=asyncio.Event(),
        delivery_timeout=30.0,
        record_replay=lambda req: pytest.fail("cancelled waiter replayed"),
        mark_self=lambda req: pytest.fail("cancelled waiter marked"),
        signal_restart=lambda: pytest.fail("cancelled waiter signaled"),
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    [request] = coordinator.scan()
    assert request.state == "armed"
    assert not coordinator.leader_dir.exists()


def test_t9j_request_id_replay_marks_are_idempotent_and_trip_at_three(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_RESTART_LOOP_THRESHOLD", "3")
    runner, _adapter = make_restart_runner()
    runner.session_store, entry = _store(tmp_path)

    assert runner._record_restart_replay_mark(
        entry.session_key, now=1.0, request_id="req-1"
    ) is False
    assert runner._record_restart_replay_mark(
        entry.session_key, now=1.1, request_id="req-1"
    ) is False
    assert runner._record_restart_replay_mark(
        entry.session_key, now=2.0, request_id="req-2"
    ) is False
    assert runner._record_restart_replay_mark(
        entry.session_key, now=3.0, request_id="req-3"
    ) is True
    assert runner.session_store._entries[entry.session_key].suspended is True

    counts = runner._load_restart_failure_counts()[entry.session_key]
    assert counts["replay_request_ids"] == ["req-1", "req-2", "req-3"]
    assert counts["replay_marks"] == [1.0, 2.0, 3.0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case_id,order",
    [
        ("T9a", ("req-1", "req-2")),
        ("T9b", ("req-2", "req-1")),
    ],
)
async def test_t9a_t9b_two_initiators_sequential_orders_coalesce_once(
    tmp_path: Path, case_id: str, order: tuple[str, str]
) -> None:
    store, first = _store(tmp_path)
    second = store.get_or_create_session(
        SessionSource(platform=Platform.TELEGRAM, chat_id="456", user_id="u2")
    )
    _submit(tmp_path, first.session_key)
    submit_deferred_restart(
        tmp_path,
        session_key=second.session_key,
        handoff="second handoff",
        boot_id="11:22",
        intent_ts=10.5,
        request_id="req-2",
    )
    coordinator = DeferredRestartCoordinator(tmp_path, boot_id="11:22")
    for key in (first.session_key, second.session_key):
        assert coordinator.arm_for_session(key, consume_breadcrumb=lambda _k: True) == "armed"

    key_by_id = {"req-1": first.session_key, "req-2": second.session_key}
    delivered = asyncio.Event()
    delivered.set()
    signals: list[str] = []
    marks: list[str] = []
    replays: list[str] = []
    tasks = []
    for request_id in order:
        tasks.append(
            coordinator.schedule_armed(
                key_by_id[request_id],
                delivery_event=delivered,
                delivery_timeout=1.0,
                record_replay=lambda req: replays.append(req.request_id),
                mark_self=lambda req: marks.append(req.request_id),
                signal_restart=lambda: signals.append(case_id),
            )
        )
        await asyncio.sleep(0)
    outcomes = await asyncio.gather(*tasks)

    assert outcomes == ["signaled", "coalesced"]
    assert signals == [case_id]
    assert sorted(marks) == ["req-1", "req-2"]
    assert sorted(replays) == ["req-1", "req-2"]


@pytest.mark.asyncio
async def test_t9c_two_concurrent_leaders_signal_once_and_preserve_both_handoffs(tmp_path: Path) -> None:
    store, first = _store(tmp_path)
    second = store.get_or_create_session(
        SessionSource(platform=Platform.TELEGRAM, chat_id="456", user_id="u2")
    )
    _submit(tmp_path, first.session_key)
    submit_deferred_restart(
        tmp_path,
        session_key=second.session_key,
        handoff="second handoff",
        boot_id="11:22",
        intent_ts=10.5,
        request_id="req-2",
    )
    coordinator = DeferredRestartCoordinator(tmp_path, boot_id="11:22")
    assert coordinator.arm_for_session(first.session_key, consume_breadcrumb=lambda _k: True) == "armed"
    assert coordinator.arm_for_session(second.session_key, consume_breadcrumb=lambda _k: True) == "armed"

    delivered = asyncio.Event()
    delivered.set()
    signals: list[int] = []
    marks: list[str] = []
    replays: list[str] = []
    tasks = [
        coordinator.schedule_armed(
            key,
            delivery_event=delivered,
            delivery_timeout=1.0,
            record_replay=lambda req: replays.append(req.request_id),
            mark_self=lambda req: marks.append(req.request_id),
            signal_restart=lambda: signals.append(1),
        )
        for key in (first.session_key, second.session_key)
    ]
    await asyncio.gather(*tasks)

    assert len(signals) == 1
    assert sorted(marks) == ["req-1", "req-2"]
    assert sorted(replays) == ["req-1", "req-2"]
    states = {req.request_id: req.state for req in coordinator.scan()}
    assert sorted(states.values()) == ["claimed", "coalesce_pending"]


@pytest.mark.asyncio
async def test_t9c_restart_waits_for_every_staggered_delivery_barrier(
    tmp_path: Path,
) -> None:
    store, first = _store(tmp_path)
    second = store.get_or_create_session(
        SessionSource(platform=Platform.TELEGRAM, chat_id="456", user_id="u2")
    )
    _submit(tmp_path, first.session_key)
    submit_deferred_restart(
        tmp_path,
        session_key=second.session_key,
        handoff="second handoff",
        boot_id="11:22",
        intent_ts=10.5,
        request_id="req-2",
    )
    coordinator = DeferredRestartCoordinator(tmp_path, boot_id="11:22")
    for key in (first.session_key, second.session_key):
        coordinator.arm_for_session(key, consume_breadcrumb=lambda _k: True)

    first_delivered = asyncio.Event()
    first_delivered.set()
    second_delivered = asyncio.Event()
    signals: list[int] = []
    first_task = coordinator.schedule_armed(
        first.session_key,
        delivery_event=first_delivered,
        delivery_timeout=1.0,
        record_replay=lambda req: None,
        mark_self=lambda req: None,
        signal_restart=lambda: signals.append(1),
    )
    second_task = coordinator.schedule_armed(
        second.session_key,
        delivery_event=second_delivered,
        delivery_timeout=1.0,
        record_replay=lambda req: None,
        mark_self=lambda req: None,
        signal_restart=lambda: signals.append(1),
    )

    await asyncio.sleep(0.05)
    assert signals == []
    second_delivered.set()
    await asyncio.gather(first_task, second_task)
    assert signals == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case_id,boundary",
    [
        ("T9n1", "after_meta_publish"),
        ("T9n2", "after_claim"),
        ("T9n3", "after_replay_mark"),
        ("T9n4", "after_self_mark"),
        ("T9n5", "before_commit_publish"),
        ("T9n6", "after_peer_deliveries"),
    ],
)
async def test_t9n_leader_cancellation_releases_latch_and_loser_re_elects(
    tmp_path: Path, case_id: str, boundary: str
) -> None:
    store, first = _store(tmp_path)
    second = store.get_or_create_session(
        SessionSource(platform=Platform.TELEGRAM, chat_id="456", user_id="u2")
    )
    _submit(tmp_path, first.session_key)
    submit_deferred_restart(
        tmp_path,
        session_key=second.session_key,
        handoff="second",
        boot_id="11:22",
        intent_ts=11.0,
        request_id="req-2",
    )
    coordinator = DeferredRestartCoordinator(tmp_path, boot_id="11:22")
    coordinator.arm_for_session(first.session_key, consume_breadcrumb=lambda _k: True)
    coordinator.arm_for_session(second.session_key, consume_breadcrumb=lambda _k: True)
    delivered = asyncio.Event()
    delivered.set()
    signals: list[int] = []
    injected = False

    def inject(name: str, req: DeferredRestartRequest) -> None:
        nonlocal injected
        if not injected and req.request_id == "req-1" and name == boundary:
            injected = True
            raise asyncio.CancelledError(case_id)

    first_task = coordinator.schedule_armed(
        first.session_key,
        delivery_event=delivered,
        delivery_timeout=1.0,
        record_replay=lambda req: None,
        mark_self=lambda req: None,
        signal_restart=lambda: signals.append(1),
        checkpoint=inject,
    )
    second_task = coordinator.schedule_armed(
        second.session_key,
        delivery_event=delivered,
        delivery_timeout=1.0,
        record_replay=lambda req: None,
        mark_self=lambda req: None,
        signal_restart=lambda: signals.append(1),
    )
    outcomes = await asyncio.gather(first_task, second_task)
    assert sorted(outcomes) == ["coalesced", "signaled"]
    assert signals == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case_id,boundary",
    [
        ("T9n1", "after_meta_publish"),
        ("T9n2", "after_claim"),
        ("T9n3", "after_replay_mark"),
        ("T9n4", "after_self_mark"),
        ("T9n5", "before_commit_publish"),
        ("T9n6", "after_peer_deliveries"),
    ],
)
async def test_t9n_single_request_retries_every_precommit_boundary(
    tmp_path: Path, case_id: str, boundary: str
) -> None:
    _store_obj, entry = _store(tmp_path)
    _submit(tmp_path, entry.session_key)
    coordinator = DeferredRestartCoordinator(tmp_path, boot_id="11:22")
    coordinator.arm_for_session(entry.session_key, consume_breadcrumb=lambda _k: True)
    delivered = asyncio.Event()
    delivered.set()
    signals: list[str] = []
    injected = False

    def inject_once(name: str, _req: DeferredRestartRequest) -> None:
        nonlocal injected
        if not injected and name == boundary:
            injected = True
            raise asyncio.CancelledError(case_id)

    outcome = await coordinator.schedule_armed(
        entry.session_key,
        delivery_event=delivered,
        delivery_timeout=1.0,
        record_replay=lambda req: None,
        mark_self=lambda req: None,
        signal_restart=lambda: signals.append(case_id),
        checkpoint=inject_once,
    )
    assert outcome == "signaled"
    assert injected is True
    assert signals == [case_id]


@pytest.mark.parametrize(
    "case_id,state",
    [
        ("T9e", "claimed"),
        ("T9f", "claimed"),
        ("T9g", "claimed"),
        ("T9h", "claimed"),
        ("T9i", "claimed"),
        ("T9d", "submitted"),
        ("T9l", "armed"),
        ("T9l", "claimed"),
        ("T9l", "coalesce_pending"),
    ],
)
def test_t9_cross_boot_states_synthesize_handoff_without_signal(
    tmp_path: Path, case_id: str, state: str
) -> None:
    store, entry = _store(tmp_path)
    request = _submit(tmp_path, entry.session_key)
    coordinator = DeferredRestartCoordinator(tmp_path, boot_id="old:boot")
    if state != "submitted":
        request = coordinator.transition(request, state)
    replayed: list[str] = []
    marked: list[str] = []
    flushed: list[str] = []
    signals: list[int] = []

    count = reconcile_deferred_restarts_at_boot(
        tmp_path,
        current_boot_id="new:boot",
        boot_started_at=20.0,
        has_durable_mark=lambda req: False,
        record_replay=lambda req: replayed.append(req.request_id),
        mark_in_memory=lambda req: marked.append(req.request_id),
        flush_sessions=lambda: flushed.append("flush"),
        signal_restart=lambda: signals.append(1),
    )

    assert count == 1
    assert replayed == ["req-1"]
    assert marked == ["req-1"]
    assert flushed == ["flush"]
    assert signals == []
    assert not any(coordinator.requests_dir.iterdir())


def test_t9m_boot_before_intent_quarantines(tmp_path: Path) -> None:
    _store_obj, entry = _store(tmp_path)
    _submit(tmp_path, entry.session_key, intent_ts=30.0)
    count = reconcile_deferred_restarts_at_boot(
        tmp_path,
        current_boot_id="new:boot",
        boot_started_at=20.0,
        has_durable_mark=lambda req: False,
        record_replay=lambda req: None,
        mark_in_memory=lambda req: None,
        flush_sessions=lambda: None,
        signal_restart=lambda: pytest.fail("cross-boot reconciliation signaled"),
    )
    assert count == 0
    [path] = list((tmp_path / "gateway" / "resume_requests").iterdir())
    assert ".rejected" in path.name


def test_t9o_boot_crash_after_in_memory_mark_before_flush_retries_once(tmp_path: Path) -> None:
    _store_obj, entry = _store(tmp_path)
    _submit(tmp_path, entry.session_key)
    durable: set[str] = set()
    volatile: set[str] = set()

    def crash(name: str, req: DeferredRestartRequest) -> None:
        if name == "after_boot_mark_before_flush":
            raise RuntimeError("crash")

    with pytest.raises(RuntimeError, match="crash"):
        reconcile_deferred_restarts_at_boot(
            tmp_path,
            current_boot_id="new:boot",
            boot_started_at=20.0,
            has_durable_mark=lambda req: req.request_id in durable,
            record_replay=lambda req: None,
            mark_in_memory=lambda req: volatile.add(req.request_id),
            flush_sessions=lambda: durable.update(volatile),
            signal_restart=lambda: pytest.fail("cross-boot reconciliation signaled"),
            checkpoint=crash,
        )
    volatile.clear()  # process died; in-memory mark vanished
    assert durable == set()

    assert reconcile_deferred_restarts_at_boot(
        tmp_path,
        current_boot_id="newer:boot",
        boot_started_at=21.0,
        has_durable_mark=lambda req: req.request_id in durable,
        record_replay=lambda req: None,
        mark_in_memory=lambda req: volatile.add(req.request_id),
        flush_sessions=lambda: durable.update(volatile),
        signal_restart=lambda: pytest.fail("cross-boot reconciliation signaled"),
    ) == 1
    assert durable == {"req-1"}


def test_t9p_durable_mark_before_consume_is_idempotent(tmp_path: Path) -> None:
    _store_obj, entry = _store(tmp_path)
    _submit(tmp_path, entry.session_key)
    marked: list[str] = []
    assert reconcile_deferred_restarts_at_boot(
        tmp_path,
        current_boot_id="new:boot",
        boot_started_at=20.0,
        has_durable_mark=lambda req: True,
        record_replay=lambda req: None,
        mark_in_memory=lambda req: marked.append(req.request_id),
        flush_sessions=lambda: None,
        signal_restart=lambda: pytest.fail("cross-boot reconciliation signaled"),
    ) == 1
    assert marked == []


def test_t9q_terminal_consumed_file_is_deleted_without_reprocess(tmp_path: Path) -> None:
    _store_obj, entry = _store(tmp_path)
    request = _submit(tmp_path, entry.session_key)
    coordinator = DeferredRestartCoordinator(tmp_path, boot_id="old:boot")
    coordinator.transition(request, "consumed")
    assert reconcile_deferred_restarts_at_boot(
        tmp_path,
        current_boot_id="new:boot",
        boot_started_at=20.0,
        has_durable_mark=lambda req: False,
        record_replay=lambda req: pytest.fail("terminal request replayed"),
        mark_in_memory=lambda req: pytest.fail("terminal request marked"),
        flush_sessions=lambda: pytest.fail("terminal request flushed"),
        signal_restart=lambda: pytest.fail("terminal request signaled"),
    ) == 0
    assert not any(coordinator.requests_dir.iterdir())
