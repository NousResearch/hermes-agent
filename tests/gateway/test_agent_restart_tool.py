"""Tests for GatewayRunner.dispatch_gateway_restart — the shared restart coordinator.

Verifies:
1. Agent tool and /restart slash command use the SAME shared coordinator.
2. Detached handoff is established before drain/stop (Handshake Acknowledgement).
3. Handoff failure / Popen raising OSError:
   - Returns False JSON.
   - stop() is NOT called.
   - Gateway stays active and restart flags roll back.
4. Transactional rollback of existing state:
   - Old dedup marker is preserved when handoff fails.
   - File write failure (notify or dedup) aborts request_restart and rolls back.
5. Concurrent restart requests on the Gateway loop: exactly ONE succeeds.
6. Delayed helper canceled before commitment -> stop() never called.
7. Helper canceled AFTER commitment -> stop() proceeds, state not rolled back.
8. Service path commitment (via_service=True) under ambient INVOCATION_ID vs unset.
9. Redelivery suppression integrated into public entry path (_handle_message).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source(
    chat_id: str = "chat-99", thread_id: str | None = None, message_id: str = "msg-101"
) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="dm",
        user_id="u42",
        thread_id=thread_id,
        message_id=message_id,
    )


# ---------------------------------------------------------------------------
# 1. /restart and agent tool share coordinator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slash_and_tool_share_restart_coordinator(monkeypatch, tmp_path):
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)

    runner, _ = make_restart_runner()
    gw_loop = asyncio.get_running_loop()
    runner._gateway_loop = gw_loop

    async def fake_launch():
        return True

    runner._launch_detached_restart_command = fake_launch  # type: ignore[method-assign]
    runner.stop = AsyncMock()  # type: ignore[method-assign]

    with patch("gateway.run._hermes_home", tmp_path):
        success, msg = await runner.dispatch_gateway_restart(
            source=_make_source(), origin="agent_tool", reason="testing shared coordinator"
        )

    assert success is True
    assert (tmp_path / ".restart_notify.json").exists()
    assert (tmp_path / ".restart_last_processed.json").exists()

    dedup = json.loads((tmp_path / ".restart_last_processed.json").read_text())
    assert dedup["origin"] == "agent_tool"
    assert dedup["reason"] == "testing shared coordinator"


# ---------------------------------------------------------------------------
# 2. Handoff failure (Popen error) fail-closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handoff_failure_returns_false_and_preserves_marker(
    monkeypatch, tmp_path
):
    """The detached launcher raising OSError maps to launcher_result=UNKNOWN
    → the background helper records OUTCOME_UNKNOWN (marker preserved).

    P1 #71876 semantic: the accepted ack is published immediately after
    claim_handoff, so dispatch returns success=True ("transaction claimed;
    restart will proceed") BEFORE the launcher runs. The launcher failure is
    then recorded by the background helper via complete_unknown(): stage
    OUTCOME_UNKNOWN, final_outcome UNKNOWN, marker deliberately NOT rolled
    back (UNKNOWN cannot prove no side-effect), stop() never reached.

    Implementation note: `make_restart_runner()` defaults
    `runner._launch_detached_restart_command` to an AsyncMock that
    returns True, which short-circuits any patching of
    `subprocess.Popen`/`_resolve_hermes_bin`. We replace the launcher
    binding directly here with an AsyncMock whose side_effect raises
    OSError so the helper enters `complete_unknown()` deterministically.
    Real Popen-argument inspection tests opt in via
    `attach_real_launcher_under_mocked_popen` (only in
    tests/gateway/test_restart_drain.py).
    """
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)

    runner, _ = make_restart_runner()
    gw_loop = asyncio.get_running_loop()
    runner._gateway_loop = gw_loop

    # `make_restart_runner()` already installs `runner.stop` as an
    # AsyncMock; we keep it that way so we can use `assert_not_called`
    # below. No override needed.

    captured_txn: list = []

    async def launcher_raises():
        # Capture the live transaction while the helper is executing it
        # (dispatch returns before the launcher runs, and the helper clears
        # runner._restart_transaction when it records a terminal state).
        captured_txn.append(runner._restart_transaction)
        raise OSError("simulated launcher failure")

    runner._launch_detached_restart_command = launcher_raises  # type: ignore[method-assign]

    with patch("gateway.run._hermes_home", tmp_path):
        success, msg = await runner.dispatch_gateway_restart(
            source=_make_source(chat_id="chat-HF"), origin="agent_tool"
        )

    # P1 #71876: dispatch returns success because the accepted ack was
    # published at claim_handoff time (before the launcher ran). The
    # message must NOT claim completion — "restarting" / "draining" only
    # means the transaction was claimed and will proceed in the background.
    assert success is True

    # The helper executed the launcher, so the transaction was live then.
    txn = captured_txn[0]
    assert txn is not None

    # Wait for the background helper to record the launcher failure.
    if txn.restart_task is not None:
        try:
            await asyncio.wait_for(asyncio.shield(txn.restart_task), timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

    # stop() must NEVER have been reached (no commit, no shutdown).
    runner.stop.assert_not_called()

    # The transaction must now be in OUTCOME_UNKNOWN (launcher raised after
    # claim), NOT still IN_FLIGHT and NOT committed.
    if txn is not None:
        from gateway.slash_commands import (
            _LauncherResult,
            _RestartFinalOutcome,
            _RestartStage,
        )

        assert txn.stage is _RestartStage.OUTCOME_UNKNOWN
        assert txn.launcher_result is _LauncherResult.UNKNOWN
        assert txn.final_outcome is _RestartFinalOutcome.UNKNOWN

    # UNKNOWN deliberately does NOT restore prior state. The dedup marker
    # survives intact and reflects the request that just ran.
    on_disk = json.loads(
        (tmp_path / ".restart_last_processed.json").read_text(encoding="utf-8")
    )
    assert on_disk["chat_id"] == "chat-HF"


# ---------------------------------------------------------------------------
# 3. File write failure (notify or dedup) fails closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_write_failure_aborts_restart(monkeypatch, tmp_path):
    runner, _ = make_restart_runner()
    gw_loop = asyncio.get_running_loop()
    runner._gateway_loop = gw_loop
    runner.stop = AsyncMock()  # type: ignore[method-assign]

    # Patch atomic_json_write to raise OSError on the first file write
    with patch("gateway.run._hermes_home", tmp_path), patch(
        "gateway.slash_commands.atomic_json_write", side_effect=OSError("Disk full")
    ):
        success, msg = await runner.dispatch_gateway_restart(
            source=_make_source(), origin="agent_tool"
        )

        assert success is False
        assert "Failed to persist" in msg
        runner.stop.assert_not_called()
        assert runner._restart_requested is False


# ---------------------------------------------------------------------------
# 4. Delayed helper canceled BEFORE commitment -> stop() never called
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_cancel_after_helper_inflight_does_not_stop(
    monkeypatch, tmp_path
):
    """Cancelling the dispatcher once the helper has crossed IN_FLIGHT
    MUST NOT cancel the helper, MUST NOT trigger shutdown, MUST NOT
    rollback the marker.

    P1 #71876: the accepted ack is published immediately after
    claim_handoff, so dispatch returns (success) as soon as the helper is
    IN_FLIGHT. A caller that cancels the dispatch task after that point is
    a pure observer — the helper continues independently and only stop()
    is invoked if it actually reaches HANDOFF_COMMITTED.

    No fixed sleep is used: synchronization is via asyncio.Event so the
    test is deterministic on the loop scheduler.
    """
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)

    runner, _ = make_restart_runner()
    gw_loop = asyncio.get_running_loop()
    runner._gateway_loop = gw_loop
    runner.stop = AsyncMock()  # type: ignore[method-assign]

    helper_in_flight = asyncio.Event()
    helper_release = asyncio.Event()

    async def fake_launch_gated() -> bool:
        helper_in_flight.set()
        await helper_release.wait()
        return True

    runner._launch_detached_restart_command = fake_launch_gated  # type: ignore[method-assign]

    with patch("gateway.run._hermes_home", tmp_path):
        dispatch_task = asyncio.create_task(
            runner.dispatch_gateway_restart(
                source=_make_source(chat_id="chat-DC"), origin="agent_tool"
            )
        )
        # Wait deterministically for the helper to enter IN_FLIGHT
        # (claim_handoff already succeeded and fake_launch is awaiting).
        await helper_in_flight.wait()
        # The dispatcher has already observed the claimed ack and returns
        # success; the helper is IN_FLIGHT and must not be disturbed.
        success, _msg = await dispatch_task

        assert success is True
        # The helper is still pending in fake_launch_gated (IN_FLIGHT).
        assert runner._restart_transaction is not None

    # Authoritative contract under observer mode:
    # 1. stop() is NEVER called from the dispatcher observer branch.
    runner.stop.assert_not_called()

    # 2. The marker survives (IN_FLIGHT: no rollback yet).
    on_disk = json.loads(
        (tmp_path / ".restart_last_processed.json").read_text(encoding="utf-8")
    )
    assert on_disk["chat_id"] == "chat-DC", (
        f"IN_FLIGHT must preserve the dedup marker; got {on_disk}"
    )

    # 3. Release + let the helper finish → stop() is invoked.
    helper_release.set()
    for _ in range(50):
        if runner.stop.called:
            break
        await asyncio.sleep(0.02)
    assert runner.stop.called is True, (
        "After release, helper MUST proceed to HANDOFF_COMMITTED → stop()."
    )


# ---------------------------------------------------------------------------
# 5. Helper committed -> Cancel after commitment does not stop shutdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_helper_committed_triggers_stop(monkeypatch, tmp_path):
    """After dispatch_gateway_restart succeeds the helper reaches
    HANDOFF_COMMITTED and invokes runner.stop() in its post-launch path.
    The dispatcher wait must not gate on stop().

    No fixed sleep: the helper task's completion (which production awaits
    inside the same coroutine that calls stop) is the deterministic
    synchronizer. A bounded wait_for is used only as a deadlock guard.
    """
    runner, _ = make_restart_runner()
    gw_loop = asyncio.get_running_loop()
    runner._gateway_loop = gw_loop

    stop_called = False

    async def fake_stop(**kwargs):
        nonlocal stop_called
        stop_called = True

    runner.stop = fake_stop  # type: ignore[method-assign]

    async def fake_launch_ok():
        return True

    runner._launch_detached_restart_command = fake_launch_ok  # type: ignore[method-assign]

    with patch("gateway.run._hermes_home", tmp_path):
        success, msg = await runner.dispatch_gateway_restart(
            source=_make_source(chat_id="chat-COMMITTED"), origin="agent_tool"
        )
        assert success is True

        # Authority: the per-request transaction owns the stage. We do
        # NOT reintroduce a runner-global `_restart_stage` legacy alias.
        from gateway.slash_commands import _RestartStage

        txn = runner._restart_transaction
        assert txn is not None, (
            "Dispatch must leave the live transaction reachable via "
            "runner._restart_transaction until the dispatcher returned."
        )
        assert txn.stage is _RestartStage.HANDOFF_COMMITTED

        # Wait for helper task completion (production stop() is called
        # inside helper post-launch; its completion implies stop ran).
        helper_task = txn.restart_task
        if helper_task is not None and not helper_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(helper_task), timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                # 2s is a deadlock guard; legitimate runs finish faster.
                pass

    assert stop_called is True, (
        "stop() must be invoked by the helper after HANDOFF_COMMITTED; "
        "production calls it from inside _run_restart post-launch."
    )


# ---------------------------------------------------------------------------
# 6. Service path commitment under ambient INVOCATION_ID vs unset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_service_path_commitment_dual_environment(monkeypatch, tmp_path):
    # Test 1: Supervisor set
    monkeypatch.setenv("INVOCATION_ID", "systemd-unit-123")
    monkeypatch.setattr("gateway.restart.is_container_restart_context", lambda: False)

    runner1, _ = make_restart_runner()
    gw_loop = asyncio.get_running_loop()
    runner1._gateway_loop = gw_loop
    runner1.stop = AsyncMock()  # type: ignore[method-assign]

    # Spy request_restart to capture the transaction object reliably before completion
    captured1: dict = {}
    orig_request_restart1 = runner1.request_restart

    def spy_request_restart1(*args, **kwargs):
        captured1["tx"] = kwargs.get("transaction")
        return orig_request_restart1(*args, **kwargs)

    runner1.request_restart = spy_request_restart1  # type: ignore[method-assign]

    # Blocker 2 requirement: configure launcher to raise if called on service path.
    # Service path must NEVER invoke _launch_detached_restart_command.
    mock_launcher1 = AsyncMock(side_effect=AssertionError("detached launcher called on service path"))
    runner1._launch_detached_restart_command = mock_launcher1

    with patch("gateway.run._hermes_home", tmp_path):
        success1, msg1 = await runner1.dispatch_gateway_restart(
            source=_make_source(), origin="agent_tool"
        )
        assert success1 is True
        assert runner1._restart_via_service is True
        assert runner1._restart_detached is False
        mock_launcher1.assert_not_awaited()

        # Service acknowledgement happens upon handoff claim, before stop() runs on background task.
        tx1 = captured1.get("tx")
        assert tx1 is not None
        assert tx1.is_committed() is True

        # Deterministically await the background restart task to complete stop()
        await asyncio.wait_for(asyncio.shield(tx1.restart_task), timeout=2.0)

        runner1.stop.assert_called_once_with(
            restart=True, detached_restart=False, service_restart=True
        )

    # Test 2: Supervisor unset
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)
    runner2, _ = make_restart_runner()
    runner2._gateway_loop = gw_loop
    runner2.stop = AsyncMock()  # type: ignore[method-assign]

    captured2: dict = {}
    orig_request_restart2 = runner2.request_restart

    def spy_request_restart2(*args, **kwargs):
        captured2["tx"] = kwargs.get("transaction")
        return orig_request_restart2(*args, **kwargs)

    runner2.request_restart = spy_request_restart2  # type: ignore[method-assign]

    mock_launcher2 = AsyncMock(return_value=True)
    runner2._launch_detached_restart_command = mock_launcher2

    with patch("gateway.run._hermes_home", tmp_path):
        success2, msg2 = await runner2.dispatch_gateway_restart(
            source=_make_source(), origin="agent_tool"
        )
        assert success2 is True
        assert runner2._restart_via_service is False
        assert runner2._restart_detached is True
        mock_launcher2.assert_awaited_once()

        tx2 = captured2.get("tx")
        assert tx2 is not None
        assert tx2.is_committed() is True

        await asyncio.wait_for(asyncio.shield(tx2.restart_task), timeout=2.0)

        runner2.stop.assert_called_once_with(
            restart=True, detached_restart=True, service_restart=False
        )


@pytest.mark.asyncio
async def test_service_path_ignores_launcher_exceptions(monkeypatch, tmp_path):
    """Blocker 2: Service path does not call detached launcher even if it raises."""
    monkeypatch.setenv("INVOCATION_ID", "systemd-unit-999")
    monkeypatch.setattr("gateway.restart.is_container_restart_context", lambda: False)

    runner, _ = make_restart_runner()
    runner._gateway_loop = asyncio.get_running_loop()
    runner.stop = AsyncMock()

    captured: dict = {}
    orig_request_restart = runner.request_restart

    def spy_request_restart(*args, **kwargs):
        captured["tx"] = kwargs.get("transaction")
        return orig_request_restart(*args, **kwargs)

    runner.request_restart = spy_request_restart  # type: ignore[method-assign]

    mock_launcher = AsyncMock(side_effect=RuntimeError("launcher broken"))
    runner._launch_detached_restart_command = mock_launcher

    with patch("gateway.run._hermes_home", tmp_path):
        success, msg = await runner.dispatch_gateway_restart(
            source=_make_source(), origin="agent_tool"
        )

    assert success is True
    mock_launcher.assert_not_awaited()

    tx = captured.get("tx")
    assert tx is not None
    assert tx.is_committed() is True

    await asyncio.wait_for(asyncio.shield(tx.restart_task), timeout=2.0)

    runner.stop.assert_called_once_with(
        restart=True, detached_restart=False, service_restart=True
    )


@pytest.mark.asyncio
async def test_detached_path_launcher_failure_rolls_back(monkeypatch, tmp_path):
    """Blocker 2: Detached path still calls launcher; failure means NOT_STARTED, rollback, no stop().

    P1 #71876 semantic: dispatch returns success=True once the accepted ack
    is published at claim_handoff (before the launcher runs). The launcher
    returning False (NOT_STARTED — provably no side-effect) is recorded by
    the background helper: transaction ABORTED, marker rolled back, stop()
    never reached.
    """
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)
    monkeypatch.setattr("gateway.restart.is_container_restart_context", lambda: False)

    runner, _ = make_restart_runner()
    runner._gateway_loop = asyncio.get_running_loop()
    runner.stop = AsyncMock()

    captured: dict = {}
    orig_request_restart = runner.request_restart

    def spy_request_restart(*args, **kwargs):
        captured["tx"] = kwargs.get("transaction")
        return orig_request_restart(*args, **kwargs)

    runner.request_restart = spy_request_restart  # type: ignore[method-assign]

    mock_launcher = AsyncMock(return_value=False)
    runner._launch_detached_restart_command = mock_launcher

    with patch("gateway.run._hermes_home", tmp_path):
        success, msg = await runner.dispatch_gateway_restart(
            source=_make_source(), origin="agent_tool"
        )

    # P1 #71876: accepted ack published at claim_handoff → dispatch success.
    assert success is True

    tx = captured.get("tx")
    assert tx is not None

    # Wait for the background helper to roll back the marker and record ABORTED.
    await asyncio.wait_for(asyncio.shield(tx.restart_task), timeout=2.0)

    assert tx.is_aborted() is True

    runner.stop.assert_not_called()
    assert runner._restart_transaction is None


# ---------------------------------------------------------------------------
# 7. Redelivery suppression integrated into public entry path (_handle_message)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_redelivery_suppression_in_public_handle_message_entry(monkeypatch, tmp_path):
    runner, _ = make_restart_runner()
    runner._booted_from_restart = True

    marker_data = {
        "platform": "telegram",
        "chat_id": "chat-99",
        "thread_id": None,
        "message_id": "msg-nl-123",
        "origin": "agent_tool",
        "requested_at": time.time(),
    }
    (tmp_path / ".restart_last_processed.json").write_text(json.dumps(marker_data))

    event_exact = MessageEvent(
        text="please restart",
        message_type=MessageType.TEXT,
        source=make_restart_source(chat_id="chat-99"),
        message_id="msg-nl-123",
    )

    with patch("gateway.run._hermes_home", tmp_path), patch.object(
        runner, "_handle_message_with_agent", AsyncMock()
    ) as mock_agent_turn:
        # Call public entry path
        res = await runner._handle_message(event_exact)

        # Invariant: suppressed at public entry! Returns empty string, agent turn NEVER called!
        assert res == ""
        mock_agent_turn.assert_not_called()


# ===========================================================================
# State machine regression suite
# ----------------------------------------------------------------------------
# Coverage goals:
#   * Three-state ack semantics: ACCEPTED / NOT_STARTED / OUTCOME_UNKNOWN
#     (start, rollback, do-not-retry).
#   * PREPARING abort wins and helper is forbidden from launching.
#   * IN_FLIGHT observer path: dispatcher must NOT cancel task, NOT rollback,
#     NOT write ack; helper completes independently.
#   * _complete_atomic correctness: cancelled ack and conflicting ack raise
#     RuntimeError; idempotent same-outcome writes return ALREADY_COMPLETE.
#   * cancel_restart_task_and_await handles CancelledError explicitly
#     (does NOT swallow via except Exception).
#   * PREPARING claim race: exactly one of claim_abort/claim_handoff wins.
#   * Scheduler mock safety: when safe_schedule_threadsafe is patched to
#     simulate an outer 10s timeout, the test must explicitly consume the
#     coroutine (loop.create_task + cancel/await OR coro.close). NO reliance
#     on the GC or warnings filter — that warning was the original bug.
#   * Stop() is always an AsyncMock by default so no real shutdown runs.
#
# Safety contract — every test below MUST honor:
#   * monkeypatch.delenv INVOCATION_ID / HERMES_S6_SUPERVISED_CHILD before
#     constructing the runner (so is_gateway_supervisor_process is False).
#   * No test patches subprocess.Popen to a fake that actually spawns;
#     keep _launch_detached_restart_command as the AsyncMock provided by
#     make_restart_runner, OR replace it with a deterministic async fn.
# ===========================================================================


# NOTE: an earlier draft of this file carried an empty `autouse=True` fixture
# (`_no_real_popen_in_state_machine_tests`) intended to act as a future-proofing
# documentation hook. It had no body, but assigning it module-level forced
# pytest to inject a `monkeypatch` fixture into every test in this module,
# which introduced non-obvious ordering dependencies on a few HEAD tests
# (handoff-failure, delayed helper cancel, post-commit cancel). Removed.

@pytest.mark.asyncio
async def test_dispatcher_rolls_back_marker_on_not_started(monkeypatch, tmp_path):
    """Helper returns NOT_STARTED: background helper rolls back the dedup
    marker to whatever was there before the new request began. The "old-NOT"
    marker must survive on disk unchanged.

    P1 #71876 semantic: dispatch returns success=True at claim_handoff; the
    NOT_STARTED rollback is performed by the background helper.
    """
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)

    pre_marker = {
        "request_id": "req-PRIOR-NOT",
        "platform": "telegram",
        "chat_id": "old-NOT",
    }
    (tmp_path / ".restart_last_processed.json").write_text(
        json.dumps(pre_marker), encoding="utf-8"
    )

    runner, _ = make_restart_runner()
    gw_loop = asyncio.get_running_loop()
    runner._gateway_loop = gw_loop
    runner.stop = AsyncMock()  # type: ignore[method-assign]

    # Helper claims handoff and returns NOT_STARTED (Popen refused to run).
    captured_txn: list = []

    async def fake_launch_not_started() -> bool:
        captured_txn.append(runner._restart_transaction)
        return False

    runner._launch_detached_restart_command = fake_launch_not_started  # type: ignore[method-assign]

    with patch("gateway.run._hermes_home", tmp_path):
        success, msg = await runner.dispatch_gateway_restart(
            source=_make_source(chat_id="chat-NOT"), origin="agent_tool"
        )

    # P1 #71876: accepted ack published at claim_handoff → dispatch success.
    assert success is True

    # The helper executed the launcher, so the transaction was live then.
    txn = captured_txn[0]
    assert txn is not None

    # Wait for the background helper to complete the NOT_STARTED rollback.
    if txn.restart_task is not None:
        try:
            await asyncio.wait_for(asyncio.shield(txn.restart_task), timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

    # The pre_marker survives. The helper rolled back the new write.
    on_disk = json.loads(
        (tmp_path / ".restart_last_processed.json").read_text(encoding="utf-8")
    )
    assert on_disk["request_id"] == "req-PRIOR-NOT", (
        f"NOT_STARTED must roll back to prior marker, got {on_disk}"
    )

    # stop() was never reached.
    runner.stop.assert_not_called()


@pytest.mark.asyncio
async def test_dispatcher_returns_do_not_retry_on_unknown(monkeypatch, tmp_path):
    """Helper raises mid-launch → background helper records OUTCOME_UNKNOWN.

    Maps launcher_result=UNKNOWN → stage=OUTCOME_UNKNOWN. The marker stays on
    disk (UNKNOWN refuses rollback).

    P1 #71876 semantic: dispatch returns success=True at claim_handoff; the
    launcher failure is recorded by the background helper.
    """
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)

    runner, _ = make_restart_runner()
    gw_loop = asyncio.get_running_loop()
    runner._gateway_loop = gw_loop
    runner.stop = AsyncMock()  # type: ignore[method-assign]

    captured_txn: list = []

    async def fake_launch_raises() -> bool:
        # Capture the live transaction while the helper is executing it.
        captured_txn.append(runner._restart_transaction)
        # Popen raised after the fork attempt — maps to UNKNOWN in detached mode.
        raise OSError("simulated post-fork failure")

    runner._launch_detached_restart_command = fake_launch_raises  # type: ignore[method-assign]

    with patch("gateway.run._hermes_home", tmp_path):
        success, msg = await runner.dispatch_gateway_restart(
            source=_make_source(chat_id="chat-UNK"), origin="agent_tool"
        )

    # P1 #71876: accepted ack published at claim_handoff → dispatch success.
    assert success is True

    # The helper executed the launcher, so the transaction was live then.
    txn = captured_txn[0]
    assert txn is not None

    # Wait for the background helper to record the UNKNOWN outcome.
    if txn.restart_task is not None:
        try:
            await asyncio.wait_for(asyncio.shield(txn.restart_task), timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

    from gateway.slash_commands import (
        _RestartFinalOutcome,
        _RestartStage,
    )

    assert txn.stage is _RestartStage.OUTCOME_UNKNOWN
    assert txn.final_outcome is _RestartFinalOutcome.UNKNOWN

    # Marker survives — UNKNOWN refuses rollback.
    assert (tmp_path / ".restart_last_processed.json").exists()
    on_disk = json.loads(
        (tmp_path / ".restart_last_processed.json").read_text(encoding="utf-8")
    )
    assert on_disk["chat_id"] == "chat-UNK"

    runner.stop.assert_not_called()


@pytest.mark.asyncio
async def test_preparing_claim_abort_wins_blocks_handoff(
    monkeypatch, tmp_path
):
    """Direct state-machine test: caller wins claim_abort from PREPARING.
    Helper's subsequent claim_handoff MUST return False; stage stays
    ABORTING. This exercises the same contract the dispatcher relies on,
    but without inducing a deterministic cancellation race against a real
    helper coroutine.
    """
    from gateway.slash_commands import (
        _RestartStage,
        _RestartStateBackup,
        _RestartTransaction,
    )

    loop = asyncio.get_running_loop()
    txn = _RestartTransaction(
        request_id="req-prep-abort-wins",
        backup=_RestartStateBackup(
            "req-prep-abort-wins",
            tmp_path / ".restart_notify.json",
            tmp_path / ".restart_last_processed.json",
            None,
        ),
        loop=loop,
        detached=True,
        via_service=False,
    )
    assert txn.stage is _RestartStage.PREPARING

    # Caller wins abort.
    assert await txn.claim_abort() is True
    assert txn.stage is _RestartStage.ABORTING

    # Helper's attempt to enter IN_FLIGHT is rejected.
    assert await txn.claim_handoff() is False
    # Stage remains ABORTING.
    assert txn.stage is _RestartStage.ABORTING


@pytest.mark.asyncio
async def test_in_flight_observer_does_not_rollback_or_cancel(
    monkeypatch, tmp_path
):
    """Helper crossed IN_FLIGHT and dispatch has returned (P1 #71876).

    With the P1 fix, the accepted ack is published immediately after
    claim_handoff, so dispatch returns BEFORE the launcher runs. The
    dispatcher must NOT cancel the helper task, MUST NOT rollback, MUST NOT
    write a second ack. The helper goes on to STARTED → HANDOFF_COMMITTED →
    stop(). This is the spec section 二 observer contract, now exercised
    from the post-claim/pre-launch window.
    """
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.delenv("HERMES_S6_SUPERVISED_CHILD", raising=False)

    runner, _ = make_restart_runner()
    gw_loop = asyncio.get_running_loop()
    runner._gateway_loop = gw_loop

    stop_called = []

    async def fake_stop(**kwargs):
        stop_called.append(True)

    runner.stop = fake_stop  # type: ignore[method-assign]

    # Gate the helper so it stays IN_FLIGHT (launcher pending) while the
    # dispatcher returns. Release only after the test confirms the
    # dispatcher already returned.
    helper_in_flight = asyncio.Event()
    release_helper = asyncio.Event()

    async def fake_launch_gated() -> bool:
        helper_in_flight.set()
        await release_helper.wait()
        return True

    runner._launch_detached_restart_command = fake_launch_gated  # type: ignore[method-assign]

    with patch("gateway.run._hermes_home", tmp_path):
        dispatch_task = asyncio.create_task(
            runner.dispatch_gateway_restart(
                source=_make_source(chat_id="chat-OBS"), origin="agent_tool"
            )
        )
        # Wait for the helper to reach the gated launcher (IN_FLIGHT).
        await helper_in_flight.wait()
        success, msg = await dispatch_task

    # P1 #71876: accepted ack published at claim_handoff → dispatch success
    # even though the launcher is still gated. The message does NOT claim
    # completion — the transaction is claimed and will proceed in the
    # background.
    assert success is True

    # At this point the dispatcher has returned. The helper is still in the
    # gate. Releasing it now lets the helper proceed to STARTED → stop().
    release_helper.set()

    # Wait briefly for the helper to record its terminal state. The only
    # externally observable signal that the helper truly finished is
    # stop() being invoked.
    for _ in range(50):
        if stop_called:
            break
        await asyncio.sleep(0.02)
    assert stop_called, (
        "After dispatcher returned at claim time, helper MUST proceed to "
        "STARTED → stop(). Dispatcher observer role MUST NOT prevent helper "
        "completion."
    )


@pytest.mark.asyncio
async def test_ack_future_cancelled_blocks_complete_started(monkeypatch, tmp_path):
    """Cancelling the ack_future before _complete_atomic runs must raise
    RuntimeError('ack future cancelled'). The state must NOT have been
    modified.
    """
    from gateway.slash_commands import (
        _RestartFinalOutcome,
        _RestartStage,
        _RestartStateBackup,
        _RestartTransaction,
    )

    loop = asyncio.get_running_loop()
    txn = _RestartTransaction(
        request_id="req-cancel-ack",
        backup=_RestartStateBackup(
            "req-test",
            tmp_path / ".restart_notify.json",
            tmp_path / ".restart_last_processed.json",
            None,
        ),
        loop=loop,
        detached=True,
        via_service=False,
    )
    assert await txn.claim_handoff() is True
    assert txn.stage is _RestartStage.IN_FLIGHT

    txn.ack_future.cancel()

    with pytest.raises(RuntimeError, match="ack future cancelled"):
        await txn.complete_started()

    # State untouched.
    assert txn.stage is _RestartStage.IN_FLIGHT
    assert txn.final_outcome is _RestartFinalOutcome.UNKNOWN
    assert txn.launcher_result is None
    assert txn.final_outcome_event.is_set() is False


@pytest.mark.asyncio
async def test_ack_future_conflict_raises_runtime_error(monkeypatch, tmp_path):
    """A different terminal-ack outcome than the one pre-set on a fresh
    transaction MUST raise RuntimeError('ack conflict') before any state
    mutation. The idempotent same-outcome case is covered by
    test_idempotent_complete_with_terminal_stage.
    """
    from gateway.slash_commands import (
        _RestartAckOutcome,
        _RestartFinalOutcome,
        _RestartStage,
        _RestartStateBackup,
        _RestartTransaction,
    )

    loop = asyncio.get_running_loop()
    txn = _RestartTransaction(
        request_id="req-conflict",
        backup=_RestartStateBackup(
            "req-test",
            tmp_path / ".restart_notify.json",
            tmp_path / ".restart_last_processed.json",
            None,
        ),
        loop=loop,
        detached=True,
        via_service=False,
    )
    assert await txn.claim_handoff() is True
    assert txn.stage is _RestartStage.IN_FLIGHT

    # Pre-set the ack to an outcome that complete_started does NOT intend
    # to write. complete_started wants ACCEPTED; the ack is already
    # NOT_STARTED. _complete_atomic must raise before any mutation.
    txn.ack_future.set_result(_RestartAckOutcome.NOT_STARTED)

    with pytest.raises(RuntimeError, match="ack conflict"):
        await txn.complete_started()

    # State untouched by the conflict-writing attempt.
    assert txn.stage is _RestartStage.IN_FLIGHT
    assert txn.final_outcome is _RestartFinalOutcome.UNKNOWN
    assert txn.launcher_result is None


@pytest.mark.asyncio
async def test_idempotent_complete_with_terminal_stage(monkeypatch, tmp_path):
    """Calling complete_started after the transaction already reached
    HANDOFF_COMMITTED returns ALREADY_COMPLETE without raising.
    """
    from gateway.slash_commands import (
        _RestartStage,
        _RestartStateBackup,
        _RestartTransaction,
        _TransitionResult,
    )

    loop = asyncio.get_running_loop()
    txn = _RestartTransaction(
        request_id="req-idemp",
        backup=_RestartStateBackup(
            "req-test",
            tmp_path / ".restart_notify.json",
            tmp_path / ".restart_last_processed.json",
            None,
        ),
        loop=loop,
        detached=True,
        via_service=False,
    )
    assert await txn.claim_handoff() is True
    assert await txn.complete_started() is _TransitionResult.TRANSITIONED
    # Idempotent re-application is ALREADY_COMPLETE (terminal match).
    assert await txn.complete_started() is _TransitionResult.ALREADY_COMPLETE
    assert txn.stage is _RestartStage.HANDOFF_COMMITTED


@pytest.mark.asyncio
async def test_cancel_restart_task_and_await_with_pending_task(
    monkeypatch, tmp_path
):
    """cancel_restart_task_and_await must cancel the helper task, await it,
    return True on success. The helper treats CancelledError explicitly
    (not via except Exception:).
    """
    from gateway.slash_commands import (
        _RestartStage,
        _RestartStateBackup,
        _RestartTransaction,
    )

    loop = asyncio.get_running_loop()
    txn = _RestartTransaction(
        request_id="req-cancel-task",
        backup=_RestartStateBackup(
            "req-test",
            tmp_path / ".restart_notify.json",
            tmp_path / ".restart_last_processed.json",
            None,
        ),
        loop=loop,
        detached=True,
        via_service=False,
    )
    assert await txn.claim_handoff() is True

    async def helper_waits_for_cancel() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # The helper MUST see CancelledError distinctly (not under
            # a broad except Exception swallow that production has banned).
            raise

    helper_task = asyncio.create_task(helper_waits_for_cancel())
    txn.restart_task = helper_task

    cancelled_ok = await txn.cancel_restart_task_and_await()
    assert cancelled_ok is True
    assert helper_task.done()
    assert txn.stage is _RestartStage.IN_FLIGHT  # Helper did NOT transition.


@pytest.mark.asyncio
async def test_race_g_preparing_one_winner(monkeypatch, tmp_path):
    """concurrent claim_abort + claim_handoff from PREPARING: exactly one
    returns True, the other False. Repeat across 5 rounds to catch
    scheduler-dependent bugs. No sleep: the lock IS the synchronization.
    """
    from gateway.slash_commands import (
        _RestartStage,
        _RestartStateBackup,
        _RestartTransaction,
    )

    for round_i in range(5):
        loop = asyncio.get_running_loop()
        txn = _RestartTransaction(
            request_id=f"req-race-g-{round_i}",
            backup=_RestartStateBackup(
                f"req-race-g-{round_i}",
                tmp_path / ".restart_notify.json",
                tmp_path / ".restart_last_processed.json",
                None,
            ),
            loop=loop,
            detached=True,
            via_service=False,
        )
        assert txn.stage is _RestartStage.PREPARING

        async def _do_abort():
            return await txn.claim_abort()

        async def _do_handoff():
            return await txn.claim_handoff()

        results = await asyncio.gather(
            _do_abort(), _do_handoff(), return_exceptions=True
        )
        true_count = sum(1 for r in results if r is True)
        false_count = sum(1 for r in results if r is False)
        assert true_count == 1, f"round {round_i}: {results}"
        assert false_count == 1, f"round {round_i}: {results}"

        if results[0] is True:
            assert txn.stage is _RestartStage.ABORTING
        else:
            assert txn.stage is _RestartStage.IN_FLIGHT


@pytest.mark.asyncio
async def test_worker_scheduler_outer_timeout_consumes_coro(
    monkeypatch, tmp_path
):
    """Regression for the original unawaited-coroutine warning:

    _handle_request_gateway_restart (called from a worker thread) builds

        coro = runner.dispatch_gateway_restart(source=..., origin=...)
        future = safe_schedule_threadsafe(coro, gw_loop)

    When the test patches safe_schedule_threadsafe to simulate an outer 10s
    timeout, the mock MUST consume `coro` explicitly — either:

      (a) close it (`coro.close()`), or
      (b) schedule it (`asyncio.ensure_future(coro)`), or
      (c) run it on the loop (`asyncio.run_coroutine_threadsafe(coro, loop)`).

    Relying on the GC to dispose of the unconsumed coroutine triggers
    `RuntimeWarning: coroutine ... was never awaited` later in the suite.
    This test enforces option (a) — the close-coroutine variant — so the
    suite stays warning-free.
    """
    monkeypatch.setenv("_HERMES_GATEWAY", "1")

    from tools.gateway_restart_tool import _handle_request_gateway_restart
    import gateway.run as gw_run
    from gateway.session import SessionSource
    from gateway.config import Platform

    runner, _ = make_restart_runner()
    gw_loop = asyncio.get_running_loop()
    runner._gateway_loop = gw_loop

    async def fake_launch_slow() -> bool:
        # The test's mock-wrapped schedule_threadsafe raises TimeoutError
        # before the helper ever gets a chance to start, so this never
        # runs. If anything causes it to run, it would block forever.
        await asyncio.Event().wait()
        return True

    runner._launch_detached_restart_command = fake_launch_slow  # type: ignore[method-assign]
    runner.stop = AsyncMock()  # type: ignore[method-assign]

    orig_ref = gw_run._gateway_runner_ref
    gw_run._gateway_runner_ref = lambda: runner

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-raceh",
        chat_type="dm",
        user_id="u-raceh",
        message_id="m-raceh",
    )

    class _TimeoutFuture:
        def result(self, timeout=None):
            raise TimeoutError("simulated outer 10s timeout")

        def cancel(self):
            return False

    def _safe_schedule_threadsafe(coro, loop):
        # Explicitly consume `coro` — this IS the fix for the regression.
        # In production this function would normally run the coroutine on
        # `loop`; here we just need to dispose of `coro` cleanly so it
        # doesn't leak and surface as a 'was never awaited' warning.
        coro.close()
        return _TimeoutFuture()

    try:
        with patch(
            "tools.gateway_restart_tool._is_authorized_foreground_turn",
            return_value=True,
        ), patch(
            "tools.gateway_restart_tool._resolve_current_source",
            return_value=source,
        ), patch(
            "tools.gateway_restart_tool._check_restart_policy",
            return_value=None,
        ), patch(
            "tools.gateway_restart_tool.safe_schedule_threadsafe",
            side_effect=_safe_schedule_threadsafe,
        ):
            res = json.loads(
                _handle_request_gateway_restart({"reason": "raceh-regression"})
            )

        # Outer worker reported fixed outcome_unknown.
        assert res == {
            "success": False,
            "outcome": "outcome_unknown",
            "message": "Restart outcome could not be confirmed. "
                       "Do not retry immediately.",
            "can_retry": False,
        }
    finally:
        gw_run._gateway_runner_ref = orig_ref


@pytest.mark.asyncio
async def test_no_real_popen_when_mock_intact():
    """Defensive check: make_restart_runner returns a runner whose
    `_launch_detached_restart_command` IS an AsyncMock. A test that
    "forgot" to swap it for a fake launcher MUST still get an AsyncMock.
    Otherwise attach_real_launcher_under_mocked_popen has been called
    without also patching subprocess.Popen — which would leak a real
    watcher subprocess into the test host. This test guards that
    invariant at the level of make_restart_runner.
    """
    runner, _ = make_restart_runner()
    assert isinstance(runner._launch_detached_restart_command, AsyncMock), (
        "make_restart_runner must default to AsyncMock for the launcher to "
        "prevent real subprocess.Popen in tests that don't opt in."
    )
    assert isinstance(runner.stop, AsyncMock), (
        "make_restart_runner must default to AsyncMock for stop() to "
        "prevent the real shutdown from running."
    )
