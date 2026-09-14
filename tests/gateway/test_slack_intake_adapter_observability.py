"""Stage B RED contracts for Slack Socket Mode intake observability.

The future bridge must observe the SDK envelope before Bolt dispatch, observe an
acknowledgement only after the SDK send succeeds, and keep listener decisions
explicit without blocking the event loop or leaking raw platform identifiers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.platforms.slack.intake_observability import (
    IntakeContext,
    SlackIntakeObserver,
    install_socket_observer,
)
from plugins.platforms.slack import intake_observability as _observability


@pytest.mark.windows_only
def test_windows_adapter_import_does_not_load_durable_ledger():
    code = """
import sys
from plugins.platforms.slack import adapter
assert not adapter._slack_intake_durability_supported()
assert 'gateway.slack_intake_ledger' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.fspath(Path(__file__).resolve().parents[2]),
        check=True,
    )


@pytest.mark.asyncio
async def test_persistence_failure_preserves_successful_listener(monkeypatch, tmp_path):
    ledger = _observability._ledger
    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    ledger._reset_initialization_for_tests()
    ledger._reset_persistence_health_for_tests()
    observer = SlackIntakeObserver()
    context = await observer.envelope_received(
        workspace_id="T-q1", envelope_id="X-q1", event_id="E-q1",
        event_type="message", channel_id="C-q1", thread_id=None, message_id="1.0",
    )
    assert context.persisted
    assert context.receipt_id is not None
    calls = []

    async def listener():
        calls.append("dispatched")

    async def reject_persistence(*_args, **_kwargs):
        raise RuntimeError("synthetic persistence outage")

    with monkeypatch.context() as fault:
        fault.setattr(ledger, "_run_ledger_work", reject_persistence)
        terminal = await observer.run_listener(context, listener)
        assert calls == ["dispatched"]
        assert terminal.state == "accepted" and not terminal.persisted
        assert ledger.persistence_degraded()
        assert {entry["stage"] for entry in ledger.read_fallback_receipts()} == {
            "listener_entered", "accepted",
        }
        assert not observer._active_listeners
        assert ledger.read_receipts()[0]["terminal_state"] is None
    assert (await ledger.mark_accepted_safely(context.receipt_id, decided_at=123)).persisted


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("listener_result", "terminal_stage", "terminal_reason"),
    [(None, "accepted", None), ("ignored_channel", "dropped", "ignored_channel")],
)
async def test_admission_busy_envelope_retains_correlated_fallback_lifecycle(
    monkeypatch, tmp_path, listener_result, terminal_stage, terminal_reason
):
    ledger = _observability._ledger
    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    ledger._reset_initialization_for_tests()
    ledger._reset_persistence_health_for_tests()
    ledger.hash_identifier("warmup", "correlation-key")
    observer = SlackIntakeObserver()
    await observer._stage_admission_lock.acquire()
    try:
        context = await observer.envelope_received(
            workspace_id="T-busy",
            envelope_id="X-busy",
            event_id=f"E-{terminal_stage}",
            event_type="message",
            channel_id="C-busy",
            thread_id=None,
            message_id=f"1-{terminal_stage}",
        )
    finally:
        observer._stage_admission_lock.release()

    result = await observer.run_listener(
        context, AsyncMock(return_value=listener_result)
    )
    receipt_id = ledger._receipt_id("T-busy", f"E-{terminal_stage}")
    fallback = ledger.read_fallback_receipts()

    assert context.receipt_id == receipt_id
    assert result.state == terminal_stage
    assert [entry["stage"] for entry in fallback] == [
        "envelope_received",
        "listener_entered",
        terminal_stage,
    ]
    assert all(entry["receipt_id"] == receipt_id for entry in fallback)
    assert fallback[-1]["reason"] == terminal_reason


@pytest.mark.asyncio
async def test_failed_envelope_persistence_retains_correlated_fallback_lifecycle(
    monkeypatch, tmp_path
):
    ledger = _observability._ledger
    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    ledger._reset_initialization_for_tests()
    ledger._reset_persistence_health_for_tests()
    ledger.hash_identifier("warmup", "correlation-key")
    observer = SlackIntakeObserver()

    async def reject_persistence(*_args, **_kwargs):
        raise RuntimeError("synthetic persistence outage")

    monkeypatch.setattr(ledger, "_run_ledger_work", reject_persistence)
    context = await observer.envelope_received(
        workspace_id="T-failed",
        envelope_id="X-failed",
        event_id="E-failed",
        event_type="message",
        channel_id="C-failed",
        thread_id=None,
        message_id="1-failed",
    )
    terminal = await observer.run_listener(context, AsyncMock(return_value=None))
    receipt_id = ledger._receipt_id("T-failed", "E-failed")
    fallback = ledger.read_fallback_receipts()

    assert context.receipt_id == receipt_id
    assert terminal.state == "accepted"
    assert [entry["stage"] for entry in fallback] == [
        "envelope_received",
        "listener_entered",
        "accepted",
    ]
    assert all(entry["receipt_id"] == receipt_id for entry in fallback)


@pytest.mark.asyncio
async def test_identity_conflict_cannot_terminalize_the_original_receipt(monkeypatch, tmp_path):
    ledger = _observability._ledger
    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    ledger._reset_initialization_for_tests()
    ledger._reset_persistence_health_for_tests()
    observer = SlackIntakeObserver()
    original = await observer.envelope_received(
        workspace_id="T", envelope_id="X-one", event_id="E",
        event_type="message", channel_id="C-one", thread_id=None, message_id="1",
    )

    conflicting = await observer.envelope_received(
        workspace_id="T", envelope_id="X-two", event_id="E",
        event_type="message", channel_id="C-two", thread_id=None, message_id="2",
    )
    terminal = await observer.run_listener(conflicting, AsyncMock(return_value=None))

    assert original.receipt_id is not None
    assert conflicting.persisted is False
    assert conflicting.receipt_id is None
    assert terminal.persisted is False
    assert terminal.failure_reason == "unknown_receipt"
    assert ledger.read_receipts(limit=1)[0]["terminal_state"] is None


@pytest.mark.asyncio
async def test_envelope_commit_and_correlation_share_one_executor_admission(monkeypatch, tmp_path):
    ledger = _observability._ledger
    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    ledger._reset_initialization_for_tests()
    ledger._reset_persistence_health_for_tests()
    original_run = ledger._run_ledger_work
    admissions = 0

    async def reject_second_admission(*args, **kwargs):
        nonlocal admissions
        admissions += 1
        if admissions == 2:
            raise TimeoutError("synthetic post-commit correlation rejection")
        return await original_run(*args, **kwargs)

    monkeypatch.setattr(ledger, "_run_ledger_work", reject_second_admission)
    observer = SlackIntakeObserver()
    context = await observer.envelope_received(
        workspace_id="T", envelope_id="X", event_id="E",
        event_type="message", channel_id="C", thread_id=None, message_id="1",
    )

    assert admissions == 1
    assert context.persisted is True
    assert context.receipt_id is not None
    assert observer.context_for_envelope("X") is context
    assert observer.context_for_event("E", workspace_id="T") is context


@pytest.mark.asyncio
async def test_repeated_real_malformed_key_has_bounded_critical_logs(monkeypatch, tmp_path, caplog):
    ledger = _observability._ledger
    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    ledger._reset_initialization_for_tests()
    ledger._reset_persistence_health_for_tests()
    (tmp_path / "correlation.key").write_bytes(b"bad-key")
    observer = SlackIntakeObserver()
    message = {"type": "events_api", "envelope_id": "envelope", "payload": {
        "team_id": "T", "event_id": "event", "event": {"type": "message", "channel": "C", "ts": "1"}}}
    for _ in range(12):
        await observer.observe_socket_message(message, "{}")
    critical = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert len(critical) <= 2, [r.getMessage() for r in critical]


@pytest.mark.asyncio
async def test_repeated_ack_stage_failure_has_bounded_critical_logs(caplog):
    ledger = _observability._ledger
    ledger._reset_persistence_health_for_tests()
    observer = SlackIntakeObserver(store=_Store())
    await observer.envelope_received(
        workspace_id="T", envelope_id="X", event_id="E", event_type="message",
        channel_id="C", thread_id=None, message_id="1",
    )
    observer.acknowledged = AsyncMock(side_effect=RuntimeError("PRIVATE-ACK-FAILURE"))
    for _ in range(12):
        await observer.acknowledged_for_envelope("X")
    critical = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert 1 <= len(critical) <= 2
    assert "PRIVATE-ACK-FAILURE" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("outage", ["malformed_key", "saturation"])
async def test_observer_outage_recovers_without_losing_bounded_fallback(monkeypatch, tmp_path, caplog, outage):
    import threading

    ledger = _observability._ledger
    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    monkeypatch.setattr(ledger, "_MAX_FALLBACK_RECEIPTS", 3)
    ledger._reset_initialization_for_tests()
    ledger._reset_persistence_health_for_tests()
    observer = SlackIntakeObserver(max_contexts=2)
    message = {"type": "events_api", "envelope_id": "PRIVATE-ENVELOPE", "payload": {
        "team_id": "PRIVATE-TEAM", "event_id": "PRIVATE-EVENT", "event": {
            "type": "message", "channel": "PRIVATE-CHANNEL", "ts": "1", "text": "PRIVATE-BODY"}}}

    with monkeypatch.context() as unavailable:
        if outage == "malformed_key":
            (tmp_path / "correlation.key").write_bytes(b"bad-key")
        else:
            unavailable.setattr(ledger, "_LEDGER_PERMITS", threading.BoundedSemaphore(0))
        for _ in range(12):
            await observer.observe_socket_message(message, "PRIVATE-BODY")
        assert ledger.persistence_degraded()
        assert len(ledger.read_fallback_receipts()) == 3
        assert ledger.fallback_eviction_count() == 9
        assert not observer._envelope_contexts and not observer._event_contexts
        critical = [r for r in caplog.records if r.levelno == logging.CRITICAL]
        assert 1 <= len(critical) <= 2
        serialized = caplog.text + repr(ledger.read_fallback_receipts())
        for raw in ("PRIVATE-ENVELOPE", "PRIVATE-TEAM", "PRIVATE-EVENT", "PRIVATE-CHANNEL", "PRIVATE-BODY", str(tmp_path)):
            assert raw not in serialized
        if outage == "malformed_key":
            (tmp_path / "correlation.key").unlink()

    await observer.observe_socket_message(message, "PRIVATE-BODY")
    context = observer.context_for_event("PRIVATE-EVENT", workspace_id="PRIVATE-TEAM")
    assert context is not None and context.persisted
    assert not ledger.persistence_degraded()
    assert len(ledger.read_fallback_receipts()) == 3
    assert len(ledger.read_receipts()) == 1


@pytest.mark.asyncio
async def test_repair_ack_dict_is_correlated_and_success_survives_instrumentation_failure():
    observer = SlackIntakeObserver(store=_Store())
    observer.acknowledged_for_envelope = AsyncMock()
    client = SimpleNamespace(message_listeners=[], send_socket_mode_response=AsyncMock(return_value="sent"))
    install_socket_observer(client, observer)
    assert await client.send_socket_mode_response({"envelope_id": "env-private"}) == "sent"
    observer.acknowledged_for_envelope.assert_awaited_once_with("env-private")
    observer.acknowledged_for_envelope.side_effect = RuntimeError("private payload")
    assert await client.send_socket_mode_response({"envelope_id": "env-private"}) == "sent"


@pytest.mark.asyncio
async def test_socket_observer_ignores_valid_unsupported_event_without_degradation():
    ledger = _observability._ledger
    ledger._reset_persistence_health_for_tests()
    observer = SlackIntakeObserver(store=_Store())
    observer.envelope_received = AsyncMock()
    observer.acknowledged_for_envelope = AsyncMock()
    bolt_listener = AsyncMock()
    original_send = AsyncMock(return_value="sent")
    client = SimpleNamespace(
        message_listeners=[bolt_listener],
        send_socket_mode_response=original_send,
    )
    install_socket_observer(client, observer)
    message = {
        "type": "events_api",
        "envelope_id": "envelope",
        "payload": {
            "team_id": "T",
            "event_id": "event",
            "event": {
                "type": "member_joined_channel",
                "channel": "C",
                "event_ts": "1",
            },
        },
    }

    for listener in tuple(client.message_listeners):
        await listener(client, message, "{}")

    observer.envelope_received.assert_not_awaited()
    bolt_listener.assert_awaited_once_with(client, message, "{}")
    assert not ledger.persistence_degraded()
    assert await client.send_socket_mode_response({"envelope_id": "envelope"}) == "sent"
    original_send.assert_awaited_once()
    observer.acknowledged_for_envelope.assert_awaited_once_with("envelope")


@pytest.mark.asyncio
async def test_repair_terminalization_cancellation_preserves_original_listener_error():
    observer = SlackIntakeObserver(store=_Store())
    context = SimpleNamespace(receipt_id="a" * 64)
    original = ValueError("original listener failure")
    observer.dropped = AsyncMock(side_effect=asyncio.CancelledError())

    async def listener():
        raise original

    with pytest.raises(ValueError) as caught:
        await observer.run_listener(context, listener)
    assert caught.value is original


@pytest.mark.asyncio
async def test_cancellation_during_listener_entry_is_terminally_classified():
    observer = SlackIntakeObserver(store=_Store())
    original = asyncio.CancelledError("synthetic entry cancellation")
    observer.listener_entered = AsyncMock(side_effect=original)
    observer.dropped = AsyncMock()
    listener = AsyncMock()
    context = IntakeContext(
        receipt_id="a" * 64, envelope_hash="b" * 64, event_hash="c" * 64,
        message_key_hash="d" * 64, persisted=True, degraded=False,
        duplicate=False, retry_attempt=0,
    )

    with pytest.raises(asyncio.CancelledError) as caught:
        await observer.run_listener(context, listener)

    assert caught.value is original
    listener.assert_not_awaited()
    observer.dropped.assert_awaited_once_with(context, reason="listener_cancelled")


@pytest.mark.asyncio
async def test_repair_stage_failure_does_not_prevent_listener_execution():
    observer = SlackIntakeObserver(store=_Store())
    observer.listener_entered = AsyncMock(side_effect=OSError("private path"))
    listener = AsyncMock(return_value=None)
    await observer.run_listener(SimpleNamespace(receipt_id="a" * 64), listener)
    listener.assert_awaited_once()


@pytest.mark.asyncio
async def test_repair_envelope_correlation_key_work_is_off_loop(monkeypatch, tmp_path):
    import threading
    ledger = _observability._ledger
    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    ledger._reset_initialization_for_tests()
    ledger._reset_persistence_health_for_tests()
    observer = SlackIntakeObserver()
    main_thread = threading.get_ident()
    original = ledger._correlation_key
    threads = []

    def key():
        threads.append(threading.get_ident())
        return original()

    monkeypatch.setattr(ledger, "_correlation_key", key)
    await observer.envelope_received(
        workspace_id="T", envelope_id="X", event_id="E", event_type="message",
        channel_id="C", thread_id=None, message_id="1",
    )
    assert threads and main_thread not in threads


@pytest.mark.asyncio
async def test_repair_same_event_ids_remain_workspace_scoped(monkeypatch):
    observer = SlackIntakeObserver(store=_Store())
    contexts = []
    for team in ("T-one", "T-two"):
        contexts.append(await observer.envelope_received(
            workspace_id=team, envelope_id=team, event_id="E-shared", event_type="message",
            channel_id="C", thread_id=None, message_id="1",
        ))
    assert contexts[0].event_hash != contexts[1].event_hash
    assert observer.context_for_event("E-shared", workspace_id="T-one") is contexts[0]
    assert observer.context_for_event("E-shared", workspace_id="T-two") is contexts[1]
    assert observer.context_for_event("E-shared") is None


def test_same_envelope_ids_have_workspace_scoped_transport_hashes(monkeypatch, tmp_path):
    ledger = _observability._ledger
    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    ledger._reset_initialization_for_tests()
    ledger._reset_persistence_health_for_tests()

    observations = [
        ledger.record_listener_received(
            workspace_id=team,
            event_id=f"E-{team}",
            transport_id="X-shared",
            event_type="message",
            channel_id="C",
            thread_id=None,
            message_id="1",
            received_at=1.0,
            stage="envelope_received",
        )
        for team in ("T-one", "T-two")
    ]

    assert observations[0].transport_hash != observations[1].transport_hash


@pytest.mark.asyncio
async def test_same_envelope_id_across_workspaces_never_attributes_ack(monkeypatch, tmp_path):
    ledger = _observability._ledger
    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    ledger._reset_initialization_for_tests()
    ledger._reset_persistence_health_for_tests()
    observer = SlackIntakeObserver()
    for team in ("T-one", "T-two"):
        await observer.envelope_received(
            workspace_id=team,
            envelope_id="X-shared",
            event_id=f"E-{team}",
            event_type="message",
            channel_id="C",
            thread_id=None,
            message_id="1",
        )
    observer.acknowledged = AsyncMock()

    await observer.acknowledged_for_envelope("X-shared")

    observer.acknowledged.assert_not_awaited()


@pytest.mark.asyncio
async def test_same_receipt_envelope_retry_remains_correlatable_for_ack(monkeypatch, tmp_path):
    ledger = _observability._ledger
    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    ledger._reset_initialization_for_tests()
    ledger._reset_persistence_health_for_tests()
    observer = SlackIntakeObserver()
    contexts = [
        await observer.envelope_received(
            workspace_id="T-one",
            envelope_id="X-retry",
            event_id="E-retry",
            event_type="message",
            channel_id="C",
            thread_id=None,
            message_id="1",
            retry_attempt=attempt,
        )
        for attempt in (0, 1)
    ]
    observer.acknowledged = AsyncMock()

    await observer.acknowledged_for_envelope("X-retry")

    assert contexts[0].receipt_id == contexts[1].receipt_id
    observer.acknowledged.assert_awaited_once_with(contexts[1])


@pytest.mark.asyncio
async def test_repair_context_lookup_never_accesses_key_filesystem(monkeypatch):
    observer = SlackIntakeObserver(store=_Store())
    context = await observer.envelope_received(
        workspace_id="T", envelope_id="X", event_id="E", event_type="message",
        channel_id="C", thread_id=None, message_id="1",
    )
    monkeypatch.setattr(_observability._ledger, "_correlation_key", lambda: pytest.fail("lookup touched filesystem"))
    assert observer.context_for_envelope("X") is context
    assert observer.context_for_event("E", workspace_id="T") is context


@pytest.mark.asyncio
async def test_repair_synthetic_terminal_explicitly_disclaims_durable_success(monkeypatch):
    store = _observability.LedgerIntakeStore()
    terminal = await store.mark_accepted(None, decided_at=1)
    assert terminal.persisted is False
    assert terminal.failure_reason == "unknown_receipt"
    monkeypatch.setattr(_observability._ledger, "mark_accepted_safely", AsyncMock(return_value=SimpleNamespace(observation=None, persisted=False, failure_reason="work_deadline_exceeded")))
    terminal = await store.mark_accepted("a" * 64, decided_at=1)
    assert terminal.persisted is False
    assert terminal.failure_reason == "work_deadline_exceeded"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "error", "cancel"])
async def test_concurrent_same_receipt_has_one_lifecycle_owner(failure):
    store = _Store()
    observer = SlackIntakeObserver(store=store)
    context = IntakeContext(
        receipt_id="a" * 64, envelope_hash="b" * 64, event_hash="c" * 64,
        message_key_hash="d" * 64, persisted=True, degraded=False,
        duplicate=False, retry_attempt=0,
    )
    entered, release, retry_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = (asyncio.CancelledError("synthetic owner cancellation") if failure == "cancel"
                else ValueError("synthetic owner error"))

    async def handler():
        entered.set()
        await release.wait()
        if failure:
            raise original

    listener = AsyncMock(side_effect=handler)

    async def invoke(retry=False):
        if retry:
            retry_started.set()
        try:
            return await observer.run_listener(context, listener)
        except BaseException as exc:
            return exc

    owner = asyncio.create_task(invoke())
    retry = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        retry = asyncio.create_task(invoke(True))
        await asyncio.wait_for(retry_started.wait(), 2)
        release.set()
        first_result, retry_result = await asyncio.wait_for(asyncio.gather(owner, retry), 2)
        listener.assert_awaited_once()
        assert first_result is retry_result
        if failure:
            assert first_result is original
        assert len([call for call in store.calls if call[0] in {"accepted", "dropped"}]) == 1
        # No retained in-flight result can suppress a later independent call.
        await invoke()
        assert listener.await_count == 2
    finally:
        release.set()
        for task in (owner, retry):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(t for t in (owner, retry) if t is not None), return_exceptions=True)


class _Store:
    def __init__(self, *, persisted: bool = True):
        self.persisted = persisted
        self.calls: list[tuple[str, dict]] = []

    async def record_envelope(self, **metadata):
        self.calls.append(("envelope_received", metadata))
        return SimpleNamespace(
            persisted=self.persisted,
            observation=(
                SimpleNamespace(
                    receipt_id=_observability._ledger._receipt_id(
                        metadata["workspace_id"], metadata["event_id"]
                    ),
                    duplicate=False,
                    transport_hash=_observability._ledger._transport_key(
                        metadata["workspace_id"], metadata["transport_id"]
                    ),
                    transport_lookup_hash=_observability._ledger.hash_identifier(
                        "transport", metadata["transport_id"]
                    ),
                    message_key_hash=_observability._ledger._message_key(
                        metadata["workspace_id"],
                        metadata["channel_id"],
                        metadata["message_id"],
                    ),
                )
                if self.persisted
                else None
            ),
            failure_reason=None if self.persisted else "persistence_failed",
        )

    async def append_stage(self, receipt_id, *, stage, observed_at, reason=None):
        self.calls.append(
            (
                stage,
                {
                    "receipt_id": receipt_id,
                    "observed_at": observed_at,
                    **({"reason": reason} if reason is not None else {}),
                },
            )
        )

    def record_unavailable_stage(
        self, receipt_id, *, stage, observed_at, reason=None
    ):
        _observability._ledger.record_unavailable_stage(
            receipt_id, stage=stage, observed_at=observed_at, reason=reason
        )

    async def mark_accepted(self, receipt_id, *, decided_at):
        self.calls.append(
            ("accepted", {"receipt_id": receipt_id, "decided_at": decided_at})
        )
        return SimpleNamespace(
            state="accepted", reason=None, related_receipt_id=None
        )

    async def mark_dropped(self, receipt_id, *, reason, decided_at):
        self.calls.append(
            (
                "dropped",
                {
                    "receipt_id": receipt_id,
                    "reason": reason,
                    "decided_at": decided_at,
                },
            )
        )
        return SimpleNamespace(
            state="dropped", reason=reason, related_receipt_id=None
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", [[], {}])
async def test_drop_reason_validation_is_total_for_unhashable_values(reason):
    store = _Store()
    observer = SlackIntakeObserver(store=store)
    context = IntakeContext(
        receipt_id="a" * 64,
        envelope_hash="b" * 64,
        event_hash="c" * 64,
        message_key_hash="d" * 64,
        persisted=True,
        degraded=False,
        duplicate=False,
        retry_attempt=0,
    )

    with pytest.raises(ValueError, match="fixed safe vocabulary"):
        await observer.dropped(context, reason=reason)

    assert store.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [[], {}, "unknown_reason"])
async def test_invalid_listener_result_terminalizes_before_reraising(result):
    store = _Store()
    observer = SlackIntakeObserver(store=store)
    context = IntakeContext(
        receipt_id="a" * 64,
        envelope_hash="b" * 64,
        event_hash="c" * 64,
        message_key_hash="d" * 64,
        persisted=True,
        degraded=False,
        duplicate=False,
        retry_attempt=0,
    )
    listener = AsyncMock(return_value=result)

    with pytest.raises(TypeError, match="listener result"):
        await observer.run_listener(context, listener)

    listener.assert_awaited_once()
    terminal_calls = [call for call in store.calls if call[0] in {"accepted", "dropped"}]
    assert terminal_calls == [
        (
            "dropped",
            {
                "receipt_id": "a" * 64,
                "reason": "exception_type_error",
                "decided_at": terminal_calls[0][1]["decided_at"],
            },
        )
    ]
    assert observer._active_listeners == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("evicted_map", ["event", "envelope", "both"])
@pytest.mark.parametrize("cancel_conflict", [False, True])
@pytest.mark.parametrize(
    ("listener_result", "terminal_stage"),
    [(None, "accepted"), ("ignored_channel", "dropped")],
)
async def test_identity_conflict_revokes_captured_stage_capabilities(
    evicted_map, cancel_conflict, listener_result, terminal_stage
):
    class _PausedConflictStore(_Store):
        def __init__(self):
            super().__init__()
            self.conflict_written = asyncio.Event()
            self.release_conflict = asyncio.Event()

        async def record_envelope(self, **metadata):
            if not self.calls:
                return await super().record_envelope(**metadata)
            self.calls.append(("envelope_received", metadata))
            self.conflict_written.set()
            await asyncio.shield(self.release_conflict.wait())
            return SimpleNamespace(
                persisted=False,
                observation=None,
                failure_reason="identity_conflict",
            )

    store = _PausedConflictStore()
    observer = SlackIntakeObserver(store=store)
    original = await observer.envelope_received(
        workspace_id="T",
        envelope_id="X-original",
        event_id="E",
        event_type="message",
        channel_id="C-original",
        thread_id=None,
        message_id="1",
    )
    if evicted_map in {"event", "both"}:
        observer._event_contexts.clear()
    if evicted_map in {"envelope", "both"}:
        observer._envelope_contexts.clear()

    conflict_task = asyncio.create_task(
        observer.envelope_received(
            workspace_id="T",
            envelope_id="X-conflict",
            event_id="E",
            event_type="message",
            channel_id="C-conflict",
            thread_id=None,
            message_id="2",
        )
    )
    await asyncio.wait_for(store.conflict_written.wait(), 2)
    if cancel_conflict:
        conflict_task.cancel()
        await asyncio.sleep(0)

    if evicted_map == "event":
        ack_task = asyncio.create_task(
            observer.acknowledged_for_envelope("X-original")
        )
    else:
        # Model an acknowledgement that resolved the context immediately before
        # envelope-map eviction and retained the returned capability.
        ack_task = asyncio.create_task(observer.acknowledged(original))
    listener = AsyncMock(return_value=listener_result)
    listener_task = asyncio.create_task(observer.run_listener(original, listener))
    await asyncio.sleep(0)
    store.release_conflict.set()
    conflict_result, _ack, terminal = await asyncio.wait_for(
        asyncio.gather(
            conflict_task, ack_task, listener_task, return_exceptions=True
        ),
        2,
    )

    if cancel_conflict:
        assert isinstance(conflict_result, asyncio.CancelledError)
    else:
        assert not isinstance(conflict_result, BaseException)
    assert not isinstance(terminal, BaseException)
    listener.assert_awaited_once()
    assert [
        stage
        for stage, _metadata in store.calls
        if stage not in {"envelope_received"}
    ] == []
    assert terminal.state == terminal_stage
    assert terminal.persisted is False
    assert terminal.failure_reason == "identity_conflict"


@pytest.mark.asyncio
@pytest.mark.parametrize("evicted_map", ["event", "envelope"])
@pytest.mark.parametrize(
    ("listener_result", "terminal_stage"),
    [(None, "accepted"), ("ignored_channel", "dropped")],
)
async def test_timed_out_envelope_observation_revokes_captured_stage_capabilities(
    evicted_map,
    listener_result,
    terminal_stage,
):
    class DeadlineConflictStore(_Store):
        def __init__(self):
            super().__init__()
            self.release_late_work = asyncio.Event()
            self.late_work_done = asyncio.Event()
            self.late_owner = None

        async def record_envelope(self, **metadata):
            if not self.calls:
                return await super().record_envelope(**metadata)
            self.calls.append(("envelope_received", metadata))

            async def finish_late_conflict():
                await self.release_late_work.wait()
                self.late_work_done.set()

            self.late_owner = asyncio.create_task(finish_late_conflict())
            return SimpleNamespace(
                persisted=False,
                observation=None,
                failure_reason="work_deadline_exceeded",
            )

    store = DeadlineConflictStore()
    observer = SlackIntakeObserver(store=store, clock=lambda: 710.0, max_contexts=2)
    original = await observer.envelope_received(
        workspace_id="T-one",
        envelope_id="X-one",
        event_id="Ev-same",
        event_type="message",
        channel_id="C-one",
        thread_id=None,
        message_id="1710000000.000100",
    )
    if evicted_map == "event":
        observer._event_contexts.clear()
    else:
        observer._envelope_contexts.clear()

    timed_out = await observer.envelope_received(
        workspace_id="T-one",
        envelope_id="X-conflict",
        event_id="Ev-same",
        event_type="message",
        channel_id="C-other",
        thread_id=None,
        message_id="1710000000.000100",
    )
    assert timed_out.persisted is False
    assert timed_out.receipt_id is None

    listener = AsyncMock(return_value=listener_result)
    acknowledgement_result, terminal = await asyncio.gather(
        observer.acknowledged(original),
        observer.run_listener(original, listener),
    )
    store.release_late_work.set()
    await store.late_work_done.wait()
    assert store.late_owner is not None
    await store.late_owner

    listener.assert_awaited_once()
    assert [
        stage
        for stage, _metadata in store.calls
        if stage not in {"envelope_received"}
    ] == []
    assert acknowledgement_result.persisted is False
    assert acknowledgement_result.failure_reason == "identity_conflict"
    assert terminal.state == terminal_stage
    assert terminal.persisted is False
    assert terminal.failure_reason == "identity_conflict"


@pytest.mark.asyncio
@pytest.mark.parametrize("evicted_map", ["event", "envelope"])
@pytest.mark.parametrize(
    ("listener_result", "terminal_stage"),
    [(None, "accepted"), ("ignored_channel", "dropped")],
)
async def test_real_late_identity_conflict_cannot_race_captured_stage_mutation(
    monkeypatch,
    tmp_path,
    evicted_map,
    listener_result,
    terminal_stage,
):
    import threading

    ledger = _observability._ledger
    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    ledger._reset_initialization_for_tests()
    ledger._reset_persistence_health_for_tests()
    observer = SlackIntakeObserver(clock=lambda: 720.0, max_contexts=2)
    original = await observer.envelope_received(
        workspace_id="T-one",
        envelope_id="X-one",
        event_id="Ev-same",
        event_type="message",
        channel_id="C-one",
        thread_id=None,
        message_id="1710000000.000100",
    )
    if evicted_map == "event":
        observer._event_contexts.clear()
    else:
        observer._envelope_contexts.clear()

    late_work_started = threading.Event()
    release_late_work = threading.Event()
    late_work_done = threading.Event()

    def delayed_identity_conflict(**_metadata):
        late_work_started.set()
        try:
            assert release_late_work.wait(1)
            raise ledger.InvalidIntakeTransition("late identity conflict")
        finally:
            late_work_done.set()

    monkeypatch.setattr(ledger, "record_listener_received", delayed_identity_conflict)
    monkeypatch.setattr(ledger, "_LEDGER_WORK_TIMEOUT", 0.05)
    timed_out = await observer.envelope_received(
        workspace_id="T-one",
        envelope_id="X-conflict",
        event_id="Ev-same",
        event_type="message",
        channel_id="C-other",
        thread_id=None,
        message_id="1710000000.000100",
    )
    assert late_work_started.is_set()
    assert timed_out.persisted is False
    assert timed_out.receipt_id is None

    listener = AsyncMock(return_value=listener_result)
    acknowledgement = asyncio.create_task(observer.acknowledged(original))
    listener_run = asyncio.create_task(observer.run_listener(original, listener))
    await asyncio.sleep(0)
    release_late_work.set()
    acknowledgement_result, terminal = await asyncio.gather(
        acknowledgement,
        listener_run,
    )
    assert await asyncio.wait_for(asyncio.to_thread(late_work_done.wait), 1)

    listener.assert_awaited_once()
    rows = ledger.read_receipts(limit=1)
    assert len(rows) == 1
    assert [event["stage"] for event in rows[0]["events"]] == [
        "envelope_received"
    ]
    assert rows[0]["terminal_state"] is None
    assert acknowledgement_result.persisted is False
    assert acknowledgement_result.failure_reason == "identity_conflict"
    assert terminal.state == terminal_stage
    assert terminal.persisted is False
    assert terminal.failure_reason == "identity_conflict"


@pytest.mark.asyncio
async def test_revocation_capacity_never_reactivates_captured_capabilities():
    class ConflictStore(_Store):
        async def record_envelope(self, **metadata):
            if metadata["channel_id"] == "C-conflict":
                self.calls.append(("envelope_received", metadata))
                return SimpleNamespace(
                    persisted=False,
                    observation=None,
                    failure_reason="identity_conflict",
                )
            return await super().record_envelope(**metadata)

    store = ConflictStore()
    observer = SlackIntakeObserver(store=store, clock=lambda: 730.0, max_contexts=2)
    captured = []
    for index in range(3):
        event_id = f"Ev-{index}"
        captured.append(
            await observer.envelope_received(
                workspace_id="T-one",
                envelope_id=f"X-{index}",
                event_id=event_id,
                event_type="message",
                channel_id="C-original",
                thread_id=None,
                message_id=f"1710000000.00010{index}",
            )
        )
        conflict = await observer.envelope_received(
            workspace_id="T-one",
            envelope_id=f"X-conflict-{index}",
            event_id=event_id,
            event_type="message",
            channel_id="C-conflict",
            thread_id=None,
            message_id=f"1710000000.00010{index}",
        )
        assert conflict.persisted is False

    stage_count = len(store.calls)
    result = await observer.acknowledged(captured[0])
    listener = AsyncMock(return_value=None)
    terminal = await observer.run_listener(captured[1], listener)

    assert len(observer._revoked_receipts) <= 2
    assert len(store.calls) == stage_count
    assert result.persisted is False
    assert result.failure_reason == "identity_conflict"
    listener.assert_awaited_once()
    assert terminal.state == "accepted"
    assert terminal.persisted is False
    assert terminal.failure_reason == "identity_conflict"


@pytest.mark.asyncio
async def test_blocked_diagnostics_fall_back_without_queueing_socket_frames(
    monkeypatch, tmp_path, caplog
):
    ledger = _observability._ledger
    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    ledger._reset_initialization_for_tests()
    ledger._reset_persistence_health_for_tests()
    started = asyncio.Event()
    release = asyncio.Event()
    original_record = ledger.record_listener_received_safely
    calls = 0

    async def blocking_record(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await release.wait()
        return await original_record(**kwargs)

    monkeypatch.setattr(ledger, "record_listener_received_safely", blocking_record)
    caplog.set_level(logging.CRITICAL, logger=ledger.__name__)

    def frame(suffix: str) -> dict:
        return {
            "type": "events_api",
            "envelope_id": f"X-{suffix}",
            "payload": {
                "team_id": "T-one",
                "event_id": f"Ev-{suffix}",
                "event": {
                    "type": "message",
                    "channel": "C-one",
                    "ts": f"1.{suffix}",
                },
            },
        }

    observer = SlackIntakeObserver()
    first = asyncio.create_task(observer.observe_socket_message(frame("1"), "{}"))
    await started.wait()

    await asyncio.wait_for(
        observer.observe_socket_message(frame("2"), "{}"), timeout=0.1
    )
    fallback = ledger.read_fallback_receipts()
    assert ledger.persistence_degraded()
    assert len(fallback) == 1
    assert fallback[0]["stage"] == "envelope_received"
    assert fallback[0]["failure_reason"] == "admission_busy"
    critical = [record for record in caplog.records if record.levelno == logging.CRITICAL]
    assert len(critical) == 1
    serialized = caplog.text + repr(fallback)
    for raw in ("T-one", "X-2", "Ev-2", "C-one", "1.2", str(tmp_path)):
        assert raw not in serialized

    release.set()
    await first
    assert calls == 1
    assert not ledger.persistence_degraded()
    assert len(ledger.read_receipts()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "call_kwargs", "expected_stage", "expected_reason"),
    [
        ("acknowledged", {}, "acknowledged", None),
        ("listener_entered", {}, "listener_entered", None),
        ("accepted", {}, "accepted", None),
        ("dropped", {"reason": "ignored_channel"}, "dropped", "ignored_channel"),
    ],
)
async def test_blocked_lifecycle_diagnostics_fall_back_without_queueing(
    monkeypatch,
    tmp_path,
    method_name,
    call_kwargs,
    expected_stage,
    expected_reason,
):
    ledger = _observability._ledger
    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "ledger.sqlite3")
    ledger._reset_initialization_for_tests()
    ledger._reset_persistence_health_for_tests()
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking_progress(*_args, **_kwargs):
        started.set()
        await release.wait()
        return SimpleNamespace(persisted=True, observation=None, failure_reason=None)

    monkeypatch.setattr(ledger, "append_stage_safely", blocking_progress)
    observer = SlackIntakeObserver()
    context = IntakeContext(
        receipt_id="a" * 64,
        envelope_hash="b" * 64,
        event_hash="a" * 64,
        message_key_hash="c" * 64,
        persisted=True,
        degraded=False,
        duplicate=False,
        retry_attempt=0,
    )
    first = asyncio.create_task(observer.listener_entered(context))
    await started.wait()

    try:
        result = await asyncio.wait_for(
            getattr(observer, method_name)(context, **call_kwargs), timeout=0.1
        )
    finally:
        release.set()
        await first

    fallback = ledger.read_fallback_receipts()
    assert result.persisted is False
    assert result.failure_reason == "admission_busy"
    assert len(fallback) == 1
    assert fallback[0]["receipt_id"] == context.receipt_id
    assert fallback[0]["stage"] == expected_stage
    assert fallback[0]["reason"] == expected_reason
    assert fallback[0]["failure_reason"] == "admission_busy"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("listener_result", "expected_state", "expected_reason"),
    [(None, "accepted", None), ("ignored_channel", "dropped", "ignored_channel")],
)
async def test_terminal_persistence_degradation_remains_socket_available(
    monkeypatch, listener_result, expected_state, expected_reason
):
    observer = SlackIntakeObserver(clock=lambda: 100.0)
    context = IntakeContext(
        receipt_id="a" * 64,
        envelope_hash="c" * 64,
        event_hash="d" * 64,
        message_key_hash="e" * 64,
        persisted=True,
        degraded=False,
        duplicate=False,
        retry_attempt=0,
    )

    async def _fail_write(*_args, **_kwargs):
        raise sqlite3.OperationalError("private database path")

    monkeypatch.setattr(_observability._ledger, "_run_ledger_work", _fail_write)

    async def _listener():
        return listener_result

    terminal = await observer.run_listener(context, _listener)
    assert terminal.state == expected_state
    assert terminal.reason == expected_reason


@pytest.mark.asyncio
async def test_sdk_listener_precedes_bolt_and_ack_is_post_send_only():
    order: list[str] = []
    store = _Store()
    observer = SlackIntakeObserver(store=store, clock=lambda: 100.0)
    original_envelope_received = observer.envelope_received

    async def _observed(**kwargs):
        order.append("envelope_received")
        return await original_envelope_received(**kwargs)

    observer.envelope_received = _observed
    original_acknowledged = observer.acknowledged

    async def _acknowledged(context):
        order.append("acknowledged")
        return await original_acknowledged(context)

    observer.acknowledged = _acknowledged

    async def bolt_listener(_client, _request):
        order.append("bolt_listener")

    async def original_send(_response):
        order.append("response_sent")

    client = SimpleNamespace(
        message_listeners=[],
        socket_mode_request_listeners=[bolt_listener],
        send_socket_mode_response=original_send,
    )
    install_socket_observer(client, observer)

    raw_message = json.dumps(
        {
            "envelope_id": "env-fake",
            "payload": {
                "team_id": "T-from-envelope",
                "event_id": "Ev-fake",
                "event": {
                    "type": "app_mention",
                    "channel": "C-fake",
                    "ts": "1712345.200",
                },
            },
        }
    )
    for listener in client.message_listeners:
        await listener(client, json.loads(raw_message), raw_message)
    for listener in client.socket_mode_request_listeners:
        await listener(client, SimpleNamespace())
    assert order == ["envelope_received", "bolt_listener"]
    assert store.calls[0][1]["workspace_id"] == "T-from-envelope"

    await client.send_socket_mode_response(
        SimpleNamespace(envelope_id="env-fake")
    )
    assert order == [
        "envelope_received",
        "bolt_listener",
        "response_sent",
        "acknowledged",
    ]

    async def failed_send(_response):
        raise RuntimeError("send failed")

    failed_observer = SlackIntakeObserver(store=_Store(), clock=lambda: 100.0)
    failed_observer.acknowledged = AsyncMock()
    failed_client = SimpleNamespace(
        message_listeners=[],
        socket_mode_request_listeners=[],
        send_socket_mode_response=failed_send,
    )
    install_socket_observer(failed_client, failed_observer)
    for listener in failed_client.message_listeners:
        await listener(failed_client, json.loads(raw_message), raw_message)
    with pytest.raises(RuntimeError, match="send failed"):
        await failed_client.send_socket_mode_response(
            SimpleNamespace(envelope_id="env-fake")
        )
    failed_observer.acknowledged.assert_not_awaited()


@pytest.mark.asyncio
async def test_observer_records_listener_and_exact_terminal_disposition():
    store = _Store()
    observer = SlackIntakeObserver(store=store, clock=lambda: 200.0)

    context = await observer.envelope_received(
        workspace_id="T-private",
        envelope_id="env-private",
        event_id="Ev-private",
        event_type="message",
        channel_id="C-private",
        thread_id=None,
        message_id="1712345.200",
    )
    await observer.listener_entered(context)
    terminal = await observer.dropped(context, reason="unauthorized")

    assert [stage for stage, _metadata in store.calls] == [
        "envelope_received",
        "listener_entered",
        "dropped",
    ]
    assert terminal.state == "dropped"
    assert terminal.reason == "unauthorized"
    representation = repr(context)
    for raw in ("T-private", "env-private", "Ev-private", "C-private"):
        assert raw not in representation


@pytest.mark.asyncio
async def test_acknowledged_envelope_continues_with_bounded_fallback_context():
    store = _Store(persisted=False)
    observer = SlackIntakeObserver(store=store, clock=lambda: 300.0)

    context = await observer.envelope_received(
        workspace_id="T-private",
        envelope_id="env-private",
        event_id="Ev-private",
        event_type="app_mention",
        channel_id="C-private",
        thread_id=None,
        message_id="1712345.300",
    )
    await observer.acknowledged(context)
    await observer.listener_entered(context)
    terminal = await observer.accepted(context)

    assert context.persisted is False
    assert context.degraded is True
    assert terminal.state == "accepted"
    assert [stage for stage, _metadata in store.calls] == [
        "envelope_received",
        "acknowledged",
        "listener_entered",
        "accepted",
    ]


@pytest.mark.asyncio
async def test_duplicate_projection_is_returned_as_explicit_sibling_drop():
    store = _Store()
    store.mark_accepted = AsyncMock(
        return_value=SimpleNamespace(
            state="dropped",
            reason="duplicate_ts",
            related_receipt_id="b" * 64,
        )
    )
    observer = SlackIntakeObserver(store=store, clock=lambda: 400.0)
    context = SimpleNamespace(receipt_id="a" * 64, persisted=True, degraded=False)

    terminal = await observer.accepted(context)

    assert terminal.state == "dropped"
    assert terminal.reason == "duplicate_ts"
    assert terminal.related_receipt_id == "b" * 64


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "raw"),
    [
        ({}, "not-json SECRET-BODY"),
        ({"type": "hello"}, '{"type":"hello","text":"SECRET-BODY"}'),
        (
            {"envelope_id": "env", "payload": {"type": "not-events-api"}},
            '{"payload":{"text":"SECRET-BODY"}}',
        ),
        (
            {"envelope_id": "env", "payload": {"event_id": "Ev"}},
            "SECRET-BODY" * 100_000,
        ),
    ],
)
async def test_malformed_or_oversized_frames_fail_open_to_bolt_without_payload_leak(
    message, raw, caplog: pytest.LogCaptureFixture
):
    store = _Store()
    observer = SlackIntakeObserver(store=store, clock=lambda: 100.0)
    reached_bolt = []

    async def bolt(_client, _request):
        reached_bolt.append(True)

    client = SimpleNamespace(
        message_listeners=[],
        socket_mode_request_listeners=[bolt],
        send_socket_mode_response=AsyncMock(),
    )
    install_socket_observer(client, observer)

    for listener in client.message_listeners:
        await listener(client, message, raw)
    for listener in client.socket_mode_request_listeners:
        await listener(client, SimpleNamespace())

    assert reached_bolt == [True]
    assert store.calls == []
    assert "SECRET-BODY" not in caplog.text


@pytest.mark.asyncio
async def test_listener_exception_and_cancellation_terminalize_then_reraise_original():
    store = _Store()
    observer = SlackIntakeObserver(store=store, clock=lambda: 500.0)
    context = await observer.envelope_received(
        workspace_id="T-private",
        envelope_id="env-private",
        event_id="Ev-private",
        event_type="message",
        channel_id="C-private",
        thread_id=None,
        message_id="1712345.500",
    )
    error = RuntimeError("SECRET-BODY")

    async def fails():
        raise error

    with pytest.raises(RuntimeError) as caught:
        await observer.run_listener(context, fails)
    assert caught.value is error
    assert store.calls[-1][0] == "dropped"
    assert store.calls[-1][1]["reason"] == "exception_runtime_error"

    second = await observer.envelope_received(
        workspace_id="T-private",
        envelope_id="env-private-2",
        event_id="Ev-private-2",
        event_type="app_mention",
        channel_id="C-private",
        thread_id=None,
        message_id="1712345.500",
    )
    cancellation = asyncio.CancelledError()

    async def cancels():
        raise cancellation

    with pytest.raises(asyncio.CancelledError) as cancelled:
        await observer.run_listener(second, cancels)
    assert cancelled.value is cancellation
    assert store.calls[-1][1]["reason"] == "listener_cancelled"


@pytest.mark.asyncio
async def test_context_maps_are_bounded_and_digest_keyed_only():
    observer = SlackIntakeObserver(store=_Store(), clock=lambda: 700.0, max_contexts=2)
    raw_ids = []
    for index in range(4):
        envelope = f"env-private-{index}"
        event = f"Ev-private-{index}"
        raw_ids.extend((envelope, event))
        await observer.envelope_received(
            workspace_id="T-private",
            envelope_id=envelope,
            event_id=event,
            event_type="message",
            channel_id="C-private",
            thread_id=None,
            message_id=f"1712345.{index}",
        )

    assert len(observer._envelope_contexts) <= 2
    assert len(observer._event_contexts) <= 2
    serialized = repr(observer._envelope_contexts) + repr(observer._event_contexts)
    assert all(raw not in serialized for raw in raw_ids)
    assert all(len(key) == 64 for key in observer._envelope_contexts)
