"""Behavioral contract for reliable direct Honcho delivery (BC-62A)."""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import (
    DeliveryOutcome,
    DeliveryState,
    HonchoSession,
    HonchoSessionManager,
    _ASYNC_SHUTDOWN,
    classify_delivery_error,
)


def _manager(write_frequency="turn") -> HonchoSessionManager:
    return HonchoSessionManager(
        honcho=MagicMock(),
        config=HonchoClientConfig(
            api_key="test-key",
            enabled=True,
            write_frequency=write_frequency,
        ),
    )


def _session() -> HonchoSession:
    return HonchoSession(
        key="cli:test",
        user_peer_id="human",
        assistant_peer_id="hermes",
        honcho_session_id="cli-test",
    )


def _provider_with_manager() -> tuple[HonchoMemoryProvider, MagicMock]:
    provider = HonchoMemoryProvider()
    provider._config = HonchoClientConfig(save_messages=True)
    manager = MagicMock()
    provider._manager = manager
    provider._session_key = "cli:test"
    provider._session_initialized = True
    return provider, manager


def _wire_delivery(mgr, session, side_effect):
    created = []
    user_peer = MagicMock()
    assistant_peer = MagicMock()

    def make(content, *, metadata, created_at):
        message = SimpleNamespace(
            content=content, metadata=metadata, created_at=created_at
        )
        created.append(message)
        return message

    user_peer.message.side_effect = make
    assistant_peer.message.side_effect = make
    mgr._get_or_create_peer = MagicMock(
        side_effect=lambda peer_id: (
            user_peer if peer_id == session.user_peer_id else assistant_peer
        )
    )
    sdk_session = MagicMock()
    sdk_session.add_messages.side_effect = side_effect
    mgr._sessions_cache[session.honcho_session_id] = sdk_session
    return created, sdk_session


def test_terminal_failure_returns_safe_outcome_and_preserves_unsynced(caplog):
    secret_content = "password=message-secret"
    secret_error = "token=error-secret https://user:pass@example.invalid/private"
    mgr = _manager()
    session = _session()
    mgr.add_source_message(session, "user", secret_content)
    _wire_delivery(mgr, session, RuntimeError(secret_error))

    with caplog.at_level(logging.ERROR, logger="plugins.memory.honcho.session"):
        outcome = mgr._flush_session(session)

    assert outcome == DeliveryOutcome(
        state=DeliveryState.FAILED,
        attempted_count=1,
        delivered_count=0,
        pending_count=1,
        error_category="sdk_error",
        http_status=None,
    )
    assert outcome is mgr.last_delivery_outcome
    assert session.messages[0]["_synced"] is False
    log_text = caplog.text
    assert secret_content not in log_text
    assert secret_error not in log_text
    assert "example.invalid" not in log_text
    assert "terminal failure" in log_text


def test_later_in_process_flush_retries_same_provenance_and_delivers():
    mgr = _manager()
    session = _session()
    mgr.add_source_message(session, "user", "retry me")
    created, sdk_session = _wire_delivery(
        mgr,
        session,
        [ConnectionError("first attempt failed"), None],
    )

    failed = mgr._flush_session(session)
    delivered = mgr.flush_all()

    assert failed.state is DeliveryState.FAILED
    assert delivered.state is DeliveryState.DELIVERED
    assert delivered.delivered_count == 1
    assert session.messages[0]["_synced"] is True
    assert sdk_session.add_messages.call_count == 2
    assert created[0].metadata == created[1].metadata
    assert created[0].created_at == created[1].created_at


def test_already_synced_is_structured_noop():
    mgr = _manager()
    session = _session()
    session.add_message("user", "already there", _synced=True)

    outcome = mgr._flush_session(session)

    assert outcome.state is DeliveryState.NOOP
    assert outcome.attempted_count == 0
    assert outcome.delivered_count == 0
    assert outcome.pending_count == 0


def test_concurrent_flushes_send_one_sdk_batch():
    mgr = _manager()
    session = _session()
    mgr.add_source_message(session, "user", "send once")
    entered = threading.Event()
    release = threading.Event()

    def blocking_delivery(_messages):
        entered.set()
        assert release.wait(timeout=5)

    _, sdk_session = _wire_delivery(mgr, session, blocking_delivery)
    outcomes = []
    first = threading.Thread(
        target=lambda: outcomes.append(mgr._flush_session(session))
    )
    second = threading.Thread(
        target=lambda: outcomes.append(mgr._flush_session(session))
    )

    first.start()
    assert entered.wait(timeout=5)
    second.start()
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert sdk_session.add_messages.call_count == 1
    assert {outcome.state for outcome in outcomes} == {
        DeliveryState.DELIVERED,
        DeliveryState.NOOP,
    }


def test_message_appended_during_flush_remains_pending_for_next_flush():
    mgr = _manager()
    session = _session()
    mgr.add_source_message(session, "user", "first")
    entered = threading.Event()
    release = threading.Event()

    def blocking_delivery(_messages):
        entered.set()
        assert release.wait(timeout=5)

    call_count = 0

    def delivery(messages):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            blocking_delivery(messages)

    _, sdk_session = _wire_delivery(mgr, session, delivery)
    first_outcome = []
    worker = threading.Thread(
        target=lambda: first_outcome.append(mgr._flush_session(session))
    )
    worker.start()
    assert entered.wait(timeout=5)
    mgr.add_source_message(session, "assistant", "appended")
    release.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert first_outcome[0].state is DeliveryState.DELIVERED
    assert first_outcome[0].pending_count == 1
    assert [message.get("_synced", False) for message in session.messages] == [
        True,
        False,
    ]
    second_outcome = mgr._flush_session(session)
    assert second_outcome.state is DeliveryState.DELIVERED
    assert sdk_session.add_messages.call_count == 2


def test_empty_flush_all_returns_fresh_noop():
    mgr = _manager()
    mgr._retain_delivery_outcome(
        DeliveryOutcome(
            state=DeliveryState.DELIVERED,
            attempted_count=1,
            delivered_count=1,
        )
    )

    outcome = mgr.flush_all()

    assert outcome == DeliveryOutcome(state=DeliveryState.NOOP)
    assert outcome is mgr.last_delivery_outcome


def test_real_honcho_transport_errors_are_classified():
    exceptions = pytest.importorskip("honcho.http.exceptions")

    assert classify_delivery_error(exceptions.TimeoutError()) == ("timeout", None)
    assert classify_delivery_error(exceptions.ConnectionError()) == (
        "connection",
        None,
    )


def test_async_writer_calls_boundary_once_and_retains_retryable_failure(caplog):
    mgr = _manager(write_frequency="async")
    session = _session()
    session.add_message("user", "password=do-not-log")
    failed = DeliveryOutcome(
        state=DeliveryState.FAILED,
        attempted_count=1,
        delivered_count=0,
        pending_count=1,
        error_category="connection",
    )
    mgr._flush_session = MagicMock(return_value=failed)
    mgr._async_queue.put(session)
    mgr._async_queue.put(_ASYNC_SHUTDOWN)

    with caplog.at_level(logging.ERROR, logger="plugins.memory.honcho.session"):
        mgr._async_writer_loop()

    mgr._flush_session.assert_called_once_with(session)
    assert mgr.last_delivery_outcome == failed
    assert session.messages[0].get("_synced") is not True
    assert "password=do-not-log" not in caplog.text
    assert "dropping" not in caplog.text.lower()


@pytest.mark.parametrize("path", ["sync_turn", "session_end"])
def test_provider_delivery_exception_logs_are_content_free(path, caplog):
    secret_error = (
        "Authorization: Bearer secret-token "
        "https://user:password@example.invalid/private"
    )
    provider, manager = _provider_with_manager()

    with caplog.at_level(logging.DEBUG, logger="plugins.memory.honcho"):
        if path == "sync_turn":
            manager.get_or_create.return_value = _session()
            manager.save.side_effect = RuntimeError(secret_error)
            provider.sync_turn("safe user", "safe assistant")
            assert provider._sync_thread is not None
            provider._sync_thread.join(timeout=5)
        else:
            manager.flush_all.side_effect = RuntimeError(secret_error)
            provider.on_session_end([])

    assert secret_error not in caplog.text
    assert "Authorization" not in caplog.text
    assert "example.invalid" not in caplog.text
    assert "user:password" not in caplog.text
    assert "category=sdk_error status=none" in caplog.text


@pytest.mark.parametrize(
    ("frequency", "action"),
    [
        ("turn", "save"),
        (2, "save-twice"),
        ("session", "flush-all"),
        ("async", "writer"),
    ],
)
def test_every_write_frequency_converges_through_one_boundary(frequency, action):
    mgr = _manager(write_frequency=frequency)
    session = _session()
    session.add_message("user", "hello")
    mgr._cache[session.key] = session
    delivered = DeliveryOutcome(
        state=DeliveryState.DELIVERED,
        attempted_count=1,
        delivered_count=1,
        pending_count=0,
    )

    with patch.object(mgr, "_flush_session", return_value=delivered) as boundary:
        if action == "save":
            assert mgr.save(session) == delivered
        elif action == "save-twice":
            assert mgr.save(session) is None
            assert mgr.save(session) == delivered
        elif action == "flush-all":
            assert mgr.save(session) is None
            assert mgr.flush_all() == delivered
        else:
            mgr._async_queue.put(session)
            mgr._async_queue.put(_ASYNC_SHUTDOWN)
            mgr._async_writer_loop()
        boundary.assert_called_once_with(session)
