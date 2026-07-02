"""
Tests for the busy-session drive-queue recovery path (drive-queue-recovery).

A reply sent to a managed session that is busy (RUNNING) must not be dropped:
it is enqueued into the persistent pending-drive store and redelivered by the
watcher once the session next becomes idle (WAITING_USER), or expired after a
TTL with a user notification.

Coverage
--------
Registry (T001, single-writer invariant):
  * enqueue_pending_drive is a plain append-only INSERT usable without a
    session_orchestration row or any lock.
  * list_pending_drive returns entries FIFO; delete/bump mutate correctly.

Gateway drive loop (T002):
  * A RUNNING session enqueues the reply (relay NOT driven) + queued ack.
  * A TimeoutError from send_message enqueues rather than dropping.

Watcher (T003 redelivery, T004 TTL):
  * Redelivery on WAITING_USER drives FIFO, deletes entries, posts confirmation.
  * A failed (TimeoutError) redelivery keeps the entry and bumps attempts.
  * A TTL-expired entry is dropped with a notice; a fresh entry is kept.
  * TTL expiry runs even when the session is not WAITING_USER.

Async tests use asyncio.run() following the repo convention.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from session_orchestration.adapters.base import AgentAdapter
from session_orchestration.registry import SessionOrchestrationRegistry
from session_orchestration.relay import LockConflictError
from session_orchestration.types import Capabilities, SessionHandle, SessionLifecycle
from session_orchestration.watcher import SessionWatcher


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "state.db"


@pytest.fixture()
def registry(db_path: Path) -> SessionOrchestrationRegistry:
    return SessionOrchestrationRegistry(db_path=db_path)


class _FakeAdapter(AgentAdapter):
    """Stub adapter recording drive calls; drive may be made to raise."""

    def __init__(
        self,
        lifecycle: SessionLifecycle = SessionLifecycle.WAITING_USER,
        drive_exc: Optional[BaseException] = None,
    ):
        self._lifecycle = lifecycle
        self._drive_exc = drive_exc
        self.drive_calls: List[str] = []

    def capabilities(self) -> Capabilities:
        return Capabilities()

    def launch(self, workdir: str, prompt: str) -> SessionHandle:
        raise NotImplementedError

    def detect(self, handle: SessionHandle) -> SessionLifecycle:
        return self._lifecycle

    def drive(self, handle: SessionHandle, message: str, *, pre_keys=None) -> None:
        if self._drive_exc is not None:
            raise self._drive_exc
        self.drive_calls.append(message)

    def resume(self, handle: SessionHandle, message: str) -> None:
        raise NotImplementedError

    def terminate(self, handle: SessionHandle) -> None:
        raise NotImplementedError


def _seed_row(
    registry: SessionOrchestrationRegistry,
    task_id: str,
    *,
    tmux_session: str = "hermes-test-sess",
    discord_thread_id: Optional[str] = None,
) -> dict:
    registry.upsert(
        task_id,
        agent="omp",
        tmux_session=tmux_session,
        state="RUNNING",
        discord_thread_id=discord_thread_id,
    )
    return registry.get(task_id)


def _make_event(text: str, thread_id: Optional[str] = None) -> MagicMock:
    event = MagicMock()
    event.text = text
    event.internal = False
    source = MagicMock()
    source.thread_id = thread_id
    event.source = source
    event.get_command.return_value = None
    return event


# ===========================================================================
# T001 — registry pending-drive store (single-writer invariant)
# ===========================================================================


def test_enqueue_pending_drive_is_append_only_without_row_or_lock(registry):
    """enqueue works with no session_orchestration row and no lock held."""
    task_id = str(uuid.uuid4())
    # No _seed_row, no acquire_lock — a pure append from any thread.
    registry.enqueue_pending_drive(task_id, "queued msg", pre_keys=["Escape"])

    entries = registry.list_pending_drive(task_id)
    assert len(entries) == 1
    assert entries[0]["message"] == "queued msg"
    assert entries[0]["pre_keys"] == '["Escape"]'
    assert entries[0]["attempts"] == 0


def test_list_pending_drive_is_fifo_and_delete_bump_work(registry):
    task_id = str(uuid.uuid4())
    registry.enqueue_pending_drive(task_id, "first")
    registry.enqueue_pending_drive(task_id, "second")
    registry.enqueue_pending_drive(task_id, "third")

    entries = registry.list_pending_drive(task_id)
    assert [e["message"] for e in entries] == ["first", "second", "third"]

    registry.bump_pending_drive_attempt(entries[0]["id"])
    registry.bump_pending_drive_attempt(entries[0]["id"])
    registry.delete_pending_drive(entries[1]["id"])

    remaining = registry.list_pending_drive(task_id)
    assert [e["message"] for e in remaining] == ["first", "third"]
    assert remaining[0]["attempts"] == 2


def test_pending_drive_is_scoped_by_task_id(registry):
    registry.enqueue_pending_drive("task-a", "a1")
    registry.enqueue_pending_drive("task-b", "b1")
    assert [e["message"] for e in registry.list_pending_drive("task-a")] == ["a1"]
    assert [e["message"] for e in registry.list_pending_drive("task-b")] == ["b1"]


# ===========================================================================
# T002 — gateway enqueue-on-busy
# ===========================================================================

from gateway.run import GatewayRunner as _GW  # noqa: E402


def test_running_session_enqueues_reply_and_acks(registry):
    """A RUNNING session enqueues the reply instead of driving it."""
    task_id = str(uuid.uuid4())
    thread_id = "discord-thread-busy"
    _seed_row(registry, task_id, discord_thread_id=thread_id)

    adapter = _FakeAdapter(lifecycle=SessionLifecycle.RUNNING)
    relay_mock = MagicMock()
    event = _make_event("reply while busy", thread_id=thread_id)

    async def _run():
        with (
            patch(
                "session_orchestration.registry.SessionOrchestrationRegistry",
                return_value=registry,
            ),
            patch("session_orchestration.spawn.get_adapter", return_value=adapter),
            patch("session_orchestration.relay.SessionRelay", return_value=relay_mock),
        ):
            stub = MagicMock()
            stub.config = {}
            return await _GW._handle_managed_thread_reply(stub, event, thread_id)

    result = asyncio.run(_run())

    # User is told it was queued; relay was NOT driven.
    assert result is not None and "queued" in result.lower()
    relay_mock.send_message.assert_not_called()
    # The reply is persisted for redelivery.
    entries = registry.list_pending_drive(task_id)
    assert [e["message"] for e in entries] == ["reply while busy"]


def test_send_timeout_enqueues_instead_of_dropping(registry):
    """A TimeoutError from send_message queues the reply, not the dead-end error."""
    task_id = str(uuid.uuid4())
    thread_id = "discord-thread-timeout"
    _seed_row(registry, task_id, discord_thread_id=thread_id)

    # Idle on pre-check so we reach send_message, which then times out.
    adapter = _FakeAdapter(lifecycle=SessionLifecycle.WAITING_USER)
    relay_mock = MagicMock()
    relay_mock.send_message.side_effect = TimeoutError("pane not ready")
    event = _make_event("reply that times out", thread_id=thread_id)

    async def _run():
        with (
            patch(
                "session_orchestration.registry.SessionOrchestrationRegistry",
                return_value=registry,
            ),
            patch("session_orchestration.spawn.get_adapter", return_value=adapter),
            patch("session_orchestration.relay.SessionRelay", return_value=relay_mock),
        ):
            stub = MagicMock()
            stub.config = {}
            return await _GW._handle_managed_thread_reply(stub, event, thread_id)

    result = asyncio.run(_run())

    assert result is not None and "queued" in result.lower()
    # It must NOT be the old dead-end message.
    assert "did not become ready" not in result
    entries = registry.list_pending_drive(task_id)
    assert [e["message"] for e in entries] == ["reply that times out"]


def test_lock_conflict_still_returns_busy_error_unchanged(registry):
    """LockConflictError path is unchanged — not enqueued, returns busy error."""
    task_id = str(uuid.uuid4())
    thread_id = "discord-thread-lock"
    _seed_row(registry, task_id, discord_thread_id=thread_id)

    adapter = _FakeAdapter(lifecycle=SessionLifecycle.WAITING_USER)
    relay_mock = MagicMock()
    relay_mock.send_message.side_effect = LockConflictError("held")
    event = _make_event("reply", thread_id=thread_id)

    async def _run():
        with (
            patch(
                "session_orchestration.registry.SessionOrchestrationRegistry",
                return_value=registry,
            ),
            patch("session_orchestration.spawn.get_adapter", return_value=adapter),
            patch("session_orchestration.relay.SessionRelay", return_value=relay_mock),
        ):
            stub = MagicMock()
            stub.config = {}
            return await _GW._handle_managed_thread_reply(stub, event, thread_id)

    result = asyncio.run(_run())
    assert result is not None and "try again" in result.lower()
    # Lock conflict is transient — the reply is NOT queued.
    assert registry.list_pending_drive(task_id) == []


# ===========================================================================
# T003 — watcher redelivery on WAITING_USER
# ===========================================================================


def _watcher(registry) -> SessionWatcher:
    return SessionWatcher(registry=registry, adapters={})


def test_redelivery_fifo_deletes_and_confirms(registry):
    task_id = str(uuid.uuid4())
    thread_id = "thread-redeliver"
    row = _seed_row(registry, task_id, discord_thread_id=thread_id)
    registry.enqueue_pending_drive(task_id, "first queued")
    registry.enqueue_pending_drive(task_id, "second queued")

    adapter = _FakeAdapter(lifecycle=SessionLifecycle.WAITING_USER)
    posts: List[str] = []
    now = datetime.now(tz=timezone.utc)

    with patch(
        "session_orchestration.feed._post_discord_message",
        side_effect=lambda ch, content, **kw: posts.append((ch, content)),
    ):
        _watcher(registry)._drain_pending_drive(
            task_id, row, adapter, SessionLifecycle.WAITING_USER.value, now
        )

    # Both delivered FIFO, both removed, one confirmation each.
    assert adapter.drive_calls == ["first queued", "second queued"]
    assert registry.list_pending_drive(task_id) == []
    assert len(posts) == 2
    assert all(ch == thread_id for ch, _ in posts)


def test_no_redelivery_when_not_waiting_user(registry):
    task_id = str(uuid.uuid4())
    row = _seed_row(registry, task_id)
    registry.enqueue_pending_drive(task_id, "still queued")

    adapter = _FakeAdapter(lifecycle=SessionLifecycle.RUNNING)
    now = datetime.now(tz=timezone.utc)

    _watcher(registry)._drain_pending_drive(
        task_id, row, adapter, SessionLifecycle.RUNNING.value, now
    )

    assert adapter.drive_calls == []
    # Entry survives (still busy) — not dropped, since it is within TTL.
    assert [e["message"] for e in registry.list_pending_drive(task_id)] == ["still queued"]


def test_failed_redelivery_keeps_entry_and_bumps_attempt(registry):
    task_id = str(uuid.uuid4())
    row = _seed_row(registry, task_id)
    registry.enqueue_pending_drive(task_id, "will fail")
    registry.enqueue_pending_drive(task_id, "behind it")

    # drive raises TimeoutError → send_message propagates it → entry retained.
    adapter = _FakeAdapter(
        lifecycle=SessionLifecycle.WAITING_USER, drive_exc=TimeoutError("busy")
    )
    now = datetime.now(tz=timezone.utc)

    _watcher(registry)._drain_pending_drive(
        task_id, row, adapter, SessionLifecycle.WAITING_USER.value, now
    )

    entries = registry.list_pending_drive(task_id)
    # FIFO preserved: the stuck head is kept (attempt bumped), the tail not skipped.
    assert [e["message"] for e in entries] == ["will fail", "behind it"]
    assert entries[0]["attempts"] == 1
    assert entries[1]["attempts"] == 0


# ===========================================================================
# T004 — TTL expiry + notification
# ===========================================================================


def test_ttl_expires_old_entry_with_notice(registry):
    task_id = str(uuid.uuid4())
    thread_id = "thread-ttl"
    row = _seed_row(registry, task_id, discord_thread_id=thread_id)
    registry.enqueue_pending_drive(task_id, "stale reply")

    adapter = _FakeAdapter(lifecycle=SessionLifecycle.RUNNING)
    posts: List[str] = []
    # Evaluate "now" one hour in the future so the just-enqueued entry is
    # older than the default 600s TTL.
    future = datetime.now(tz=timezone.utc) + timedelta(hours=1)

    with patch(
        "session_orchestration.feed._post_discord_message",
        side_effect=lambda ch, content, **kw: posts.append((ch, content)),
    ):
        _watcher(registry)._drain_pending_drive(
            task_id, row, adapter, SessionLifecycle.RUNNING.value, future
        )

    # Dropped + user notified, even though the session never went idle.
    assert registry.list_pending_drive(task_id) == []
    assert adapter.drive_calls == []
    assert len(posts) == 1
    assert "expired" in posts[0][1].lower()


def test_ttl_keeps_fresh_entry(registry):
    task_id = str(uuid.uuid4())
    row = _seed_row(registry, task_id)
    registry.enqueue_pending_drive(task_id, "fresh reply")

    adapter = _FakeAdapter(lifecycle=SessionLifecycle.RUNNING)
    now = datetime.now(tz=timezone.utc)  # age ~0 < TTL

    _watcher(registry)._drain_pending_drive(
        task_id, row, adapter, SessionLifecycle.RUNNING.value, now
    )

    assert [e["message"] for e in registry.list_pending_drive(task_id)] == ["fresh reply"]
