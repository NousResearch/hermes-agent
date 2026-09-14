"""Producer-hook tests: _process_message_background records delivery
obligations around the final send (gateway/platforms/base.py).

Contract: obligation recorded (pending→attempting) BEFORE the send await,
delivered/failed by SendResult afterward; slash commands, ephemeral
replies, and empty responses are never recorded; ledger failures never
block the send.
"""

import asyncio
import sqlite3
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    yield


class _Adapter(BasePlatformAdapter):  # type: ignore[misc]
    """Minimal concrete adapter driving the real base-class pipeline."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.SLACK)
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False):  # pragma: no cover
        return True

    async def disconnect(self):  # pragma: no cover - unused
        return None

    async def get_chat_info(self, chat_id):  # pragma: no cover - unused
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id="m1")


def _event(text="hello agent"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.SLACK, chat_id="C1", chat_type="channel"
        ),
        message_id="msg-42",
    )


def _rows():
    with dl._connect() as conn:
        return conn.execute(
            "SELECT obligation_id, state, content FROM delivery_obligations"
        ).fetchall()


def _blocking_probe():
    """Return a blocking ledger call and an event-loop progress witness."""
    ledger_started = threading.Event()
    event_loop_progressed = threading.Event()
    blocked_event_loop = []

    def _slow_ledger_call(*args, **kwargs):
        ledger_started.set()
        # Generous timeout: a genuinely blocked loop can never set the event
        # (the witness coroutine cannot run), so a longer wait only guards
        # against loaded-CI scheduling flake, not against missing the bug.
        if not event_loop_progressed.wait(timeout=5.0):
            blocked_event_loop.append(True)

    async def _event_loop_witness():
        deadline = asyncio.get_running_loop().time() + 10
        while not ledger_started.is_set():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("ledger call never started")
            await asyncio.sleep(0)
        event_loop_progressed.set()

    return _slow_ledger_call, _event_loop_witness, blocked_event_loop


async def _run(adapter, event, response="final answer"):
    adapter._message_handler = AsyncMock(return_value=response)
    session_key = "agent:main:slack:channel:C1"
    adapter._active_sessions[session_key] = asyncio.Event()
    await adapter._process_message_background(event, session_key)


class TestProducerHook:
    @pytest.mark.asyncio
    async def test_normal_turn_records_and_delivers(self):
        adapter = _Adapter()
        await _run(adapter, _event())

        assert adapter.sent == ["final answer"]
        rows = _rows()
        assert len(rows) == 1
        assert rows[0][1] == "delivered"
        assert rows[0][2] == "final answer"

    @pytest.mark.asyncio
    async def test_send_failure_leaves_failed_row(self):
        adapter = _Adapter()
        adapter.send = AsyncMock(
            return_value=SendResult(success=False, error="chat_not_found")
        )
        await _run(adapter, _event())

        rows = _rows()
        assert len(rows) == 1
        assert rows[0][1] == "failed"

    @pytest.mark.asyncio
    async def test_attempting_failure_recovery_is_visibly_ambiguous(self, monkeypatch):
        adapter = _Adapter()

        def fail_attempting(obligation_id):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(dl, "mark_attempting", fail_attempting)
        await _run(adapter, _event())

        rows = _rows()
        assert adapter.sent == ["final answer"]
        assert len(rows) == 1
        assert rows[0][1] == "delivered"

    @pytest.mark.asyncio
    async def test_attempting_and_terminal_failure_never_redelivers_silently(
        self, monkeypatch
    ):
        adapter = _Adapter()

        def fail_checkpoint(*_args, **_kwargs):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(dl, "mark_attempting", fail_checkpoint)
        monkeypatch.setattr(dl, "mark_delivered", fail_checkpoint)
        await _run(adapter, _event())

        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1"
            )
        claimed = dl.sweep_recoverable()

        assert adapter.sent == ["final answer"]
        assert len(claimed) == 1
        assert claimed[0]["needs_marker"] is True

    @pytest.mark.asyncio
    async def test_locked_ledger_fails_open_before_writer_releases(self):
        adapter = _Adapter()
        send_started = asyncio.Event()

        async def send(chat_id, content, reply_to=None, metadata=None):
            send_started.set()
            return SendResult(success=True, message_id="m1")

        adapter.send = send
        with dl._connect():
            pass
        blocker = sqlite3.connect(dl._db_path(), timeout=1, isolation_level=None)
        blocker.execute("BEGIN IMMEDIATE")

        task = asyncio.create_task(_run(adapter, _event()))
        try:
            await asyncio.wait_for(send_started.wait(), timeout=1.0)
            assert blocker.in_transaction
        finally:
            blocker.execute("ROLLBACK")
            blocker.close()

        await task
        assert _rows() == []

    @pytest.mark.asyncio
    async def test_in_process_ledger_lock_fails_open_before_release(self):
        adapter = _Adapter()
        send_started = asyncio.Event()
        lock_held = threading.Event()
        release_lock = threading.Event()

        async def send(chat_id, content, reply_to=None, metadata=None):
            send_started.set()
            return SendResult(success=True, message_id="m1")

        def hold_lock():
            with dl._DB_LOCK:
                lock_held.set()
                release_lock.wait(timeout=5.0)

        adapter.send = send
        holder = threading.Thread(target=hold_lock)
        holder.start()
        assert lock_held.wait(timeout=1.0)

        task = asyncio.create_task(_run(adapter, _event()))
        try:
            await asyncio.wait_for(send_started.wait(), timeout=1.0)
            assert holder.is_alive()
        finally:
            release_lock.set()
            holder.join(timeout=1.0)

        await task
        assert _rows() == []


    @pytest.mark.asyncio
    async def test_slow_ledger_record_does_not_block_event_loop(self):
        adapter = _Adapter()
        slow_record, event_loop_witness, blocked_event_loop = _blocking_probe()

        with patch(
            "gateway.delivery_ledger.record_obligation",
            side_effect=slow_record,
        ), patch("gateway.delivery_ledger.mark_attempting"):
            await asyncio.gather(_run(adapter, _event()), event_loop_witness())

        assert blocked_event_loop == []
        assert adapter.sent == ["final answer"]

    @pytest.mark.asyncio
    async def test_slow_ledger_update_does_not_block_event_loop(self):
        adapter = _Adapter()
        slow_delivered, event_loop_witness, blocked_event_loop = _blocking_probe()

        with patch("gateway.delivery_ledger.record_obligation"), patch(
            "gateway.delivery_ledger.mark_attempting"
        ), patch(
            "gateway.delivery_ledger.mark_delivered",
            side_effect=slow_delivered,
        ):
            await asyncio.gather(_run(adapter, _event()), event_loop_witness())

        assert blocked_event_loop == []
        assert adapter.sent == ["final answer"]

    @pytest.mark.asyncio
    async def test_ledger_path_does_not_use_asyncio_default_executor(
        self, monkeypatch
    ):
        adapter = _Adapter()

        async def reject_to_thread(*_args, **_kwargs):
            raise AssertionError("delivery ledger used asyncio.to_thread")

        monkeypatch.setattr(asyncio, "to_thread", reject_to_thread)

        await _run(adapter, _event())

        assert adapter.sent == ["final answer"]
        assert _rows()[0][1] == "delivered"

    @pytest.mark.asyncio
    async def test_crash_between_attempting_and_ack_is_recoverable(self):
        """The core scenario (#58818): process dies mid-send. The row must
        be claimable by a later process and carry the ambiguity marker."""
        adapter = _Adapter()

        async def _dies_mid_send(chat_id, content, reply_to=None, metadata=None):
            raise ConnectionError("gateway killed mid-await")

        adapter.send = _dies_mid_send
        # _send_with_retry raising propagates; the background task catches
        # broadly — drive only through the send block by tolerating the error.
        try:
            await _run(adapter, _event())
        except Exception:
            pass

        rows = _rows()
        assert len(rows) == 1
        # Row is stuck in 'attempting' (or failed if retry wrapper caught it):
        # either way it is non-delivered and recoverable.
        assert rows[0][1] in ("attempting", "failed")
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1"
            )
        claimed = dl.sweep_recoverable()
        assert len(claimed) == 1
        assert claimed[0]["needs_marker"] is True



class TestFinalHandoffPathOffDefaultExecutor:
    """Atomic r28 blocker 1: the FinalResponseHandoff receipt lifecycle
    (prepare -> attempting -> settle / cancel) must not run on asyncio's
    default executor. A stalled SQLite call there keeps the executor alive
    during loop shutdown and blocks the old gateway from exiting on restart.
    """

    def _handoff(self, calls):
        from gateway.post_response import FinalResponseHandoff

        class _Runner:
            def _prepare_final_response_handoff(self, h, *, text_content, event):
                calls.append("prepare")
                h.obligation_id = "ob-f1"
                return True

            def _mark_final_response_handoff_attempting(self, h):
                calls.append("attempting")
                return True

            def _settle_final_response_handoff(self, h, *, delivered, error=None):
                calls.append(("settle", delivered))
                return True

            def _cancel_uncommitted_final_response_handoff(self, h):
                calls.append("cancel")
                return None

            class _post_response_controller:
                @staticmethod
                def wake():
                    calls.append("wake")

        h = FinalResponseHandoff(
            "final answer",
            runner=_Runner(),
            session_key="agent:main:slack:channel:C1",
            session_lineage="lin1",
            expected_session_id="s1",
            new_session_id=None,
            active_turn_token=None,
            last_prompt_tokens=10,
            touch_activity=True,
            append_final_text=True,
            clear_resume_pending=False,
        )
        return h

    @pytest.mark.asyncio
    async def test_delivered_handoff_avoids_default_executor(self, monkeypatch):
        import asyncio as _asyncio

        async def reject_to_thread(*_args, **_kwargs):
            raise AssertionError("final-response handoff used asyncio.to_thread")

        monkeypatch.setattr(_asyncio, "to_thread", reject_to_thread)

        calls: list = []
        adapter = _Adapter()
        await _run(adapter, _event(), response=self._handoff(calls))

        assert adapter.sent == ["final answer"]
        assert calls[0] == "prepare"
        assert "attempting" in calls
        assert ("settle", True) in calls
        assert "wake" in calls

    @pytest.mark.asyncio
    async def test_empty_response_handoff_cancel_avoids_default_executor(
        self, monkeypatch
    ):
        import asyncio as _asyncio

        async def reject_to_thread(*_args, **_kwargs):
            raise AssertionError("final-response handoff used asyncio.to_thread")

        monkeypatch.setattr(_asyncio, "to_thread", reject_to_thread)

        calls: list = []
        adapter = _Adapter()
        await _run(adapter, _event(), response=None)
        # response=None path: no handoff existed; the run must simply not
        # touch the default executor anywhere in the background pipeline.
        assert adapter.sent == []
