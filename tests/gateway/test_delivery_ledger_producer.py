"""Producer-hook tests: _process_message_background records delivery
obligations around the final send (gateway/platforms/base.py).

Contract: obligation recorded (pending→attempting) BEFORE the send await,
delivered/failed by SendResult afterward; slash commands, ephemeral
replies, and empty responses are never recorded; ledger failures never
block the send.
"""

import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    yield home


class _Adapter(BasePlatformAdapter):  # type: ignore[misc]
    """Minimal concrete adapter driving the real base-class pipeline."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.SLACK)
        self.sent = []
        self.sent_metadata = []

    async def connect(self, *, is_reconnect: bool = False):  # pragma: no cover
        return True

    async def disconnect(self):  # pragma: no cover - unused
        return None

    async def get_chat_info(self, chat_id):  # pragma: no cover - unused
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        self.sent_metadata.append(dict(metadata or {}))
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
            """SELECT obligation_id, state, content, adapter_profile
               FROM delivery_obligations"""
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
    async def test_late_transient_failure_signals_reconnected_runner(self):
        """A replacement installed mid-send must trigger another ledger sweep."""
        adapter = _Adapter()
        adapter._owner_profile = "reviewer"
        replacement = _Adapter()
        replacement._owner_profile = "reviewer"
        runner = MagicMock()
        runner._adapter_for_source.side_effect = [adapter, replacement]
        runner._redeliver_failed_obligations_for_platform = AsyncMock(return_value=1)
        adapter.gateway_runner = runner
        adapter.send = AsyncMock(
            return_value=SendResult(
                success=False,
                error="send_path_degraded",
                retryable=True,
            )
        )

        await _run(adapter, _event())

        assert _rows()[0][1] == "failed"
        assert _rows()[0][3] == "reviewer"
        runner._redeliver_failed_obligations_for_platform.assert_awaited_once_with(
            Platform.SLACK, profile="reviewer"
        )


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

    @pytest.mark.asyncio
    async def test_claimed_result_reuses_pre_staged_obligation(self):
        """Final delivery must consume claim-owned output, not enqueue a copy."""
        from gateway.platforms.base import DeliveryOwnedReply

        obligation_id = dl.record_claimed_result(
            session_key="agent:main:slack:channel:C1",
            claim_id="claim-owned",
            claim_event_id="event-owned",
            platform="slack",
            chat_id="C1",
            thread_id=None,
            content="final answer",
            adapter_profile=None,
        )
        adapter = _Adapter()

        await _run(
            adapter,
            _event(),
            response=DeliveryOwnedReply("final answer", obligation_id),
        )

        assert adapter.sent == ["final answer"]
        rows = _rows()
        assert rows == [(obligation_id, "delivered", "final answer", "default")]

    @pytest.mark.asyncio
    async def test_recovered_precomputed_result_bypasses_agent_handler(self):
        """Restart publication reuses the final-response pipeline without execution."""
        obligation_id = dl.record_claimed_result(
            session_key="agent:main:slack:channel:C1",
            claim_id="claim-recovered",
            claim_event_id="event-recovered",
            platform="slack",
            chat_id="C1",
            thread_id=None,
            content="recovered answer",
            adapter_profile=None,
        )
        adapter = _Adapter()
        adapter._message_handler = AsyncMock(
            side_effect=AssertionError("agent handler must not execute")
        )
        event = _event(text="")
        event._hermes_precomputed_response = "recovered answer"
        event._hermes_precomputed_obligation_id = obligation_id
        session_key = "agent:main:slack:channel:C1"
        adapter._active_sessions[session_key] = asyncio.Event()

        await adapter._process_message_background(event, session_key)

        adapter._message_handler.assert_not_awaited()
        assert adapter.sent == ["recovered answer"]
        assert _rows() == [
            (obligation_id, "delivered", "recovered answer", "default")
        ]

    @pytest.mark.asyncio
    async def test_restart_redelivery_keeps_unavailable_raw_attachment_incomplete_without_agent(
        self,
    ):
        from gateway.run import GatewayRunner

        source = _event(text="").source
        source_payload = source.to_dict()
        source_payload["is_bot"] = bool(source.is_bot)
        source_payload["role_authorized"] = False
        source_json = json.dumps(
            source_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        obligation_id = dl.record_claimed_result(
            session_key="agent:main:slack:channel:C1",
            claim_id="claim-restart-pipeline",
            claim_event_id="event-restart-pipeline",
            platform="slack",
            chat_id="C1",
            thread_id=None,
            content="recovered answer",
            raw_content="recovered answer\nMEDIA:C:/private/recovered.png",
            source_json=source_json,
            message_ref="original-result",
            adapter_profile=None,
        )
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET owner_pid=999999999, "
                "owner_started_at=1 WHERE obligation_id=?",
                (obligation_id,),
            )
        claimed = dl.sweep_recoverable(deliverable_platforms={"slack"})
        assert len(claimed) == 1

        adapter = _Adapter()
        adapter._message_handler = AsyncMock(
            side_effect=AssertionError("agent handler must not execute")
        )
        runner = object.__new__(GatewayRunner)
        runner.adapters = {Platform.SLACK: adapter}
        runner._profile_adapters = {}
        runner._authorization_adapter = lambda _platform, _profile=None: adapter

        delivered = await runner._redeliver_claimed_obligations(claimed)

        assert delivered == 0
        adapter._message_handler.assert_not_awaited()
        assert adapter.sent == ["recovered answer"]
        assert _rows() == [
            (
                obligation_id,
                "failed",
                "recovered answer",
                "default",
            )
        ]
        assert [
            (row["kind"], row["state"])
            for row in dl.get_claimed_result_parts(obligation_id)
        ] == [("text", "delivered"), ("image", "failed")]

    @pytest.mark.asyncio
    async def test_completed_result_staging_sanitizes_missing_bare_path_before_crash(
        self,
        _fresh_db,
        tmp_path,
    ):
        """The first durable visible projection must not depend on availability."""
        from gateway.goal_continuation_claims import (
            event_claim_identity,
            load_claims,
            publish_claim,
        )
        from gateway.run import GatewayRunner

        missing = (tmp_path / "PRIVATE_STAGING_PATH.pdf").resolve()
        explicit = (tmp_path / "PRIVATE_EXPLICIT_EXTENSIONLESS").resolve()
        spaced = (tmp_path / "My Documents" / "Caddyfile").resolve()
        assert not missing.exists()
        assert not explicit.exists()
        assert not spaced.exists()
        response = (
            f"recovered answer\n{missing}\nMEDIA:{explicit}\nMEDIA:{spaced}"
        )
        event = _event(
            text="[Continuing toward your standing goal]\nGoal: stage safely"
        )
        event.goal_continuation = True
        event.allow_gateway_control = False
        session_key = "agent:main:slack:channel:C1"
        publish_claim(
            session_key,
            "sid-staging-privacy",
            [event],
            home=_fresh_db,
        )
        _claim_id, event_id = event_claim_identity(event)

        adapter = _Adapter()
        runner = object.__new__(GatewayRunner)
        runner._goal_continuation_claim_home = _fresh_db
        runner._adapter_for_source = lambda _source: adapter

        def _crash_before_claim_ack(*_args, **_kwargs):
            raise SystemExit(73)

        runner._complete_goal_continuation_claim_event = _crash_before_claim_ack
        result = {"final_response": response}

        with pytest.raises(SystemExit, match="73"):
            await runner._commit_goal_continuation_result(
                session_key=session_key,
                source=event.source,
                event=event,
                result=result,
            )

        claims = load_claims(home=_fresh_db)
        assert len(claims) == 1
        attachment_snapshot = getattr(
            event,
            "_hermes_claimed_response_parts_snapshot",
        )
        assert attachment_snapshot.visible_text == "recovered answer"
        assert attachment_snapshot.media_files == (
            (str(explicit), False),
            (str(spaced), False),
        )
        assert attachment_snapshot.local_files == (str(missing),)
        assert claims[0].completed_delivery_texts[event_id] == "recovered answer"
        assert str(missing) not in claims[0].completed_delivery_texts[event_id]
        assert str(explicit) not in claims[0].completed_delivery_texts[event_id]
        assert str(spaced) not in claims[0].completed_delivery_texts[event_id]
        assert str(Path("Documents") / "Caddyfile") not in (
            claims[0].completed_delivery_texts[event_id]
        )
        assert claims[0].completed_results[event_id] == response

        obligation_id = result["_delivery_obligation_id"]
        with dl._connect() as conn:
            staged = conn.execute(
                "SELECT content, raw_content, last_error, source_json, message_ref "
                "FROM delivery_obligations WHERE obligation_id=?",
                (obligation_id,),
            ).fetchone()
        assert staged[0] == "recovered answer"
        assert staged[1] == response
        assert str(missing) not in repr((staged[0], staged[2], staged[3], staged[4]))
        assert str(explicit) not in repr((staged[0], staged[2], staged[3], staged[4]))
        assert str(spaced) not in repr((staged[0], staged[2], staged[3], staged[4]))

        restart = object.__new__(GatewayRunner)
        restart._goal_continuation_claim_home = _fresh_db
        assert restart._reconcile_completed_goal_continuation_claims() == 1
        assert load_claims(home=_fresh_db) == []

        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET owner_pid=999999999, "
                "owner_started_at=1 WHERE obligation_id=?",
                (obligation_id,),
            )
        claimed = dl.sweep_recoverable(deliverable_platforms={"slack"})
        assert len(claimed) == 1

        recovery_adapter = _Adapter()
        recovery_adapter._message_handler = AsyncMock(
            side_effect=AssertionError("restart must not invoke the agent")
        )
        restart.adapters = {Platform.SLACK: recovery_adapter}
        restart._profile_adapters = {}
        restart._authorization_adapter = (
            lambda _platform, _profile=None: recovery_adapter
        )

        delivered = await restart._redeliver_claimed_obligations(claimed)

        assert delivered == 0
        recovery_adapter._message_handler.assert_not_awaited()
        assert recovery_adapter.sent == ["recovered answer"]
        with dl._connect() as conn:
            visible_row = conn.execute(
                "SELECT content, last_error, source_json, message_ref "
                "FROM delivery_obligations WHERE obligation_id=?",
                (obligation_id,),
            ).fetchone()
            part_rows = conn.execute(
                "SELECT part_id, part_ordinal, kind, state, last_error, "
                "remote_receipt FROM delivery_obligation_parts "
                "WHERE obligation_id=? ORDER BY part_ordinal",
                (obligation_id,),
            ).fetchall()
        provider_projection = repr(
            (
                recovery_adapter.sent,
                recovery_adapter.sent_metadata,
                visible_row,
                part_rows,
            )
        )
        assert str(missing) not in provider_projection
        assert str(explicit) not in provider_projection
        assert str(spaced) not in provider_projection
        assert str(Path("Documents") / "Caddyfile") not in provider_projection
        assert "PRIVATE_STAGING_PATH" not in provider_projection
        assert "PRIVATE_EXPLICIT_EXTENSIONLESS" not in provider_projection
        assert [(row[2], row[3]) for row in part_rows] == [
            ("text", "delivered"),
            ("document", "failed"),
            ("document", "pending"),
            ("document", "pending"),
        ]
