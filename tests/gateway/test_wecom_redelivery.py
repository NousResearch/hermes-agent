"""Reconnect-gap redelivery for the WeCom adapter.

A send that hits errcode 846609 (subscription lost) during the
``closed→reconnect→subscribed`` websocket gap used to return
``SendResult(success=False)`` and the message was lost forever — including final
responses, which the delivery ledger then stranded as ``abandoned`` because the
raw WeCom error is not on its runtime-redelivery allowlist.

Contract under test (all against the REAL ``WeComAdapter._send_inner``; only the
websocket byte-writer / ack layer is faked):

1. A transient (provably-undelivered) failure waits out the reconnect and retries
   the SAME payload — delivered exactly once, no duplicate send.
2. When the reconnect wait is exhausted the send fails closed as
   ``send_path_degraded`` + retryable — the delivery ledger's runtime-redelivery
   token — so the obligation is replayed with a visible marker later.
3. A successful ``_listen_loop`` reconnect kicks the gateway's
   ``_redeliver_failed_obligations_for_platform`` (WeCom self-heals its websocket
   in-process, so the adapter-swap replay path never fires for it).
4. A definitive WeCom errcode is NOT treated as transient: it keeps the original
   fail-closed shape and never becomes ``send_path_degraded``.
5. A passive-path timeout still falls back to proactive exactly once (timeouts
   may have delivered — they are never retried as transient).
6. An ack-path drop AFTER a successful frame write ("connection interrupted" from
   ``_fail_all``) is ambiguous — the frame may have landed — so it is never
   transient-retried either: proactive path fails closed once, passive path falls
   back to proactive once, no duplicate payload ever reaches the wire.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from plugins.platforms.wecom import adapter as wecom_adapter
from plugins.platforms.wecom.adapter import WeComAdapter

CHAT_ID = "wmXXXXchat"
REQ_ID = "req-inbound-1"


def _make_adapter() -> WeComAdapter:
    adapter = WeComAdapter(PlatformConfig(enabled=True))
    adapter._ws = MagicMock(closed=False)
    adapter._last_chat_req_ids[CHAT_ID] = REQ_ID
    return adapter


@pytest.fixture
def fast_redelivery(monkeypatch):
    """Collapse the redelivery timers so tests stay event-based, not wall-clock."""
    monkeypatch.setattr(wecom_adapter, "REDELIVERY_WAIT_SECONDS", 0.3)
    monkeypatch.setattr(wecom_adapter, "REDELIVERY_WAIT_POLL_SECONDS", 0.02)
    monkeypatch.setattr(wecom_adapter, "REDELIVERY_RETRY_BACKOFF_SECONDS", 0.0)


class TestTransientRedelivery:
    @pytest.mark.asyncio
    async def test_subscription_lost_waits_reconnect_and_delivers_once(self, fast_redelivery):
        """846609 on the passive path → stale purge → proactive retry after the
        gap → success. The payload reaches the wire exactly once (no duplicate)."""
        adapter = _make_adapter()

        async def _fail_846609(reply_req_id: str, content: str):
            raise RuntimeError(
                "send reply markdown failed: WeCom errcode 846609: aibot websocket not subscribed"
            )

        adapter._send_reply_markdown = _fail_846609

        proactive_sends: list[dict] = []

        async def _fake_proactive(chat_id: str, content: str):
            proactive_sends.append({"chatid": chat_id, "content": content})
            return {"errcode": 0, "errmsg": "ok", "headers": {"req_id": "sent-1"}}

        adapter._send_proactive_markdown = _fake_proactive

        result = await adapter._send_inner(CHAT_ID, "hello after the gap")

        assert result.success, result.error
        assert len(proactive_sends) == 1  # delivered exactly once — no duplicate
        assert proactive_sends[0]["content"] == "hello after the gap"
        assert proactive_sends[0]["chatid"] == CHAT_ID
        # The stale-req_id purge ran (the retry went proactive for a DM).
        assert adapter._last_chat_req_ids == {}

    @pytest.mark.asyncio
    async def test_exhausted_reconnect_wait_fails_closed_as_send_path_degraded(self, fast_redelivery):
        """Websocket never comes back: the send must fail closed with the ledger's
        runtime-redelivery token instead of a silent permanent loss."""
        adapter = _make_adapter()
        adapter._ws = MagicMock(closed=True)  # down for the whole budget
        adapter._last_chat_req_ids.clear()  # no cached req_id → proactive path

        async def _require_ws_fail(chat_id: str, content: str):
            raise RuntimeError("WeCom websocket is not connected")

        adapter._send_proactive_markdown = _require_ws_fail

        result = await adapter._send_inner(CHAT_ID, "lost without the fix")

        assert not result.success
        assert result.error == "send_path_degraded"
        assert result.retryable is True


class TestReconnectTriggersLedgerReplay:
    @pytest.mark.asyncio
    async def test_reconnect_kicks_failed_obligation_redelivery(self):
        """WeCom reconnects in-process, so the adapter must kick the gateway's
        runtime sweep itself — otherwise ``send_path_degraded`` rows wait for the
        next restart."""
        adapter = _make_adapter()
        calls: list[tuple] = []

        class _FakeRunner:
            async def _redeliver_failed_obligations_for_platform(self, platform, *, profile=None):
                calls.append((platform, profile))
                return 0

        adapter.gateway_runner = _FakeRunner()  # type: ignore[assignment]

        adapter._trigger_failed_obligation_redelivery()
        await asyncio.sleep(0)  # let the spawned task run
        await asyncio.sleep(0)

        assert calls == [(Platform.WECOM, None)]

    @pytest.mark.asyncio
    async def test_redelivery_trigger_without_runner_is_noop(self):
        """No gateway_runner (standalone adapter): the trigger must not raise."""
        adapter = _make_adapter()
        adapter._trigger_failed_obligation_redelivery()
        await asyncio.sleep(0)


class TestNonTransientFailClosed:
    @pytest.mark.asyncio
    async def test_definitive_errcode_is_never_send_path_degraded(self, fast_redelivery):
        """A permission-style rejection is permanent: it keeps the original
        error shape (errcode preserved), is not retried as transient, and never
        becomes ``send_path_degraded``."""
        adapter = _make_adapter()

        async def _fail_permission(reply_req_id: str, content: str):
            raise RuntimeError("send reply markdown failed: WeCom errcode 600011: no permission")

        proactive_calls: list[str] = []

        async def _proactive(chat_id: str, content: str):
            proactive_calls.append(content)
            return {"errcode": 600011, "errmsg": "no permission"}

        adapter._send_reply_markdown = _fail_permission
        adapter._send_proactive_markdown = _proactive

        result = await adapter._send_inner(CHAT_ID, "forbidden")

        assert not result.success
        assert "600011" in (result.error or "")
        assert result.error != "send_path_degraded"
        assert len(proactive_calls) == 1  # original single fallback, no transient retries

    @pytest.mark.asyncio
    async def test_passive_timeout_falls_back_to_proactive_once(self, fast_redelivery):
        """A passive-reply timeout may have delivered: it must fall back to the
        proactive path exactly once (original behaviour), never transient-retry."""
        adapter = _make_adapter()

        async def _timeout(reply_req_id: str, content: str):
            raise asyncio.TimeoutError()

        proactive_calls: list[str] = []

        async def _proactive(chat_id: str, content: str):
            proactive_calls.append(content)
            return {"errcode": 0, "errmsg": "ok", "headers": {"req_id": "sent-2"}}

        adapter._send_reply_markdown = _timeout
        adapter._send_proactive_markdown = _proactive

        result = await adapter._send_inner(CHAT_ID, "maybe already delivered")

        assert result.success, result.error
        assert len(proactive_calls) == 1  # single fallback — no duplicate from retries


class TestAmbiguousAckDropIsNeverTransient:
    """The reviewer's duplicate-send case (#114751 review): ``_request`` writes the frame
    FIRST, then awaits the ack. If the websocket drops after the write, ``_listen_loop``'s
    ``_fail_all(RuntimeError("WeCom connection interrupted"))`` fails the pending ack future —
    but the frame may already have been delivered; what was lost is the acknowledgement, not
    the message. Such a failure is ambiguous, so it must NOT be retried as transient (that
    duplicates). These drive the REAL ``_request``/``_send_json`` path; only the ws byte-writer
    is faked, and it fires the drop right after the frame lands on the wire."""

    @staticmethod
    def _make_dropping_ws(adapter, written: list):
        class _FakeWS:
            closed = False

            async def send_json(self, payload):
                written.append(payload)  # the frame is now on the wire …
                # … and the connection drops immediately after: _listen_loop fails the ack.
                adapter._fail_all(RuntimeError("WeCom connection interrupted"))

        return _FakeWS()

    def test_marker_set_excludes_ambiguous_ack_drops(self):
        """Contract: only server-rejected (846609) and pre-write refusals are provably
        undelivered; timeouts and post-write ack drops ("connection interrupted" /
        "websocket closed") are ambiguous and must never be transient."""
        is_transient = WeComAdapter._is_transient_send_error
        # provably undelivered → transient (safe to retry)
        assert is_transient("WeCom errcode 846609: aibot websocket not subscribed")
        assert is_transient("WeCom websocket is not connected")
        # ambiguous (may have delivered) → NOT transient
        assert not is_transient("WeCom connection interrupted")
        assert not is_transient("WeCom websocket closed")

    @pytest.mark.asyncio
    async def test_proactive_ack_drop_after_write_fails_closed_once(self, fast_redelivery):
        """Proactive path: frame written, ack dropped → fails closed with the original error,
        never ``send_path_degraded`` / retryable, and the frame reaches the wire exactly once."""
        adapter = _make_adapter()
        adapter._last_chat_req_ids.clear()  # no cached req_id + DM → proactive path
        written: list = []
        adapter._ws = self._make_dropping_ws(adapter, written)

        result = await adapter._send_inner(CHAT_ID, "maybe already on the wire")

        assert len(written) == 1  # written once — no duplicate frame
        assert not result.success
        assert result.error != "send_path_degraded"  # ambiguous → not the retry-ledger token
        assert result.retryable is not True
        assert "connection interrupted" in (result.error or "")

    @pytest.mark.asyncio
    async def test_passive_ack_drop_after_write_falls_back_once(self, fast_redelivery):
        """Passive path: frame written, ack dropped → not transient, so it takes the ORIGINAL
        passive→proactive fallback exactly once instead of transient-retrying the same payload."""
        adapter = _make_adapter()  # cached REQ_ID → passive path
        written: list = []
        adapter._ws = self._make_dropping_ws(adapter, written)
        proactive_calls: list[str] = []

        async def _proactive(chat_id: str, content: str):
            proactive_calls.append(content)
            return {"errcode": 0, "errmsg": "ok", "headers": {"req_id": "sent-3"}}

        adapter._send_proactive_markdown = _proactive

        result = await adapter._send_inner(CHAT_ID, "passive then drop")

        assert result.success, result.error
        assert len(written) == 1  # the passive frame was written once, never re-sent
        assert len(proactive_calls) == 1  # single fallback — no duplicate from transient retries
