"""Behavior-contract tests for the WeCom proactive-send fallback fixes.

Two production bug classes are covered, each toggle-validated so the test
reproduces the failure when the fix is disabled and passes when it is
enabled — proving the assertions track behavior, not a frozen snapshot.

Fix A — a cached per-chat ``req_id`` is only usable while its passive-reply
window is still live.  WeCom's req_id window is ~6 min (errcode 846604); a
scheduled job (e.g. an 08:00 cron report) reusing a req_id recorded hours or
days earlier always fails, and the consumer falls back only **after** burning a
guaranteed-fail passive attempt.  A chat already known-expired, or whose req_id
was recorded longer ago than the req_id window (``REQ_ID_MAX_AGE_SECONDS``,
falling back to WeCom's documented ~6 min), must resolve to ``None``
so the caller goes straight to the proactive path.  Toggle =
``adapter._last_chat_req_ids_ts`` (absent → old unconditional behaviour).

Fix B — the proactive APP_CMD_SEND fallback must retry.  By the time the
proactive path runs the passive attempt has already failed, so a single
transient WeCom/network hiccup silently drops the whole report.  Toggle =
``adapter._send_proactive_with_retry`` vs the old bare
``_send_proactive_markdown`` call.

These drive the REAL ``WeComAdapter._cached_reply_req_id`` and the REAL
``send()`` fallback control flow with only the wire-level
``_send_proactive_markdown`` / stream seams faked, so the actual branch logic
runs.  Assertions read observable adapter state: the resolved req_id, how many
proactive attempts actually reached the wire, and the resulting ``SendResult``.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.wecom.adapter import WeComAdapter


CHAT_ID = "chat-reqid"
REQ_ID = "req-reqid"

# WeCom's documented passive-reply window is ~6 min (errcode 846604).
# Read defensively so this module still imports against unfixed code and the
# toggle produces a BEHAVIOURAL failure, not a collection error.
WINDOW_SECONDS = getattr(
    __import__("plugins.platforms.wecom.adapter", fromlist=["x"]),
    "WINDOW_SECONDS",
    300.0,
)


def _make_adapter() -> WeComAdapter:
    """Real adapter with only the wire-level proactive sender faked."""
    return WeComAdapter(PlatformConfig(enabled=True, extra={}))


def _stamp(adapter: WeComAdapter, chat_id: str, when: float) -> None:
    """Record when a chat's req_id arrived.

    Uses setdefault so the test stays runnable against unfixed code (where the
    attribute does not exist), turning the toggle into a behavioural failure
    rather than an AttributeError.
    """
    adapter.__dict__.setdefault("_last_chat_req_ids_ts", {})[chat_id] = when


# ===========================================================================
# Fix A — a cached req_id is only valid inside its passive-reply window
# ===========================================================================


class TestCachedReqIdWindow:
    """A stale req_id must not be handed to a guaranteed-fail passive attempt."""

    @pytest.mark.asyncio
    async def test_fresh_req_id_is_returned(self):
        """FIX ENABLED: a just-recorded req_id is inside the window → returned.

        Post-fix contract: the passive path is still preferred while it can
        possibly work, so a freshly received inbound message is answered
        passively (no extra proactive call).
        """
        adapter = _make_adapter()
        try:
            adapter._last_chat_req_ids[CHAT_ID] = REQ_ID
            _stamp(adapter, CHAT_ID, time.time())

            assert adapter._cached_reply_req_id(CHAT_ID, None) == REQ_ID
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_stale_req_id_is_dropped(self):
        """FIX ENABLED: a req_id older than the window → None (proactive path).

        This is the cron-report case: the req_id was recorded when the user
        last spoke, hours/days before the scheduled send.  Returning it burns a
        passive attempt that always fails with 846604.
        """
        adapter = _make_adapter()
        try:
            adapter._last_chat_req_ids[CHAT_ID] = REQ_ID
            _stamp(adapter, CHAT_ID, time.time() - WINDOW_SECONDS - 60)

            assert adapter._cached_reply_req_id(CHAT_ID, None) is None
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_known_expired_chat_is_dropped(self):
        """FIX ENABLED: a chat already marked stream-expired → None.

        ``_stream_expired_chats`` is populated when the reply flow is confirmed
        dead (846604/846609).  Re-offering its cached req_id contradicts that
        knowledge, so the chat must short-circuit to the proactive path.
        """
        adapter = _make_adapter()
        try:
            adapter._last_chat_req_ids[CHAT_ID] = REQ_ID
            _stamp(adapter, CHAT_ID, time.time())  # fresh on purpose
            adapter._stream_expired_chats.add(CHAT_ID)

            assert adapter._cached_reply_req_id(CHAT_ID, None) is None
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_no_recorded_timestamp_preserves_old_behaviour(self):
        """TOGGLE: with no timestamp recorded, the cached req_id is returned.

        Proves Fix A is gated on the recorded timestamp rather than being an
        unconditional refusal — a req_id that arrived without one is not aged
        out, so behaviour is unchanged for callers that never record a time.
        """
        adapter = _make_adapter()
        try:
            adapter._last_chat_req_ids[CHAT_ID] = REQ_ID

            assert adapter._cached_reply_req_id(CHAT_ID, None) == REQ_ID
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_explicit_reply_to_wins_over_age(self):
        """An explicit reply_to mapping is authoritative and never aged out.

        ``reply_to`` is a per-message instruction, not a chat-level cache; the
        window check must not silently drop it.
        """
        adapter = _make_adapter()
        try:
            adapter._reply_req_ids["msg-1"] = REQ_ID
            _stamp(adapter, CHAT_ID, time.time() - WINDOW_SECONDS - 60)
            adapter._stream_expired_chats.add(CHAT_ID)

            assert adapter._cached_reply_req_id(CHAT_ID, "msg-1") == REQ_ID
        finally:
            await adapter.disconnect()


# ===========================================================================
# Fix B — the proactive fallback retries transient failures
# ===========================================================================


class TestProactiveRetry:
    """A transient proactive-send failure must not silently drop the message."""

    @pytest.mark.asyncio
    async def test_transient_failure_then_success(self):
        """FIX ENABLED: first proactive attempt fails, retry succeeds.

        Post-fix contract: the message still lands.  Without the retry the
        single hiccup returned a failed SendResult and the report vanished.
        """
        adapter = _make_adapter()
        try:
            calls = {"n": 0}

            async def flaky(chat_id: str, content: str):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise asyncio.TimeoutError("transient")
                return {"errcode": 0}

            adapter._send_proactive_markdown = flaky

            result = await adapter._send_proactive_with_retry(CHAT_ID, "report")

            assert result == {"errcode": 0}
            assert calls["n"] == 2, (
                "a transient first failure must be retried, not returned as a "
                "final failure that silently drops a cron report"
            )
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_error_errcode_is_treated_as_failure(self):
        """An errcode payload is a failure even when no exception is raised.

        WeCom reports errors in-body; treating a non-zero errcode as success
        would mark a dropped report as sent.
        """
        adapter = _make_adapter()
        try:
            calls = {"n": 0}

            async def errcode_then_ok(chat_id: str, content: str):
                calls["n"] += 1
                if calls["n"] == 1:
                    return {"errcode": 846609, "errmsg": "subscription lost"}
                return {"errcode": 0}

            adapter._send_proactive_markdown = errcode_then_ok

            result = await adapter._send_proactive_with_retry(CHAT_ID, "report")

            assert result == {"errcode": 0}
            assert calls["n"] == 2
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_persistent_failure_raises_after_bounded_attempts(self):
        """FIX ENABLED: the retry is bounded — it gives up and raises.

        An unbounded retry would hang the sender on a genuinely dead channel;
        the caller (send()) must still see a failure it can report.
        """
        adapter = _make_adapter()
        try:
            calls = {"n": 0}

            async def always_fail(chat_id: str, content: str):
                calls["n"] += 1
                raise RuntimeError("down")

            adapter._send_proactive_markdown = always_fail

            with pytest.raises(RuntimeError):
                await adapter._send_proactive_with_retry(CHAT_ID, "report", attempts=3)

            assert calls["n"] == 3, "attempts must be bounded by the parameter"
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_retry_backoff_is_awaited(self, monkeypatch):
        """Retries are spaced, not hammered back-to-back.

        A rate-limited channel (846607) must not receive the retries at full
        speed, so the backoff sleep is asserted rather than assumed.
        """
        adapter = _make_adapter()
        try:
            slept: list[float] = []

            async def record_sleep(delay: float) -> None:
                slept.append(delay)

            monkeypatch.setattr(asyncio, "sleep", record_sleep)

            async def always_fail(chat_id: str, content: str):
                raise RuntimeError("down")

            adapter._send_proactive_markdown = always_fail

            with pytest.raises(RuntimeError):
                await adapter._send_proactive_with_retry(CHAT_ID, "report", attempts=3)

            assert len(slept) == 2, "one backoff between each pair of attempts"
            assert slept == sorted(slept) and slept[0] < slept[1], (
                "backoff must increase between attempts"
            )
        finally:
            await adapter.disconnect()


# ===========================================================================
# End-to-end: send() routes a stale-req_id chat through the retrying proactive path
# ===========================================================================


class TestSendFallsBackToProactive:
    """The real send() control flow, driven through the real cached-req_id gate."""

    @pytest.mark.asyncio
    async def test_stale_req_id_chat_sends_proactively_without_passive_attempt(self):
        """A chat whose only req_id is stale goes straight to the proactive path.

        End-to-end contract: with the req_id aged out, send() must NOT attempt a
        passive reply (that attempt is a guaranteed 846604) and must deliver via
        the proactive path exactly once — no wasted round-trip, no lost report.
        """
        adapter = _make_adapter()
        try:
            adapter._last_chat_req_ids[CHAT_ID] = REQ_ID
            _stamp(adapter, CHAT_ID, time.time() - WINDOW_SECONDS - 60)

            passive_attempts = {"n": 0}

            async def passive_should_not_run(*args, **kwargs):
                passive_attempts["n"] += 1
                raise AssertionError("passive path must not be attempted for a stale req_id")

            adapter._send_reply_markdown = passive_should_not_run
            proactive = AsyncMock(return_value={"errcode": 0})
            adapter._send_proactive_markdown = proactive

            result = await adapter._send_inner(CHAT_ID, "08:00 巡检报告")

            assert result.success is True
            assert passive_attempts["n"] == 0, (
                "a stale req_id must be filtered before the passive attempt, "
                "not discovered by failing it"
            )
            assert proactive.await_count == 1
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_proactive_retry_rescues_send_after_transient_failure(self):
        """End-to-end: a transient proactive failure still delivers via retry."""
        adapter = _make_adapter()
        try:
            calls = {"n": 0}

            async def flaky(chat_id: str, content: str):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise asyncio.TimeoutError("transient")
                return {"errcode": 0}

            adapter._send_proactive_markdown = flaky

            result = await adapter._send_inner(CHAT_ID, "08:00 巡检报告")

            assert result.success is True, (
                "a single transient hiccup must not turn a delivered report "
                "into a silent failure"
            )
            assert calls["n"] == 2
        finally:
            await adapter.disconnect()
