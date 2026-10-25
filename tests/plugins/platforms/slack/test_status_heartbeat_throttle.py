"""Slack status heartbeat throttling.

The base ``_keep_typing`` heartbeat (``gateway/platforms/base.py``) ticks every 2s for every
platform. For Slack each tick is an ``agents.sessions.setStatus`` / ``assistant.threads.setStatus``
call — Slack's Tier 3 limit (~50+/min per workspace) — so a few concurrent turns exceed it, and the
resulting throttling can reject the final ``active`` clear too, leaving "Working…" on the message
until Slack's own ~90 min status expiry. See ``plugins/platforms/slack/adapter_status.py``:
``_status_send_due`` throttles ``send_typing`` against what Slack last accepted for the target;
``stop_typing``'s clear is never throttled and retries a rate limit within one bounded budget.

Non-Slack platforms are unaffected: this file touches no code in ``gateway/platforms/base.py``, so
the base 2s-per-tick cadence for every other adapter is unchanged (also exercised directly here and
throughout ``tests/gateway/test_keep_typing_timeout.py``).
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import plugins.platforms.slack.adapter as _slack_mod
import plugins.platforms.slack.adapter_status as _status_mod
from gateway.config import PlatformConfig
from gateway.platforms.base_status_anchor import STATUS_ANCHOR_MESSAGE_ID, STATUS_OWNER_KEY
from plugins.platforms.slack.adapter import SlackAdapter
from tests.gateway.slack_status_clock import SimulatedTurnClock
from tests.gateway.test_keep_typing_timeout import _StubAdapter
from tests.waiting import async_wait_until


def _ratelimit_error(retry_after: str) -> Exception:
    exc = Exception("ratelimited")
    exc.response = SimpleNamespace(status_code=429, headers={"Retry-After": retry_after})
    return exc


def _adapter(**extra) -> SlackAdapter:
    config = PlatformConfig(enabled=True, token="xoxb-fake-token", extra=extra)
    a = SlackAdapter(config)
    a._app = MagicMock()
    a._app.client = AsyncMock()
    return a


async def _three_minute_turn(a: SlackAdapter, clock: SimulatedTurnClock) -> None:
    """The real ``_keep_typing`` loop over ticks t=0, 2, ..., 180 (``clock.turn_seconds``); the
    timeout only guards against a hang — the turn ends by tick count in ``SimulatedTurnClock``."""
    await asyncio.wait_for(
        SlackAdapter._keep_typing(a, "C123", interval=2.0, metadata={"thread_id": "parent_ts"},
                                  stop_event=None),
        timeout=30.0)
    assert clock.now == 180.0 and len(clock.sleeps) == 90  # the whole turn really ran


@pytest.fixture(autouse=True)
def _sessions_api_supported_by_default():
    prev = _slack_mod._AGENT_SESSIONS_SUPPORTED
    _slack_mod._AGENT_SESSIONS_SUPPORTED = True
    yield
    _slack_mod._AGENT_SESSIONS_SUPPORTED = prev


class TestSendTypingThrottle:
    """Per status target: first set, a changed legacy label (<=30s), or a 60s keep-alive —
    never a plain re-send of an unchanged sessions-API `processing` payload."""

    @pytest.mark.asyncio
    async def test_first_set_always_sends(self):
        a = _adapter()
        await a.send_typing("C123", metadata={"thread_id": "parent_ts"})
        a._app.client.agents_sessions_setStatus.assert_called_once_with(
            channel_id="C123", thread_ts="parent_ts", status="processing")

    @pytest.mark.asyncio
    async def test_repeat_tick_same_instant_is_throttled(self):
        a = _adapter()
        md = {"thread_id": "parent_ts"}
        await a.send_typing("C123", metadata=md)
        a._app.client.agents_sessions_setStatus.reset_mock()
        await a.send_typing("C123", metadata=md)  # same instant, unchanged payload: throttled
        a._app.client.agents_sessions_setStatus.assert_not_called()

    @pytest.mark.asyncio
    async def test_sessions_api_3_minute_turn_sends_initial_plus_60s_keepalives(self, monkeypatch):
        """A naive 2s-per-tick heartbeat over a 3-minute turn would be 91 calls; throttled, it is
        exactly the t=0 set, the keep-alives at t=60/120/180, and the turn-end clear."""
        a = _adapter()
        clock = SimulatedTurnClock(turn_seconds=180.0).install(monkeypatch)
        sent = clock.record(a._app.client.agents_sessions_setStatus)
        await _three_minute_turn(a, clock)
        assert sent == [(0.0, "processing"), (60.0, "processing"), (120.0, "processing"),
                        (180.0, "processing"), (180.0, "active")]
        a._app.client.assistant_threads_setStatus.assert_not_called()

    @pytest.mark.asyncio
    async def test_legacy_label_updates_every_30s_over_a_long_turn(self, monkeypatch):
        """The legacy free-text elapsed label ("still working… (2m03s)") changes every tick past
        30s elapsed, but re-calls the legacy API exactly every 30s."""
        _slack_mod._AGENT_SESSIONS_SUPPORTED = False
        a = _adapter()
        clock = SimulatedTurnClock(turn_seconds=180.0).install(monkeypatch)
        sent = clock.record(a._app.client.assistant_threads_setStatus)
        await _three_minute_turn(a, clock)
        labels, clear = sent[:-1], sent[-1]
        assert [at for at, _ in labels] == [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
        assert labels[0][1] == "is thinking..." and "3m00s" in labels[-1][1]
        assert len({label for _, label in labels}) == len(labels)
        assert clear == (180.0, "")
        a._app.client.agents_sessions_setStatus.assert_not_called()


class TestThrottleFollowsTheAcceptedEndpoint:
    """A session-less thread: the sessions API rejects it, the legacy fallback accepts the free
    text. The throttle must compare that free text, not the constant sessions ``processing``."""

    @staticmethod
    def _sessionless_adapter() -> SlackAdapter:
        a = _adapter()
        a._app.client.agents_sessions_setStatus = AsyncMock(side_effect=Exception("not_in_session"))
        return a

    @staticmethod
    def _legacy_labels(a: SlackAdapter) -> list:
        return [c.kwargs["status"] for c in a._app.client.assistant_threads_setStatus.call_args_list]

    @pytest.mark.asyncio
    async def test_live_status_change_sends_immediately_via_legacy(self, monkeypatch):
        a = self._sessionless_adapter()
        clock = SimulatedTurnClock().install(monkeypatch)
        md = {"thread_id": "parent_ts"}
        await a.send_typing("C123", metadata=md)
        clock.now += 2
        a._status_text["C123"] = "reading files…"
        await a.send_typing("C123", metadata=md)
        assert self._legacy_labels(a) == ["is thinking...", "reading files…"]

    @pytest.mark.asyncio
    async def test_elapsed_label_follows_the_30s_cadence_via_legacy(self, monkeypatch):
        a = self._sessionless_adapter()
        clock = SimulatedTurnClock().install(monkeypatch)
        md = {"thread_id": "parent_ts"}
        for _ in range(23):  # t = 0, 2, ..., 44: the label changes every tick from t=30
            await a.send_typing("C123", metadata=md)
            clock.now += 2
        labels = self._legacy_labels(a)
        assert labels[0] == "is thinking..."
        assert len(labels) == 2 and "30s" in labels[1], labels


class TestHeartbeatStateSurvivesEviction:
    """``_active_status_threads`` evicts at its cap; the throttle state of targets whose
    ``_keep_typing`` loop is still ticking must not go with it."""

    @pytest.mark.asyncio
    async def test_more_live_targets_than_the_cap_do_not_each_resend_every_tick(self, monkeypatch):
        a = _adapter()
        a._ACTIVE_STATUS_THREADS_MAX = 4
        clock = SimulatedTurnClock().install(monkeypatch)
        targets = [{"thread_id": f"17000000{i:02d}.000100"} for i in range(10)]
        for md in targets:
            await a.send_typing("C123", metadata=md)
        assert a._app.client.agents_sessions_setStatus.call_count == len(targets)
        assert len(a._active_status_threads) <= 4  # the tracked map did evict
        a._app.client.agents_sessions_setStatus.reset_mock()
        for _ in range(5):  # five more 2s ticks of every live loop
            clock.now += 2
            for md in targets:
                await a.send_typing("C123", metadata=md)
        a._app.client.agents_sessions_setStatus.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_typing_drops_the_heartbeat_state(self):
        a = _adapter()
        md = {"thread_id": "parent_ts"}
        await a.send_typing("C123", metadata=md)
        await a.stop_typing("C123", metadata=md)
        assert a._status_heartbeats == {}


class TestHeartbeatRecordsOnlyDelivered:
    """Only a delivered set starts the keep-alive / label window; a failed or abandoned one is
    retried after the short backoff, not after the 60s keep-alive."""

    @pytest.mark.asyncio
    async def test_failed_first_set_is_retried_after_the_short_backoff(self, monkeypatch):
        a = _adapter()
        setter = a._app.client.agents_sessions_setStatus = AsyncMock(
            side_effect=[_ratelimit_error("1"), None])
        a._app.client.assistant_threads_setStatus = AsyncMock(side_effect=_ratelimit_error("1"))
        clock = SimulatedTurnClock().install(monkeypatch)
        md = {"thread_id": "parent_ts"}
        for _ in range(4):  # t = 0, 2, 4, 6
            await a.send_typing("C123", metadata=md)
            clock.now += 2
        assert setter.call_count == 2  # t=0 failed; t=2,4 back off; t=6 retried and delivered

    @pytest.mark.asyncio
    async def test_set_abandoned_by_the_tick_timeout_is_retried_after_the_short_backoff(self, monkeypatch):
        a = _adapter()
        hang = asyncio.Event()
        calls = []

        async def setter(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                await hang.wait()  # _keep_typing's per-tick wait_for cancels this one

        a._app.client.agents_sessions_setStatus = setter
        clock = SimulatedTurnClock().install(monkeypatch)
        md = {"thread_id": "parent_ts"}
        tick = asyncio.create_task(a.send_typing("C123", metadata=md))
        await async_wait_until(lambda: calls, what="first status set in flight")
        tick.cancel()  # what _keep_typing's wait_for does to a tick past its budget
        with pytest.raises(asyncio.CancelledError):
            await tick
        for _ in range(3):  # t = 2, 4, 6
            clock.now += 2
            await a.send_typing("C123", metadata=md)
        assert len(calls) == 2


class TestClearAlwaysSentAndRetried:
    """``stop_typing``'s clear is never throttled, and retries a Slack rate-limit response after
    ``Retry-After`` within ONE 10s budget for the whole clear, so it isn't dropped itself."""

    @pytest.mark.asyncio
    async def test_clear_sent_even_right_after_a_throttled_tick(self):
        a = _adapter()
        md = {"thread_id": "parent_ts"}
        await a.send_typing("C123", metadata=md)
        await a.send_typing("C123", metadata=md)  # throttled: no call
        a._app.client.agents_sessions_setStatus.reset_mock()
        await a.stop_typing("C123", metadata=md)
        a._app.client.agents_sessions_setStatus.assert_called_once_with(
            channel_id="C123", thread_ts="parent_ts", status="active")

    @pytest.mark.asyncio
    async def test_ratelimited_clear_is_retried_once_after_retry_after(self, monkeypatch):
        a = _adapter()
        a._app.client.agents_sessions_setStatus = AsyncMock(
            side_effect=[None, _ratelimit_error("0.5"), None])
        clock = SimulatedTurnClock().install(monkeypatch)
        md = {"thread_id": "parent_ts"}
        await a.send_typing("C123", metadata=md)  # call #1: succeeds
        await a.stop_typing("C123", metadata=md)  # call #2: 429, call #3: retried clear succeeds
        assert a._app.client.agents_sessions_setStatus.call_count == 3
        assert clock.sleeps == [0.5]
        last = a._app.client.agents_sessions_setStatus.call_args_list[-1]
        assert last.kwargs == {"channel_id": "C123", "thread_ts": "parent_ts", "status": "active"}

    @pytest.mark.asyncio
    async def test_retry_after_past_the_budget_waits_it_out_and_does_not_retry(self, monkeypatch):
        """A huge Retry-After is cut to the 10s budget, and once that wait has spent it no retry
        (and no legacy fallback) starts at or past the deadline."""
        a = _adapter()
        setter = a._app.client.agents_sessions_setStatus = AsyncMock(
            side_effect=[None, _ratelimit_error("999"), None])
        clock = SimulatedTurnClock().install(monkeypatch)
        md = {"thread_id": "parent_ts"}
        await a.send_typing("C123", metadata=md)
        await a.stop_typing("C123", metadata=md)
        assert clock.sleeps == [10.0] and clock.now == 10.0
        assert [c.kwargs["status"] for c in setter.call_args_list] == ["processing", "active"]
        a._app.client.assistant_threads_setStatus.assert_not_called()

    @pytest.mark.asyncio
    async def test_queued_flat_clears_share_the_one_spent_budget(self, monkeypatch):
        """The opener's clear spends the whole budget on a Retry-After: 999 wait, so the same
        stop's clear of the ``/queue`` follow-up's flat status is not started past it — and, never
        attempted, it is not deduped either: the next stop sends it."""
        a = _adapter(reply_in_thread=False)
        setter = a._app.client.agents_sessions_setStatus = AsyncMock(
            side_effect=[None, None, _ratelimit_error("999"), None])
        clock = SimulatedTurnClock().install(monkeypatch)
        opener = {STATUS_ANCHOR_MESSAGE_ID: "1761300000.000100", STATUS_OWNER_KEY: "session-1"}
        queued = {STATUS_ANCHOR_MESSAGE_ID: "1761300050.000300", STATUS_OWNER_KEY: "session-1"}
        await a.send_typing("C123", metadata=opener)
        await a.send_typing("C123", metadata=queued)
        await a.stop_typing("C123", metadata=opener)
        assert clock.now == 10.0
        calls = [(c.kwargs["thread_ts"], c.kwargs["status"]) for c in setter.call_args_list]
        assert calls == [("1761300000.000100", "processing"), ("1761300050.000300", "processing"),
                         ("1761300000.000100", "active")]
        await a.stop_typing("C123", metadata=queued)
        assert setter.call_args_list[-1].kwargs["thread_ts"] == "1761300050.000300"
        assert setter.call_args_list[-1].kwargs["status"] == "active"

    @pytest.mark.asyncio
    async def test_the_retried_clear_request_is_cut_off_at_the_deadline(self, monkeypatch):
        """Real time: a retry that hangs is abandoned when the budget runs out, not awaited."""
        monkeypatch.setattr(_status_mod, "_STATUS_CLEAR_BUDGET", 0.5)
        a = _adapter()
        outcomes = []

        async def setter(**kwargs):
            if kwargs["status"] == "processing":
                return
            if not outcomes:
                outcomes.append("ratelimited")
                raise _ratelimit_error("0.1")
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                outcomes.append("cut off")
                raise

        a._app.client.agents_sessions_setStatus = setter
        md = {"thread_id": "parent_ts"}
        await a.send_typing("C123", metadata=md)
        await asyncio.wait_for(a.stop_typing("C123", metadata=md), timeout=5.0)
        assert outcomes == ["ratelimited", "cut off"]

    @pytest.mark.asyncio
    async def test_ratelimited_threaded_clear_with_fallback_fits_one_budget(self, monkeypatch):
        """Sessions rate-limits the clear twice, then the legacy fallback is rate-limited too: the
        whole clear (both endpoints, every retry wait) finishes within one 10s budget, not 10s per
        endpoint."""
        a = _adapter()
        a._app.client.agents_sessions_setStatus = AsyncMock(
            side_effect=[_ratelimit_error("8"), _ratelimit_error("8")])
        legacy = a._app.client.assistant_threads_setStatus = AsyncMock(
            side_effect=[_ratelimit_error("8"), None])
        clock = SimulatedTurnClock().install(monkeypatch)
        await a.stop_typing("C123", metadata={"thread_id": "parent_ts"})
        assert clock.sleeps == [8.0, 2.0] and clock.now == 10.0  # legacy's 8s wait cut to the 2s left
        assert legacy.call_args_list == [
            ((), {"channel_id": "C123", "thread_ts": "parent_ts", "status": ""})]  # no retry at 10s

    @pytest.mark.asyncio
    async def test_repeated_clears_of_one_target_do_not_repeat_the_wait(self, monkeypatch):
        """A turn end clears the same target several times (``_keep_typing``'s finally and
        ``_stop_typing_refresh``'s attempts); only the first pays a rate-limit wait."""
        a = _adapter()
        setter = a._app.client.agents_sessions_setStatus = AsyncMock(
            side_effect=[None, _ratelimit_error("4"), None])
        a._app.client.assistant_threads_setStatus = AsyncMock(side_effect=_ratelimit_error("4"))
        clock = SimulatedTurnClock().install(monkeypatch)
        md = {"thread_id": "parent_ts"}
        await a.send_typing("C123", metadata=md)
        await a._stop_typing_refresh("C123", None, metadata=md, stop_attempts=3)
        await a.stop_typing("C123", metadata=md)
        assert clock.now == 4.0
        assert setter.call_count == 3

    @pytest.mark.asyncio
    async def test_a_status_set_again_after_the_clear_is_cleared_again(self):
        a = _adapter()
        md = {"thread_id": "parent_ts"}
        await a.send_typing("C123", metadata=md)
        await a.stop_typing("C123", metadata=md)
        await a.send_typing("C123", metadata=md)  # a late tick re-set the status
        await a.stop_typing("C123", metadata=md)
        statuses = [c.kwargs["status"] for c in a._app.client.agents_sessions_setStatus.call_args_list]
        assert statuses == ["processing", "active", "processing", "active"]

    @pytest.mark.asyncio
    async def test_a_cancelled_set_after_the_clear_still_gets_cleared(self):
        """A new turn's set whose response never arrives (cancelled by the tick timeout) may
        still have landed on Slack: the quick turn-end clear right after must be sent, not
        deduped against the previous turn's clear."""
        a = _adapter()
        statuses = []

        async def setter(**kwargs):
            statuses.append(kwargs["status"])
            if len(statuses) == 3:
                await asyncio.Event().wait()

        a._app.client.agents_sessions_setStatus = setter
        md = {"thread_id": "parent_ts"}
        await a.send_typing("C123", metadata=md)
        await a.stop_typing("C123", metadata=md)
        tick = asyncio.create_task(a.send_typing("C123", metadata=md))
        await async_wait_until(lambda: len(statuses) == 3, what="new turn's set in flight")
        tick.cancel()
        with pytest.raises(asyncio.CancelledError):
            await tick
        await a.stop_typing("C123", metadata=md)
        assert statuses == ["processing", "active", "processing", "active"]

    @pytest.mark.asyncio
    async def test_non_ratelimit_clear_failure_is_not_retried(self, monkeypatch):
        """Only a rate limit retries; any other clear failure behaves as before (swallowed,
        legacy fallback attempted)."""
        a = _adapter()
        a._app.client.agents_sessions_setStatus = AsyncMock(
            side_effect=[None, Exception("invalid_arguments")])
        clock = SimulatedTurnClock().install(monkeypatch)
        md = {"thread_id": "parent_ts"}
        await a.send_typing("C123", metadata=md)
        await a.stop_typing("C123", metadata=md)
        assert clock.sleeps == []
        assert a._app.client.agents_sessions_setStatus.call_count == 2


class TestQueuedAnchorSurvivesAnExhaustedClearBudget:
    """A queued ``/queue`` anchor whose clear was never attempted
    because the opener's clear already spent the whole shared budget must stay tracked — not be
    dropped as if it had been cleared — so the next stop call for it still sends the clear."""

    @pytest.mark.asyncio
    async def test_rate_limited_opener_clear_leaves_the_queued_anchor_tracked(self, monkeypatch):
        a = _adapter(reply_in_thread=False)
        setter = a._app.client.agents_sessions_setStatus = AsyncMock(
            side_effect=[None, None, _ratelimit_error("999"), None])
        clock = SimulatedTurnClock().install(monkeypatch)
        opener = {STATUS_ANCHOR_MESSAGE_ID: "1761300000.000100", STATUS_OWNER_KEY: "session-1"}
        queued = {STATUS_ANCHOR_MESSAGE_ID: "1761300050.000300", STATUS_OWNER_KEY: "session-1"}
        await a.send_typing("C123", metadata=opener)
        await a.send_typing("C123", metadata=queued)
        await a.stop_typing("C123", metadata=opener)
        assert clock.now == 10.0  # the opener's clear ate the whole budget on a Retry-After
        # Never attempted: the queued anchor must still be tracked, not dropped.
        assert len([k for k in a._active_status_threads if k[2] == "1761300050.000300"]) == 1
        # The next stop call for it really clears it (not a no-op / dedup skip).
        await a.stop_typing("C123", metadata=queued)
        assert setter.call_args_list[-1].kwargs == {
            "channel_id": "C123", "thread_ts": "1761300050.000300", "status": "active"}
        assert not any(k[2] == "1761300050.000300" for k in a._active_status_threads)

    @pytest.mark.asyncio
    async def test_failed_queued_clear_is_retried_by_the_next_opener_stop(self):
        a = _adapter(reply_in_thread=False)
        queued_ts = "1761300050.000300"
        fail_queued_clear_once = True

        async def setter(**kwargs):
            nonlocal fail_queued_clear_once
            if (kwargs.get("thread_ts"), kwargs.get("status")) == (queued_ts, "active") and fail_queued_clear_once:
                fail_queued_clear_once = False
                raise Exception("internal_error")

        a._app.client.agents_sessions_setStatus = AsyncMock(side_effect=setter)
        opener = {STATUS_ANCHOR_MESSAGE_ID: "1761300000.000100", STATUS_OWNER_KEY: "session-1"}
        queued = {STATUS_ANCHOR_MESSAGE_ID: queued_ts, STATUS_OWNER_KEY: "session-1"}
        await a.send_typing("C123", metadata=opener)
        await a.send_typing("C123", metadata=queued)
        await a.stop_typing("C123", metadata=opener)
        assert not fail_queued_clear_once
        assert any(k[2] == queued_ts for k in a._active_status_threads)

        await a.stop_typing("C123", metadata=opener)
        queued_clears = [
            c.kwargs
            for c in a._app.client.agents_sessions_setStatus.call_args_list
            if c.kwargs.get("thread_ts") == queued_ts and c.kwargs.get("status") == "active"]
        assert len(queued_clears) == 2
        assert not any(k[2] == queued_ts for k in a._active_status_threads)


class TestClearDedupReleasedWhenNoEndpointAccepted:
    """The dedup marker set before a clear attempt must not survive
    a clear no endpoint accepted (e.g. both fail with a transient non-429 error), or the next
    stop call would skip the retry and leave `processing` stuck on Slack's side."""

    @pytest.mark.asyncio
    async def test_both_endpoints_failing_is_not_deduped_on_the_next_stop(self):
        a = _adapter()
        setter = a._app.client.agents_sessions_setStatus = AsyncMock(
            side_effect=Exception("internal_error"))
        a._app.client.assistant_threads_setStatus = AsyncMock(side_effect=Exception("internal_error"))
        md = {"thread_id": "parent_ts"}
        await a.send_typing("C123", metadata=md)
        setter.reset_mock()
        await a.stop_typing("C123", metadata=md)
        assert setter.call_count == 1  # first clear attempt: not accepted by either endpoint
        await a.stop_typing("C123", metadata=md)
        assert setter.call_count == 2  # not deduped: the clear is sent again

    @pytest.mark.asyncio
    async def test_a_delivered_clear_is_still_deduped(self):
        a = _adapter()
        setter = a._app.client.agents_sessions_setStatus
        md = {"thread_id": "parent_ts"}
        await a.send_typing("C123", metadata=md)
        setter.reset_mock()
        await a.stop_typing("C123", metadata=md)  # delivered
        assert setter.call_count == 1
        await a.stop_typing("C123", metadata=md)  # deduped: no new call within the 10s window
        assert setter.call_count == 1


class TestNonSlackPlatformsUnchanged:
    """The Slack-only throttle lives in the Slack adapter's ``send_typing``; the base
    ``_keep_typing`` heartbeat other platforms use keeps refreshing every ``interval``."""

    @pytest.mark.asyncio
    async def test_generic_adapter_keeps_ticking_every_interval(self):
        adapter = _StubAdapter()
        ticks = []

        async def recording_send_typing(chat_id, metadata=None):
            ticks.append(chat_id)

        adapter.send_typing = recording_send_typing
        stop_event = asyncio.Event()
        task = asyncio.create_task(
            adapter._keep_typing("123", interval=0.02, stop_event=stop_event))
        await async_wait_until(lambda: len(ticks) >= 3, what="generic adapter typing ticks")
        stop_event.set()
        await asyncio.wait_for(task, timeout=2.0)
        assert len(ticks) >= 3
