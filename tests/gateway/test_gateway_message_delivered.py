"""``gateway_message_delivered`` observer hook (#64176 delivery half): payload contract, dispatch,
and the kind="final" fire sites (non-streamed ledgered send + confirmed streamed final)."""

import asyncio
import contextvars
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway import delivery_hooks as dh
from gateway.platforms.base import BasePlatformAdapter, Platform, PlatformConfig, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run_turn import GatewayTurnMixin


@pytest.fixture
def fired():
    """Capture dispatched payloads; pretend a plugin subscribes."""
    calls = []

    def _sync(name, **kw):
        calls.append((name, kw))
        return []

    async def _async(name, **kw):
        calls.append((name, kw))
        return []

    with patch.object(dh, "_has_subscribers", return_value=True), \
         patch("hermes_cli.lifecycle.invoke_hook", _sync), \
         patch("hermes_cli.lifecycle.ainvoke_hook", _async), \
         patch.object(dh, "_active_profile", return_value="jarvis"):
        yield calls


# --------------------------------------------------------------------------- payload contract


class TestDeliveredMessageIds:
    def test_prefers_the_adapters_ordered_list(self):
        result = SendResult(success=True, message_id="1", raw_response={"message_ids": ["1", "2", "3"]})
        assert dh.delivered_message_ids(result) == ["1", "2", "3"]

    def test_falls_back_to_continuation_plus_message_id(self):
        result = SendResult(success=True, message_id="3", continuation_message_ids=("1", "2"))
        assert dh.delivered_message_ids(result) == ["1", "2", "3"]

    def test_dedupes_and_drops_sentinels_and_empties(self):
        result = SendResult(success=True, message_id="2", continuation_message_ids=("2", None, "__no_edit__"))
        assert dh.delivered_message_ids(result) == ["2"]
        assert dh.delivered_message_ids(SendResult(success=True)) == []

    def test_accepts_dict_results_and_ints(self):
        assert dh.delivered_message_ids({"message_id": 42}) == ["42"]


class TestBuildPayload:
    def test_normalized_bounded_fields(self):
        with patch.object(dh, "_active_profile", return_value="team-b"):
            payload = dh.build_payload(
                kind="cron", platform=Platform.TELEGRAM, chat_id=-100, thread_id=7,
                message_ids=["5", "6", "6", "x" * 500], text="t" * 10_000, job_id="job-1")
        assert payload["platform"] == "telegram"
        assert payload["chat_id"] == "-100" and payload["thread_id"] == "7"
        assert payload["message_ids"][:2] == ["5", "6"] and len(payload["message_ids"][2]) == 128
        assert payload["last_message_id"] == payload["message_ids"][-1]
        assert len(payload["text"]) == 8192
        assert payload["job_id"] == "job-1" and payload["streamed"] is False
        assert payload["session_key"] is None and payload["profile"] == "team-b"

    def test_unknown_values_are_none_not_guessed(self):
        payload = dh.build_payload(kind="final", platform="Discord", chat_id="1")
        assert payload["message_ids"] == [] and payload["last_message_id"] is None
        assert payload["thread_id"] is None and payload["text"] is None and payload["platform"] == "discord"


class TestDispatch:
    def test_no_subscriber_short_circuits_before_building(self):
        async def main():
            dh.notify_message_delivered(kind="final", platform="telegram", chat_id="1")

        with patch.object(dh, "_has_subscribers", return_value=False), \
             patch.object(dh, "build_payload") as build:
            dh.notify_message_delivered_from_thread(None, kind="cron", platform="telegram", chat_id="1")
            asyncio.run(main())
        build.assert_not_called()

    def test_callback_failure_never_raises(self):
        async def main():
            dh.notify_message_delivered(kind="final", platform="telegram", chat_id="1")
            await dh.wait_for_pending_notifications()

        def boom(*a, **k):
            raise RuntimeError("plugin exploded")

        with patch.object(dh, "_has_subscribers", return_value=True), \
             patch("hermes_cli.lifecycle.invoke_hook", boom), \
             patch("hermes_cli.lifecycle.ainvoke_hook", AsyncMock(side_effect=RuntimeError("x"))):
            dh.notify_message_delivered_from_thread(None, kind="cron", platform="telegram", chat_id="1")
            asyncio.run(main())

    def test_no_running_loop_dispatches_synchronously(self, fired):
        dh.notify_message_delivered_from_thread(None, kind="cron", platform="telegram", chat_id="1")
        assert [name for name, _ in fired] == ["gateway_message_delivered"]

    def test_thread_dispatch_runs_on_the_gateway_loop_with_the_callers_context(self):
        """Async callbacks must share the loop that owns the adapters' clients, and see the
        caller's profile scope (contextvars travel with the scheduled callback)."""
        scope = contextvars.ContextVar("scope", default=None)
        loop = asyncio.new_event_loop()
        runner = threading.Thread(target=loop.run_forever, daemon=True)
        runner.start()
        seen, done = {}, threading.Event()

        async def callback(name, **kw):
            seen.update(loop=asyncio.get_running_loop(), scope=scope.get(), thread=threading.current_thread())
            done.set()
            return []

        def cron_thread():
            scope.set("profile-b")
            dh.notify_message_delivered_from_thread(loop, kind="cron", platform="telegram", chat_id="1")

        try:
            with patch.object(dh, "_has_subscribers", return_value=True), \
                 patch("hermes_cli.lifecycle.ainvoke_hook", callback):
                worker = threading.Thread(target=cron_thread)
                worker.start()
                worker.join(5)
                assert done.wait(5)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            runner.join(5)
            loop.close()
        assert seen["loop"] is loop and seen["thread"] is runner and seen["scope"] == "profile-b"

    def test_real_plugin_registration_receives_the_payload(self):
        from hermes_cli.plugins import VALID_HOOKS, PluginContext, PluginManager, PluginManifest
        from hermes_cli.plugins_dispatch import _HOOK_TIMEOUT_BOUNDED_HOOKS

        assert "gateway_message_delivered" in VALID_HOOKS
        assert "gateway_message_delivered" in _HOOK_TIMEOUT_BOUNDED_HOOKS
        manager = PluginManager()
        got = []
        ctx = PluginContext(PluginManifest(name="delivered-fixture", source="user"), manager)
        ctx.register_hook("gateway_message_delivered", lambda **kw: got.append(kw))
        manager.invoke_hook("gateway_message_delivered",
                            **dh.build_payload(kind="final", platform="telegram", chat_id="1", message_ids=["9"]))
        assert got and got[0]["last_message_id"] == "9" and got[0]["kind"] == "final"


# --------------------------------------------------------------------------- kind="final", non-streamed


class _Adapter(BasePlatformAdapter):
    def __init__(self, result):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
        self.result = result

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return self.result

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


def _send(adapter, *args, **kwargs):
    async def main():
        result = await adapter.send_final_ledgered(*args, **kwargs)
        await dh.wait_for_pending_notifications(timeout=5)
        return result

    return asyncio.run(main())


def _event(adapter):
    return MessageEvent(text="hi", message_type=MessageType.TEXT, message_id="in-1",
                        source=adapter.build_source(chat_id="8", user_id="u1", thread_id="77"))


class TestNonStreamedFinal:
    def test_fires_once_with_all_ids_after_a_successful_send(self, fired):
        adapter = _Adapter(SendResult(success=True, message_id="11", raw_response={"message_ids": ["11", "12"]}))
        result, _ = _send(adapter, _event(adapter), "sess-1", "final answer", {}, reply_to=None)
        assert result.success
        assert len(fired) == 1
        name, payload = fired[0]
        assert name == "gateway_message_delivered"
        assert payload["kind"] == "final" and payload["streamed"] is False
        assert payload["message_ids"] == ["11", "12"] and payload["last_message_id"] == "12"
        assert payload["chat_id"] == "8" and payload["thread_id"] == "77"
        assert payload["text"] == "final answer" and payload["session_key"] == "sess-1"

    def test_failed_send_does_not_fire(self, fired):
        adapter = _Adapter(SendResult(success=False, error="Forbidden: bot was blocked", error_kind="forbidden"))
        _send(adapter, _event(adapter), "k", "x", {}, reply_to=None)
        assert fired == []

    def test_ephemeral_system_reply_does_not_fire(self, fired):
        adapter = _Adapter(SendResult(success=True, message_id="1"))
        _send(adapter, _event(adapter), "k", "Stopped.", {}, reply_to=None, is_ephemeral_response=True)
        assert fired == []

    def test_slow_callback_never_delays_the_delivery(self):
        adapter = _Adapter(SendResult(success=True, message_id="1"))
        started = []

        async def hang(name, **kw):
            started.append(name)
            await asyncio.sleep(3600)

        async def main():
            t0 = time.monotonic()
            # wait_for turns a regression (delivery awaiting the observer) into a failure, not a hang.
            result, _ = await asyncio.wait_for(
                adapter.send_final_ledgered(_event(adapter), "k", "x", {}, reply_to=None), timeout=2)
            elapsed = time.monotonic() - t0
            await asyncio.sleep(0)            # let the background dispatch start
            for task in list(dh._PENDING):
                task.cancel()
            return result, elapsed

        with patch.object(dh, "_has_subscribers", return_value=True), \
             patch("hermes_cli.lifecycle.ainvoke_hook", hang):
            result, elapsed = asyncio.run(main())
        assert result.success and elapsed < 1.0 and started == ["gateway_message_delivered"]

    def test_hook_failure_leaves_the_delivery_result_intact(self):
        adapter = _Adapter(SendResult(success=True, message_id="1"))
        with patch.object(dh, "_has_subscribers", return_value=True), \
             patch("hermes_cli.lifecycle.ainvoke_hook", AsyncMock(side_effect=RuntimeError("boom"))):
            result, _ = _send(adapter, _event(adapter), "k", "x", {}, reply_to=None)
        assert result.success and result.message_id == "1"


# --------------------------------------------------------------------------- kind="final", streamed


class _Harness(GatewayTurnMixin):
    def __init__(self, confirmed):
        self._confirmed = confirmed

    def _run_agent_stream_confirmed_final_delivery(self, _sc, _final, *, previewed=False):
        return self._confirmed


def _mark(consumer, *, confirmed, response=None):
    turn_ctx = SimpleNamespace(
        stream_consumer_holder=[consumer],
        source=SimpleNamespace(platform=Platform.TELEGRAM, chat_id="c1", thread_id="t1"),
        session_key="sess-s")
    response = response or {"final_response": "complete answer"}

    async def main():
        await _Harness(confirmed)._run_agent_mark_streamed_delivery(response, turn_ctx)
        await dh.wait_for_pending_notifications(timeout=5)

    asyncio.run(main())
    return response


def _consumer(**over):
    base = dict(final_content_delivered=False, delivered_final_matches=None, message_id="55",
                stream_deltas_enabled=True, _delivered_final_text="complete answer (as shown)")
    base.update(over)
    return SimpleNamespace(**base)


class TestStreamedFinal:
    def test_confirmed_stream_fires_with_final_id_and_shown_text(self, fired):
        response = _mark(_consumer(), confirmed=True)
        assert response["already_sent"] is True
        (_, payload), = fired
        assert payload["streamed"] is True and payload["kind"] == "final"
        assert payload["message_ids"] == ["55"] and payload["text"] == "complete answer (as shown)"
        assert payload["chat_id"] == "c1" and payload["thread_id"] == "t1" and payload["session_key"] == "sess-s"

    def test_ambiguous_fallback_resend_is_not_reported_as_delivered(self, fired):
        response = _mark(_consumer(_delivery_ambiguous=True), confirmed=True)
        assert response["already_sent"] is True and fired == []

    def test_stale_finalize_reconcile_edit_fires_with_complete_text(self, fired):
        adapter = SimpleNamespace(edit_message=AsyncMock(return_value=SimpleNamespace(success=True)))
        consumer = _consumer(final_content_delivered=True, delivered_final_matches=lambda _f: False, adapter=adapter)
        _mark(consumer, confirmed=False)
        (_, payload), = fired
        assert payload["text"] == "complete answer" and payload["message_ids"] == ["55"]

    def test_failed_reconcile_edit_does_not_fire(self, fired):
        adapter = SimpleNamespace(edit_message=AsyncMock(return_value=SimpleNamespace(success=False, error="x")))
        consumer = _consumer(final_content_delivered=True, delivered_final_matches=lambda _f: False, adapter=adapter)
        response = _mark(consumer, confirmed=False)
        assert "already_sent" not in response and fired == []

    def test_unconfirmed_stream_leaves_it_to_the_normal_send(self, fired):
        _mark(_consumer(), confirmed=False)
        assert fired == []   # the ledgered normal send fires instead — never both
