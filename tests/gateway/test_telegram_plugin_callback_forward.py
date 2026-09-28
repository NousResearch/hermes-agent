"""Telegram inline-button taps with plugin-owned prefixes (``ok:`` / ``edit:``) reach plugins.

Contract (samimizer approval cards): the adapter publishes a ``gateway_platform_event`` of type
``callback_query`` through the runner's auth boundary, does not answer the tap itself when a plugin
claims it, and answers "Niet verwerkt" when nothing claims it within the window. Other prefixes keep
their existing routing.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from plugins.platforms.telegram import adapter as adapter_mod  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402
from gateway.run import GatewayRunner  # noqa: E402
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest  # noqa: E402


def _adapter() -> TelegramAdapter:
    a = object.__new__(TelegramAdapter)
    a.platform = Platform.TELEGRAM
    a.config = SimpleNamespace(extra={"allow_from": ["*"]})
    a.gateway_runner = None
    a._platform_event_handler = None
    a._background_tasks = set()
    return a


def _query(data="ok:abc123", *, user_id=555000111, chat_id=555000111, message_id=4242, qid="987654321"):
    q = MagicMock()
    q.id = qid
    q.data = data
    q.from_user = SimpleNamespace(id=user_id, username="sam", full_name="Sam", first_name="Sam")
    q.message = SimpleNamespace(
        chat=SimpleNamespace(id=chat_id, type="private", is_forum=False),
        chat_id=chat_id, message_id=message_id, message_thread_id=None, is_topic_message=False,
        date=SimpleNamespace(timestamp=lambda: 1.0))
    q.answer = AsyncMock()
    return q


def _update(query):
    return SimpleNamespace(callback_query=query)


@pytest.fixture(autouse=True)
def _hook_present(monkeypatch):
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)


async def _tap(a, query):
    await a._handle_callback_query(_update(query), context=MagicMock())
    tasks = list(a._background_tasks)
    if tasks:
        await asyncio.gather(*tasks)


class TestNormalize:
    def test_envelope_matches_contract(self):
        a = _adapter()
        event = a._normalize_callback_query_event(_query("edit:xyz"), 1727000000.5)
        assert event == {
            "platform": "telegram",
            "event_type": "callback_query",
            "payload": {
                "platform": "telegram", "internal": False, "user_id": "555000111",
                "chat_id": "555000111", "message_id": "4242", "data": "edit:xyz",
                "date": 1727000000.5, "callback_query_id": "987654321"},
        }

    def test_missing_card_message_yields_none(self):
        q = _query()
        q.message = None
        assert _adapter()._normalize_callback_query_event(q, 1.0) is None

    def test_foreign_prefix_yields_none(self):
        assert _adapter()._normalize_callback_query_event(_query("ea:once:1"), 1.0) is None

    def test_source_carries_tapper_and_chat(self):
        source = _adapter()._source_from_callback_query_for_auth(_query(user_id=7, chat_id=-100))
        assert source.user_id == "7"
        assert source.chat_id == "-100"

    def test_source_without_tapper_fails_closed(self):
        q = _query()
        q.from_user = None
        with pytest.raises(ValueError):
            _adapter()._source_from_callback_query_for_auth(q)


class TestForward:
    def test_claimed_tap_is_not_answered_by_adapter(self):
        a = _adapter()
        seen = []

        async def handler(event, source):
            seen.append((event, source))
            return [{"_callback_query_claim": True}]

        a.set_platform_event_handler(handler)
        q = _query()
        before = time.time()
        asyncio.run(_tap(a, q))

        assert len(seen) == 1
        event, source = seen[0]
        assert event["event_type"] == "callback_query"
        assert event["payload"]["data"] == "ok:abc123"
        assert before <= event["payload"]["date"] <= time.time()
        assert source.user_id == "555000111"
        q.answer.assert_not_awaited()

    @pytest.mark.parametrize("results", [None, [], [None, False]])
    def test_unclaimed_tap_answers_niet_verwerkt(self, results):
        a = _adapter()
        a.set_platform_event_handler(AsyncMock(return_value=results))
        q = _query()
        asyncio.run(_tap(a, q))
        q.answer.assert_awaited_once_with(text="Niet verwerkt")

    def test_no_handler_answers_niet_verwerkt(self):
        a = _adapter()
        q = _query("edit:1")
        asyncio.run(_tap(a, q))
        q.answer.assert_awaited_once_with(text="Niet verwerkt")

    def test_no_subscriber_answers_without_dispatch(self):
        a = _adapter()
        handler = AsyncMock(return_value=[{"_callback_query_claim": True}])
        a.set_platform_event_handler(handler)
        q = _query()
        with patch("hermes_cli.lifecycle.has_hook", return_value=False):
            asyncio.run(_tap(a, q))
        handler.assert_not_awaited()
        q.answer.assert_awaited_once_with(text="Niet verwerkt")

    def test_slow_plugin_gets_fallback_answer(self, monkeypatch):
        monkeypatch.setattr(adapter_mod, "_PLUGIN_CALLBACK_ANSWER_WINDOW", 0.05)
        a = _adapter()

        async def slow(event, source):
            await asyncio.sleep(1)
            return [{"_callback_query_claim": True}]

        a.set_platform_event_handler(slow)
        q = _query()
        asyncio.run(_tap(a, q))
        q.answer.assert_awaited_once_with(text="Niet verwerkt")

    def test_handler_error_gets_fallback_answer(self):
        a = _adapter()
        a.set_platform_event_handler(AsyncMock(side_effect=RuntimeError("boom")))
        q = _query()
        asyncio.run(_tap(a, q))
        q.answer.assert_awaited_once_with(text="Niet verwerkt")

    def test_other_prefixes_keep_existing_routing(self):
        a = _adapter()
        handler = AsyncMock(return_value=[{"_callback_query_claim": True}])
        a.set_platform_event_handler(handler)
        a._handle_exec_approval_callback = AsyncMock()
        q = _query("ea:once:abc")
        asyncio.run(_tap(a, q))
        a._handle_exec_approval_callback.assert_awaited_once()
        handler.assert_not_awaited()
        q.answer.assert_not_awaited()

    def test_unknown_prefix_still_ignored(self):
        a = _adapter()
        handler = AsyncMock(return_value=[{"_callback_query_claim": True}])
        a.set_platform_event_handler(handler)
        q = _query("zz:1")
        asyncio.run(_tap(a, q))
        handler.assert_not_awaited()
        q.answer.assert_not_awaited()


class TestRunnerBoundary:
    def test_callback_query_runs_hook_off_loop_and_returns_results(self):
        runner = object.__new__(GatewayRunner)
        runner._is_user_authorized = lambda source: True
        loop_thread = {}
        hook_thread = {}

        def invoke(name, **event):
            hook_thread["id"] = threading.get_ident()
            return [{"_callback_query_claim": True, "value": "claimed"}]

        async def run():
            loop_thread["id"] = threading.get_ident()
            source = _adapter()._source_from_callback_query_for_auth(_query())
            event = _adapter()._normalize_callback_query_event(_query(), 1.0)
            return await runner._handle_gateway_platform_event(event, source)

        with patch("hermes_cli.lifecycle.invoke_hook", invoke):
            assert asyncio.run(run()) == [{"_callback_query_claim": True, "value": "claimed"}]
        assert hook_thread["id"] != loop_thread["id"]

    def test_unauthorized_tapper_never_reaches_hooks(self):
        runner = object.__new__(GatewayRunner)
        runner._is_user_authorized = lambda source: False
        invoke = MagicMock()
        a = _adapter()
        a.set_platform_event_handler(runner._handle_gateway_platform_event)
        q = _query(user_id=666)
        with patch("hermes_cli.lifecycle.invoke_hook", invoke):
            asyncio.run(_tap(a, q))
        invoke.assert_not_called()
        q.answer.assert_awaited_once_with(text="Niet verwerkt")

    def test_reaches_real_plugin_callback(self):
        manager = PluginManager()
        context = PluginContext(PluginManifest(name="cb-fixture", source="user"), manager)
        seen = []

        def on_event(platform, event_type, payload):
            seen.append((platform, event_type, payload))
            return {"_callback_query_claim": True}

        context.register_hook("gateway_platform_event", on_event)
        runner = object.__new__(GatewayRunner)
        runner._is_user_authorized = lambda source: source.user_id == "555000111"
        a = _adapter()
        a.set_platform_event_handler(runner._handle_gateway_platform_event)
        q = _query("ok:draft-1")
        with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
            asyncio.run(_tap(a, q))

        assert len(seen) == 1
        platform, event_type, payload = seen[0]
        assert (platform, event_type) == ("telegram", "callback_query")
        assert payload["data"] == "ok:draft-1"
        assert payload["message_id"] == "4242"
        assert payload["callback_query_id"] == "987654321"
        q.answer.assert_not_awaited()
