"""In-memory Bot API transport; Telegram SDK and adapter remain real."""
import asyncio
import json
from collections import defaultdict, deque
from types import SimpleNamespace

import pytest
import pytest_asyncio

pytest.importorskip("telegram", reason="requires the declared telegram extra")
from telegram import Bot
from telegram.request import BaseRequest

from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource
from plugins.platforms.telegram.adapter import TelegramAdapter
from tests.gateway.test_todo_checklist_delivery import make_turn, snapshot


class BotAPI(BaseRequest):
    def __init__(self):
        self.calls, self.messages = [], {}
        self.failures = defaultdict(deque)
        self.accepted = asyncio.Event()
        self.hold_receipt = False
        self.inflight = 0
        self.next_id = 100

    @property
    def read_timeout(self):
        return 5

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    def reject(self, endpoint, code, description, *, retry_after=None, times=1):
        error = dict(ok=False, error_code=code, description=description)
        if retry_after is not None:
            error["parameters"] = {"retry_after": retry_after}
        self.failures[endpoint].extend([(code, error)] * times)

    def requests(self, endpoint):
        return [payload for name, payload in self.calls if name == endpoint]

    async def do_request(self, url, method, request_data=None, **kwargs):
        endpoint = url.rsplit("/", 1)[-1]
        payload = request_data.parameters if request_data else {}
        self.calls.append((endpoint, payload))
        if self.failures[endpoint]:
            failure = self.failures[endpoint].popleft()
            if isinstance(failure, Exception):
                raise failure
            code, body = failure
            return code, json.dumps(body).encode()
        if endpoint == "getMe":
            result = dict(id=123456, is_bot=True, first_name="Offline", username="offline_test_bot")
        elif endpoint in {"sendMessage", "editMessageText"}:
            chat = int(payload["chat_id"])
            if endpoint == "sendMessage":
                self.next_id += 1
                mid = self.next_id
            else:
                mid = int(payload["message_id"])
                assert (chat, mid) in self.messages, "edit must target a message this bot sent"
            result = dict(message_id=mid, date=0, chat={"id": chat, "type": "supergroup"}, text=payload["text"])
            if endpoint == "sendMessage":
                result["message_thread_id"] = payload.get("message_thread_id")
            else:
                result["message_thread_id"] = self.messages[chat, mid].get("message_thread_id")
            self.messages[chat, mid] = result
            if endpoint == "sendMessage" and self.hold_receipt:
                self.inflight += 1
                self.accepted.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    self.inflight -= 1
        elif endpoint == "sendChatAction":
            result = True
        else:
            raise AssertionError(f"Unexpected Bot API operation: {endpoint}")
        return 200, json.dumps({"ok": True, "result": result}).encode()


@pytest_asyncio.fixture
async def telegram_api():
    api = BotAPI()
    # Both SDK request channels are in-memory; no HTTP client or real token is used.
    async with Bot("123456:offline-test-token", request=api, get_updates_request=BotAPI()) as bot:
        adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123456:offline-test-token"))
        adapter._bot = bot
        adapter._telegram_chat_outbound_slot_secs = 0
        try:
            yield adapter, api
        finally:
            tasks = list(adapter._background_tasks)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def turn_for(adapter, topic="11", chat="-100123", *, grouping="accumulate"):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id=chat, thread_id=topic, chat_type="group")
    return make_turn(adapter, source, grouping=grouping)


def status_for(adapter, monkeypatch):
    from gateway.todo_progress import TodoProgress

    clock = [10.0]
    monkeypatch.setattr("gateway.todo_progress.time", SimpleNamespace(monotonic=lambda: clock[0]))
    turn = turn_for(adapter)
    return TodoProgress(turn, adapter), clock
