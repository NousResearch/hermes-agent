"""Full progress through the native Telegram formatter and python-telegram-bot HTTP path."""

import json
import re
import sys
from copy import deepcopy
from queue import Queue
from types import SimpleNamespace
from urllib.parse import parse_qs
from unittest.mock import Mock

import httpx
import pytest
import pytest_asyncio

# Gateway conftest eagerly installs SDK stubs even when PTB is installed. This
# file runs in its own process and requires the real optional dependency.
for _name, _module in tuple(sys.modules.items()):
    if (_name == "telegram" or _name.startswith("telegram.")) and isinstance(_module, Mock):
        del sys.modules[_name]

from telegram import Bot
from telegram.request import HTTPXRequest

from gateway.config import PlatformConfig
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext
from plugins.platforms.telegram.adapter import TelegramAdapter


def _units(text):
    return len(text.encode("utf-16-le")) // 2


def _visible(payload):
    text = payload["text"]
    if payload.get("parse_mode") == "MarkdownV2":
        # These fixtures contain literal JSON, no Markdown constructs. Undo wire
        # escapes only; do not call the formatter/stripper being tested.
        text = re.sub(r"\\([_*\[\]()~`>#+\-=|{}.!\\])", r"\1", text)
    return text


class TelegramHTTP:
    """Only external I/O is replaced; Bot builds requests and decodes real Message objects."""

    def __init__(self):
        self.cap = TelegramAdapter.MAX_MESSAGE_LENGTH
        self.requests = []
        self.posts = {}
        self.acknowledged = []
        self.reject_send_at = None
        self.flood = False
        self.send_attempts = 0

    def __call__(self, request):
        method = request.url.path.rsplit("/", 1)[-1]
        payload = {key: values[0] for key, values in parse_qs(
            request.content.decode(), keep_blank_values=True
        ).items()}
        self.requests.append((method, payload))
        if method == "getMe":
            result = {"id": 123456, "is_bot": True, "first_name": "Full test", "username": "full_test_bot"}
        elif method == "sendChatAction":
            result = True
        elif method in ("sendMessage", "editMessageText"):
            if _units(payload["text"]) > self.cap:
                return httpx.Response(400, json={"ok": False, "error_code": 400,
                                                 "description": "Bad Request: message is too long"})
            if method == "sendMessage":
                self.send_attempts += 1
                if self.reject_send_at is not None and self.send_attempts >= self.reject_send_at:
                    if self.flood:
                        return httpx.Response(429, json={"ok": False, "error_code": 429,
                                                        "description": "Too Many Requests: retry after 1000",
                                                        "parameters": {"retry_after": 1000}})
                    return httpx.Response(400, json={"ok": False, "error_code": 400,
                                                     "description": "Bad Request: synthetic refusal"})
                message_id = str(len(self.posts) + 100)
                self.acknowledged.append(message_id)
            else:
                message_id = payload["message_id"]
                assert message_id in self.posts
            self.posts[message_id] = deepcopy(payload)
            result = {"message_id": int(message_id), "date": 1,
                      "chat": {"id": int(payload["chat_id"]), "type": "supergroup"},
                      "text": _visible(payload)}
        elif method == "deleteMessage":
            assert payload["message_id"] in self.posts
            del self.posts[payload["message_id"]]
            result = True
        else:
            raise AssertionError(f"Unexpected Telegram method: {method}")
        return httpx.Response(200, json={"ok": True, "result": result})


@pytest_asyncio.fixture
async def native_turn():
    wire = TelegramHTTP()
    request = HTTPXRequest(httpx_kwargs={"transport": httpx.MockTransport(wire)})
    # Both SDK request slots are isolated; getUpdates is never called.
    updates = HTTPXRequest(httpx_kwargs={"transport": httpx.MockTransport(wire)})
    async with Bot("123456:synthetic-full-progress", request=request, get_updates_request=updates) as bot:
        adapter = TelegramAdapter(PlatformConfig(enabled=True, token="synthetic-full-progress"))
        adapter._bot = bot
        ctx = TurnContext(
            source=SimpleNamespace(chat_id="-100123", thread_id="73"),
            _run_still_current=lambda: True,
            progress_mode="full", tool_progress_enabled=True, progress_queue=Queue(),
            _cleanup_progress=True, _progress_reply_to="17",
            _progress_metadata={"thread_id": "73", "_interim_send": True},
        )
        turn = TurnRunner(SimpleNamespace(_adapter_for_source=lambda _s: adapter), ctx)
        yield turn, adapter, wire


def _entry(turn, name, args):
    turn.progress_callback("tool.started", name, "short", args)
    marker, content = turn._ctx.progress_queue.get_nowait()
    assert marker == "__full__"
    return content


def _assert_arguments(text, names, originals):
    remaining = text
    for name, expected in zip(names, originals):
        header, separator, body = remaining.partition("\n")
        assert separator and header.endswith(name)
        decoded, end = json.JSONDecoder().raw_decode(body)
        assert decoded == expected
        remaining = body[end:].removeprefix("\n")
    assert remaining == "", "progress content was replayed or acquired a native chunk marker"


@pytest.mark.asyncio
@pytest.mark.parametrize("grouping", ["grouped", "separate"])
async def test_full_native_telegram_fits_formatted_send_and_edit_losslessly(native_turn, grouping):
    turn, adapter, wire = native_turn
    turn._ctx.progress_grouping = grouping
    st = turn._progress_edit_state(adapter)
    names = ["initial_payload_tool", "next_payload_tool", "large_payload_tool", "last_payload_tool"]
    originals = [
        {"question": "visible-initial"},
        {"command": "printf 'safe-value'", "path": r"/tmp/a-b\c", "enabled": False},
        {"question": ("a-" * 1500) + "🧪 漢字\n\t  \"quoted\"", "nested": [1, None, "end."]},
        {"question": "visible-final"},
    ]
    before = deepcopy(originals)
    entries = [_entry(turn, name, args) for name, args in zip(names, originals)]
    assert _units(entries[2]) < adapter.MAX_MESSAGE_LENGTH
    assert _units(adapter.format_message(entries[2])) > adapter.MAX_MESSAGE_LENGTH
    for entry in entries:
        await turn._send_full_progress_entry(st, entry)
    assert originals == before
    assert turn._ctx.progress_queue.empty()
    assert turn._ctx._progress_metadata == {"thread_id": "73", "_interim_send": True}
    text_requests = [(method, body) for method, body in wire.requests if "text" in body]
    assert all(_units(body["text"]) <= wire.cap for _, body in text_requests)
    assert all(body["parse_mode"] == "MarkdownV2" for _, body in text_requests)
    if grouping == "grouped":
        assert any(method == "editMessageText" for method, _ in text_requests)
    else:
        assert all(method == "sendMessage" for method, _ in text_requests)
    for method, body in text_requests:
        assert body["chat_id"] == turn._ctx.source.chat_id
        if method == "sendMessage":
            assert body["message_thread_id"] == "73"
            assert json.loads(body["reply_parameters"])["message_id"] == 17
            assert body["disable_notification"] == "true"
    _assert_arguments("".join(_visible(body) for body in wire.posts.values()), names, originals)
    assert turn._ctx._cleanup_msg_ids == wire.acknowledged


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["send", "partial-send", "partial-send-flood", "edit", "partial-edit"])
async def test_full_native_telegram_multipart_receipts_close_edit_ownership(native_turn, operation):
    turn, adapter, wire = native_turn
    st = turn._progress_edit_state(adapter)
    prefix = _entry(turn, "initial_tool", {"question": "initial"})
    if operation.endswith("edit"):
        await turn._send_full_progress_entry(st, prefix)
    # The native cap can change after the consumer cached its budget. Exercise
    # real adapter splitting/partial receipts rather than fabricating SendResult.
    adapter.MAX_MESSAGE_LENGTH = wire.cap = 512
    if operation.startswith("partial"):
        wire.reject_send_at = wire.send_attempts + 3
        wire.flood = operation.endswith("flood")
    large = _entry(turn, "large_tool", {"question": "x" * 1600})
    await turn._send_full_progress_entry(st, large)
    assert len(wire.acknowledged) > 1
    assert turn._ctx._cleanup_msg_ids == wire.acknowledged
    assert st.progress_msg_id is None and st.progress_lines == []
    frozen = deepcopy(wire.posts)
    attempts_before = wire.send_attempts
    wire.reject_send_at = None
    next_entry = _entry(turn, "next_tool", {"question": "next"})
    await turn._send_full_progress_entry(st, next_entry)
    assert all(wire.posts[mid] == body for mid, body in frozen.items())
    assert _visible(wire.posts[wire.acknowledged[-1]]) == next_entry
    assert turn._ctx._cleanup_msg_ids == wire.acknowledged
    if operation.startswith("partial"):
        # Refused chunks are not replayed by Full, even if native delivery was partial.
        assert wire.send_attempts == attempts_before + 1
    for mid in turn._ctx._cleanup_msg_ids:
        assert await adapter.delete_message(turn._ctx.source.chat_id, mid)
    assert wire.posts == {}
