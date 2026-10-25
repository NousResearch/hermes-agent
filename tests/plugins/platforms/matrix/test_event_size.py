"""Every event the Matrix plugin builds fits the homeserver's byte cap, whatever the script or shape of
the text: sends, stream/progress edits at the adapter's advertised budget, and the standalone sender."""

import json
import sys
import types

import pytest

from gateway.config import PlatformConfig

# The homeserver rejects an event whose canonical JSON exceeds 65,536 bytes (M_TOO_LARGE). E2EE
# base64-inflates the content by 4/3, and the envelope (ids, hashes, signatures) needs room too.
_EVENT_CAP = 65_536
_LINES = {
    "english": "The **quick** brown fox jumps over the *lazy* dog, and `code` too.\n",
    "cyrillic": "Быстрая **коричневая** лиса прыгает через *ленивую* собаку, и `код` тоже.\n",
    "cjk": "敏捷的**棕色**狐狸跳过了*懒惰的*狗，还有`代码`也是。这是一个很长的段落。\n",
    "code": "```html\n<a href=\"x&amp;y\">it's <b>&lt;bold&gt;</b></a>\n```\n",
}


def _texts(chars: int) -> dict:
    """One text per script/shape, about *chars* characters long."""
    texts = {name: line * (chars // len(line)) for name, line in _LINES.items()}
    # Distinct Matrix IDs: each lands in m.mentions, which an edit carries twice.
    texts["mentions"] = " ".join(f"@u{i}:a.b" for i in range(chars // 8))
    return texts


def _fits_one_event(content: dict) -> bool:
    size = len(json.dumps(content, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return size * 4 // 3 + 4096 <= _EVENT_CAP


def _make_adapter():
    from plugins.platforms.matrix.adapter import MatrixAdapter

    return MatrixAdapter(PlatformConfig(enabled=True, token="syt_test_token", extra={
        "homeserver": "https://matrix.example.org", "user_id": "@bot:example.org"}))


@pytest.mark.asyncio
async def test_sends_and_budget_sized_edits_fit_one_event():
    from gateway.platforms.base import SendResult, _custom_unit_to_cp

    adapter = _make_adapter()
    sent, edits = [], []

    async def send_room_message(chat_id, content):
        sent.append(content)
        return "$sent"

    async def send_content_event(chat_id, content):
        edits.append(content)
        return SendResult(success=True, message_id="$edit")

    adapter._send_room_message = send_room_message
    adapter._send_content_event = send_content_event
    chat = "!room:example.org"
    # Stream previews and progress edits stay within the adapter's advertised per-chat budget.
    limit, len_fn = adapter.max_message_length_for_chat(chat), adapter.message_len_fn_for_chat(chat)
    for name, text in _texts(40_000).items():
        assert (await adapter.send(chat, text)).success, name
        preview = text[: _custom_unit_to_cp(text, limit, len_fn)]
        assert (await adapter.edit_message(chat, "$preview", preview)).success, name

    oversized = [len(json.dumps(c, ensure_ascii=False)) for c in sent + edits if not _fits_one_event(c)]
    assert oversized == []
    # Chunking keeps Markdown rendering: an ordinary chunk still carries its HTML.
    assert all("formatted_body" in c for c in sent if "quick" in c["body"])


@pytest.mark.asyncio
async def test_standalone_sender_events_fit_one_event(monkeypatch):
    from plugins.platforms.matrix.adapter import DEFAULT_MAX_MESSAGE_LENGTH, _standalone_send

    payloads = []

    class _Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def json(self):
            return {"event_id": f"$e{len(payloads)}"}

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        def put(self, url, **kwargs):
            payloads.append(kwargs["json"])
            return _Response()

    monkeypatch.setitem(sys.modules, "aiohttp", types.SimpleNamespace(ClientSession=_Session))
    pconfig = PlatformConfig(enabled=True, token="syt_test_token", extra={"homeserver": "https://matrix.example.org"})
    # send_message chunks plugin platforms on characters at the registry's max_message_length.
    for name, text in _texts(DEFAULT_MAX_MESSAGE_LENGTH).items():
        result = await _standalone_send(pconfig, "!room:example.org", text[:DEFAULT_MAX_MESSAGE_LENGTH])
        assert result["success"], (name, result)

    assert payloads
    assert [len(json.dumps(p, ensure_ascii=False)) for p in payloads if not _fits_one_event(p)] == []
