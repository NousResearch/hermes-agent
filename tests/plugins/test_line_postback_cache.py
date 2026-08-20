"""Regression tests for LINE slow-response postback cache delivery."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from plugins.platforms.line.adapter import LineAdapter, State


class _ReplyRecorder:
    def __init__(self) -> None:
        self.replies: list[tuple[str, list[dict[str, object]]]] = []
        self.pushes: list[tuple[str, list[dict[str, object]]]] = []

    async def reply(self, reply_token: str, messages: list[dict[str, object]]) -> None:
        self.replies.append((reply_token, messages))

    async def push(self, chat_id: str, messages: list[dict[str, object]]) -> None:
        self.pushes.append((chat_id, messages))


def _adapter() -> LineAdapter:
    return LineAdapter(SimpleNamespace(extra={}))


def _postback_event(chat_id: str, reply_token: str, request_id: str) -> dict[str, object]:
    return {
        "replyToken": reply_token,
        "source": {"type": "user", "userId": chat_id},
        "postback": {
            "data": json.dumps(
                {"action": "show_response", "request_id": request_id}
            )
        },
    }


def _cache_entry_in_state(adapter: LineAdapter, state: State) -> str:
    request_id = adapter._cache.register_pending("U-chat-a")
    if state is State.READY:
        adapter._cache.set_ready(request_id, "cached private answer")
    elif state is State.ERROR:
        adapter._cache.set_error(request_id, "cached private error")
    elif state is State.DELIVERED:
        adapter._cache.set_ready(request_id, "cached private answer")
        adapter._cache.mark_delivered(request_id)
    return request_id


@pytest.mark.parametrize("state", list(State))
@pytest.mark.asyncio
async def test_postback_cache_rejects_every_state_from_different_chat(
    state: State,
) -> None:
    adapter = _adapter()
    client = _ReplyRecorder()
    adapter._client = client

    request_id = _cache_entry_in_state(adapter, state)
    entry = adapter._cache.get(request_id)
    original = (entry.state, entry.payload, entry.updated_at)
    adapter._pending_buttons["U-chat-a"] = request_id

    await adapter._handle_postback_event(
        _postback_event("U-chat-b", "reply-token-b", request_id)
    )

    assert client.replies == []
    assert client.pushes == []
    entry = adapter._cache.get(request_id)
    assert (entry.state, entry.payload, entry.updated_at) == original
    assert adapter._pending_buttons == {"U-chat-a": request_id}


@pytest.mark.parametrize(
    ("state", "expected_text", "expected_final_state"),
    [
        (State.READY, "cached private answer", State.DELIVERED),
        (State.ERROR, "cached private error", State.DELIVERED),
        (State.DELIVERED, None, State.DELIVERED),
        (State.PENDING, None, State.PENDING),
    ],
)
@pytest.mark.asyncio
async def test_postback_cache_preserves_origin_chat_behavior(
    state: State,
    expected_text: str | None,
    expected_final_state: State,
) -> None:
    adapter = _adapter()
    client = _ReplyRecorder()
    adapter._client = client

    request_id = _cache_entry_in_state(adapter, state)
    if expected_text is None:
        expected_text = (
            adapter.delivered_text
            if state is State.DELIVERED
            else adapter.pending_text
        )

    await adapter._handle_postback_event(
        _postback_event("U-chat-a", "reply-token-a", request_id)
    )

    assert client.replies == [
        ("reply-token-a", [{"type": "text", "text": expected_text}])
    ]
    assert client.pushes == []
    assert adapter._cache.get(request_id).state is expected_final_state
