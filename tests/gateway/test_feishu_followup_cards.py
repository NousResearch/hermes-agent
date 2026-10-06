"""Feishu renders ``::followup`` as safe, clickable next-step cards."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


def _ensure_feishu_mocks() -> None:
    """Allow adapter imports in the minimal test environment."""
    if importlib.util.find_spec("lark_oapi") is None and "lark_oapi" not in sys.modules:
        module = MagicMock()
        for name in (
            "lark_oapi",
            "lark_oapi.api.im.v1",
            "lark_oapi.event",
            "lark_oapi.event.callback_type",
        ):
            sys.modules.setdefault(name, module)
    if importlib.util.find_spec("aiohttp") is None and "aiohttp" not in sys.modules:
        module = MagicMock()
        sys.modules.setdefault("aiohttp", module)
        sys.modules.setdefault("aiohttp.web", module.web)


_ensure_feishu_mocks()

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from plugins.platforms.feishu.adapter import FeishuAdapter, _split_followup_directive


def _response(message_id: str = "om_message") -> SimpleNamespace:
    return SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id=message_id))


def _adapter() -> FeishuAdapter:
    adapter = FeishuAdapter(PlatformConfig(enabled=True))
    adapter._client = MagicMock()
    return adapter


def _card_data(prompt: str, *, token: str = "card-token") -> SimpleNamespace:
    return SimpleNamespace(
        event=SimpleNamespace(
            token=token,
            context=SimpleNamespace(open_chat_id="oc_test"),
            operator=SimpleNamespace(open_id="ou_test"),
            action=SimpleNamespace(tag="button", value={"hermes_followup_prompt": prompt}),
        )
    )


def test_split_followup_removes_only_directive_and_bounds_unique_prompts():
    body, prompts = _split_followup_directive(
        'Done.\n\n::followup{p1="Run tests" p2="Open a PR" p3="Run tests" '
        'p4="Read logs" p5="Inspect diff" p6="Ignored"}\n::preview{file="x.html"}'
    )

    assert body == 'Done.\n\n\n::preview{file="x.html"}'
    assert prompts == ["Run tests", "Open a PR", "Read logs", "Inspect diff", "Ignored"]


def test_send_strips_directive_from_text_and_sends_full_width_prompt_cards():
    adapter = _adapter()
    adapter._feishu_send_with_retry = AsyncMock(side_effect=[_response("body"), _response("card")])
    original_card_sender = adapter._send_followup_card
    adapter._send_followup_card = AsyncMock(wraps=original_card_sender)

    result = asyncio.run(adapter.send(
        "oc_test",
        'Finished.\n\n::followup{p1="Run the focused tests" p2="Open the pull request"}',
    ))

    assert result.success is True
    assert result.message_id == "body"
    calls = adapter._feishu_send_with_retry.await_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["msg_type"] == "text"
    assert json.loads(calls[0].kwargs["payload"]) == {"text": "Finished."}
    assert calls[1].kwargs["msg_type"] == "interactive"
    card = json.loads(calls[1].kwargs["payload"])
    assert card["header"]["title"]["content"] == "Bước tiếp"
    assert [element["actions"][0]["value"]["hermes_followup_prompt"] for element in card["elements"][1:]] == [
        "Run the focused tests",
        "Open the pull request",
    ]
    assert all(element["actions"][0]["width"] == "fill" for element in card["elements"][1:])


def test_send_does_not_attach_card_when_body_delivery_fails():
    adapter = _adapter()
    adapter._feishu_send_with_retry = AsyncMock(return_value=SimpleNamespace(success=lambda: False, code=500, msg="no"))

    result = asyncio.run(adapter.send("oc_test", 'Body\n::followup{p1="Retry"}'))

    assert result.success is False
    assert all(call.kwargs["msg_type"] != "interactive" for call in adapter._feishu_send_with_retry.await_args_list)


def test_card_callback_schedules_followup_as_text_not_card_command():
    adapter = _adapter()
    loop = MagicMock()
    loop.is_closed.return_value = False
    adapter._loop = loop
    scheduled = []

    def submit(_loop, coro):
        scheduled.append(coro)
        coro.close()
        return True

    adapter._submit_on_loop = submit
    adapter._card_response = MagicMock(return_value="response")

    assert adapter._on_card_action_trigger(_card_data("Run the focused tests")) == "response"
    assert len(scheduled) == 1


def test_followup_click_dispatches_prompt_as_text_once():
    adapter = _adapter()
    adapter._resolve_sender_profile = AsyncMock(return_value={})
    adapter.get_chat_info = AsyncMock(return_value={})
    adapter._dispatch_synthetic_event = AsyncMock()

    async def _run():
        await adapter._handle_followup_card_click(_card_data("Run the focused tests").event, "Run the focused tests")
        await adapter._handle_followup_card_click(_card_data("Run the focused tests").event, "Run the focused tests")

    asyncio.run(_run())

    adapter._dispatch_synthetic_event.assert_awaited_once()
    assert adapter._dispatch_synthetic_event.await_args.kwargs["text"] == "Run the focused tests"
    assert adapter._dispatch_synthetic_event.await_args.kwargs["message_type"] is MessageType.TEXT
