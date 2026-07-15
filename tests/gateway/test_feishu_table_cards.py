"""Feishu CardKit v2 rendering and fallback tests for GFM tables."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter, _build_markdown_table_card

TABLE = "| Name | Status |\n| --- | --- |\n| Alpha | **Ready** |"


def _success_response():
    return SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_ok"))


def _failure_response(message: str = "invalid card content"):
    return SimpleNamespace(success=lambda: False, code=230099, msg=message)


def test_builds_native_table_with_surrounding_markdown():
    card = _build_markdown_table_card(f"**Summary**\n\n{TABLE}\n\nDone")

    assert card is not None
    assert card["schema"] == "2.0"
    elements = card["body"]["elements"]
    assert [element["tag"] for element in elements] == ["markdown", "table", "markdown"]
    table = elements[1]
    assert [column["display_name"] for column in table["columns"]] == ["Name", "Status"]
    assert table["columns"][1]["data_type"] == "lark_md"
    assert table["rows"] == [{"col_0": "Alpha", "col_1": "**Ready**"}]


def test_routes_only_valid_tables_to_interactive():
    adapter = FeishuAdapter(PlatformConfig())

    msg_type, payload = adapter._build_outbound_payload(TABLE)
    assert msg_type == "interactive"
    assert json.loads(payload)["schema"] == "2.0"

    msg_type, _ = adapter._build_outbound_payload("status | value\nnot a table")
    assert msg_type == "text"

    fenced = f"```markdown\n{TABLE}\n```"
    msg_type, _ = adapter._build_outbound_payload(fenced)
    assert msg_type == "post"


def test_send_falls_back_to_original_text_when_card_is_rejected():
    adapter = FeishuAdapter(PlatformConfig())
    adapter._client = object()
    calls = []

    async def fake_send(**kwargs):
        calls.append(kwargs)
        return _failure_response() if len(calls) == 1 else _success_response()

    adapter._feishu_send_with_retry = fake_send
    with patch.object(adapter, "truncate_message", return_value=[TABLE]):
        result = asyncio.run(adapter.send("oc_chat", TABLE))

    assert result.success
    assert [call["msg_type"] for call in calls] == ["interactive", "text"]
    assert json.loads(calls[1]["payload"])["text"] == TABLE


def test_send_falls_back_when_card_creation_raises():
    adapter = FeishuAdapter(PlatformConfig())
    adapter._client = object()
    calls = []

    async def fake_send(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("Failed to create card content, code=230099")
        return _success_response()

    adapter._feishu_send_with_retry = fake_send
    with patch.object(adapter, "truncate_message", return_value=[TABLE]):
        result = asyncio.run(adapter.send("oc_chat", TABLE))

    assert result.success
    assert [call["msg_type"] for call in calls] == ["interactive", "text"]


def test_edit_falls_back_to_original_text_when_card_is_rejected():
    adapter = FeishuAdapter(PlatformConfig())
    requests = []

    class MessageAPI:
        def update(self, request):
            requests.append(request)
            return _failure_response() if len(requests) == 1 else _success_response()

    adapter._client = SimpleNamespace(
        im=SimpleNamespace(v1=SimpleNamespace(message=MessageAPI()))
    )

    async def direct(func, *args, **kwargs):
        return func(*args, **kwargs)

    with patch("plugins.platforms.feishu.adapter.asyncio.to_thread", side_effect=direct):
        result = asyncio.run(adapter.edit_message("oc_chat", "om_progress", TABLE))

    assert result.success
    assert [request.request_body.msg_type for request in requests] == ["interactive", "text"]
    assert json.loads(requests[1].request_body.content)["text"] == TABLE
