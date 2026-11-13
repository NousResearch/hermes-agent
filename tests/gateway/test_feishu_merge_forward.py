"""REST enrichment regressions for merge_forward placeholders (PR #38686)."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tests.gateway.feishu_helpers import make_adapter_skeleton, make_message


def child(kind, content, sender=None, parent="om_xxx"):
    return SimpleNamespace(
        upper_message_id=parent, msg_type=kind,
        body=SimpleNamespace(content=json.dumps(content)), sender=sender,
    )


def extract(items=None, error=None, success=True, populated=False):
    adapter = make_adapter_skeleton()
    response = SimpleNamespace(success=lambda: success, data=SimpleNamespace(items=items))
    adapter._client = Mock()
    adapter._client.im.v1.message.get = Mock(return_value=response, side_effect=error)
    adapter._build_get_message_request = Mock(return_value=object())
    # Keep the executor seam visible while executing the mocked SDK call inline.
    adapter._run_blocking = AsyncMock(side_effect=lambda func, *args: func(*args))
    adapter._download_feishu_message_resources = AsyncMock(return_value=([], []))
    message = make_message()
    message.message_type = "merge_forward"
    message.content = json.dumps({"messages": [{"text": "already populated"}]}) if populated else "Merged and Forwarded Message"
    result = asyncio.run(adapter._extract_message_content(message))
    return result, adapter


def test_rest_children_expanded_in_order_with_senders():
    parent = SimpleNamespace(upper_message_id=None, msg_type="merge_forward")
    items = [parent, child("text", {"text": "hello"}, SimpleNamespace(id="ou_original")),
             child("text", {"text": "world"}, SimpleNamespace(name="Alice", id="ou_alice")),
             child("text", {"text": "unrelated"}, parent="other")]
    result, adapter = extract(items)
    assert result[0] == "[ou_original] hello\n[Alice] world"
    adapter._build_get_message_request.assert_called_once_with("om_xxx")
    adapter._run_blocking.assert_awaited_once_with(
        adapter._client.im.v1.message.get, adapter._build_get_message_request.return_value,
    )


def test_image_and_file_children_are_text_only():
    result, _ = extract([child("image", {"image_key": "img_secret"}),
                         child("file", {"file_key": "file_secret", "file_name": "report.pdf"})])
    assert result[0] == "[图片]\n[文件]"
    assert result[2:5] == ([], [], [])


@pytest.mark.parametrize("kwargs", [{"error": RuntimeError("lookup failed")},
                                    {"success": False}, {"items": []},
                                    {"items": [SimpleNamespace(upper_message_id=None)]}])
def test_lookup_failure_keeps_placeholder(kwargs):
    from plugins.platforms.feishu.adapter import FALLBACK_FORWARD_TEXT

    result, _ = extract(**kwargs)
    assert result[0] == FALLBACK_FORWARD_TEXT


def test_transcript_is_bounded_with_explicit_marker():
    result, _ = extract([child("text", {"text": "x" * 20000})])
    assert len(result[0]) <= 12000
    assert result[0].endswith("\n[转发内容已截断]")


def test_populated_forward_does_not_fetch():
    result, adapter = extract(populated=True)
    assert "already populated" in result[0]
    adapter._client.im.v1.message.get.assert_not_called()
