"""_fetch_last_message_in_thread must ask the API for newest-first.

Regression: the request carried no sort_type, so the API default (oldest first)
applied and items[0] was the first message of the thread, not the last one.
"""
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter


@pytest.mark.asyncio
async def test_fetch_last_message_in_thread_requests_newest_first():
    adapter = FeishuAdapter(PlatformConfig())
    captured = {}

    async def fake_run_blocking(fn, request, *args, **kwargs):
        captured["request"] = request
        return SimpleNamespace(
            code=0,
            data=SimpleNamespace(items=[SimpleNamespace(message_id="om_newest")]),
        )

    adapter._run_blocking = fake_run_blocking
    adapter._response_succeeded = lambda response: True
    adapter._client = SimpleNamespace(
        im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(list=lambda req: None))),
    )

    message_id = await adapter._fetch_last_message_in_thread("th_1")

    assert message_id == "om_newest"
    request = captured["request"]
    assert getattr(request, "sort_type", None) == "ByCreateTimeDesc", (
        "thread message list must be requested newest-first; otherwise items[0] is the "
        "OLDEST message and reply anchoring targets the wrong message"
    )
