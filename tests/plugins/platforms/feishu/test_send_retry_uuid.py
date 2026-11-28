"""A lost Feishu send response must not make the retry a new message."""

from collections import defaultdict
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("reply_to", [None, "om_parent"], ids=["create", "reply"])
async def test_retry_reuses_each_chunks_uuid(reply_to, monkeypatch):
    adapter = FeishuAdapter(PlatformConfig())
    attempts = defaultdict(list)
    delivered = {}

    def accept_then_lose_first_response(request):
        body = request.request_body
        attempts[body.content].append(body.uuid)
        message_id = delivered.setdefault(body.uuid, f"om_{len(delivered)}")
        if len(attempts[body.content]) == 1:
            raise TimeoutError("the server accepted the request but its response was lost")
        return SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id=message_id))

    api = SimpleNamespace(create=accept_then_lose_first_response, reply=accept_then_lose_first_response)
    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=api)))

    async def run_directly(func, *args):
        return func(*args)

    async def no_retry_delay(_seconds):
        pass

    monkeypatch.setattr(adapter, "_run_blocking", run_directly)
    monkeypatch.setattr(adapter, "truncate_message", lambda _content, _limit: ["first", "second"])
    monkeypatch.setattr("plugins.platforms.feishu.adapter.asyncio.sleep", no_retry_delay)

    result = await adapter.send("oc_chat", "first second", reply_to=reply_to)

    assert result.success
    assert len(attempts) == 2
    assert all(len(uuids) == 2 and uuids[0] == uuids[1] for uuids in attempts.values())
    assert len(delivered) == 2
