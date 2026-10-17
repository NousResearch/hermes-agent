"""Regression for #119033: deleted bot messages cannot carry the next turn's badge."""

import asyncio
from types import SimpleNamespace

import pytest

from gateway.platforms.event import MessageEvent
from tests.gateway import test_feishu

fake_lark_requests = test_feishu.fake_lark_requests


@pytest.mark.parametrize("thread_id", [None, "omt_current"])
@pytest.mark.parametrize("newer_reply", ["none", "before", "during"])
@pytest.mark.parametrize("delete_outcome", ["success", "rejected", "exception"])
def test_send_delete_next_turn_preserves_only_live_badge_targets(
    fake_lark_requests, monkeypatch, thread_id, newer_reply, delete_outcome,
):
    monkeypatch.setenv("FEISHU_REACTIONS", "true")
    adapter, tracker = test_feishu._reaction_adapter()
    deleted_messages = []
    metadata = {"thread_id": thread_id}
    event = MessageEvent(
        text="next turn", message_id="om_user",
        source=adapter.build_source(chat_id="oc_chat", thread_id=thread_id),
    )

    def delete(request):
        deleted_messages.append(request.message_id)
        if delete_outcome == "exception":
            raise OSError("transport unavailable")
        return SimpleNamespace(success=lambda: delete_outcome == "success", code=99, msg="rejected")

    adapter._client.im.v1.message.delete = delete
    run_blocking = adapter._run_blocking

    async def send_newer():
        tracker.response = SimpleNamespace(
            success=lambda: True, data=SimpleNamespace(message_id="om_new", chat_id="oc_chat"),
        )
        assert (await adapter.send("oc_chat", "newer answer", metadata=metadata)).success

    async def transport(func, *args):
        if func is delete and newer_reply == "during":
            await send_newer()
        return await run_blocking(func, *args)

    monkeypatch.setattr(adapter, "_run_blocking", transport)

    async def run():
        try:
            assert (await adapter.send("oc_chat", "preview", metadata=metadata)).message_id == "om_bot"
            adapter._last_bot_messages[("oc_other", None)] = "om_other"
            if newer_reply == "before":
                await send_newer()
            assert await adapter.delete_message("oc_chat", "om_bot") is (delete_outcome == "success")
            expected = "om_bot"
            if newer_reply != "none":
                expected = "om_new"
            elif delete_outcome == "success":
                expected = None
            assert adapter._last_bot_messages.get(("oc_chat", thread_id)) == expected
            assert adapter._last_bot_messages[("oc_other", None)] == "om_other"
            await adapter.on_processing_start(event)
            assert tracker.created_on == ([expected] if expected else [])
        finally:
            adapter._shutdown_sdk_executor()

    asyncio.run(run())
    assert deleted_messages == ["om_bot"]


def test_delete_retires_every_alias_of_the_exact_message(fake_lark_requests):
    adapter, _tracker = test_feishu._reaction_adapter()
    adapter._last_bot_messages.update({
        ("oc_chat", None): "om_preview",
        ("oc_chat", "omt_preview"): "om_preview",
        ("oc_chat", "omt_newer"): "om_new",
    })
    adapter._client.im.v1.message.delete = lambda request: SimpleNamespace(success=lambda: True)

    async def run():
        try:
            assert await adapter.delete_message("feishu_user_id:u_human", "om_preview")
            assert dict(adapter._last_bot_messages) == {("oc_chat", "omt_newer"): "om_new"}
        finally:
            adapter._shutdown_sdk_executor()

    asyncio.run(run())
