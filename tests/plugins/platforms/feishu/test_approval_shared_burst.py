"""Regression for #151762: a Feishu shared group burst must not bind two senders' work to one requester."""

import pytest

from hermes_agent.gateway.approval_actor import approval_refusal, refusal_text
from hermes_agent.gateway.config import GatewayConfig, Platform, PlatformConfig
from hermes_agent.gateway.platforms.event import MessageEvent, MessageType
from hermes_agent.gateway.run import GatewayRunner
from hermes_agent.gateway.session import SessionSource, build_session_context
from plugins.platforms.feishu.adapter import FeishuAdapter
from tests.gateway.test_approval_requester_binding import write_approvers
from hermes_agent.tools import approval as approval_mod
from hermes_agent.tools.approval_gateway_wait import _ApprovalEntry


@pytest.fixture(autouse=True)
def _clean_approval_queue():
    approval_mod._gateway_queues.clear()
    yield
    approval_mod._gateway_queues.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["text", "photo"])
async def test_feishu_shared_batch_mixed_senders_only_admin_can_approve(monkeypatch, kind):
    write_approvers([])
    adapter = FeishuAdapter(PlatformConfig(extra={"group_sessions_per_user": False}))

    def incoming(user_id: str, name: str, content: str) -> MessageEvent:
        source = SessionSource(
            platform=Platform.FEISHU, chat_id="oc_shared", chat_type="group",
            user_id=user_id, user_name=name,
        )
        if kind == "photo":
            return MessageEvent(
                text=content, message_type=MessageType.PHOTO, source=source,
                media_urls=[f"/cache/{user_id}.jpg"], media_types=["image/jpeg"],
            )
        return MessageEvent(text=content, message_type=MessageType.TEXT, source=source)

    alice = incoming("111", "Alice", "Summarize these notes")
    bob = incoming("222", "Bob", "Delete the backups")
    assert adapter._text_batch_key(alice) == adapter._text_batch_key(bob)

    if kind == "photo":
        monkeypatch.setattr(adapter, "_schedule_media_batch_flush", lambda _key: None)
        await adapter._enqueue_media_event(alice)
        await adapter._enqueue_media_event(bob)
        merged = adapter._pending_media_batches[adapter._media_batch_key(alice)]
    else:
        monkeypatch.setattr(adapter, "_schedule_text_batch_flush", lambda _key: None)
        await adapter._enqueue_text_event(alice)
        await adapter._enqueue_text_event(bob)
        merged = adapter._pending_text_batches[adapter._text_batch_key(alice)]

    assert "Delete the backups" in merged.text
    runner = object.__new__(GatewayRunner)
    tokens = runner._set_session_env(build_session_context(merged.source, GatewayConfig(), event=merged))
    try:
        entry = _ApprovalEntry({"command": "rm -rf backups"})
    finally:
        runner._clear_session_env(tokens)

    assert entry.authority["requester_id"] == ""
    approval_mod._gateway_queues["feishu-shared"] = [entry]
    for member in ("111", "222"):
        refusal = approval_refusal("feishu-shared", member)
        assert refusal is not None
        assert refusal_text(refusal) == "Only an admin or a listed approver can answer this approval."
    assert approval_refusal(
        "feishu-shared", "999", extra={"group_allow_admin_from": ["999"]},
    ) is None
