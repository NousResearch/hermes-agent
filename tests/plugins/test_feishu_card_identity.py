"""Card callbacks must keep the identity authenticated on the message path."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.pairing import PairingStore
from plugins.platforms.feishu import adapter as feishu
from plugins.platforms.feishu.adapter_identity import SenderIdentityCache
from tools import approval
from tools.approval_gateway_wait import _ApprovalEntry


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "")
    monkeypatch.setattr(feishu, "P2CardActionTriggerResponse", lambda: SimpleNamespace(card=None, toast=None))
    monkeypatch.setattr(feishu, "CallBackCard", SimpleNamespace)
    monkeypatch.setattr(feishu, "CallBackToast", SimpleNamespace)
    instance = feishu.FeishuAdapter(PlatformConfig(enabled=True))
    monkeypatch.setattr(instance, "_resolve_sender_name_from_api", AsyncMock(return_value="Alice"))
    return instance


def card_event(open_id="ou_alice", chat_id="oc_dm"):
    return SimpleNamespace(event=SimpleNamespace(
        operator=SimpleNamespace(open_id=open_id),
        context=SimpleNamespace(open_chat_id=chat_id),
        action=SimpleNamespace(value={"hermes_action": "approve_once", "approval_id": 1}),
    ))


@pytest.mark.asyncio
@pytest.mark.parametrize("paired_id", ["u_alice", "ou_alice", "on_alice"])
async def test_open_id_callback_resolves_paired_requester_and_rechecks_revocation(adapter, paired_id):
    store = PairingStore()
    code = store.generate_code("feishu", paired_id)
    assert store.approve_code("feishu", code)
    adapter.set_authorization_check(lambda user_id, *_args: store.is_approved("feishu", user_id))
    await adapter._resolve_sender_profile(SimpleNamespace(open_id="ou_alice", user_id="u_alice", union_id="on_alice"))
    adapter._loop = asyncio.get_running_loop()
    session = "agent:main:feishu:dm:oc_dm"
    entry = _ApprovalEntry({"command": "echo approved", "request_id": "request-1"})
    entry.authority = {"requester_id": "u_alice", "approvers": [], "chat_type": "dm"}
    approval._gateway_queues[session] = [entry]
    adapter._approval_state[1] = {"session_key": session, "chat_id": "oc_dm", "request_id": "request-1"}
    try:
        response = await asyncio.to_thread(adapter._on_card_action_trigger, card_event())
        assert response.card is not None
        assert await asyncio.to_thread(entry.event.wait, 2)
        assert entry.result == "once"

        # An alias is identity evidence, never a cached authorization grant.
        assert store.revoke("feishu", paired_id)
        assert adapter._validate_card_action(
            event=card_event().event, state={"chat_id": "oc_dm"}, label="approval", ident=2,
        ) is None
        adapter._approval_state[2] = {"session_key": session, "chat_id": "oc_dm"}
        await adapter._resolve_approval(
            2, "once", "Alice", open_id="ou_alice", user_id="u_alice", union_id="on_alice", chat_id="oc_dm",
        )
        assert 2 in adapter._approval_state
    finally:
        approval._gateway_queues.pop(session, None)


@pytest.mark.parametrize("case", ["unknown_user", "wrong_chat", "expired"])
def test_rejected_approval_click_has_visible_feedback(adapter, case):
    adapter._loop = SimpleNamespace(is_closed=lambda: False)
    adapter.set_authorization_check(lambda *_args: case != "unknown_user")
    if case != "expired":
        adapter._approval_state[1] = {"session_key": "session", "chat_id": "oc_dm"}
    response = adapter._on_card_action_trigger(card_event(chat_id="oc_other" if case == "wrong_chat" else "oc_dm"))
    assert response.card is None
    assert response.toast is not None
    assert response.toast.type == "warning"
    assert response.toast.content


def test_identity_evidence_is_bounded_and_does_not_cross_adapters():
    cache = SenderIdentityCache(max_entries=2)
    other = SenderIdentityCache()
    cache.remember("ou_alice", "u_alice", "on_alice")
    cache.remember("ou_bob", "u_bob", None)
    assert cache.resolve("ou_alice", "", "") == ("u_alice", "on_alice")
    assert other.resolve("ou_alice", "", "") == ("", "")
    cache.remember("ou_charlie", "u_charlie", None)
    assert cache.resolve("ou_alice", "", "") == ("", "")
    assert cache.resolve("ou_bob", "", "") == ("u_bob", "")
    assert cache.resolve("ou_alice", "u_alice", "on_alice") == ("u_alice", "on_alice")
