"""Inbound SimpleX files: the daemon is told to /freceive only for senders the gateway authorizes.

Drives the real adapter with the real GatewayRunner authorization callback wired as startup wires it;
only the daemon WebSocket write is stubbed.
"""
import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig

OWNER, STRANGER = 4, 9


def _chat_item(contact_id, *, file_id, file_name, msg_type="file", text="hi"):
    return {
        "chatInfo": {"type": "direct", "contact": {"contactId": contact_id, "localDisplayName": "someone"}},
        "chatItem": {
            "chatDir": {"type": "directRcv"},
            "meta": {"itemTs": "2026-01-01T00:00:00Z"},
            "content": {"type": "rcvMsgContent", "msgContent": {"type": msg_type, "text": text}},
            "file": {"fileId": file_id, "fileName": file_name},
        },
    }


@pytest.fixture
def simplex(tmp_path, monkeypatch):
    from gateway.pairing import PairingStore
    from gateway.run import GatewayRunner
    from plugins.platforms.simplex.adapter import SimplexAdapter

    for key in ("GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS", "SIMPLEX_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SIMPLEX_ALLOWED_USERS", str(OWNER))
    monkeypatch.setattr("gateway.pairing.PAIRING_DIR", tmp_path / "pairing")

    platform = Platform("simplex")
    adapter = SimplexAdapter(PlatformConfig(enabled=True, extra={"ws_url": "ws://127.0.0.1:1"}))
    sent, events = [], []

    async def send(command):
        sent.append(command)
    adapter._send_fire_and_forget = send

    async def capture(event):
        events.append(event)
    adapter.handle_message = capture
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={platform: adapter.config})
    runner.adapters = {platform: adapter}
    runner.pairing_store = PairingStore()
    adapter.set_authorization_check(runner._make_adapter_auth_check(platform))
    return adapter, sent, events


@pytest.mark.asyncio
async def test_xftp_file_is_received_only_for_authorized_senders(simplex):
    adapter, sent, _events = simplex

    for contact_id, file_id in ((STRANGER, 11), (OWNER, 12)):
        await adapter._handle_event({"resp": {
            "type": "rcvFileDescrReady", "rcvFileTransfer": {"fileId": file_id},
            "chatItem": _chat_item(contact_id, file_id=file_id, file_name="doc.pdf")}})

    assert sent == ["/freceive 12"]


@pytest.mark.asyncio
async def test_voice_note_is_received_only_for_authorized_senders(simplex):
    adapter, sent, events = simplex

    await adapter._handle_event({"resp": {"type": "newChatItems", "chatItems": [
        _chat_item(STRANGER, file_id=21, file_name="note.m4a", msg_type="voice", text="")]}})
    assert sent == []
    assert [e.source.user_id for e in events] == [str(STRANGER)]  # still dispatched so the runner can deny/pair
    assert events[0].media_urls == []

    await adapter._handle_event({"resp": {"type": "newChatItems", "chatItems": [
        _chat_item(OWNER, file_id=22, file_name="note.m4a", msg_type="voice", text="")]}})
    assert sent == ["/freceive 22"]
    assert 22 in adapter._pending_file_transfers


def _group_item(group_id, member_id, *, file_id):
    item = _chat_item(0, file_id=file_id, file_name="doc.pdf")
    item["chatInfo"] = {"type": "group", "groupInfo": {"groupId": group_id}}
    item["chatItem"]["chatDir"] = {"type": "groupRcv", "groupMember": {"memberId": member_id}}
    return item


async def _descr_ready(adapter, chat_item, file_id):
    await adapter._handle_event({"resp": {
        "type": "rcvFileDescrReady", "rcvFileTransfer": {"fileId": file_id}, "chatItem": chat_item}})


@pytest.mark.asyncio
async def test_group_file_is_received_only_from_authorized_members_of_allowed_groups(simplex):
    adapter, sent, _events = simplex
    adapter.group_allow_from = {"7"}

    await _descr_ready(adapter, _group_item(7, str(STRANGER), file_id=31), 31)
    await _descr_ready(adapter, _group_item(8, str(OWNER), file_id=32), 32)  # group not allowed
    await _descr_ready(adapter, _group_item(7, str(OWNER), file_id=33), 33)

    assert sent == ["/freceive 33"]


@pytest.mark.asyncio
async def test_unknown_chat_or_unwired_check_receives_nothing(simplex):
    adapter, sent, _events = simplex
    unknown = _chat_item(OWNER, file_id=41, file_name="doc.pdf")
    unknown["chatInfo"]["type"] = "local"
    await _descr_ready(adapter, unknown, 41)

    adapter.set_authorization_check(None)
    await _descr_ready(adapter, _chat_item(OWNER, file_id=42, file_name="doc.pdf"), 42)

    assert sent == []
