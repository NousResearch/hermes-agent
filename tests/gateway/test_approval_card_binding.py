"""A stale approval button must never approve a newer command in the same session.

Native approval cards previously recorded only the session key, and
``resolve_gateway_approval(session_key, choice)`` falls back to the oldest queued
entry when no ``request_id`` is given. A tap on a card for command A therefore
resolved whatever command B happened to be pending.

Each test deals its card through the adapter's real send path, then denies the
old request by text (the card stays visible), queues a new sensitive request,
and taps the stale card. The new request must stay pending.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.slack.adapter import SlackAdapter
from tests.gateway.relay.stub_connector import StubConnector
from tests.gateway.relay.test_relay_interactive import _event
from tools import approval
from tools.approval_gateway_wait import _ApprovalEntry

SESSION = "agent:main:discord:group:123"


@pytest.fixture
def stale_card_and_new_request():
    """An old card stays visible; a NEW request owns the queue."""
    old = _ApprovalEntry({"command": "old command", "request_id": "old"})
    new = _ApprovalEntry({"command": "new sensitive command", "request_id": "new"})
    approval._gateway_queues[SESSION] = [old]
    # The old request is resolved by text; its card remains on screen.
    assert approval.resolve_gateway_approval(SESSION, "deny", request_id="old") == 1
    approval._gateway_queues[SESSION] = [new]
    yield new
    approval._gateway_queues.pop(SESSION, None)


def _assert_not_approved(new):
    assert new.result is None, "a stale card approved a different pending command"
    assert approval._gateway_queues[SESSION] == [new]


@pytest.mark.asyncio
async def test_discord_stale_button_does_not_approve_next_request(stale_card_and_new_request):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test"))
    sent = {}

    async def send(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(id=42)

    adapter._client = SimpleNamespace(
        get_channel=lambda _: SimpleNamespace(send=send), fetch_channel=AsyncMock())
    adapter._allowed_user_ids = {"123"}
    await adapter.send_exec_approval("123", "old command", SESSION, request_id="old")

    view = sent["view"]
    view._check_auth = lambda _: True
    interaction = SimpleNamespace(
        user=SimpleNamespace(display_name="Owner"), message=SimpleNamespace(embeds=[]),
        response=SimpleNamespace(edit_message=AsyncMock()))
    with patch("plugins.platforms.discord.adapter.discord.Color.dark_grey",
               return_value=None, create=True):
        await view._resolve(interaction, "once", None, "Approved once")
    _assert_not_approved(stale_card_and_new_request)


@pytest.mark.asyncio
async def test_slack_stale_button_does_not_approve_next_request(stale_card_and_new_request):
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test-token"))
    adapter._app = MagicMock()
    client = AsyncMock()
    client.chat_postMessage = AsyncMock(return_value={"ts": "1234.1"})
    adapter._team_clients = {"T1": client}
    adapter._team_bot_user_ids = {"T1": "U_BOT"}
    adapter._channel_team = {"C1": "T1"}
    assert (await adapter.send_exec_approval(
        "C1", "old command", SESSION, request_id="old")).success
    button = client.chat_postMessage.call_args.kwargs["blocks"][1]["elements"][0]

    adapter._is_interactive_user_authorized = lambda *a, **kw: True
    body = {"message": {"ts": "1234.1", "blocks": []}, "channel": {"id": "C1"},
            "user": {"name": "Owner", "id": "U_OWNER"}}
    await adapter._handle_approval_action(AsyncMock(), body, button)
    _assert_not_approved(stale_card_and_new_request)


@pytest.mark.asyncio
async def test_whatsapp_cloud_stale_button_does_not_approve_next_request(stale_card_and_new_request):
    adapter = WhatsAppCloudAdapter.__new__(WhatsAppCloudAdapter)
    adapter._exec_approval_state = {}
    adapter._reply_best_effort = AsyncMock()
    adapter._post_message_result = AsyncMock(return_value=SimpleNamespace(success=True))
    await adapter.send_exec_approval("15551234567", "old command", SESSION, request_id="old")
    approval_id = next(iter(adapter._exec_approval_state))

    await adapter._handle_approval_tap("15551234567", {}, ["appr", approval_id, "approve"])
    _assert_not_approved(stale_card_and_new_request)


@pytest.mark.asyncio
async def test_relay_stale_button_does_not_approve_next_request(stale_card_and_new_request):
    descriptor = CapabilityDescriptor(
        contract_version=CONTRACT_VERSION, platform="telegram", label="Telegram",
        max_message_length=4096, supports_draft_streaming=False, supports_edit=True,
        supports_threads=True, markdown_dialect="markdown_v2", len_unit="utf16",
        supported_ops=("send", "prompt"))
    stub = StubConnector(descriptor)
    adapter = RelayAdapter(PlatformConfig(), descriptor, transport=stub)
    assert (await adapter.send_exec_approval(
        "c1", "old command", SESSION, request_id="old")).success
    prompt_id = stub.sent[-1]["prompt_id"]

    adapter._send_lifecycle_ack = lambda *a, **kw: None
    await adapter._consume_prompt_response(_event({"prompt_id": prompt_id, "option_id": "once"}))
    _assert_not_approved(stale_card_and_new_request)


@pytest.mark.asyncio
async def test_text_slash_approve_stays_fifo(stale_card_and_new_request):
    """Text /approve carries no request_id and must keep its FIFO behavior.

    The card path fails closed when unbound; the text path is a deliberate
    "resolve whatever is oldest" and must not be broken by this change.
    """
    assert approval.resolve_gateway_approval(SESSION, "once") == 1
    assert stale_card_and_new_request.result == "once"
