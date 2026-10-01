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
def queued_entries(card_state):
    old = _ApprovalEntry({"command": "old command", "request_id": "old"})
    new = _ApprovalEntry({"command": "new sensitive command", "request_id": "new"})
    approval._gateway_queues[SESSION] = [old, new] if card_state == "current" else [new]
    yield old, new
    approval._gateway_queues.pop(SESSION, None)


def card_arguments(adapter, card_state, identity_source="both"):
    import inspect

    request_id = "new" if card_state == "current" else "old"
    if card_state == "unbound":
        return {}
    result = {}
    if identity_source != "keyword":
        result["metadata"] = {"approval_id": request_id}
    if (
        identity_source != "metadata"
        and "request_id" in inspect.signature(adapter.send_exec_approval).parameters
    ):
        result["request_id"] = request_id
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["discord", "slack", "whatsapp", "relay"])
@pytest.mark.parametrize("card_state", ["current", "stale", "unbound"])
@pytest.mark.parametrize("identity_source", ["both", "metadata", "keyword"])
async def test_native_card_resolves_only_its_exact_current_request(
    platform, card_state, identity_source, queued_entries
):
    if platform == "discord":
        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test"))
        sent = {}

        async def send(**kwargs):
            sent.update(kwargs)
            return SimpleNamespace(id=42)

        adapter._client = SimpleNamespace(
            get_channel=lambda _: SimpleNamespace(send=send), fetch_channel=AsyncMock()
        )
        adapter._allowed_user_ids = {"123"}
        await adapter.send_exec_approval(
            "123",
            "displayed command",
            SESSION,
            **card_arguments(adapter, card_state, identity_source),
        )
        view = sent["view"]
        view._check_auth = lambda _: True
        interaction = SimpleNamespace(
            user=SimpleNamespace(display_name="Owner"),
            message=SimpleNamespace(embeds=[]),
            response=SimpleNamespace(edit_message=AsyncMock()),
        )
        with patch(
            "plugins.platforms.discord.adapter.discord.Color.dark_grey",
            return_value=None,
            create=True,
        ):
            await view._resolve(interaction, "once", None, "Approved once")
    elif platform == "slack":
        adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test-token"))
        adapter._app = MagicMock()
        client = AsyncMock()
        client.chat_postMessage = AsyncMock(return_value={"ts": "1234.1"})
        adapter._team_clients = {"T1": client}
        adapter._team_bot_user_ids = {"T1": "U_BOT"}
        adapter._channel_team = {"C1": "T1"}
        assert (
            await adapter.send_exec_approval(
                "C1",
                "displayed command",
                SESSION,
                **card_arguments(adapter, card_state, identity_source),
            )
        ).success
        button = client.chat_postMessage.call_args.kwargs["blocks"][1]["elements"][0]
        adapter._is_interactive_user_authorized = lambda *a, **kw: True
        body = {
            "message": {"ts": "1234.1", "blocks": []},
            "channel": {"id": "C1"},
            "user": {"name": "Owner", "id": "U_OWNER"},
        }
        await adapter._handle_approval_action(AsyncMock(), body, button)
    elif platform == "whatsapp":
        adapter = WhatsAppCloudAdapter.__new__(WhatsAppCloudAdapter)
        adapter._exec_approval_state = {}
        adapter._reply_best_effort = AsyncMock()
        adapter._post_message_result = AsyncMock(
            return_value=SimpleNamespace(success=True)
        )
        await adapter.send_exec_approval(
            "15551234567",
            "displayed command",
            SESSION,
            **card_arguments(adapter, card_state, identity_source),
        )
        approval_id = next(iter(adapter._exec_approval_state))
        await adapter._handle_approval_tap(
            "15551234567", {}, ["appr", approval_id, "approve"]
        )
    else:
        descriptor = CapabilityDescriptor(
            contract_version=CONTRACT_VERSION,
            platform="telegram",
            label="Telegram",
            max_message_length=4096,
            supports_draft_streaming=False,
            supports_edit=True,
            supports_threads=True,
            markdown_dialect="markdown_v2",
            len_unit="utf16",
            supported_ops=("send", "prompt"),
        )
        stub = StubConnector(descriptor)
        adapter = RelayAdapter(PlatformConfig(), descriptor, transport=stub)
        assert (
            await adapter.send_exec_approval(
                "c1",
                "displayed command",
                SESSION,
                **card_arguments(adapter, card_state, identity_source),
            )
        ).success
        prompt_id = stub.sent[-1]["prompt_id"]
        adapter._send_lifecycle_ack = lambda *a, **kw: None
        await adapter._consume_prompt_response(
            _event({"prompt_id": prompt_id, "option_id": "once"})
        )

    old, new = queued_entries
    expected_new = (
        ("approve" if platform == "whatsapp" else "once")
        if card_state == "current"
        else None
    )
    assert {
        "old_result": old.result,
        "new_result": new.result,
        "remaining": [
            entry.data["request_id"]
            for entry in approval._gateway_queues.get(SESSION, [])
        ],
    } == {
        "old_result": None,
        "new_result": expected_new,
        "remaining": ["old"] if card_state == "current" else ["new"],
    }


def test_text_approval_stays_fifo():
    old = _ApprovalEntry({"command": "old command", "request_id": "old"})
    new = _ApprovalEntry({"command": "new command", "request_id": "new"})
    approval._gateway_queues[SESSION] = [old, new]
    try:
        assert approval.resolve_gateway_approval(SESSION, "once") == 1
        assert {
            "old": old.result,
            "new": new.result,
            "remaining": approval._gateway_queues[SESSION],
        } == {
            "old": "once",
            "new": None,
            "remaining": [new],
        }
    finally:
        approval._gateway_queues.pop(SESSION, None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "identity_source", ["keyword", "metadata", "both", "conflicting"]
)
async def test_native_prompt_identity_is_bound_before_delivery(identity_source):
    from gateway.platforms.base import SendResult

    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test-token"))
    adapter._app = MagicMock()
    client = AsyncMock()
    client.chat_postMessage = AsyncMock(return_value={"ts": "1234.1"})
    adapter._team_clients = {"T1": client}
    adapter._team_bot_user_ids = {"T1": "U_BOT"}
    adapter._channel_team = {"C1": "T1"}
    arguments = {}
    if identity_source in {"metadata", "both", "conflicting"}:
        arguments["metadata"] = {"approval_id": "shown-request"}
    if identity_source in {"keyword", "both", "conflicting"}:
        arguments["request_id"] = (
            "other-request" if identity_source == "conflicting" else "shown-request"
        )

    result = await adapter.send_exec_approval(
        "C1", "displayed command", SESSION, **arguments
    )
    if identity_source == "conflicting":
        assert {
            "result": result,
            "delivery_calls": client.chat_postMessage.call_args_list,
        } == {
            "result": SendResult(
                success=False, error="Approval request identities disagree"
            ),
            "delivery_calls": [],
        }
        return

    buttons = client.chat_postMessage.call_args.kwargs["blocks"][1]["elements"]
    assert {
        "success": result.success,
        "button_values": [button["value"] for button in buttons],
    } == {
        "success": True,
        "button_values": [f"ea:shown-request:{SESSION}"] * 4,
    }
