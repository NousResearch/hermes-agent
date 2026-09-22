"""Real SDK regression for the targeted-delivery suggestion on PR #79466."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, PlatformConfig
from gateway.pairing import PairingStore
from gateway.run import GatewayRunner
from plugins.platforms.teams.adapter import TeamsAdapter

api = pytest.importorskip("microsoft_teams.api")


def _adapter(require_mention=True):
    adapter = TeamsAdapter(PlatformConfig(enabled=True, extra={
        "client_id": "bot-id", "client_secret": "test-secret", "tenant_id": "test-tenant",
        "require_mention": require_mention,
    }))
    adapter._app = SimpleNamespace(id="bot-id")
    adapter.handle_message = AsyncMock()
    adapter._fetch_attachment_bytes = AsyncMock(return_value=b"test attachment")
    return adapter


def _activity(context, **kwargs):
    return api.MessageActivity(
        id="incoming", from_=api.Account(id="29:sender", aad_object_id="aad-sender"),
        recipient=api.Account(id="28:bot-id"),
        conversation=api.ConversationAccount(id="19:chat", conversation_type=context),
        text="hello", attachments=[api.Attachment(name="probe.txt", content_type="text/plain", content_url="https://example.com/probe.txt")],
        **kwargs,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("context", ["channel", "groupChat", "personal"])
@pytest.mark.parametrize("signal, addressed", [
    ("targeted", True), ("false", False), ("absent", False), ("truthy", False),
    ("mock", False), ("mention", True), ("bare-id", True), ("other", False),
    ("targeted-other", True), ("reply", True), ("unknown-reply", False), ("text-only", True),
])
async def test_only_addressed_shared_messages_reach_attachments(context, signal, addressed):
    adapter = _adapter()
    activity = _activity(context)
    # Assignment intentionally tests the adapter's strict boolean boundary without Pydantic coercion.
    activity.recipient.is_targeted = {"targeted": True, "targeted-other": True, "false": False,
                                      "truthy": "true", "mock": MagicMock()}.get(signal)
    mention = {"mention": "28:bot-id", "bare-id": "bot-id", "other": "29:other",
               "targeted-other": "29:other"}.get(signal)
    if mention:
        activity.entities = [api.MentionEntity(mentioned=api.Account(id=mention), text="<at>name</at>")]
        activity.text = "<at>name</at> hello"
    if signal == "text-only":
        activity.text = "<at>Hermes</at> hello"
    if signal in {"reply", "unknown-reply"}:
        activity.reply_to_id = signal
        adapter._remember_sent(SimpleNamespace(id="reply"))
    expected = addressed or context == "personal"
    await adapter._on_message(SimpleNamespace(activity=activity, conversation_ref=None))
    assert adapter.handle_message.await_count == int(expected)
    assert adapter._fetch_attachment_bytes.await_count == int(expected)


@pytest.mark.anyio
@pytest.mark.parametrize("require_mention", [True, False])
async def test_targeted_delivery_does_not_authorize_the_sender(monkeypatch, require_mention):
    for key in ("TEAMS_ALLOWED_USERS", "TEAMS_ALLOW_ALL_USERS", "TEAMS_REQUIRE_MENTION",
                "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)
    adapter = _adapter(require_mention)
    activity = _activity("channel")
    activity.recipient.is_targeted = True
    await adapter._on_message(SimpleNamespace(activity=activity, conversation_ref=None))
    adapter.handle_message.assert_awaited_once()
    source = adapter.handle_message.call_args.args[0].source
    assert source.role_authorized is False
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {adapter.platform: adapter}
    runner.pairing_store = PairingStore()
    assert runner._is_user_authorized(source) is False
    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "aad-sender")
    assert runner._is_user_authorized(source) is True
