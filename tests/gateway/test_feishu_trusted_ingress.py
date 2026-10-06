import asyncio
import json
import time
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, call

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
import plugins.platforms.feishu.adapter as feishu_module
from plugins.platforms.feishu.adapter import FeishuAdapter, TrustedFeishuIngressTicket


class _Body:
    def __init__(self, body: bytes):
        self.body = body

    async def readexactly(self, size: int) -> bytes:
        if len(self.body) < size:
            raise asyncio.IncompleteReadError(self.body, size)
        return self.body[:size]


class _Response:
    def __init__(self, *, status: int = 200, text=None, body=None):
        self.status = status
        self.text = text
        self.body = body


class _Web:
    @staticmethod
    def Response(*, status=200, text=None):
        return _Response(status=status, text=text)

    @staticmethod
    def json_response(body, *, status=200):
        return _Response(status=status, body=body)


@pytest.fixture(autouse=True)
def _reset_admitter(monkeypatch):
    monkeypatch.setattr(FeishuAdapter, "_trusted_ingress_admitter", None)
    if feishu_module.web is None:
        monkeypatch.setattr(feishu_module, "web", _Web)


def _adapter() -> FeishuAdapter:
    return FeishuAdapter(
        PlatformConfig(enabled=True, extra={"app_id": "cli_trusted", "app_secret": "secret"})
    )


def _event(kind: str, key: str = "evt_1") -> NS:
    header = NS(event_id=key)
    if kind == "message":
        return NS(
            header=header,
            event=NS(
                message=NS(message_id="om_1", chat_id="oc_1", thread_id="omt_1"),
                sender=NS(sender_id=NS(open_id="ou_actor"), sender_type="user"),
            ),
        )
    if kind == "reaction":
        return NS(
            header=header,
            event=NS(
                message_id="om_1",
                user_id=NS(open_id="ou_actor"),
                operator_type="user",
                reaction_type=NS(emoji_type="OK"),
            ),
        )
    if kind in {"button", "form"}:
        return NS(
            header=header,
            event=NS(
                token=key,
                operator=NS(open_id="ou_actor"),
                context=NS(open_chat_id="oc_1", open_message_id="om_1"),
                action=NS(
                    tag="button",
                    name="submit",
                    value={"action": "submit"},
                    form_value={} if kind == "form" else None,
                ),
            ),
        )
    if kind == "comment":
        return NS(header=header, event=NS(comment_id="comment_1"))
    return NS(header=header, event=NS(operator=NS(open_id="ou_actor"), meeting_id="meeting_1"))


def _request(payload: dict) -> NS:
    body = json.dumps(payload).encode()
    return NS(
        remote="127.0.0.1",
        headers={"Content-Type": "application/json"},
        content_length=len(body),
        content=_Body(body),
    )


@pytest.mark.parametrize(
    ("kind", "event_type", "handler_name"),
    [
        ("message", "im.message.receive_v1", "_on_message_event"),
        ("reaction", "im.message.reaction.created_v1", "_on_reaction_event"),
        ("button", "card.action.trigger", "_on_card_action_trigger"),
        ("comment", "drive.notice.comment_add_v1", "_on_drive_comment_event"),
        ("vc", "vc.bot.meeting_invited_v1", "_on_meeting_invited_event"),
    ],
)
def test_no_admitter_preserves_native_dispatch(monkeypatch, kind, event_type, handler_name):
    adapter = _adapter()
    raw = _event(kind)
    handler = Mock(return_value=object())
    monkeypatch.setattr(adapter, handler_name, handler)

    adapter._dispatch_trusted_ingress(event_type, raw, transport="websocket")

    if kind == "reaction":
        handler.assert_called_once_with(event_type, raw)
    else:
        handler.assert_called_once_with(raw)


def test_websocket_handlers_cross_the_optional_dispatch_seam(monkeypatch):
    adapter = _adapter()
    callbacks = {}

    class _Builder:
        def __getattr__(self, name):
            if not name.startswith("register_"):
                raise AttributeError(name)

            def register(*args):
                key = args[0] if name == "register_p2_customized_event" else name
                callbacks[key] = args[-1]
                return self

            return register

        def build(self):
            return self

    class _Dispatcher:
        @staticmethod
        def builder(_encrypt_key, _verification_token):
            return _Builder()

    dispatch = Mock()
    reaction = Mock()
    monkeypatch.setattr(feishu_module, "EventDispatcherHandler", _Dispatcher)
    monkeypatch.setattr(adapter, "_dispatch_trusted_ingress", dispatch)
    monkeypatch.setattr(adapter, "_on_reaction_event", reaction)
    adapter._build_event_handler()

    message = _event("message")
    card = _event("button")
    comment = _event("comment")
    meeting = _event("vc")
    reaction_data = _event("reaction")
    callbacks["register_p2_im_message_receive_v1"](message)
    callbacks["register_p2_card_action_trigger"](card)
    callbacks["drive.notice.comment_add_v1"](comment)
    callbacks["vc.bot.meeting_invited_v1"](meeting)
    callbacks["register_p2_im_message_reaction_created_v1"](reaction_data)

    assert dispatch.call_args_list == [
        call("im.message.receive_v1", message, transport="websocket"),
        call("card.action.trigger", card, transport="websocket"),
        call("drive.notice.comment_add_v1", comment, transport="websocket"),
        call("vc.bot.meeting_invited_v1", meeting, transport="websocket"),
    ]
    reaction.assert_called_once_with("im.message.reaction.created_v1", reaction_data)


@pytest.mark.parametrize(
    ("kind", "event_type"),
    [("message", "im.message.receive_v1"), ("button", "card.action.trigger"), ("form", "card.action.trigger")],
)
@pytest.mark.parametrize("transport", ["websocket", "webhook"])
def test_admitted_message_and_card_receive_signed_envelope(monkeypatch, kind, event_type, transport):
    adapter = _adapter()
    admission = object()
    admitted = []
    downstream = Mock(return_value="handled")
    handler_name = "_on_message_event" if kind == "message" else "_on_card_action_trigger"
    monkeypatch.setattr(
        FeishuAdapter,
        "_trusted_ingress_admitter",
        staticmethod(lambda *, ticket, adapter: admitted.append(ticket) or admission),
    )
    monkeypatch.setattr(adapter, handler_name, downstream)

    result = adapter._dispatch_trusted_ingress(event_type, _event(kind), transport=transport)

    assert result == "handled"
    assert len(admitted) == 1
    envelope = downstream.call_args.args[0]
    ticket = envelope.trusted_feishu_ingress_ticket
    assert ticket is admitted[0]
    assert ticket.event_kind == kind
    assert ticket.transport == transport
    assert ticket.is_valid(account_id="cli_trusted")
    assert envelope.trusted_feishu_ingress_admission is admission


@pytest.mark.parametrize(("kind", "event_type"), [("message", "im.message.receive_v1"), ("button", "card.action.trigger")])
def test_rejected_message_and_card_fail_closed_without_identifier_logs(monkeypatch, caplog, kind, event_type):
    adapter = _adapter()
    downstream = Mock()
    denied_ack = object()
    handler_name = "_on_message_event" if kind == "message" else "_on_card_action_trigger"
    monkeypatch.setattr(FeishuAdapter, "_trusted_ingress_admitter", staticmethod(lambda **_kwargs: None))
    monkeypatch.setattr(adapter, handler_name, downstream)
    monkeypatch.setattr(adapter, "_card_response", Mock(return_value=denied_ack))

    result = adapter._dispatch_trusted_ingress(event_type, _event(kind, "evt_private"), transport="websocket")

    downstream.assert_not_called()
    assert result is (denied_ack if kind == "button" else None)
    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert "ou_actor" not in rendered
    assert "evt_private" not in rendered


def test_admission_exception_fails_closed(monkeypatch):
    adapter = _adapter()
    downstream = Mock()

    def broken(**_kwargs):
        raise RuntimeError("admission unavailable")

    monkeypatch.setattr(FeishuAdapter, "_trusted_ingress_admitter", staticmethod(broken))
    monkeypatch.setattr(adapter, "_on_message_event", downstream)

    assert adapter._dispatch_trusted_ingress(
        "im.message.receive_v1", _event("message"), transport="websocket"
    ) is None
    downstream.assert_not_called()


def test_enabled_admitter_keeps_reactions_native_and_disables_comment_vc_bridges(monkeypatch, caplog):
    adapter = _adapter()
    admitter = Mock(return_value=object())
    reaction = Mock()
    comment = Mock()
    meeting = Mock()
    monkeypatch.setattr(FeishuAdapter, "_trusted_ingress_admitter", staticmethod(admitter))
    monkeypatch.setattr(adapter, "_on_reaction_event", reaction)
    monkeypatch.setattr(adapter, "_on_drive_comment_event", comment)
    monkeypatch.setattr(adapter, "_on_meeting_invited_event", meeting)

    reaction_data = _event("reaction")
    adapter._dispatch_trusted_ingress(
        "im.message.reaction.created_v1", reaction_data, transport="websocket"
    )
    adapter._dispatch_trusted_ingress(
        "drive.notice.comment_add_v1", _event("comment"), transport="websocket"
    )
    adapter._dispatch_trusted_ingress(
        "vc.bot.meeting_invited_v1", _event("vc"), transport="websocket"
    )

    reaction.assert_called_once_with("im.message.reaction.created_v1", reaction_data)
    admitter.assert_not_called()
    comment.assert_not_called()
    meeting.assert_not_called()
    assert "bridge_disabled kind=comment" in caplog.text
    assert "bridge_disabled kind=vc" in caplog.text


def test_ticket_is_signed_scoped_and_time_bounded():
    adapter = _adapter()
    raw = _event("message", "evt_same")
    websocket = adapter._issue_trusted_ingress_ticket(
        "im.message.receive_v1", raw, transport="websocket"
    )
    webhook = adapter._issue_trusted_ingress_ticket(
        "im.message.receive_v1", raw, transport="webhook"
    )

    assert websocket and webhook
    assert (websocket.namespace, websocket.event_key) == (webhook.namespace, webhook.event_key)
    assert websocket.is_valid(account_id="cli_trusted")
    assert not replace(websocket, actor_id="ou_forged").is_valid(account_id="cli_trusted")
    assert not replace(websocket, expires_at=0).is_valid(account_id="cli_trusted")
    assert not replace(websocket, chat_id="").is_valid(account_id="cli_trusted")
    assert not websocket.is_valid(account_id="cli_other")

    fields = {
        name: getattr(websocket, name)
        for name in (
            "transport", "event_kind", "event_type", "event_key", "account_id", "namespace",
            "actor_id", "actor_id_type", "principal_kind", "chat_id", "thread_id", "message_id",
        )
    }
    now = time.time()
    assert not TrustedFeishuIngressTicket.issue(
        **fields, issued_at=now, expires_at=now + 301
    ).is_valid(account_id="cli_trusted", now=now)
    assert not TrustedFeishuIngressTicket.issue(
        **fields, issued_at=now + 31, expires_at=now + 60
    ).is_valid(account_id="cli_trusted", now=now)
    assert not TrustedFeishuIngressTicket.issue(
        **fields, issued_at=float("nan"), expires_at=float("inf")
    ).is_valid(account_id="cli_trusted", now=now)


def test_trust_metadata_reaches_message_and_source_under_chat_lock(monkeypatch):
    adapter = _adapter()
    ticket = object()
    admission = object()
    raw = NS(
        trusted_feishu_ingress_ticket=ticket,
        trusted_feishu_ingress_admission=admission,
    )
    source = adapter.build_source(chat_id="oc_1", user_id="ou_actor")
    event = MessageEvent(text="hello", source=source, raw_message=raw)
    monkeypatch.setattr(adapter, "handle_message", AsyncMock())

    asyncio.run(adapter._handle_message_with_guards(event))

    assert event.trusted_feishu_ingress_ticket is ticket
    assert event.trusted_feishu_ingress_admission is admission
    assert source.trusted_feishu_ingress_ticket is ticket
    assert source.trusted_feishu_ingress_admission is admission
    adapter.handle_message.assert_awaited_once_with(event)


def test_admitted_events_bypass_text_and_media_batching(monkeypatch):
    adapter = _adapter()
    source = adapter.build_source(chat_id="oc_1", user_id="ou_actor")
    raw = NS(trusted_feishu_ingress_ticket=object(), trusted_feishu_ingress_admission=object())
    text_event = MessageEvent(text="one", source=source, raw_message=raw)
    media_event = MessageEvent(
        text="photo",
        source=source,
        raw_message=raw,
        message_type=MessageType.PHOTO,
        media_urls=["/tmp/a.jpg"],
        media_types=["image/jpeg"],
    )
    guarded = AsyncMock()
    text_batch = AsyncMock()
    media_batch = AsyncMock()
    monkeypatch.setattr(adapter, "_handle_message_with_guards", guarded)
    monkeypatch.setattr(adapter, "_enqueue_text_event", text_batch)
    monkeypatch.setattr(adapter, "_enqueue_media_event", media_batch)

    async def run():
        await adapter._dispatch_inbound_event(text_event)
        await adapter._dispatch_inbound_event(media_event)

    asyncio.run(run())

    assert guarded.await_count == 2
    text_batch.assert_not_awaited()
    media_batch.assert_not_awaited()


def test_normalized_message_retains_sender_open_id(monkeypatch):
    adapter = _adapter()
    sender_id = NS(open_id="ou_actor", user_id="u_actor", union_id="on_actor")
    message = NS(chat_id="oc_1", thread_id=None, parent_id=None, message_id="om_1")
    monkeypatch.setattr(
        adapter,
        "_extract_message_content",
        AsyncMock(return_value=("hello", MessageType.TEXT, [], [], [], [])),
    )
    monkeypatch.setattr(adapter, "get_chat_info", AsyncMock(return_value={"name": "Chat", "type": "dm"}))
    monkeypatch.setattr(
        adapter,
        "_resolve_sender_profile",
        AsyncMock(return_value={"user_id": "u_actor", "user_name": "Actor", "user_id_alt": "on_actor"}),
    )
    monkeypatch.setattr(adapter, "_dispatch_inbound_event", AsyncMock())

    asyncio.run(
        adapter._process_inbound_message(
            data=NS(event=NS()),
            message=message,
            sender_id=sender_id,
            chat_type="p2p",
            message_id="om_1",
        )
    )

    normalized = adapter._dispatch_inbound_event.await_args.args[0]
    assert normalized.sender_open_id == "ou_actor"


def test_webhook_authentication_requirement_only_applies_with_admitter(monkeypatch):
    adapter = _adapter()
    native = Mock()
    monkeypatch.setattr(adapter, "_on_message_event", native)
    payload = {
        "header": {"event_type": "im.message.receive_v1", "event_id": "evt_webhook"},
        "event": {
            "message": {"message_id": "om_1", "chat_id": "oc_1"},
            "sender": {"sender_id": {"open_id": "ou_actor"}, "sender_type": "user"},
        },
    }

    response = asyncio.run(adapter._handle_webhook_request(_request(payload)))
    assert response.status == 200
    native.assert_called_once()

    native.reset_mock()
    monkeypatch.setattr(FeishuAdapter, "_trusted_ingress_admitter", staticmethod(lambda **_kwargs: object()))
    response = asyncio.run(adapter._handle_webhook_request(_request(payload)))
    assert response.status == 503
    native.assert_not_called()


def test_authenticated_webhook_uses_trusted_dispatch_with_webhook_transport(monkeypatch):
    adapter = _adapter()
    adapter._verification_token = "verify"
    dispatch = Mock()
    monkeypatch.setattr(FeishuAdapter, "_trusted_ingress_admitter", staticmethod(lambda **_kwargs: object()))
    monkeypatch.setattr(adapter, "_dispatch_trusted_ingress", dispatch)
    payload = {
        "header": {
            "event_type": "im.message.receive_v1",
            "event_id": "evt_webhook",
            "token": "verify",
        },
        "event": {
            "message": {"message_id": "om_1", "chat_id": "oc_1"},
            "sender": {"sender_id": {"open_id": "ou_actor"}, "sender_type": "user"},
        },
    }

    response = asyncio.run(adapter._handle_webhook_request(_request(payload)))

    assert response.status == 200
    dispatch.assert_called_once()
    assert dispatch.call_args.args[0] == "im.message.receive_v1"
    assert dispatch.call_args.kwargs == {"transport": "webhook"}
