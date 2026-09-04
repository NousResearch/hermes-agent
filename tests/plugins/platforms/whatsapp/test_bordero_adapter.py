from __future__ import annotations

import asyncio
import os

from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
from plugins.platforms.whatsapp.adapter import _bordero_pre_tool_call
from plugins.platforms.whatsapp.adapter import _apply_yaml_config
from gateway.config import Platform
from plugins.platforms.whatsapp.bordero_reader import (
    bordero_send_message_block,
    load_bordero_reader_config,
)


UBBO = "120363000000000001@g.us"
SALDANHA = "120363000000000002@g.us"


def _config():
    return load_bordero_reader_config(
        {
            "bordero_read_only": True,
            "bordero_routes": [
                {
                    "group_jid": UBBO,
                    "store": "PTT",
                    "location": "UBBO",
                    "telegram_chat_id": "-1003743117566",
                    "telegram_thread_id": "101",
                },
                {
                    "group_jid": SALDANHA,
                    "store": "ODI",
                    "location": "Saldanha",
                    "telegram_chat_id": "-1003743117566",
                    "telegram_thread_id": "102",
                },
            ],
        }
    )


def _adapter():
    adapter = object.__new__(WhatsAppAdapter)
    adapter._bordero_reader = _config()
    setattr(adapter, "platform", Platform.WHATSAPP)
    return adapter


def test_enabled_adapter_accepts_only_allowlisted_bordero_groups():
    adapter = _adapter()

    assert adapter._should_process_message({"isGroup": True, "chatId": UBBO}) is True
    assert adapter._should_process_message({"isGroup": True, "chatId": SALDANHA}) is True
    assert adapter._should_process_message({"isGroup": True, "chatId": "999@g.us"}) is False
    assert adapter._should_process_message({"isGroup": False, "chatId": UBBO}) is False


def test_every_whatsapp_text_send_to_bordero_group_is_suppressed_before_network():
    adapter = _adapter()
    result = asyncio.run(adapter.send(UBBO, "não pode sair"))

    assert result.success is False
    assert result.error == "bordero_read_only_whatsapp"
    assert result.raw_response == {
        "suppressed": True,
        "reason": "bordero_read_only_whatsapp",
        "chat_id": UBBO,
    }


def test_every_whatsapp_text_send_is_suppressed_when_bordero_reader_is_enabled():
    adapter = _adapter()
    result = asyncio.run(adapter.send("999@g.us", "não pode sair"))

    assert result.success is False
    assert result.error == "bordero_read_only_whatsapp"


def test_media_and_typing_paths_are_silent_for_bordero_group():
    adapter = _adapter()

    media_result = asyncio.run(
        adapter._send_media_to_bridge(UBBO, "/definitely/not/a/real/file.pdf", "document")
    )
    poll_result = asyncio.run(
        adapter.send_poll(UBBO, "confirmar?", ["sim", "não"])
    )
    asyncio.run(adapter.send_typing(UBBO))

    assert media_result.success is False
    assert media_result.error == "bordero_read_only_whatsapp"
    assert poll_result.success is False
    assert poll_result.error == "bordero_read_only_whatsapp"


def test_all_whatsapp_egress_primitives_are_suppressed_for_unknown_destination():
    adapter = _adapter()

    edit_result = asyncio.run(adapter.edit_message("999@g.us", "msg", "não editar"))
    media_result = asyncio.run(
        adapter._send_media_to_bridge("999@g.us", "/definitely/not/a/real/file.pdf", "document")
    )
    poll_result = asyncio.run(adapter.send_poll("999@g.us", "confirmar?", ["sim", "não"]))
    location_result = asyncio.run(adapter.send_location("999@g.us", 38.7, -9.1))
    clarify_result = asyncio.run(
        adapter.send_clarify("999@g.us", "escolher", ["sim", "não"], "c1", "s1")
    )

    assert [
        edit_result.error,
        media_result.error,
        poll_result.error,
        location_result.error,
        clarify_result.error,
    ] == ["bordero_read_only_whatsapp"] * 5


def test_bordero_media_routes_to_telegram_adapter_without_whatsapp_network():
    from types import SimpleNamespace

    class TelegramMediaAdapter:
        platform = Platform.TELEGRAM

        def __init__(self):
            self.calls = []

        async def send_document(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(success=True, message_id="tg-doc")

    adapter = _adapter()
    telegram = TelegramMediaAdapter()
    setattr(
        adapter,
        "gateway_runner",
        SimpleNamespace(adapters={Platform.TELEGRAM: telegram}),
    )
    metadata = {
        "hermes_delivery_route": {
            "group_jid": UBBO,
            "telegram_chat_id": "-1003743117566",
            "telegram_thread_id": "101",
            "telegram_target": "telegram:-1003743117566:101",
            "read_only": True,
        },
        "notify": True,
    }

    result = asyncio.run(
        adapter.send_document(
            UBBO,
            "/tmp/input.pdf",
            caption="borderô",
            file_name="input.pdf",
            metadata=metadata,
        )
    )

    assert result.success is True
    assert telegram.calls == [{
        "chat_id": "-1003743117566",
        "caption": "borderô",
        "reply_to": None,
        "metadata": {"thread_id": "101", "notify": True},
        "file_path": "/tmp/input.pdf",
        "file_name": "input.pdf",
    }]


def test_bordero_delivery_resolves_routed_secondary_telegram_profile():
    from types import SimpleNamespace

    adapter = _adapter()
    default_telegram = object()
    secondary_telegram = object()
    setattr(
        adapter,
        "gateway_runner",
        SimpleNamespace(
            adapters={Platform.TELEGRAM: default_telegram},
            _profile_adapters={"ops": {Platform.TELEGRAM: secondary_telegram}},
        ),
    )
    metadata = {
        "hermes_profile": "ops",
        "hermes_delivery_route": {
            "group_jid": UBBO,
            "telegram_target": "telegram:-1003743117566:101",
            "read_only": True,
        },
    }

    target, chat_id, target_metadata, error = adapter._resolve_bordero_delivery(metadata)

    assert target is secondary_telegram
    assert chat_id == "-1003743117566"
    assert target_metadata == {"thread_id": "101", "notify": True}
    assert error is None
    config = _config()
    assert bordero_send_message_block(
        "send_message",
        {"action": "send", "target": "telegram:-1003743117566:101", "message": "ok"},
        platform="whatsapp",
        chat_id=UBBO,
        config=config,
    ) is None


def test_resume_rehydrates_only_currently_configured_bordero_route():
    from types import SimpleNamespace
    from gateway.run import _rehydrate_bordero_delivery_route

    adapter = _adapter()
    source = SimpleNamespace(platform=Platform.WHATSAPP, chat_id=UBBO)
    _rehydrate_bordero_delivery_route(source, adapter)

    assert source._delivery_route == {
        "group_jid": UBBO,
        "telegram_chat_id": "-1003743117566",
        "telegram_thread_id": "101",
        "telegram_target": "telegram:-1003743117566:101",
        "read_only": True,
    }
    assert "somente leitura" in source._channel_prompt
    assert "telegram:-1003743117566:101" in source._channel_prompt

    unknown = SimpleNamespace(
        platform=Platform.WHATSAPP,
        chat_id="999@g.us",
        _delivery_route={"read_only": True, "telegram_target": "stale"},
    )
    _rehydrate_bordero_delivery_route(unknown, adapter)
    assert not hasattr(unknown, "_delivery_route")


def test_bordero_egress_blocks_wrong_transport_topic_and_reactions():
    config = _config()
    for args in (
        {"action": "send", "target": f"whatsapp:{UBBO}", "message": "no"},
        {"action": "send", "target": "telegram:-1003743117566:102", "message": "wrong topic"},
        {"action": "list"},
        {"action": "react", "target": "telegram:-1003743117566:101", "emoji": "✅"},
    ):
        reason = bordero_send_message_block(
            "send_message",
            args,
            platform="whatsapp",
            chat_id=UBBO,
            config=config,
        )
        assert isinstance(reason, str) and reason.startswith("BLOCKED:")

    enum_block = bordero_send_message_block(
        "send_message",
        {"action": "send", "target": f"whatsapp:{UBBO}", "message": "no"},
        platform=Platform.WHATSAPP,
        chat_id=UBBO,
        config=config,
    )
    assert isinstance(enum_block, str) and enum_block.startswith("BLOCKED:")


def test_egress_gate_does_not_change_ordinary_non_bordero_turns():
    config = _config()
    assert bordero_send_message_block(
        "send_message",
        {"action": "send", "target": "telegram:-1003743117566:101", "message": "ok"},
        platform="telegram",
        chat_id="-1003743117566",
        config=config,
    ) is None


def test_registered_hook_blocks_before_send_message_transport(monkeypatch):
    from types import SimpleNamespace
    from gateway.config import Platform

    extra = {
        "bordero_read_only": True,
        "bordero_routes": [
            {
                "group_jid": UBBO,
                "store": "PTT",
                "location": "UBBO",
                "telegram_chat_id": "-1003743117566",
                "telegram_thread_id": "101",
            },
            {
                "group_jid": SALDANHA,
                "store": "ODI",
                "location": "Saldanha",
                "telegram_chat_id": "-1003743117566",
                "telegram_thread_id": "102",
            },
        ],
    }
    monkeypatch.setattr(
        "gateway.session_context.get_session_env",
        lambda name, default="": {
            "HERMES_SESSION_PLATFORM": "whatsapp",
            "HERMES_SESSION_CHAT_ID": UBBO,
        }.get(name, default),
    )
    monkeypatch.setattr(
        "gateway.config.load_gateway_config",
        lambda: SimpleNamespace(
            platforms={Platform.WHATSAPP: SimpleNamespace(extra=extra)}
        ),
    )

    blocked = _bordero_pre_tool_call(
        tool_name="send_message",
        args={"action": "send", "target": f"whatsapp:{UBBO}", "message": "no"},
    )
    allowed = _bordero_pre_tool_call(
        tool_name="send_message",
        args={"action": "send", "target": "telegram:-1003743117566:101", "message": "ok"},
    )

    assert blocked["action"] == "block"
    assert blocked["message"].startswith("BLOCKED:")
    assert allowed is None


def test_whatsapp_final_text_routes_to_telegram_adapter_and_not_bridge(monkeypatch):
    class FakeTelegramAdapter:
        name = "telegram"
        platform = "telegram"

        def __init__(self):
            self.calls = []

        async def _send_with_retry(self, **kwargs):
            self.calls.append(kwargs)
            return type("Result", (), {"success": True, "message_id": "tg-1"})()

    adapter = _adapter()
    telegram = FakeTelegramAdapter()
    setattr(adapter, "gateway_runner", type("Runner", (), {"adapters": {Platform.TELEGRAM: telegram}})())
    metadata = {
        "hermes_delivery_route": {
            "group_jid": UBBO,
            "telegram_target": "telegram:-1003743117566:101",
            "read_only": True,
        },
        "notify": True,
    }

    result = asyncio.run(adapter.send(UBBO, "relatório", metadata=metadata))

    assert result.success is True
    assert telegram.calls == [{
        "chat_id": "-1003743117566",
        "content": "relatório",
        "reply_to": None,
        "metadata": {"thread_id": "101", "notify": True},
    }]


def test_standalone_sender_cannot_write_to_bordero_group():
    from types import SimpleNamespace
    from plugins.platforms.whatsapp.adapter import _standalone_send

    result = asyncio.run(
        _standalone_send(
            SimpleNamespace(extra={"bordero_read_only": True, "bordero_routes": [
                {"group_jid": UBBO, "store": "PTT", "location": "UBBO", "telegram_chat_id": "-1003743117566", "telegram_thread_id": "101"},
                {"group_jid": SALDANHA, "store": "ODI", "location": "Saldanha", "telegram_chat_id": "-1003743117566", "telegram_thread_id": "102"},
            ]}),
            UBBO,
            "não enviar",
        )
    )

    assert result == {"error": "WhatsApp Borderô reader is read-only; standalone outbound delivery is blocked"}


def test_standalone_sender_cannot_write_to_unknown_destination_when_bordero_enabled():
    from types import SimpleNamespace
    from plugins.platforms.whatsapp.adapter import _standalone_send

    result = asyncio.run(
        _standalone_send(
            SimpleNamespace(extra={"bordero_read_only": True, "bordero_routes": [
                {"group_jid": UBBO, "store": "PTT", "location": "UBBO", "telegram_chat_id": "-1003743117566", "telegram_thread_id": "101"},
                {"group_jid": SALDANHA, "store": "ODI", "location": "Saldanha", "telegram_chat_id": "-1003743117566", "telegram_thread_id": "102"},
            ]}),
            "999@g.us",
            "não enviar",
        )
    )

    assert result == {"error": "WhatsApp Borderô reader is read-only; standalone outbound delivery is blocked"}


def test_message_event_carries_origin_and_explicit_delivery_route():
    adapter = _adapter()
    event = asyncio.run(
        adapter._build_message_event({
            "isGroup": True,
            "chatId": UBBO,
            "chatName": "nome informativo que não é identidade",
            "senderId": "351900000000",
            "senderName": "Operador",
            "body": "fecho de hoje",
            "messageId": "wa-message-1",
        })
    )

    assert event is not None
    assert event.source.chat_id == UBBO
    assert event.source.platform == Platform.WHATSAPP
    assert event.metadata["bordero_reader"]["telegram_target"] == "telegram:-1003743117566:101"
    assert event.metadata["suppress_whatsapp_egress"] is True
    assert event.channel_prompt is not None
    assert "telegram:-1003743117566:101" in event.channel_prompt
    assert getattr(event.source, "_delivery_route")["group_jid"] == UBBO


def test_thread_metadata_propagates_delivery_route_without_changing_origin():
    from types import SimpleNamespace
    from gateway.platforms.base import _delivery_ledger_target, _thread_metadata_for_source

    source = SimpleNamespace(
        platform=Platform.WHATSAPP,
        chat_id=UBBO,
        chat_type="group",
        thread_id=None,
        _delivery_route={
            "group_jid": UBBO,
            "telegram_chat_id": "-1003743117566",
            "telegram_thread_id": "101",
            "telegram_target": "telegram:-1003743117566:101",
            "read_only": True,
        },
    )

    metadata = _thread_metadata_for_source(source)
    assert metadata is not None

    assert metadata["hermes_delivery_route"]["group_jid"] == UBBO
    assert metadata["disable_streaming"] is True
    assert _delivery_ledger_target(source, metadata) == (
        "telegram",
        "-1003743117566",
        "101",
    )
    assert source.platform == Platform.WHATSAPP
    assert source.chat_id == UBBO


def test_turn_runner_metadata_preserves_bordero_route_for_stream_and_status():
    from types import SimpleNamespace
    from gateway.run import GatewayRunner, _is_read_only_delivery_route

    runner = object.__new__(GatewayRunner)
    runner._thread_metadata_for_target = lambda *args, **kwargs: {"thread_id": "source"}
    source = SimpleNamespace(
        platform=Platform.WHATSAPP,
        chat_id=UBBO,
        chat_type="group",
        thread_id=None,
        message_id="wa-message-1",
        profile=None,
        _delivery_route={
            "group_jid": UBBO,
            "telegram_chat_id": "-1003743117566",
            "telegram_thread_id": "101",
            "telegram_target": "telegram:-1003743117566:101",
            "read_only": True,
        },
    )

    metadata = runner._thread_metadata_for_source(source, "wa-message-1")

    assert metadata["hermes_delivery_route"]["group_jid"] == UBBO
    assert metadata["disable_streaming"] is True
    assert _is_read_only_delivery_route(metadata) is True
    assert _is_read_only_delivery_route({"thread_id": "normal"}) is False


def test_yaml_bridge_forces_internal_safety_settings(monkeypatch):
    for name in (
        "WHATSAPP_MODE",
        "WHATSAPP_DM_POLICY",
        "WHATSAPP_GROUP_POLICY",
        "WHATSAPP_SEND_READ_RECEIPTS",
        "WHATSAPP_BORDERO_READ_ONLY",
        "WHATSAPP_BORDERO_GROUP_JIDS",
    ):
        monkeypatch.delenv(name, raising=False)

    _apply_yaml_config({}, {"bordero_read_only": True})

    assert os.environ["WHATSAPP_MODE"] == "bot"
    assert os.environ["WHATSAPP_DM_POLICY"] == "disabled"
    assert os.environ["WHATSAPP_GROUP_POLICY"] == "allowlist"
    assert os.environ["WHATSAPP_SEND_READ_RECEIPTS"] == "false"
    assert os.environ["WHATSAPP_BORDERO_READ_ONLY"] == "true"


def test_yaml_bridge_unwraps_documented_nested_platform_config(monkeypatch):
    for name in (
        "WHATSAPP_MODE",
        "WHATSAPP_DM_POLICY",
        "WHATSAPP_GROUP_POLICY",
        "WHATSAPP_SEND_READ_RECEIPTS",
        "WHATSAPP_BORDERO_READ_ONLY",
        "WHATSAPP_BORDERO_GROUP_JIDS",
    ):
        monkeypatch.delenv(name, raising=False)

    _apply_yaml_config(
        {},
        {
            "enabled": True,
            "extra": {
                "bordero_read_only": True,
                "mode": "bot",
                "dm_policy": "disabled",
                "group_policy": "allowlist",
                "send_read_receipts": False,
            },
        },
    )

    assert os.environ["WHATSAPP_MODE"] == "bot"
    assert os.environ["WHATSAPP_DM_POLICY"] == "disabled"
    assert os.environ["WHATSAPP_GROUP_POLICY"] == "allowlist"
    assert os.environ["WHATSAPP_SEND_READ_RECEIPTS"] == "false"
    assert os.environ["WHATSAPP_BORDERO_READ_ONLY"] == "true"
