from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.group_routing import (
    GroupRoutingConfig,
    WhatsAppGroupRoutingPolicy,
    route_whatsapp_group_event,
)
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource


class FakeSessionStore:
    def __init__(self):
        self.entries = []

    def get_or_create_session(self, source):
        self.source = source
        return SimpleNamespace(session_id="session-1")

    def append_to_transcript(self, session_id, entry):
        self.entries.append((session_id, entry))


def make_event(
    text: str,
    *,
    chat_type: str = "group",
    sender_id: str = "5211111111111@s.whatsapp.net",
    mentioned_ids: list[str] | None = None,
    bot_ids: list[str] | None = None,
    quoted_participant: str | None = None,
    chat_id: str = "120363000000000@g.us",
) -> MessageEvent:
    raw = {
        "body": text,
        "chatId": chat_id,
        "chatName": "Ops Group",
        "isGroup": chat_type == "group",
        "senderId": sender_id,
        "senderName": "Sender",
        "messageId": "msg-1",
        "mentionedIds": mentioned_ids or [],
        "botIds": bot_ids or ["999999999999@s.whatsapp.net"],
    }
    if quoted_participant:
        raw["quotedParticipant"] = quoted_participant
    source = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id=chat_id,
        chat_name="Ops Group",
        chat_type=chat_type,
        user_id=sender_id,
        user_name="Sender",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        raw_message=raw,
        message_id="msg-1",
    )


def evaluate(event: MessageEvent, store: FakeSessionStore | None = None):
    return route_whatsapp_group_event(
        event,
        gateway_config={
            "gateway": {
                "whatsapp_group_routing": {
                    "enabled": True,
                    "group_mode": "mention_only",
                    "allowed_public_reply": False,
                    "allowed_dm_followup": False,
                    "noise_tolerance": "low",
                    "default_followup_channel": "private_jacob",
                    "aliases": {"hermes": ["jack", "hermes"], "jacob": ["jacob"]},
                }
            }
        },
        session_store=store,
    )


def test_group_message_addressed_to_other_people_does_not_trigger_hermes_reply():
    store = FakeSessionStore()
    event = make_event(
        "@5212222222222 @5213333333333 please coordinate delivery and confirm today",
        mentioned_ids=["5212222222222@s.whatsapp.net", "5213333333333@s.whatsapp.net"],
    )

    decision = evaluate(event, store)

    assert decision.addressee_classification == "addressed_to_specific_other_people"
    assert decision.selected_action == "no_reply_task_update"
    assert decision.public_reply_allowed is False
    assert decision.should_call_llm_reply_generator is False
    assert decision.should_call_send_message is False
    assert decision.context_updated is True
    assert decision.task_update_created is True
    assert store.entries


@pytest.mark.parametrize(
    "text",
    [
        "@5212222222222 can you help him with the setup?",
        "@5212222222222 puedes apoyarlo con la entrega?",
    ],
)
def test_can_you_puedes_resolves_to_tagged_staff_member(text):
    decision = evaluate(make_event(text, mentioned_ids=["5212222222222@s.whatsapp.net"]))

    assert decision.addressee_classification == "addressed_to_specific_other_people"
    assert decision.public_reply_allowed is False
    assert decision.should_extract_task is True
    assert decision.task_update_created is True


@pytest.mark.parametrize("text", ["Jack, follow up with the owner if they do not confirm.", "Hermes: summarize this for the group"])
def test_jacob_directly_addresses_hermes_allows_dispatch(text):
    decision = evaluate(make_event(text))

    assert decision.addressee_classification == "addressed_to_hermes"
    assert decision.gateway_action == "allow"
    assert decision.public_reply_allowed is True
    assert decision.should_call_llm_reply_generator is True


def test_participant_directly_addresses_hermes_question_allowed():
    decision = evaluate(make_event("Hermes, can you confirm who owns the next step?", sender_id="521555@s.whatsapp.net"))

    assert decision.addressee_classification == "addressed_to_hermes"
    assert decision.public_reply_allowed is True


def test_reply_to_hermes_message_allowed():
    decision = evaluate(make_event("What did you mean by that?", quoted_participant="999999999999@s.whatsapp.net"))

    assert decision.addressee_classification == "addressed_to_hermes"
    assert decision.reason == "reply_to_hermes_message"
    assert decision.gateway_action == "allow"


def test_linked_device_bot_id_matches_native_mention_id():
    event = make_event(
        "@Jack Assistant voy a sacar la batería #4068 para trae la moto de plaza galerías",
        mentioned_ids=["277365414441152@lid"],
        bot_ids=["447752478277:6@s.whatsapp.net", "277365414441152:6@lid"],
    )

    decision = evaluate(event)

    assert decision.addressee_classification == "addressed_to_hermes"
    assert decision.reason == "mentioned_hermes_metadata"
    assert decision.gateway_action == "allow"


def test_adapter_direct_trigger_marker_survives_cleaned_mention_text():
    event = make_event(
        "voy a sacar la batería #4068 para trae la moto de plaza galerías",
    )
    event.raw_message["_hermes_direct_group_trigger_reason"] = "direct_hermes_text_address"

    decision = evaluate(event)

    assert decision.addressee_classification == "addressed_to_hermes"
    assert decision.reason == "direct_hermes_text_address"
    assert decision.gateway_action == "allow"


def test_raw_jack_display_mention_survives_cleaned_event_text():
    event = make_event("voy a sacar la batería #4068 para trae la moto de plaza galerías")
    event.raw_message["body"] = "@Jack Assistant voy a sacar la batería #4068 para trae la moto de plaza galerías"

    decision = evaluate(event)

    assert decision.addressee_classification == "addressed_to_hermes"
    assert decision.reason == "direct_hermes_text_address"
    assert decision.gateway_action == "allow"


def test_raw_numeric_bot_mention_survives_cleaned_event_text():
    event = make_event(
        "voy a sacar la batería #4068 para trae la moto de plaza galerías",
        bot_ids=["447752478277:6@s.whatsapp.net", "277365414441152:6@lid"],
    )
    event.raw_message["body"] = "@277365414441152 voy a sacar la batería #4068 para trae la moto de plaza galerías"

    decision = evaluate(event)

    assert decision.addressee_classification == "addressed_to_hermes"
    assert decision.reason == "mentioned_hermes_text"
    assert decision.gateway_action == "allow"


def test_ambient_group_chatter_skips_without_task():
    decision = evaluate(make_event("Traffic is lighter than yesterday."))

    assert decision.selected_action == "ignore_non_actionable_noise"
    assert decision.should_call_llm_reply_generator is False
    assert decision.should_call_send_message is False
    assert decision.should_extract_task is False


@pytest.mark.parametrize(
    "text",
    [
        "The hydraulic jack is in the truck.",
        "Use the jack stand, not the floor jack.",
        "That jack adapter is missing.",
    ],
)
def test_word_jack_as_normal_word_is_not_assistant_invocation(text):
    decision = evaluate(make_event(text))

    assert decision.addressee_classification != "addressed_to_hermes"
    assert decision.public_reply_allowed is False
    assert decision.should_call_llm_reply_generator is False


def test_urgent_blocker_prefers_private_jacob_alert_and_updates_state():
    decision = evaluate(make_event("We are blocked because the vendor has not sent the confirmation."))

    assert decision.selected_action == "private_alert_to_jacob"
    assert decision.should_alert_jacob_privately is True
    assert decision.public_reply_allowed is False
    assert decision.task_update_created is True


@pytest.mark.parametrize(
    "text,owner,deadline,pending",
    [
        ("@521222 please handle pickup logistics tomorrow", "521222@s.whatsapp.net", "tomorrow", []),
        ("@vendor please confirm availability today", "vendor@s.whatsapp.net", "today", ["confirmation"]),
        ("@ana dale seguimiento al pago el viernes", "ana@s.whatsapp.net", "viernes", []),
    ],
)
def test_generic_task_extraction_across_scenarios(text, owner, deadline, pending):
    decision = evaluate(make_event(text, mentioned_ids=[owner]))

    task = decision.extracted_task
    assert task is not None
    assert task.owner == owner
    assert task.requested_action
    assert task.due_date == deadline
    for item in pending:
        assert item in task.pending_confirmations
    assert task.source_platform == "whatsapp"
    assert task.source_message_id == "msg-1"


@pytest.mark.parametrize(
    "text",
    [
        "@luis puedes apoyarlo",
        "@luis me confirmas",
        "@luis tienes disponibilidad mañana?",
        "@luis queda pendiente la entrega",
        "@luis avísame cuando esté listo",
        "@luis dale seguimiento",
        "@luis necesito que confirmes",
        "@luis por favor confirma",
    ],
)
def test_spanish_action_phrasing_extracts_task_without_public_reply(text):
    decision = evaluate(make_event(text, mentioned_ids=["luis@s.whatsapp.net"]))

    assert decision.public_reply_allowed is False
    assert decision.should_call_llm_reply_generator is False
    assert decision.should_extract_task is True


@pytest.mark.parametrize(
    "text",
    [
        "@sam can you help him",
        "@sam please confirm",
        "@sam are you available tomorrow?",
        "@sam follow up",
        "@sam pending confirmation",
        "@sam let me know",
        "@sam please support",
        "@sam can you handle this",
    ],
)
def test_english_action_phrasing_extracts_task_without_public_reply(text):
    decision = evaluate(make_event(text, mentioned_ids=["sam@s.whatsapp.net"]))

    assert decision.public_reply_allowed is False
    assert decision.should_call_llm_reply_generator is False
    assert decision.should_extract_task is True


class DummyAdapter(BasePlatformAdapter):
    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: dict | None = None,
    ) -> SendResult:  # pragma: no cover - should not be called
        raise AssertionError("send should not be called for skipped group messages")

    async def _keep_typing(
        self,
        chat_id: str,
        interval: float = 2.0,
        metadata=None,
        stop_event: asyncio.Event | None = None,
    ) -> None:  # pragma: no cover - should not be called
        raise AssertionError("typing should not start for skipped group messages")

    async def get_chat_info(self, chat_id: str):
        return {"name": chat_id, "type": "group"}

@pytest.mark.asyncio
async def test_no_reply_path_sends_absolutely_nothing_and_does_not_call_llm():
    adapter = DummyAdapter(PlatformConfig(enabled=True, extra={}), Platform.WHATSAPP)
    handler = AsyncMock(return_value="should not run")
    adapter.set_message_handler(handler)
    adapter.set_pre_dispatch_handler(AsyncMock(return_value={"action": "skip", "reason": "group_message_not_addressed_to_hermes"}))

    await adapter.handle_message(make_event("@sam please confirm today", mentioned_ids=["sam@s.whatsapp.net"]))
    await asyncio.sleep(0)

    handler.assert_not_awaited()
    assert adapter._background_tasks == set()


def test_direct_message_behavior_remains_intact():
    decision = evaluate(make_event("hello jack", chat_type="dm", chat_id="5211111111111@s.whatsapp.net"))

    assert decision.gateway_action == "allow"
    assert decision.should_call_llm_reply_generator is True


def test_anti_parrot_suppresses_acknowledgments_and_restatements():
    policy = WhatsAppGroupRoutingPolicy(GroupRoutingConfig())

    assert policy.should_suppress_public_reply("Noted") is True
    assert policy.should_suppress_public_reply(
        "Please coordinate delivery and confirm today",
        "@sam please coordinate delivery and confirm today",
    ) is True
    assert policy.should_suppress_public_reply("Can you confirm the delivery window?", "Please coordinate delivery") is False
