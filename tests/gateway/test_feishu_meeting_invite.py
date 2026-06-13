"""Tests for Feishu vc.bot.meeting_invited_v1 event handling."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gateway.platforms.base import MessageEvent
from gateway.platforms.feishu_meeting_invite import (
    build_meeting_invite_prompt,
    handle_meeting_invited_event,
    parse_meeting_invited_event,
)


def _user_id(open_id, union_id="on_1", user_id="e65g874e"):
    return {
        "open_id": open_id,
        "union_id": union_id,
        "user_id": user_id,
    }


def _make_payload(event_id="evt_1"):
    return {
        "schema": "2.0",
        "header": {
            "event_id": event_id,
            "event_type": "vc.bot.meeting_invited_v1",
        },
        "event": {
            "meeting": {
                "id": "7646677832873577404",
                "topic": "赵磊的视频会议",
                "meeting_no": "884264377",
                "start_time": "1780384522000",
                "end_time": "1780384522000",
                "host_user": {
                    "id": _user_id("ou_390b35dca44816efc9afa812aaff3a69", "on_host", "e65g874e"),
                    "user_type": 1,
                    "user_role": 2,
                    "user_name": "赵磊",
                },
            },
            "bot": {
                "id": _user_id("ou_4398906db1bc4a2d7ed91b95ffb308d0", "on_bot", ""),
                "user_type": 10,
                "user_role": 0,
                "user_name": "Hermes龙虾",
            },
            "inviter": {
                "id": _user_id(
                    "ou_390b35dca44816efc9afa812aaff3a69",
                    "on_e19a19e6ffafbd54fbb3c4d251d6fa19",
                    "e65g874e",
                ),
                "user_type": 1,
                "user_role": 0,
                "user_name": "赵磊",
            },
            "invite_time": "1780388292",
        },
    }


def _make_payload_with_numeric_inviter_id():
    payload = _make_payload()
    payload["event"]["inviter"]["id"] = "3001"
    return payload


class _Adapter:
    def __init__(self, duplicate=False):
        self.duplicate = duplicate
        self.events = []
        self.dedup_keys = []
        self.profile_requests = []

    def _is_duplicate(self, key):
        self.dedup_keys.append(key)
        return self.duplicate

    def build_source(self, **kwargs):
        return SimpleNamespace(**kwargs)

    async def _resolve_sender_profile(self, sender_id):
        self.profile_requests.append(sender_id)
        return {
            "user_id": getattr(sender_id, "user_id", None) or getattr(sender_id, "open_id", None),
            "user_name": "Resolved Inviter",
            "user_id_alt": getattr(sender_id, "union_id", None),
        }

    async def _handle_message_with_guards(self, event):
        self.events.append(event)


class TestMeetingInviteParsing(unittest.TestCase):
    def test_parse_actual_payload_string_int64_fields(self):
        parsed = parse_meeting_invited_event(_make_payload())

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.event_id, "evt_1")
        self.assertEqual(parsed.meeting.id, "7646677832873577404")
        self.assertEqual(parsed.meeting.start_time_ms, 1780384522000)
        self.assertEqual(parsed.meeting.end_time_ms, 1780384522000)
        self.assertEqual(parsed.inviter.open_id, "ou_390b35dca44816efc9afa812aaff3a69")
        self.assertEqual(parsed.inviter.user_id, "e65g874e")
        self.assertEqual(parsed.inviter.union_id, "on_e19a19e6ffafbd54fbb3c4d251d6fa19")
        self.assertEqual(parsed.invite_time_s, 1780388292)

    def test_parse_body_content_payload(self):
        payload = _make_payload()
        wrapped = {
            "header": payload["header"],
            "event": {
                "body": {
                    "content": [
                        {
                            "contentType": "application/json",
                            "data": payload["event"],
                        }
                    ]
                }
            },
        }
        parsed = parse_meeting_invited_event(wrapped)

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.meeting.meeting_no, "884264377")
        self.assertEqual(parsed.inviter.open_id, "ou_390b35dca44816efc9afa812aaff3a69")

    def test_parse_requires_inviter(self):
        payload = _make_payload()
        del payload["event"]["inviter"]

        self.assertIsNone(parse_meeting_invited_event(payload))

    def test_parse_requires_meeting_no(self):
        payload = _make_payload()
        payload["event"]["meeting"]["meeting_no"] = ""

        self.assertIsNone(parse_meeting_invited_event(payload))

    def test_prompt_contains_meeting_and_inviter_context(self):
        parsed = parse_meeting_invited_event(_make_payload())
        prompt = build_meeting_invite_prompt(parsed)

        self.assertIn("You have been invited to join a meeting: 赵磊的视频会议", prompt)
        self.assertIn("Meeting Number: 884264377", prompt)
        self.assertIn("Inviter: 赵磊", prompt)
        self.assertIn("Join the meeting directly.", prompt)
        self.assertIn("You may use lark-cli and the relevant Lark/Feishu meeting skills", prompt)
        self.assertIn("Do not ask the user for confirmation", prompt)
        self.assertIn("If you cannot join the meeting", prompt)
        self.assertNotIn("ou_390b35dca44816efc9afa812aaff3a69", prompt)
        self.assertNotIn("user_id", prompt)
        self.assertNotIn("Use the Meeting Number as the primary credential", prompt)
        self.assertNotIn("meeting_id:", prompt)
        self.assertNotIn("start_time:", prompt)
        self.assertNotIn("end_time:", prompt)
        self.assertNotIn("Invite time:", prompt)


class TestMeetingInviteHandler(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_routes_as_synthetic_message_to_inviter_open_id(self):
        adapter = _Adapter()

        self._run(handle_meeting_invited_event(adapter, _make_payload()))

        self.assertEqual(adapter.dedup_keys, ["vc_invite:evt_1"])
        self.assertEqual(len(adapter.events), 1)
        event = adapter.events[0]
        self.assertIsInstance(event, MessageEvent)
        self.assertEqual(event.source.chat_id, "ou_390b35dca44816efc9afa812aaff3a69")
        self.assertEqual(event.source.chat_type, "dm")
        self.assertEqual(event.source.user_id, "e65g874e")
        self.assertEqual(event.source.user_name, "Resolved Inviter")
        self.assertEqual(event.source.chat_name, "Resolved Inviter")
        self.assertEqual(event.source.user_id_alt, "on_e19a19e6ffafbd54fbb3c4d251d6fa19")
        self.assertEqual(len(adapter.profile_requests), 1)
        self.assertEqual(adapter.profile_requests[0].open_id, "ou_390b35dca44816efc9afa812aaff3a69")
        self.assertEqual(adapter.profile_requests[0].user_id, "e65g874e")
        self.assertEqual(adapter.profile_requests[0].union_id, "on_e19a19e6ffafbd54fbb3c4d251d6fa19")
        self.assertIsNone(event.message_id)
        self.assertIn("You have been invited to join a meeting: 赵磊的视频会议", event.text)
        self.assertNotIn("{'open_id'", event.text)

    def test_duplicate_event_is_dropped(self):
        adapter = _Adapter(duplicate=True)

        self._run(handle_meeting_invited_event(adapter, _make_payload()))

        self.assertEqual(adapter.dedup_keys, ["vc_invite:evt_1"])
        self.assertEqual(adapter.events, [])

    def test_inviter_without_open_id_is_dropped(self):
        payload = _make_payload_with_numeric_inviter_id()
        adapter = _Adapter()

        self._run(handle_meeting_invited_event(adapter, payload))

        self.assertEqual(adapter.events, [])


class TestMeetingInviteSendRouting(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_feishu_user_id_prefix_sends_with_user_id_receive_type(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        created_requests = []

        class _Message:
            @staticmethod
            def create(request):
                created_requests.append(request)
                return SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_1"))

        adapter = FeishuAdapter(PlatformConfig())
        adapter._client = SimpleNamespace(
            im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(create=_Message.create)))
        )

        with patch.object(
            FeishuAdapter,
            "_build_create_message_body",
            staticmethod(lambda **kwargs: SimpleNamespace(**kwargs)),
        ), patch.object(
            FeishuAdapter,
            "_build_create_message_request",
            staticmethod(lambda receive_id_type, request_body: SimpleNamespace(
                receive_id_type=receive_id_type,
                request_body=request_body,
            )),
        ):
            self._run(adapter._send_raw_message(
                chat_id="feishu_user_id:3001",
                msg_type="text",
                payload='{"text":"ok"}',
                reply_to=None,
                metadata=None,
            ))

        self.assertEqual(created_requests[0].receive_id_type, "user_id")
        self.assertEqual(created_requests[0].request_body.receive_id, "3001")


if __name__ == "__main__":
    unittest.main()
