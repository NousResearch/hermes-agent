"""Card-action synthetic events must carry the operator's stable IDs.

Session keys prefer union_id (source.user_id_alt) over open_id — see
gateway/session.py _canonical_participant. _handle_card_action_event used to
pass union_id=None/user_id=None, so card clicks landed in open_id-keyed
sessions while text messages from the same person landed in union_id-keyed
sessions: one human, two sessions. Same union_id under two Feishu apps
(different open_ids) split the same way.
"""

import asyncio
import json
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, "/home/ubuntu/hermes-pr")

from gateway.session import Platform, SessionSource, build_session_key


class _Operator(types.SimpleNamespace):
    pass


def _make_event(operator_open_id, operator_union_id, operator_user_id=None):
    """Stub a lark_oapi P2CardActionTrigger event the way the adapter reads it."""
    operator = _Operator(
        open_id=operator_open_id, union_id=operator_union_id, user_id=operator_user_id
    )
    context = types.SimpleNamespace(open_chat_id="oc_group123")
    action = types.SimpleNamespace(tag="button", value={"cmd": "approve"})
    event = types.SimpleNamespace(token="tok-1", operator=operator, context=context, action=action)
    return types.SimpleNamespace(event=event)


class TestCardActionUnionId(unittest.TestCase):
    def test_card_action_carries_operator_union_id(self):
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter.__new__(FeishuAdapter)
        captured = {}

        async def _fake_dispatch(**kwargs):
            captured.update(kwargs)

        adapter._dispatch_synthetic_event = AsyncMock(side_effect=_fake_dispatch)
        adapter._is_card_action_duplicate = lambda token: False

        event = _make_event("ou_AAA", "on_UNION_SAME", "u_1")
        asyncio.get_event_loop().run_until_complete(adapter._handle_card_action_event(event))

        sender = captured["sender_id"]
        self.assertEqual(sender.open_id, "ou_AAA")
        self.assertEqual(sender.union_id, "on_UNION_SAME")
        self.assertEqual(sender.user_id, "u_1")

    def test_same_union_id_different_open_ids_share_session(self):
        """Two clicks from one human (same union_id, different app open_ids)
        must resolve to one session key — via the same source build the
        adapter's _dispatch_synthetic_event performs (_resolve_sender_profile
        maps union_id -> user_id_alt)."""
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter.__new__(FeishuAdapter)
        captured = []

        async def _fake_dispatch(**kwargs):
            captured.append(kwargs)

        adapter._dispatch_synthetic_event = AsyncMock(side_effect=_fake_dispatch)
        adapter._is_card_action_duplicate = lambda token: False

        for open_id in ("ou_AAA", "ou_BBB"):
            event = _make_event(open_id, "on_UNION_SAME")
            asyncio.get_event_loop().run_until_complete(adapter._handle_card_action_event(event))

        def _session_key(sender):
            # Mirror _resolve_sender_profile: primary = user_id or open_id; alt = union_id.
            primary = sender.user_id or sender.open_id
            source = SessionSource(
                platform=Platform.FEISHU, chat_id="oc_group123", chat_type="group",
                chat_name="g", user_id=primary, user_name="u",
                user_id_alt=sender.union_id,
            )
            return build_session_key(source)

        self.assertEqual(_session_key(captured[0]["sender_id"]), _session_key(captured[1]["sender_id"]))

    def test_missing_union_id_still_routes_on_open_id(self):
        """Operators whose event lacks union_id must not be dropped."""
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter.__new__(FeishuAdapter)
        captured = {}

        async def _fake_dispatch(**kwargs):
            captured.update(kwargs)

        adapter._dispatch_synthetic_event = AsyncMock(side_effect=_fake_dispatch)
        adapter._is_card_action_duplicate = lambda token: False

        event = _make_event("ou_ONLY", None, None)
        asyncio.get_event_loop().run_until_complete(adapter._handle_card_action_event(event))
        self.assertEqual(captured["sender_id"].open_id, "ou_ONLY")
        self.assertIsNone(captured["sender_id"].union_id)


if __name__ == "__main__":
    unittest.main()
