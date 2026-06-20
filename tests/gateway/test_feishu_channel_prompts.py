"""Tests for Feishu per-chat ephemeral channel_prompts.

These tests lock down two contracts:

1. ``gateway.platforms.base.resolve_channel_prompt`` works for
   Feishu-shaped ``open_chat_id`` keys (``oc_…``) when called with the
   adapter's ``config.extra`` dict.
2. ``gateway.platforms.feishu.FeishuAdapter._process_inbound_message``
   populates ``MessageEvent.channel_prompt`` from the resolved value.

We do not duplicate the resolver's full unit-test surface (already
covered indirectly by ``test_discord_channel_prompts.py``); we only
pin down the Feishu key shape and the integration site.
"""

import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.platforms.base import resolve_channel_prompt


class TestResolveChannelPromptForFeishu:
    """The shared resolver works for Feishu's ``oc_…`` chat_id namespace."""

    def test_no_channel_prompts_returns_none(self):
        assert resolve_channel_prompt({}, "oc_test", None) is None

    def test_match_by_chat_id(self):
        extra = {"channel_prompts": {"oc_research": "Research mode"}}
        assert resolve_channel_prompt(extra, "oc_research", None) == "Research mode"

    def test_blank_prompt_treated_as_absent(self):
        extra = {"channel_prompts": {"oc_research": "   "}}
        assert resolve_channel_prompt(extra, "oc_research", None) is None


class TestComposeChannelPromptWithAtTutorial:
    """``_compose_channel_prompt_for_chat`` appends the @-syntax tutorial after the configured prompt."""

    def test_no_configured_prompt_returns_tutorial_only(self):
        from gateway.platforms.feishu import (
            _FEISHU_AT_TUTORIAL,
            _compose_channel_prompt_for_chat,
        )

        assert _compose_channel_prompt_for_chat({}, "oc_test") == _FEISHU_AT_TUTORIAL

    def test_configured_prompt_appears_before_tutorial(self):
        from gateway.platforms.feishu import (
            _FEISHU_AT_TUTORIAL,
            _compose_channel_prompt_for_chat,
        )

        result = _compose_channel_prompt_for_chat(
            {"channel_prompts": {"oc_test": "Persona X"}}, "oc_test"
        )
        assert result == f"Persona X\n\n{_FEISHU_AT_TUTORIAL}"

    def test_blank_configured_prompt_treated_as_absent(self):
        from gateway.platforms.feishu import (
            _FEISHU_AT_TUTORIAL,
            _compose_channel_prompt_for_chat,
        )

        result = _compose_channel_prompt_for_chat(
            {"channel_prompts": {"oc_test": "   "}}, "oc_test"
        )
        assert result == _FEISHU_AT_TUTORIAL


class TestInboundMessageCarriesChannelPrompt(unittest.TestCase):
    """``_process_inbound_message`` must populate ``MessageEvent.channel_prompt``
    from ``self.config.extra['channel_prompts']`` keyed by ``chat_id``.
    """

    def _make_adapter_with_extra(self, extra: dict):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        config = PlatformConfig()
        config.extra = extra
        adapter = FeishuAdapter(config)

        adapter._dispatch_inbound_event = AsyncMock()
        adapter.get_chat_info = AsyncMock(
            return_value={"chat_id": "oc_test", "name": "Feishu Test", "type": "group"}
        )
        adapter._resolve_sender_profile = AsyncMock(
            return_value={"user_id": "ou_user", "user_name": "Tester", "user_id_alt": None}
        )
        return adapter

    def _make_message(self, chat_id: str = "oc_test"):
        return SimpleNamespace(
            chat_id=chat_id,
            thread_id=None,
            parent_id=None,
            upper_message_id=None,
            root_id=None,
            message_type="text",
            content='{"text":"hello"}',
            message_id="om_msg_1",
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_inbound_event_carries_channel_prompt_when_chat_id_matches(self):
        from gateway.platforms.feishu import _FEISHU_AT_TUTORIAL

        adapter = self._make_adapter_with_extra(
            {"channel_prompts": {"oc_test": "Persona X"}}
        )
        message = self._make_message(chat_id="oc_test")

        asyncio.run(
            adapter._process_inbound_message(
                data=SimpleNamespace(event=SimpleNamespace(message=message)),
                message=message,
                sender_id=SimpleNamespace(open_id="ou_user", user_id=None, union_id=None),
                is_bot=False,
                chat_type="group",
                message_id="om_msg_1",
            )
        )

        adapter._dispatch_inbound_event.assert_awaited_once()
        event = adapter._dispatch_inbound_event.await_args.args[0]
        self.assertEqual(event.channel_prompt, f"Persona X\n\n{_FEISHU_AT_TUTORIAL}")

    @patch.dict(os.environ, {}, clear=True)
    def test_inbound_event_channel_prompt_is_tutorial_only_when_chat_id_unknown(self):
        from gateway.platforms.feishu import _FEISHU_AT_TUTORIAL

        adapter = self._make_adapter_with_extra(
            {"channel_prompts": {"oc_other": "Persona X"}}
        )
        message = self._make_message(chat_id="oc_test")

        asyncio.run(
            adapter._process_inbound_message(
                data=SimpleNamespace(event=SimpleNamespace(message=message)),
                message=message,
                sender_id=SimpleNamespace(open_id="ou_user", user_id=None, union_id=None),
                is_bot=False,
                chat_type="group",
                message_id="om_msg_1",
            )
        )

        adapter._dispatch_inbound_event.assert_awaited_once()
        event = adapter._dispatch_inbound_event.await_args.args[0]
        self.assertEqual(event.channel_prompt, _FEISHU_AT_TUTORIAL)
