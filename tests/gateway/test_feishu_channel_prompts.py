"""Tests for Feishu per-channel prompt resolution.

Feishu previously ignored ``channel_prompts`` config (unlike Discord/Slack).
These tests verify that ``_resolve_channel_prompt`` reads the adapter's
``config.extra`` and that the resolved prompt is attached to the dispatched
``MessageEvent`` for the inbound, reaction, and card-action paths.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from gateway.config import PlatformConfig


def _build_adapter(extra=None):
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = FeishuAdapter.__new__(FeishuAdapter)
    adapter.config = PlatformConfig(extra=extra or {})
    adapter._bot_open_id = "ou_bot"
    adapter._bot_user_id = ""
    adapter._bot_name = "Hermes"
    adapter._download_feishu_message_resources = AsyncMock(return_value=([], []))
    adapter._fetch_message_text = AsyncMock(return_value=None)
    adapter.get_chat_info = AsyncMock(return_value={"name": "Test Chat"})
    adapter._resolve_sender_profile = AsyncMock(
        return_value={"user_id": "u1", "user_name": "Alice", "user_id_alt": None}
    )
    adapter._resolve_source_chat_type = Mock(return_value="group")
    adapter.build_source = Mock(return_value=SimpleNamespace(thread_id=None))
    adapter._dispatch_inbound_event = AsyncMock()
    return adapter


def _run_inbound(adapter, chat_id="oc_chat"):
    message = SimpleNamespace(
        content=json.dumps({"text": "plain message"}),
        message_type="text",
        message_id="m",
        mentions=[],
        chat_id=chat_id,
        parent_id=None,
        upper_message_id=None,
        thread_id=None,
    )
    asyncio.run(
        adapter._process_inbound_message(
            data=message, message=message, sender_id=None, chat_type="group", message_id="m",
        )
    )
    return adapter._dispatch_inbound_event.call_args.args[0]


def test_resolve_channel_prompt_exact_match():
    adapter = _build_adapter({"channel_prompts": {"oc_chat": "Be terse."}})
    assert adapter._resolve_channel_prompt("oc_chat") == "Be terse."


def test_resolve_channel_prompt_parent_fallback():
    adapter = _build_adapter({"channel_prompts": {"oc_parent": "Inherit me."}})
    assert adapter._resolve_channel_prompt("oc_thread", "oc_parent") == "Inherit me."


def test_resolve_channel_prompt_no_match_returns_none():
    adapter = _build_adapter({"channel_prompts": {"oc_other": "Nope."}})
    assert adapter._resolve_channel_prompt("oc_chat") is None


def test_resolve_channel_prompt_missing_config_is_safe():
    # __new__ adapter without a config attribute (defensive getattr path).
    from plugins.platforms.feishu.adapter import FeishuAdapter

    bare = FeishuAdapter.__new__(FeishuAdapter)
    assert bare._resolve_channel_prompt("oc_chat") is None


def test_inbound_event_carries_channel_prompt():
    adapter = _build_adapter({"channel_prompts": {"oc_chat": "Feishu role prompt."}})
    event = _run_inbound(adapter, chat_id="oc_chat")
    assert event.channel_prompt == "Feishu role prompt."


def test_inbound_event_no_prompt_when_unconfigured():
    adapter = _build_adapter({"channel_prompts": {"oc_other": "Different chat."}})
    event = _run_inbound(adapter, chat_id="oc_chat")
    assert event.channel_prompt is None
