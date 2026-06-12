"""Tests for Feishu group message fixes:

1. ``group_policy`` readable from config ``extra`` dict (not only env var)
2. WebSocket fallback captures dropped group messages
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from tests.gateway.feishu_helpers import (
    install_dedup_state,
    make_adapter_skeleton,
    make_message,
    make_sender,
    stub_mention,
)


# ---------------------------------------------------------------------------
# group_policy config reading
# ---------------------------------------------------------------------------


class TestGroupPolicyConfigReading:
    """Verify that group_policy is read from config ``extra`` dict,
    falling back to the env var, then to ``'allowlist'``."""

    @patch.dict("os.environ", {"FEISHU_APP_ID": "cli_t", "FEISHU_APP_SECRET": "s"}, clear=False)
    def test_reads_from_extra(self, monkeypatch):
        monkeypatch.delenv("FEISHU_GROUP_POLICY", raising=False)
        from gateway.platforms.feishu import FeishuAdapter

        settings = FeishuAdapter._load_settings(extra={"group_policy": "open"})
        assert settings.group_policy == "open"

    @patch.dict("os.environ", {"FEISHU_APP_ID": "cli_t", "FEISHU_APP_SECRET": "s"}, clear=False)
    def test_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("FEISHU_GROUP_POLICY", "blacklist")
        from gateway.platforms.feishu import FeishuAdapter

        settings = FeishuAdapter._load_settings(extra={})
        assert settings.group_policy == "blacklist"

    @patch.dict("os.environ", {"FEISHU_APP_ID": "cli_t", "FEISHU_APP_SECRET": "s"}, clear=False)
    def test_defaults_to_allowlist(self, monkeypatch):
        monkeypatch.delenv("FEISHU_GROUP_POLICY", raising=False)
        from gateway.platforms.feishu import FeishuAdapter

        settings = FeishuAdapter._load_settings(extra={})
        assert settings.group_policy == "allowlist"

    @patch.dict("os.environ", {"FEISHU_APP_ID": "cli_t", "FEISHU_APP_SECRET": "s"}, clear=False)
    def test_extra_overrides_env(self, monkeypatch):
        monkeypatch.setenv("FEISHU_GROUP_POLICY", "allowlist")
        from gateway.platforms.feishu import FeishuAdapter

        settings = FeishuAdapter._load_settings(extra={"group_policy": "open"})
        assert settings.group_policy == "open"


# ---------------------------------------------------------------------------
# Group message admission with group_policy=open
# ---------------------------------------------------------------------------


class TestGroupPolicyOpen:
    """When group_policy is 'open', all group messages should pass admission."""

    def test_group_message_admitted_when_open(self):
        adapter = make_adapter_skeleton(group_policy="open")
        install_dedup_state(adapter)
        stub_mention(adapter, mentions_self=True)

        sender = make_sender(open_id="ou_user1")
        message = make_message(chat_type="group", chat_id="oc_group1")

        result = adapter._admit(sender, message)
        assert result is None  # None = admitted

    def test_group_message_rejected_when_allowlist_empty(self):
        adapter = make_adapter_skeleton(group_policy="allowlist")
        install_dedup_state(adapter)
        stub_mention(adapter, mentions_self=True)

        sender = make_sender(open_id="ou_user1")
        message = make_message(chat_type="group", chat_id="oc_group1")

        result = adapter._admit(sender, message)
        assert result == "group_policy_rejected"


# ---------------------------------------------------------------------------
# WebSocket fallback
# ---------------------------------------------------------------------------


class TestWebSocketFallback:
    """Verify the fallback captures group messages when the SDK dispatcher fails."""

    def test_fallback_captures_group_message_on_dispatch_error(self):
        from gateway.platforms.feishu_group_fallback import (
            patch_ws_client_for_group_messages,
        )

        # Create a mock adapter
        adapter = MagicMock()
        adapter._on_message_event = MagicMock()

        # Create a mock WS client whose _handle_data_frame raises
        ws_client = MagicMock()

        group_payload = {
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "message": {
                    "chat_type": "group",
                    "chat_id": "oc_group1",
                    "message_id": "om_test123",
                },
                "sender": {"sender_id": {"open_id": "ou_user1"}},
            },
        }

        frame = MagicMock()
        frame.payload = json.dumps(group_payload).encode("utf-8")

        original_error = Exception("dispatcher failed")

        async def failing_handle_data_frame(f):
            raise original_error

        ws_client._handle_data_frame = failing_handle_data_frame

        # Apply the patch
        patch_ws_client_for_group_messages(ws_client, adapter)

        # Run the patched handler
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(ws_client._handle_data_frame(frame))
        finally:
            loop.close()

        # Verify the fallback captured the group message
        adapter._on_message_event.assert_called_once()
        call_args = adapter._on_message_event.call_args[0][0]
        assert call_args.event.message.chat_type == "group"
        assert call_args.event.message.chat_id == "oc_group1"

    def test_fallback_does_not_capture_p2p_message(self):
        from gateway.platforms.feishu_group_fallback import (
            patch_ws_client_for_group_messages,
        )

        adapter = MagicMock()
        adapter._on_message_event = MagicMock()

        ws_client = MagicMock()

        p2p_payload = {
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "message": {
                    "chat_type": "p2p",
                    "chat_id": "oc_dm1",
                    "message_id": "om_dm123",
                },
            },
        }

        frame = MagicMock()
        frame.payload = json.dumps(p2p_payload).encode("utf-8")

        async def failing_handle_data_frame(f):
            raise Exception("dispatcher failed")

        ws_client._handle_data_frame = failing_handle_data_frame

        patch_ws_client_for_group_messages(ws_client, adapter)

        import asyncio

        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(Exception, match="dispatcher failed"):
                loop.run_until_complete(ws_client._handle_data_frame(frame))
        finally:
            loop.close()

        # P2P messages should NOT be captured by the fallback
        adapter._on_message_event.assert_not_called()

    def test_fallback_passes_through_when_dispatcher_succeeds(self):
        from gateway.platforms.feishu_group_fallback import (
            patch_ws_client_for_group_messages,
        )

        adapter = MagicMock()
        ws_client = MagicMock()

        frame = MagicMock()
        frame.payload = b'{"header": {"event_type": "im.message.receive_v1"}}'

        async def successful_handle_data_frame(f):
            return "ok"

        ws_client._handle_data_frame = successful_handle_data_frame

        patch_ws_client_for_group_messages(ws_client, adapter)

        import asyncio

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(ws_client._handle_data_frame(frame))
        finally:
            loop.close()

        assert result == "ok"

    def test_dict_to_namespace_recursive(self):
        from gateway.platforms.feishu_group_fallback import _dict_to_namespace

        data = {"a": 1, "b": {"c": "hello", "d": [1, {"e": 2}]}}
        ns = _dict_to_namespace(data)

        assert ns.a == 1
        assert ns.b.c == "hello"
        assert ns.b.d[0] == 1
        assert ns.b.d[1].e == 2
