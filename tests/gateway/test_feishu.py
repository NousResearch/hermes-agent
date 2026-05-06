"""Tests for the Feishu gateway integration."""

import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from gateway.platforms.base import ProcessingOutcome

try:
    import lark_oapi
    _HAS_LARK_OAPI = True
except ImportError:
    _HAS_LARK_OAPI = False


def _mock_event_dispatcher_builder(mock_handler_class):
    mock_builder = Mock()
    mock_builder.register_p2_im_message_message_read_v1 = Mock(return_value=mock_builder)
    mock_builder.register_p2_im_message_receive_v1 = Mock(return_value=mock_builder)
    mock_builder.register_p2_im_message_reaction_created_v1 = Mock(return_value=mock_builder)
    mock_builder.register_p2_im_message_reaction_deleted_v1 = Mock(return_value=mock_builder)
    mock_builder.register_p2_card_action_trigger = Mock(return_value=mock_builder)
    mock_builder.build = Mock(return_value=object())
    mock_handler_class.builder = Mock(return_value=mock_builder)
    return mock_builder


class TestConfigEnvOverrides(unittest.TestCase):
    @patch.dict(os.environ, {
        "FEISHU_APP_ID": "cli_xxx",
        "FEISHU_APP_SECRET": "secret_xxx",
        "FEISHU_CONNECTION_MODE": "websocket",
        "FEISHU_DOMAIN": "feishu",
    }, clear=False)
    def test_feishu_config_loaded_from_env(self):
        from gateway.config import GatewayConfig, Platform, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        self.assertIn(Platform.FEISHU, config.platforms)
        self.assertTrue(config.platforms[Platform.FEISHU].enabled)
        self.assertEqual(config.platforms[Platform.FEISHU].extra["app_id"], "cli_xxx")
        self.assertEqual(config.platforms[Platform.FEISHU].extra["connection_mode"], "websocket")

    @patch.dict(os.environ, {
        "FEISHU_APP_ID": "cli_xxx",
        "FEISHU_APP_SECRET": "secret_xxx",
        "FEISHU_HOME_CHANNEL": "oc_xxx",
    }, clear=False)
    def test_feishu_home_channel_loaded(self):
        from gateway.config import GatewayConfig, Platform, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        home = config.platforms[Platform.FEISHU].home_channel
        self.assertIsNotNone(home)
        self.assertEqual(home.chat_id, "oc_xxx")

    @patch.dict(os.environ, {
        "FEISHU_APP_ID": "cli_xxx",
        "FEISHU_APP_SECRET": "secret_xxx",
    }, clear=False)
    def test_feishu_in_connected_platforms(self):
        from gateway.config import GatewayConfig, Platform, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)

        self.assertIn(Platform.FEISHU, config.get_connected_platforms())


class TestAdapterModule(unittest.TestCase):
    """``ws_reconnect_*`` / ``ws_ping_*`` are no longer fields on
    ``FeishuAdapterSettings``: SDK ``TransportConfig`` owns those concerns.
    The remaining test below pins the graceful-ignore behavior that lets
    deprecated env vars / extra keys not crash deployments mid-upgrade.
    """

    def test_load_settings_gracefully_ignores_deprecated_ws_keys(self):
        """Deprecated WS env vars and extra keys are read-and-discarded with
        a debug log — no crash, no attribute on the returned dataclass."""
        import logging
        from gateway.platforms.feishu import FeishuAdapter

        with self.assertLogs("gateway.platforms.feishu.adapter", level=logging.DEBUG) as captured:
            settings = FeishuAdapter._load_settings(
                {
                    "ws_reconnect_nonce": 99,
                    "ws_reconnect_interval": 7,
                    "ws_ping_interval": 5,
                    "ws_ping_timeout": 3,
                }
            )

        # Dataclass no longer carries these fields.
        self.assertFalse(hasattr(settings, "ws_reconnect_nonce"))
        self.assertFalse(hasattr(settings, "ws_reconnect_interval"))
        self.assertFalse(hasattr(settings, "ws_ping_interval"))
        self.assertFalse(hasattr(settings, "ws_ping_timeout"))

        # Each deprecated extra key triggered an "ignoring" debug log.
        joined = "\n".join(captured.output)
        self.assertIn("ws_reconnect_nonce", joined)
        self.assertIn("ws_reconnect_interval", joined)
        self.assertIn("ws_ping_interval", joined)
        self.assertIn("ws_ping_timeout", joined)


class TestProcessingReactions(unittest.TestCase):
    """Typing on start → removed on SUCCESS, swapped for CrossMark on FAILURE,
    removed (no replacement) on CANCELLED.

    ``_add_reaction`` / ``_remove_reaction`` go through
    ``adapter._channel.add_reaction`` / ``remove_reaction`` on the SDK
    ``FeishuChannel``. The mock wires async functions onto
    ``adapter._channel`` directly; no ``asyncio.to_thread`` patch is needed.
    """

    @staticmethod
    def _run(coro):
        return asyncio.run(coro)

    def _build_adapter(
        self,
        create_success: bool = True,
        delete_success: bool = True,
        next_reaction_id: str = "r1",
    ):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        tracker = SimpleNamespace(
            create_calls=[],
            delete_calls=[],
            next_reaction_id=next_reaction_id,
            create_success=create_success,
            delete_success=delete_success,
        )

        async def _add_reaction(message_id, emoji_type):
            tracker.create_calls.append(emoji_type)
            if tracker.create_success:
                # SDK SendResult shape expected by _add_reaction:
                # sdk_result.success, sdk_result.raw["data"]["reaction_id"]
                return SimpleNamespace(
                    success=True,
                    message_id=message_id,
                    error=None,
                    raw={"code": 0, "msg": "ok", "data": {"reaction_id": tracker.next_reaction_id}},
                )
            return SimpleNamespace(success=False, message_id=message_id, error="rejected", raw=None)

        async def _remove_reaction(message_id, reaction_id):
            tracker.delete_calls.append(reaction_id)
            return SimpleNamespace(
                success=tracker.delete_success,
                message_id=message_id,
                error=None if tracker.delete_success else "rejected",
                raw=None,
            )

        mock_channel = SimpleNamespace(
            add_reaction=_add_reaction,
            remove_reaction=_remove_reaction,
        )
        adapter._channel = mock_channel
        return adapter, tracker

    @staticmethod
    def _event(message_id: str = "om_msg"):
        return SimpleNamespace(message_id=message_id)

    # ------------------------------------------------------------------ start
    @patch.dict(os.environ, {}, clear=True)
    def test_start_is_idempotent_for_same_message_id(self):
        # Calling on_processing_start twice for the same message_id must only
        # add the Typing reaction once (Hermes-side cache guard).
        adapter, tracker = self._build_adapter(next_reaction_id="r_typing")
        self._run(adapter.on_processing_start(self._event()))
        self._run(adapter.on_processing_start(self._event()))
        self.assertEqual(tracker.create_calls, ["Typing"])

    @patch.dict(os.environ, {}, clear=True)
    def test_start_does_not_cache_when_create_fails(self):
        # If add_reaction returns failure, no reaction_id is cached so a
        # subsequent complete() cannot attempt to remove a non-existent badge.
        adapter, tracker = self._build_adapter(create_success=False)
        self._run(adapter.on_processing_start(self._event()))
        self.assertEqual(tracker.create_calls, ["Typing"])
        self.assertNotIn("om_msg", adapter._pending_processing_reactions)

    # --------------------------------------------------------------- complete
    @patch.dict(os.environ, {}, clear=True)
    def test_success_without_preceding_start_is_full_noop(self):
        adapter, tracker = self._build_adapter()
        self._run(
            adapter.on_processing_complete(self._event(), ProcessingOutcome.SUCCESS)
        )
        self.assertEqual(tracker.create_calls, [])
        self.assertEqual(tracker.delete_calls, [])

    @patch.dict(os.environ, {}, clear=True)
    def test_failure_without_preceding_start_still_adds_cross_mark(self):
        # No prior start → no typing badge to remove, but CrossMark must still
        # be added to signal failure to the user.
        adapter, tracker = self._build_adapter()
        self._run(
            adapter.on_processing_complete(self._event(), ProcessingOutcome.FAILURE)
        )
        self.assertEqual(tracker.create_calls, ["CrossMark"])
        self.assertEqual(tracker.delete_calls, [])

    # ------------------------- delete failure: don't stack badges -----------
    @patch.dict(os.environ, {}, clear=True)
    def test_delete_failure_on_failure_outcome_skips_cross_mark(self):
        # Removing Typing is best-effort — but if it fails, we must NOT
        # additionally add CrossMark, or the UI would show two contradictory
        # badges. The handle stays in the cache for LRU to clean up later.
        adapter, tracker = self._build_adapter(
            next_reaction_id="r_typing", delete_success=False,
        )
        self._run(adapter.on_processing_start(self._event()))
        self._run(
            adapter.on_processing_complete(self._event(), ProcessingOutcome.FAILURE)
        )
        self.assertEqual(tracker.create_calls, ["Typing"])  # CrossMark NOT added
        self.assertEqual(tracker.delete_calls, ["r_typing"])  # delete was attempted
        self.assertEqual(
            adapter._pending_processing_reactions["om_msg"], "r_typing",
        )  # handle retained

    @patch.dict(os.environ, {}, clear=True)
    def test_delete_failure_on_success_outcome_retains_handle(self):
        adapter, tracker = self._build_adapter(
            next_reaction_id="r_typing", delete_success=False,
        )
        self._run(adapter.on_processing_start(self._event()))
        self._run(
            adapter.on_processing_complete(self._event(), ProcessingOutcome.SUCCESS)
        )
        self.assertEqual(tracker.create_calls, ["Typing"])
        self.assertEqual(tracker.delete_calls, ["r_typing"])
        self.assertEqual(
            adapter._pending_processing_reactions["om_msg"], "r_typing",
        )

    # ------------------------------------------------------------- env toggle
    @patch.dict(os.environ, {"FEISHU_REACTIONS": "false"}, clear=True)
    def test_env_disable_short_circuits_both_hooks(self):
        adapter, tracker = self._build_adapter()
        self._run(adapter.on_processing_start(self._event()))
        self._run(
            adapter.on_processing_complete(self._event(), ProcessingOutcome.FAILURE)
        )
        self.assertEqual(tracker.create_calls, [])
        self.assertEqual(tracker.delete_calls, [])

    # ------------------------------------------------------------- LRU bounds
    @patch.dict(os.environ, {}, clear=True)
    def test_cache_evicts_oldest_entry_beyond_size_limit(self):
        from gateway.platforms.feishu import _FEISHU_PROCESSING_REACTION_CACHE_SIZE

        adapter, _ = self._build_adapter()
        counter = {"n": 0}

        async def _add_reaction_counting(message_id, emoji_type):
            counter["n"] += 1
            return SimpleNamespace(
                success=True,
                message_id=message_id,
                error=None,
                raw={"code": 0, "msg": "ok", "data": {"reaction_id": f"r{counter['n']}"}},
            )

        adapter._channel.add_reaction = _add_reaction_counting

        for i in range(_FEISHU_PROCESSING_REACTION_CACHE_SIZE + 1):
            self._run(adapter.on_processing_start(self._event(f"om_{i}")))

        self.assertNotIn("om_0", adapter._pending_processing_reactions)
        self.assertIn(
            f"om_{_FEISHU_PROCESSING_REACTION_CACHE_SIZE}",
            adapter._pending_processing_reactions,
        )
        self.assertEqual(
            len(adapter._pending_processing_reactions),
            _FEISHU_PROCESSING_REACTION_CACHE_SIZE,
        )


class TestFeishuMentionHint(unittest.TestCase):
    def test_hint_single_user(self):
        from gateway.platforms.feishu import FeishuMentionRef, _build_mention_hint

        refs = [FeishuMentionRef(name="Alice", open_id="ou_alice")]
        self.assertEqual(
            _build_mention_hint(refs),
            "[Mentioned: Alice (open_id=ou_alice)]",
        )

    def test_hint_multiple_users(self):
        from gateway.platforms.feishu import FeishuMentionRef, _build_mention_hint

        refs = [
            FeishuMentionRef(name="Alice", open_id="ou_alice"),
            FeishuMentionRef(name="Bob", open_id="ou_bob"),
        ]
        self.assertEqual(
            _build_mention_hint(refs),
            "[Mentioned: Alice (open_id=ou_alice), Bob (open_id=ou_bob)]",
        )

    def test_hint_at_all(self):
        from gateway.platforms.feishu import FeishuMentionRef, _build_mention_hint

        refs = [FeishuMentionRef(is_all=True)]
        self.assertEqual(_build_mention_hint(refs), "[Mentioned: @all]")

    def test_hint_filters_self_mentions(self):
        from gateway.platforms.feishu import FeishuMentionRef, _build_mention_hint

        refs = [
            FeishuMentionRef(name="Hermes", open_id="ou_bot", is_self=True),
            FeishuMentionRef(name="Alice", open_id="ou_alice"),
        ]
        self.assertEqual(
            _build_mention_hint(refs),
            "[Mentioned: Alice (open_id=ou_alice)]",
        )

    def test_hint_returns_empty_when_only_self(self):
        from gateway.platforms.feishu import FeishuMentionRef, _build_mention_hint

        refs = [FeishuMentionRef(name="Hermes", open_id="ou_bot", is_self=True)]
        self.assertEqual(_build_mention_hint(refs), "")

    def test_hint_returns_empty_for_no_refs(self):
        from gateway.platforms.feishu import _build_mention_hint

        self.assertEqual(_build_mention_hint([]), "")

    def test_hint_falls_back_when_open_id_missing(self):
        from gateway.platforms.feishu import FeishuMentionRef, _build_mention_hint

        refs = [FeishuMentionRef(name="Alice", open_id="")]
        self.assertEqual(_build_mention_hint(refs), "[Mentioned: Alice]")

    def test_hint_uses_unknown_placeholder_when_name_missing(self):
        from gateway.platforms.feishu import FeishuMentionRef, _build_mention_hint

        refs = [FeishuMentionRef(name="", open_id="ou_xxx")]
        self.assertEqual(_build_mention_hint(refs), "[Mentioned: unknown (open_id=ou_xxx)]")

    def test_hint_dedupes_repeated_user(self):
        from gateway.platforms.feishu import FeishuMentionRef, _build_mention_hint

        refs = [
            FeishuMentionRef(name="Alice", open_id="ou_alice"),
            FeishuMentionRef(name="Alice", open_id="ou_alice"),
            FeishuMentionRef(name="Bob", open_id="ou_bob"),
        ]
        self.assertEqual(
            _build_mention_hint(refs),
            "[Mentioned: Alice (open_id=ou_alice), Bob (open_id=ou_bob)]",
        )

    def test_hint_dedupes_repeated_at_all(self):
        from gateway.platforms.feishu import FeishuMentionRef, _build_mention_hint

        refs = [FeishuMentionRef(is_all=True), FeishuMentionRef(is_all=True)]
        self.assertEqual(_build_mention_hint(refs), "[Mentioned: @all]")


class TestFeishuStripLeadingSelf(unittest.TestCase):
    def _make_refs(self, *, self_name="Hermes", other_name=None):
        from gateway.platforms.feishu import FeishuMentionRef

        refs = [FeishuMentionRef(name=self_name, open_id="ou_bot", is_self=True)]
        if other_name:
            refs.append(FeishuMentionRef(name=other_name, open_id="ou_alice"))
        return refs

    def test_strips_leading_self(self):
        from gateway.platforms.feishu import _strip_edge_self_mentions

        result = _strip_edge_self_mentions("@Hermes /help", self._make_refs())
        self.assertEqual(result, "/help")

    def test_strips_consecutive_leading_self(self):
        from gateway.platforms.feishu import _strip_edge_self_mentions

        result = _strip_edge_self_mentions("@Hermes @Hermes hi", self._make_refs())
        self.assertEqual(result, "hi")

    def test_stops_at_first_non_self_token(self):
        from gateway.platforms.feishu import _strip_edge_self_mentions

        result = _strip_edge_self_mentions(
            "@Hermes @Alice make a group", self._make_refs(other_name="Alice")
        )
        self.assertEqual(result, "@Alice make a group")

    def test_preserves_mid_text_self(self):
        from gateway.platforms.feishu import _strip_edge_self_mentions

        result = _strip_edge_self_mentions("check @Hermes said yesterday", self._make_refs())
        self.assertEqual(result, "check @Hermes said yesterday")

    def test_strips_trailing_self_at_end_of_text(self):
        from gateway.platforms.feishu import _strip_edge_self_mentions

        result = _strip_edge_self_mentions("look up docs @Hermes", self._make_refs())
        self.assertEqual(result, "look up docs")

    def test_strips_trailing_self_with_terminal_punct(self):
        from gateway.platforms.feishu import _strip_edge_self_mentions

        # Terminal punct after the mention — strip the mention, keep the punct.
        result = _strip_edge_self_mentions("look up docs @Hermes.", self._make_refs())
        self.assertEqual(result, "look up docs.")

    def test_preserves_trailing_self_before_non_terminal_char(self):
        from gateway.platforms.feishu import _strip_edge_self_mentions

        # Non-terminal char (here a Chinese particle) follows — preserve.
        result = _strip_edge_self_mentions(
            "please don't @Hermes anymore", self._make_refs()
        )
        self.assertEqual(result, "please don't @Hermes anymore")

    def test_returns_input_when_refs_empty(self):
        from gateway.platforms.feishu import _strip_edge_self_mentions

        self.assertEqual(_strip_edge_self_mentions("@Hermes /help", []), "@Hermes /help")

    def test_returns_input_when_no_self_refs(self):
        from gateway.platforms.feishu import _strip_edge_self_mentions, FeishuMentionRef

        refs = [FeishuMentionRef(name="Alice", open_id="ou_alice")]
        self.assertEqual(_strip_edge_self_mentions("@Alice hi", refs), "@Alice hi")

    def test_uses_open_id_fallback_when_name_missing(self):
        from gateway.platforms.feishu import _strip_edge_self_mentions, FeishuMentionRef

        refs = [FeishuMentionRef(name="", open_id="ou_bot", is_self=True)]
        self.assertEqual(_strip_edge_self_mentions("@ou_bot hi", refs), "hi")

    def test_word_boundary_prevents_prefix_collision(self):
        """A bot named 'Al' must not eat the leading '@Alice' of a different user."""
        from gateway.platforms.feishu import _strip_edge_self_mentions, FeishuMentionRef

        refs = [FeishuMentionRef(name="Al", open_id="ou_bot", is_self=True)]
        self.assertEqual(_strip_edge_self_mentions("@Alice hi", refs), "@Alice hi")
