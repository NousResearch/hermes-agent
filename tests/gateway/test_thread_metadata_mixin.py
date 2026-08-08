"""Regression tests for the thread-metadata mixin lifted out of ``gateway/run.py``.

Covers the pure helpers extracted into ``gateway/thread_metadata_mixin.py``
(god-file decomposition campaign, shard s4, cluster c16):

* ``_thread_metadata_for_target`` — thread-metadata dict construction for
  synthetic sends (DM-topic fallback keys, Slack message_id).
* ``_thread_metadata_for_source`` — source-event wrapper that adds the Slack
  ``slack_team_id`` key.
* ``_reply_anchor_for_event`` — static passthrough to the module-level helper.

``_is_telegram_dm_topic_target`` stays on ``GatewayRunner`` (covered by the
open threads-mixin extraction PR), so the harness stubs it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import _reply_anchor_for_event as _module_reply_anchor
from gateway.thread_metadata_mixin import GatewayThreadMetadataMixin


class _ThreadMetadataHarness(GatewayThreadMetadataMixin):
    """Bare harness: only the staying ``_is_telegram_dm_topic_target`` is stubbed."""

    def __init__(self, dm_topic_target: bool = False):
        self._dm_topic_target = dm_topic_target

    def _is_telegram_dm_topic_target(
        self,
        platform,
        chat_id,
        thread_id,
        *,
        chat_type=None,
        adapter=None,
    ) -> bool:
        return self._dm_topic_target


class TestThreadMetadataForTarget:
    def test_none_thread_id_returns_none(self):
        h = _ThreadMetadataHarness()
        assert h._thread_metadata_for_target(Platform.SLACK, "c1", None) is None

    def test_plain_thread_id(self):
        h = _ThreadMetadataHarness()
        meta = h._thread_metadata_for_target(Platform.SLACK, "c1", "42")
        assert meta == {"thread_id": "42"}

    def test_telegram_dm_topic_fallback_keys(self):
        h = _ThreadMetadataHarness(dm_topic_target=True)
        meta = h._thread_metadata_for_target(
            Platform.TELEGRAM, "c1", "77", chat_type="dm", reply_to_message_id="9"
        )
        assert meta["thread_id"] == "77"
        assert meta["telegram_dm_topic_reply_fallback"] is True
        assert meta["direct_messages_topic_id"] == "77"
        assert meta["telegram_reply_to_message_id"] == "9"

    def test_telegram_dm_topic_ignores_trivial_thread_ids(self):
        # thread ids "1"/"" are the general topic — never treated as a DM lane id.
        h = _ThreadMetadataHarness(dm_topic_target=True)
        meta = h._thread_metadata_for_target(Platform.TELEGRAM, "c1", "1", chat_type="dm")
        assert meta["telegram_dm_topic_reply_fallback"] is True
        assert "direct_messages_topic_id" not in meta

    def test_slack_reply_anchor_message_id(self):
        h = _ThreadMetadataHarness()
        meta = h._thread_metadata_for_target(Platform.SLACK, "c1", "42", reply_to_message_id="5")
        assert meta["message_id"] == "5"

    def test_slack_without_reply_to_keeps_plain_metadata(self):
        h = _ThreadMetadataHarness()
        meta = h._thread_metadata_for_target(Platform.SLACK, "c1", "42")
        assert meta == {"thread_id": "42"}


class TestThreadMetadataForSource:
    def _source(self, **kwargs):
        defaults = dict(
            platform=Platform.SLACK,
            chat_id="c1",
            thread_id="42",
            chat_type=None,
            message_id="7",
            scope_id="T123",
        )
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_slack_team_id_added_from_scope_id(self):
        h = _ThreadMetadataHarness()
        meta = h._thread_metadata_for_source(self._source())
        assert meta["thread_id"] == "42"
        assert meta["slack_team_id"] == "T123"

    def test_slack_without_scope_id_keeps_plain_metadata(self):
        h = _ThreadMetadataHarness()
        source = self._source(scope_id=None)
        meta = h._thread_metadata_for_source(source)
        assert "slack_team_id" not in meta
        assert meta["thread_id"] == "42"

    def test_non_slack_platform_no_team_id(self):
        h = _ThreadMetadataHarness()
        source = self._source(platform=Platform.TELEGRAM, scope_id="T123")
        meta = h._thread_metadata_for_source(source)
        assert "slack_team_id" not in meta

    def test_reply_to_message_id_defaults_to_source_message_id(self):
        h = _ThreadMetadataHarness()
        meta = h._thread_metadata_for_source(self._source())
        # slack path adds message_id from the reply anchor
        assert meta["message_id"] == "7"


class TestReplyAnchorForEvent:
    def test_passthrough_matches_module_helper(self):
        # A bare event with only a message_id — module helper falls through to it.
        event = SimpleNamespace(
            source=None,
            raw_message=None,
            message_id="msg-1",
            reply_to_message_id=None,
        )
        assert (
            GatewayThreadMetadataMixin._reply_anchor_for_event(event)
            == _module_reply_anchor(event)
            == "msg-1"
        )

    def test_slack_no_thread_response_returns_none(self):
        event = SimpleNamespace(
            source=SimpleNamespace(platform=Platform.SLACK, thread_id=None, chat_type=None),
            raw_message={"_hermes_no_thread_response": True},
            message_id="msg-1",
            reply_to_message_id=None,
        )
        assert GatewayThreadMetadataMixin._reply_anchor_for_event(event) is None
