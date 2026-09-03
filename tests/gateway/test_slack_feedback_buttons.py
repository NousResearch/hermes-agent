"""Durable sink + hook event for Slack AI feedback-button clicks (#99809)."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

handler_mod = MagicMock()
handler_mod.AsyncSocketModeHandler = MagicMock
sys.modules.setdefault("slack_bolt.adapter", MagicMock())
sys.modules.setdefault("slack_bolt.adapter.socket_mode", MagicMock())
sys.modules.setdefault(
    "slack_bolt.adapter.socket_mode.async_handler", handler_mod
)
sdk_mod = MagicMock()
sdk_mod.web = MagicMock()
sdk_mod.web.async_client = MagicMock()
sdk_mod.web.async_client.AsyncWebClient = MagicMock
sys.modules.setdefault("slack_sdk", sdk_mod)
sys.modules.setdefault("slack_sdk.web", sdk_mod.web)
sys.modules.setdefault("slack_sdk.web.async_client", sdk_mod.web.async_client)
bolt_mod = MagicMock()
bolt_mod.async_app = MagicMock()
bolt_mod.async_app.AsyncApp = MagicMock
sys.modules.setdefault("slack_bolt", bolt_mod)
sys.modules.setdefault("slack_bolt.async_app", bolt_mod.async_app)

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


def _make_adapter():
    config = PlatformConfig(enabled=True, token="xoxb-test")
    adapter = SlackAdapter(config)
    adapter._app = MagicMock()
    adapter._bot_user_id = "U_BOT"
    adapter._team_clients = {"T1": AsyncMock()}
    adapter._team_bot_user_ids = {"T1": "U_BOT"}
    adapter._channel_team = {"C1": "T1"}
    return adapter


def _click_body(value="good", thread_ts=""):
    message = {"ts": "1700.0002", "text": "Answer"}
    if thread_ts:
        message["thread_ts"] = thread_ts
    return {
        "team_id": "T1",
        "channel": {"id": "C1"},
        "user": {"id": "U1"},
        "message": message,
    }


class TestFeedbackButtonsSink:
    @pytest.mark.asyncio
    async def test_click_appends_jsonl_record_under_home(self, tmp_path):
        adapter = _make_adapter()
        acked = []

        async def _ack():
            acked.append(True)

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            await adapter._handle_feedback_action(
                _ack, _click_body(thread_ts="1700.0001"), {"value": "good"}
            )

        assert acked == [True]
        sink = tmp_path / "logs" / "feedback.jsonl"
        assert sink.exists()
        records = [json.loads(line) for line in sink.read_text().splitlines()]
        assert len(records) == 1
        record = records[0]
        assert record["platform"] == "slack"
        assert record["value"] == "good"
        assert record["user_id"] == "U1"
        assert record["channel_id"] == "C1"
        assert record["message_ts"] == "1700.0002"
        assert record["thread_ts"] == "1700.0001"
        assert record["team_id"] == "T1"
        assert isinstance(record["timestamp"], float)

    @pytest.mark.asyncio
    async def test_click_without_thread_records_empty_thread_ts(self, tmp_path):
        adapter = _make_adapter()

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            await adapter._handle_feedback_action(
                _noop_ack, _click_body(), {"value": "bad"}
            )

        record = json.loads(
            (tmp_path / "logs" / "feedback.jsonl").read_text().splitlines()[0]
        )
        assert record["thread_ts"] == ""
        assert record["value"] == "bad"

    @pytest.mark.asyncio
    async def test_persist_failure_does_not_break_ack(self, tmp_path):
        adapter = _make_adapter()
        acked = []

        async def _ack():
            acked.append(True)

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            with patch(
                "plugins.platforms.slack.adapter.json.dumps",
                side_effect=OSError("disk full"),
            ):
                await adapter._handle_feedback_action(
                    _ack, _click_body(), {"value": "good"}
                )

        assert acked == [True]
        sink = tmp_path / "logs" / "feedback.jsonl"
        assert sink.read_text().strip() == ""


class TestFeedbackButtonsHook:
    @pytest.mark.asyncio
    async def test_click_emits_feedback_received_hook(self):
        adapter = _make_adapter()
        hook_events = []

        async def _hook(ctx):
            hook_events.append(ctx)

        adapter.set_reaction_handler(_hook)

        await adapter._handle_feedback_action(
            _noop_ack, _click_body(thread_ts="1700.0001"), {"value": "bad"}
        )

        assert len(hook_events) == 1
        event = hook_events[0]
        assert event["event_name"] == "feedback:received"
        assert event["platform"] == "slack"
        assert event["value"] == "bad"
        assert event["user_id"] == "U1"
        assert event["channel_id"] == "C1"
        assert event["message_ts"] == "1700.0002"
        assert event["thread_ts"] == "1700.0001"
        assert event["team_id"] == "T1"
        assert event["raw_event"]["user"]["id"] == "U1"

    @pytest.mark.asyncio
    async def test_click_without_handler_is_silent_noop(self, tmp_path):
        adapter = _make_adapter()
        assert adapter._reaction_handler is None

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            await adapter._handle_feedback_action(
                _noop_ack, _click_body(), {"value": "good"}
            )

    @pytest.mark.asyncio
    async def test_hook_failure_does_not_raise(self, tmp_path):
        adapter = _make_adapter()

        async def _boom(ctx):
            raise RuntimeError("hook consumer crashed")

        adapter.set_reaction_handler(_boom)

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            await adapter._handle_feedback_action(
                _noop_ack, _click_body(), {"value": "good"}
            )


async def _noop_ack():
    return None
