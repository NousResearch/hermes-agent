"""Tests for Slack CLI helpers."""

import json

import pytest

from hermes_cli.slack_cli import (
    _build_full_manifest,
    _invite_bot_to_channel,
    _list_channels,
    _render_manifest,
)


class TestSlackFullManifest:
    """Generated full Slack app manifest used by `hermes slack manifest`."""

    def test_app_home_messages_are_writable(self):
        manifest = _build_full_manifest("Hermes", "Your Hermes agent on Slack")

        assert manifest["features"]["app_home"] == {
            "home_tab_enabled": False,
            "messages_tab_enabled": True,
            "messages_tab_read_only_enabled": False,
        }

    def test_private_channel_directory_scope_is_included(self):
        manifest = _build_full_manifest("Hermes", "Your Hermes agent on Slack")

        bot_scopes = manifest["oauth_config"]["scopes"]["bot"]
        assert "groups:read" in bot_scopes

    def test_public_channel_join_scope_is_included(self):
        # channels:join lets the bot self-join public channels via
        # conversations.join (powering `hermes slack invite --all`).
        manifest = _build_full_manifest("Hermes", "Your Hermes agent on Slack")

        assert "channels:join" in manifest["oauth_config"]["scopes"]["bot"]

    def test_assistant_features_remain_enabled(self):
        manifest = _build_full_manifest("Hermes", "Your Hermes agent on Slack")

        assert "assistant_view" in manifest["features"]
        assert "assistant:write" in manifest["oauth_config"]["scopes"]["bot"]
        bot_events = manifest["settings"]["event_subscriptions"]["bot_events"]
        assert "assistant_thread_started" in bot_events

    def test_message_metadata_events_included(self):
        # metadata_subscriptions + bot_events enable Slack message tagging support.
        manifest = _build_full_manifest("Hermes", "Your Hermes agent on Slack")

        event_subscriptions = manifest["settings"]["event_subscriptions"]

        # metadata_subscriptions declares the event types we want to receive.
        assert "metadata_subscriptions" in event_subscriptions, (
            "manifest must declare metadata_subscriptions for Slack tagging to work"
        )
        subscription_event_types = [
            sub["event_type"] for sub in event_subscriptions["metadata_subscriptions"]
        ]
        assert "messages:hermes" in subscription_event_types

        # message_metadata_posted/updated must be in bot_events to receive the events.
        bot_events = event_subscriptions["bot_events"]
        assert "message_metadata_posted" in bot_events
        assert "message_metadata_updated" in bot_events

    def test_metadata_read_scope_included(self):
        # metadata.message:read is required to receive message_metadata events.
        manifest = _build_full_manifest("Hermes", "Your Hermes agent on Slack")

        bot_scopes = manifest["oauth_config"]["scopes"]["bot"]
        assert "metadata.message:read" in bot_scopes


class TestRenderManifest:
    def test_json_is_default(self):
        out = _render_manifest({"a": 1}, as_yaml=False)
        assert json.loads(out) == {"a": 1}

    def test_yaml_round_trips(self):
        yaml = pytest.importorskip("yaml")
        out = _render_manifest({"display_information": {"name": "Hermes"}}, as_yaml=True)
        assert "display_information:" in out
        assert yaml.safe_load(out) == {"display_information": {"name": "Hermes"}}


class _FakeSlackResponse(dict):
    """Minimal stand-in for slack_sdk's SlackResponse (behaves like a dict)."""


class _FakeClient:
    """Records calls and returns canned conversations.list pages."""

    def __init__(self, pages):
        self._pages = pages
        self.joined = []
        self.invited = []
        self.list_kwargs = []

    def conversations_list(self, **kwargs):
        self.list_kwargs.append(kwargs)
        cursor = kwargs.get("cursor", "")
        idx = int(cursor or 0)
        page = self._pages[idx]
        next_cursor = str(idx + 1) if idx + 1 < len(self._pages) else ""
        return _FakeSlackResponse(
            channels=page,
            response_metadata={"next_cursor": next_cursor},
        )

    def conversations_join(self, channel):
        self.joined.append(channel)
        return _FakeSlackResponse(ok=True)

    def conversations_invite(self, channel, users):
        self.invited.append((channel, users))
        return _FakeSlackResponse(ok=True)


class TestListChannels:
    def test_paginates_and_normalizes(self):
        client = _FakeClient(
            pages=[
                [
                    {"id": "C1", "name": "general", "is_member": True},
                    {"id": "C2", "name": "random", "is_member": False},
                ],
                [
                    {"id": "G1", "name": "secret", "is_member": False, "is_private": True},
                ],
            ]
        )
        channels = _list_channels(client)
        ids = {c["id"] for c in channels}
        assert ids == {"C1", "C2", "G1"}
        secret = next(c for c in channels if c["id"] == "G1")
        assert secret["is_private"] is True
        assert secret["is_member"] is False

    def test_no_private_excludes_groups_type(self):
        client = _FakeClient(pages=[[{"id": "C1", "name": "general", "is_member": True}]])
        _list_channels(client, include_private=False)
        assert client.list_kwargs[0]["types"] == "public_channel"

    def test_private_requests_both_types(self):
        client = _FakeClient(pages=[[{"id": "C1", "name": "general", "is_member": True}]])
        _list_channels(client, include_private=True)
        assert "private_channel" in client.list_kwargs[0]["types"]


class TestInviteBotToChannel:
    def test_public_channel_is_joined(self):
        client = _FakeClient(pages=[[]])
        ch = {"id": "C1", "name": "general", "is_private": False}
        ok, msg = _invite_bot_to_channel(client, ch, bot_user_id="U1")
        assert ok is True
        assert client.joined == ["C1"]
        assert "joined" in msg

    def test_private_without_user_token_is_skipped(self):
        client = _FakeClient(pages=[[]])
        ch = {"id": "G1", "name": "secret", "is_private": True}
        ok, msg = _invite_bot_to_channel(client, ch, bot_user_id="U1")
        assert ok is False
        assert client.joined == []
        assert "needs" in msg or "invite" in msg

    def test_private_with_user_token_is_invited(self):
        client = _FakeClient(pages=[[]])
        user_client = _FakeClient(pages=[[]])
        ch = {"id": "G1", "name": "secret", "is_private": True}
        ok, msg = _invite_bot_to_channel(
            client, ch, bot_user_id="U1", user_client=user_client
        )
        assert ok is True
        assert user_client.invited == [("G1", "U1")]

    def test_already_in_channel_is_success(self):
        # Mimic slack_sdk.errors.SlackApiError without importing the SDK:
        # any exception with a ``response`` mapping carrying ``error`` works.
        class _FakeApiError(Exception):
            def __init__(self, response):
                super().__init__(response.get("error", "error"))
                self.response = response

        class _Boom(_FakeClient):
            def conversations_join(self, channel):
                raise _FakeApiError(_FakeSlackResponse(error="already_in_channel"))

        client = _Boom(pages=[[]])
        ch = {"id": "C1", "name": "general", "is_private": False}
        ok, msg = _invite_bot_to_channel(client, ch, bot_user_id="U1")
        assert ok is True
        assert "already" in msg
