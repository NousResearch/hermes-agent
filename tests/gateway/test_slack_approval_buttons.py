"""Tests for Slack Block Kit approval buttons and thread context fetching."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is importable
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


# ---------------------------------------------------------------------------
# Minimal Slack SDK mock so SlackAdapter can be imported
# ---------------------------------------------------------------------------
def _ensure_slack_mock():
    """Wire up the minimal mocks required to import SlackAdapter."""
    if "slack_bolt" in sys.modules:
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    sys.modules["slack_bolt"] = slack_bolt
    sys.modules["slack_bolt.async_app"] = slack_bolt.async_app
    handler_mod = MagicMock()
    handler_mod.AsyncSocketModeHandler = MagicMock
    sys.modules["slack_bolt.adapter"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode.async_handler"] = handler_mod
    sdk_mod = MagicMock()
    sdk_mod.web = MagicMock()
    sdk_mod.web.async_client = MagicMock()
    sdk_mod.web.async_client.AsyncWebClient = MagicMock
    sys.modules["slack_sdk"] = sdk_mod
    sys.modules["slack_sdk.web"] = sdk_mod.web
    sys.modules["slack_sdk.web.async_client"] = sdk_mod.web.async_client


_ensure_slack_mock()

from plugins.platforms.slack.adapter import SlackAdapter
from gateway.config import PlatformConfig, Platform


def _make_adapter():
    """Create a SlackAdapter instance with mocked internals."""
    config = PlatformConfig(enabled=True, token="xoxb-test-token")
    adapter = SlackAdapter(config)
    adapter._app = MagicMock()
    adapter._bot_user_id = "U_BOT"
    adapter._team_clients = {"T1": AsyncMock()}
    adapter._team_bot_user_ids = {"T1": "U_BOT"}
    adapter._channel_team = {"C1": "T1"}
    return adapter


class _AuthRunner:
    def __init__(self, auth_fn=None):
        self._auth_fn = auth_fn or (lambda _source: True)
        self.seen_sources = []

    async def handle(self, event):
        return None

    def _is_user_authorized(self, source):
        self.seen_sources.append(source)
        return self._auth_fn(source)


def _attach_auth_runner(adapter, auth_fn=None):
    runner = _AuthRunner(auth_fn=auth_fn)
    adapter.set_message_handler(runner.handle)
    return runner


# ===========================================================================
# send_exec_approval — Block Kit buttons
# ===========================================================================

class TestSlackExecApproval:
    """Test the send_exec_approval method sends Block Kit buttons."""

    @pytest.mark.asyncio
    async def test_sends_blocks_with_buttons(self):
        adapter = _make_adapter()
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_postMessage = AsyncMock(return_value={"ts": "1234.5678"})

        result = await adapter.send_exec_approval(
            chat_id="C1",
            command="rm -rf /important",
            session_key="agent:main:slack:group:C1:1111",
            description="dangerous deletion",
        )

        assert result.success is True
        assert result.message_id == "1234.5678"

        # Verify chat_postMessage was called with blocks
        mock_client.chat_postMessage.assert_called_once()
        kwargs = mock_client.chat_postMessage.call_args[1]
        assert "blocks" in kwargs
        blocks = kwargs["blocks"]
        assert len(blocks) == 2
        assert blocks[0]["type"] == "section"
        assert "rm -rf /important" in blocks[0]["text"]["text"]
        assert "dangerous deletion" in blocks[0]["text"]["text"]
        assert blocks[1]["type"] == "actions"
        elements = blocks[1]["elements"]
        assert len(elements) == 4
        action_ids = [e["action_id"] for e in elements]
        assert "hermes_approve_once" in action_ids
        assert "hermes_approve_session" in action_ids
        assert "hermes_approve_always" in action_ids
        assert "hermes_deny" in action_ids
        # Each button carries the session key as value
        for e in elements:
            assert e["value"] == "agent:main:slack:group:C1:1111"

    @pytest.mark.asyncio
    async def test_smart_deny_owner_override_hides_persistent_buttons(self):
        adapter = _make_adapter()
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_postMessage = AsyncMock(return_value={"ts": "1234.5678"})

        await adapter.send_exec_approval(
            chat_id="C1", command="rm -rf /", session_key="s",
            allow_permanent=False, smart_denied=True,
        )

        kwargs = mock_client.chat_postMessage.call_args.kwargs
        elements = kwargs["blocks"][1]["elements"]
        assert [element["action_id"] for element in elements] == [
            "hermes_approve_once", "hermes_deny",
        ]
        assert "one operation" in kwargs["blocks"][0]["text"]["text"].lower()


# ===========================================================================
# _handle_approval_action — button click handler
# ===========================================================================

class TestSlackApprovalAction:
    """Test the approval button click handler."""


    @pytest.mark.asyncio
    async def test_truncates_inflated_original_text(self):
        """Interaction payload re-escapes HTML entities; text must be capped."""
        adapter = _make_adapter()
        _attach_auth_runner(adapter)
        adapter._approval_resolved["1.2"] = False

        # Simulate Slack re-escaping: original was ~2990 chars, but & → &amp;
        # etc. inflates it past 3000.
        inflated_text = "a" * 2990 + "&amp;" * 10  # 2990 + 50 = 3040 chars

        ack = AsyncMock()
        body = {
            "message": {"ts": "1.2", "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": inflated_text}},
            ]},
            "channel": {"id": "C1"},
            "user": {"name": "alice", "id": "U_ALICE"},
        }
        action = {"action_id": "hermes_approve_once", "value": "session-key"}

        mock_client = adapter._team_clients["T1"]
        mock_client.chat_update = AsyncMock()

        with patch("tools.approval.resolve_gateway_approval", return_value=1):
            await adapter._handle_approval_action(ack, body, action)

        update_kwargs = mock_client.chat_update.call_args[1]
        section_text = update_kwargs["blocks"][0]["text"]["text"]
        assert len(section_text) <= 3000

    @pytest.mark.asyncio
    async def test_global_allowlist_blocks_unauthorized_click(self, monkeypatch):
        adapter = _make_adapter()
        adapter._approval_resolved["1234.5678"] = False
        monkeypatch.delenv("SLACK_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("SLACK_ALLOW_ALL_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
        monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "U_OWNER")

        ack = AsyncMock()
        body = {
            "message": {"ts": "1234.5678", "blocks": []},
            "channel": {"id": "C1"},
            "user": {"name": "mallory", "id": "U_ATTACKER"},
        }
        action = {
            "action_id": "hermes_approve_once",
            "value": "agent:main:slack:group:C1:1111",
        }

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            await adapter._handle_approval_action(ack, body, action)

        ack.assert_called_once()
        mock_resolve.assert_not_called()


class TestSlackInteractiveAuth:
    def test_delegates_to_gateway_runner_auth(self):
        adapter = _make_adapter()
        runner = _attach_auth_runner(adapter, auth_fn=lambda source: source.user_id == "U_OK")

        assert adapter._is_interactive_user_authorized(
            "U_OK",
            channel_id="C1",
            user_name="operator",
        ) is True
        assert adapter._is_interactive_user_authorized(
            "U_BAD",
            channel_id="C1",
            user_name="intruder",
        ) is False

        assert len(runner.seen_sources) == 2
        assert runner.seen_sources[0].platform == Platform.SLACK
        assert runner.seen_sources[0].chat_id == "C1"
        assert runner.seen_sources[0].chat_type == "group"


class TestSlackSlashConfirmAction:


    @pytest.mark.asyncio
    async def test_truncates_inflated_original_text(self):
        """Interaction payload re-escapes HTML entities; text must be capped."""
        adapter = _make_adapter()
        _attach_auth_runner(adapter)
        adapter._approval_resolved["2222.3333"] = False

        # Simulate Slack re-escaping inflating text past 3000 chars.
        inflated_text = "b" * 2990 + "&lt;" * 10  # 2990 + 40 = 3030 chars

        ack = AsyncMock()
        body = {
            "message": {"ts": "2222.3333", "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": inflated_text}},
            ]},
            "channel": {"id": "C1"},
            "user": {"name": "owner", "id": "U_OWNER"},
        }
        action = {
            "action_id": "hermes_confirm_once",
            "value": "agent:main:slack:group:C1:1111|confirm-1",
        }

        mock_client = adapter._team_clients["T1"]
        mock_client.chat_update = AsyncMock()
        mock_client.chat_postMessage = AsyncMock()

        with patch("tools.slash_confirm.resolve", new=AsyncMock(return_value="ok")):
            await adapter._handle_slash_confirm_action(ack, body, action)

        update_kwargs = mock_client.chat_update.call_args[1]
        section_text = update_kwargs["blocks"][0]["text"]["text"]
        assert len(section_text) <= 3000


# ===========================================================================
# _fetch_thread_context
# ===========================================================================

class TestSlackThreadContext:
    """Test thread context fetching."""


    @pytest.mark.asyncio
    async def test_includes_self_bot_replies_as_assistant_on_cold_start(self):
        """Cold-start contract (issue #38861): self-bot replies in the thread
        must be included in the context, labelled with an ``[assistant]``
        prefix so the agent can reconstruct its own prior turns. This method
        only runs on the cold-start path (guarded at the call site by
        ``_has_active_session_for_thread``) — when an active session exists,
        the session history already carries those replies, so there is no
        risk of circular duplication.

        Third-party bots (e.g. deploy notifications) must still be kept,
        attributed to their display name."""
        adapter = _make_adapter()
        mock_client = adapter._team_clients["T1"]
        mock_client.conversations_replies = AsyncMock(return_value={
            "messages": [
                {"ts": "1000.0", "user": "U1", "text": "Parent"},
                # Self-bot reply -> kept on cold-start, prefixed [assistant]
                {
                    "ts": "1000.1",
                    "bot_id": "B_SELF",
                    "user": "U_BOT",
                    "text": "Previous bot self-reply",
                },
                # Third-party bot child -> kept (useful context)
                {
                    "ts": "1000.15",
                    "bot_id": "B_OTHER",
                    "user": "U_OTHER_BOT",
                    "text": "Deploy succeeded",
                },
                {"ts": "1000.2", "user": "U1", "text": "Current"},
            ]
        })
        adapter._user_name_cache = {
            ("T1", "U1"): "Alice",
            ("T1", "U_OTHER_BOT"): "DeployBot",
        }

        context = await adapter._fetch_thread_context(
            channel_id="C1", thread_ts="1000.0", current_ts="1000.2", team_id="T1"
        )

        assert "Alice: Parent" in context
        # Self-bot reply must now be included with [assistant] label
        assert "[assistant] Previous bot self-reply" in context
        # Third-party bot message must still be included
        assert "Deploy succeeded" in context
        # The [assistant] label must NOT leak to user messages
        assert "[assistant] Alice" not in context


    @pytest.mark.asyncio
    async def test_fetch_thread_context_extracts_block_kit_parent(self):
        """Bot-posted parents that put their content in ``blocks`` (Honeycomb,
        PagerDuty, Datadog, GitHub bot, etc.) used to be reduced to just the
        ``text`` field — typically only the alert title — which dropped the
        URL/button payload that makes the alert useful to an agent replying
        in the thread. The fetched context must now include bounded display
        text and actionable URLs so section text and button URLs survive."""
        adapter = _make_adapter()
        mock_client = adapter._team_clients["T1"]
        mock_client.conversations_replies = AsyncMock(return_value={
            "messages": [
                # Bot-posted alert: title in `text`, URL only in `blocks`.
                # Mirrors what Honeycomb, PagerDuty, etc. actually send.
                {
                    "ts": "1000.0",
                    "bot_id": "B_ALERT",
                    "subtype": "bot_message",
                    "username": "alertbot",
                    "text": "low_alerts (checkout)",
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "*Trigger fired:* low_alerts",
                            },
                        },
                        {
                            "type": "actions",
                            "elements": [
                                {
                                    "type": "button",
                                    "text": {"type": "plain_text", "text": "View graph"},
                                    "url": "https://example.example/view/abc123",
                                },
                            ],
                        },
                    ],
                },
                # User reply that triggered the fetch.
                {"ts": "1000.1", "user": "U1", "text": "what's going on?"},
            ]
        })
        adapter._user_name_cache = {"U1": "Alice"}

        context = await adapter._fetch_thread_context(
            channel_id="C1",
            thread_ts="1000.0",
            current_ts="1000.1",
            team_id="T1",
        )

        # Title still present.
        assert "low_alerts (checkout)" in context
        # URL from the action button must now surface.
        assert "https://example.example/view/abc123" in context
        # Marked as the thread parent.
        assert "[thread parent]" in context


    @pytest.mark.asyncio
    async def test_fetch_thread_context_includes_self_bot_replies_with_assistant_label(self):
        """Cold-start: parent (non-self bot) kept with [thread parent],
        self-bot child replies kept with [assistant] label, user replies
        kept unchanged. The cold-start path is the ONLY caller of this
        method; circular-context risk does not apply here (see #38861)."""
        adapter = _make_adapter()
        mock_client = adapter._team_clients["T1"]
        mock_client.conversations_replies = AsyncMock(return_value={
            "messages": [
                {"ts": "1000.0", "bot_id": "B_CRON", "text": "Cron summary"},
                # Self-bot child reply -> kept with [assistant] label
                {
                    "ts": "1000.1",
                    "bot_id": "B_SELF",
                    "user": "U_BOT",  # matches adapter._bot_user_id
                    "text": "Previous self reply",
                },
                # User reply -> kept
                {"ts": "1000.2", "user": "U1", "text": "Follow-up question"},
                # Current trigger (excluded by current_ts match)
                {"ts": "1000.3", "user": "U1", "text": "Current"},
            ]
        })
        adapter._user_name_cache = {("T1", "U1"): "Alice"}

        context = await adapter._fetch_thread_context(
            channel_id="C1", thread_ts="1000.0", current_ts="1000.3", team_id="T1"
        )

        assert "Cron summary" in context
        assert "[thread parent]" in context
        # Self-bot child reply is now kept with the [assistant] label
        assert "[assistant] Previous self reply" in context
        assert "Follow-up question" in context
        assert "Current" not in context

    @pytest.mark.asyncio
    async def test_fetch_thread_context_multi_workspace(self):
        """Self-bot filtering must use the per-workspace bot user id so a
        self-bot id that belongs to a different workspace does not accidentally
        filter out a legitimate message in the current workspace."""
        adapter = _make_adapter()
        # Add a second workspace with a different bot user id
        adapter._team_clients["T2"] = AsyncMock()
        adapter._team_bot_user_ids = {"T1": "U_BOT_T1", "T2": "U_BOT_T2"}
        adapter._bot_user_id = "U_BOT_T1"
        adapter._channel_team["C2"] = "T2"

        mock_client = adapter._team_clients["T2"]
        mock_client.conversations_replies = AsyncMock(return_value={
            "messages": [
                {"ts": "2000.0", "user": "U2", "text": "Parent T2"},
                # This has the *T1* bot's user id — from T2's perspective this
                # is a third-party bot, so it must be kept.
                {
                    "ts": "2000.1",
                    "bot_id": "B_FOREIGN",
                    "user": "U_BOT_T1",
                    "team": "T2",
                    "text": "Cross-workspace bot reply",
                },
                # Self-bot for T2 — kept with the [assistant] label
                {
                    "ts": "2000.2",
                    "bot_id": "B_SELF_T2",
                    "user": "U_BOT_T2",
                    "team": "T2",
                    "text": "Own T2 bot reply",
                },
                {"ts": "2000.3", "user": "U2", "text": "Current"},
            ]
        })
        adapter._user_name_cache = {("T2", "U2"): "Bob"}

        context = await adapter._fetch_thread_context(
            channel_id="C2", thread_ts="2000.0", current_ts="2000.3", team_id="T2"
        )

        assert "Parent T2" in context
        assert "Cross-workspace bot reply" in context
        # T2's own self-bot reply is kept with the [assistant] label
        # (cold-start path includes self-replies; see #38861). The
        # per-workspace filter still applies: this assertion confirms
        # we use T2's bot id, not T1's, when deciding what counts as
        # self-bot.
        assert "[assistant] Own T2 bot reply" in context


# ===========================================================================
# _has_active_session_for_thread — session key fix (#5833)
# ===========================================================================

class TestSessionKeyFix:
    """Test that _has_active_session_for_thread uses build_session_key."""


    def test_stale_session_returns_false(self):
        """A session key that exists but would be rolled by the reset policy
        must NOT count as active — otherwise the reset-time first turn skips
        the thread-history reseed (#55239)."""
        adapter = _make_adapter()

        class Store:
            config = MagicMock()
            config.group_sessions_per_user = False
            config.thread_sessions_per_user = False
            _entries = {
                "agent:main:slack:group:C1:1000.0": MagicMock()
            }

            def _ensure_loaded(self):
                return None

            def _should_reset(self, entry, source):
                return "idle"

        adapter._session_store = Store()

        result = adapter._has_active_session_for_thread(
            channel_id="C1", thread_ts="1000.0", user_id="U123"
        )

        assert result is False


class TestSessionKeyChatType:
    """Test that _has_active_session_for_thread passes event-derived chat_type.

    Regression for #39527: the old code hardcoded ``chat_type="group"``,
    which produced wrong session keys for DM and MPIM threads.  The fix
    passes the event-derived ``chat_type`` so ``build_session_key()``
    constructs the correct key for every channel type.
    """


    def test_dm_thread_not_found_with_group_type(self):
        """Without chat_type='dm', a DM session key would not match.

        This is the exact bug that the old ``hardcoded "group"`` code caused:
        the lookup builds ``group:…`` while the real session is ``dm:…``.
        """
        adapter = _make_adapter()
        mock_store = MagicMock()
        mock_store._entries = {
            "agent:main:slack:dm:D0DMCHANNEL:2000.0": MagicMock()
        }
        mock_store._ensure_loaded = MagicMock()
        mock_store.config = MagicMock()
        mock_store.config.group_sessions_per_user = True
        mock_store.config.thread_sessions_per_user = False
        adapter._session_store = mock_store

        # Default chat_type="group" should NOT find the DM session
        result = adapter._has_active_session_for_thread(
            channel_id="D0DMCHANNEL",
            thread_ts="2000.0",
            user_id="U_USER",
        )
        assert result is False


# ===========================================================================
# Thread engagement — bot-started threads & mentioned threads
# ===========================================================================

class TestThreadEngagement:
    """Test _bot_message_ts and _mentioned_threads tracking."""


    def test_mentioned_threads_cap_evicts_oldest_timestamps(self):
        """Mentioned-thread tracking evicts the oldest Slack timestamps first."""
        adapter = _make_adapter()
        adapter._MENTIONED_THREADS_MAX = 4
        adapter._mentioned_threads.update(
            {
                "1000.000002",
                "999.999999",
                "1000.000004",
                "1000.000001",
                "1000.000003",
            }
        )

        adapter._trim_mentioned_threads()

        assert adapter._mentioned_threads == {
            "1000.000002",
            "1000.000003",
            "1000.000004",
        }


# ===========================================================================
# _handle_slack_reaction — reaction_added forwarding
# ===========================================================================

class TestSlackReactionForwarding:
    """Reactions should flow through the same pipeline as typed messages,
    gated by the ``reaction_triggers`` opt-in (default: off)."""

    @staticmethod
    def _enable_triggers(adapter, value=True):
        adapter.config.extra["reaction_triggers"] = value

    @pytest.mark.asyncio
    async def test_default_off_reaction_acked_and_dropped(self):
        """Without the reaction_triggers opt-in, reaction events are acked
        and dropped — the historical behavior — so busy channels don't wake
        the agent on every emoji."""
        adapter = _make_adapter()
        forwarded: list[dict] = []

        async def _capture(event):
            forwarded.append(event)

        with patch.object(adapter, "_handle_slack_message", new=_capture):
            await adapter._handle_slack_reaction({
                "type": "reaction_added",
                "user": "U1",
                "reaction": "thumbsup",
                "item": {"type": "message", "channel": "C1", "ts": "2000.0"},
                "item_user": "U_BOT",
                "event_ts": "3000.0",
            })

        assert forwarded == []

    @pytest.mark.asyncio
    async def test_reaction_synthesizes_message_in_thread(self):
        """A 👍 reaction on a message in a thread should produce a synthesized
        MessageEvent that lands in that thread with text ``reaction:added:👍``,
        going through _handle_slack_message so the auth gate, thread-context
        fetch, and skill routing all apply unchanged."""
        adapter = _make_adapter()
        self._enable_triggers(adapter)
        mock_client = adapter._team_clients["T1"]
        # Reacted-to message is itself a reply inside a thread; its thread_ts
        # points at the thread parent.
        mock_client.conversations_replies = AsyncMock(return_value={
            "messages": [
                {"ts": "2000.0", "thread_ts": "1000.0", "user": "U_BOT", "text": "Proposal"}
            ]
        })
        forwarded: list[dict] = []

        async def _capture(event):
            forwarded.append(event)

        with patch.object(adapter, "_handle_slack_message", new=_capture):
            await adapter._handle_slack_reaction({
                "type": "reaction_added",
                "user": "U1",
                "reaction": "thumbsup",
                "item": {"type": "message", "channel": "C1", "ts": "2000.0"},
                "item_user": "U_BOT",
                "event_ts": "3000.0",
            })

        assert len(forwarded) == 1
        synth = forwarded[0]
        assert synth["type"] == "message"
        assert synth["user"] == "U1"
        assert synth["text"] == "reaction:added:👍"
        assert synth["channel"] == "C1"
        # Threaded back to the parent of the reacted-to message, not to the
        # reacted-to message itself.
        assert synth["thread_ts"] == "1000.0"
        # Distinct synthetic ts so dedup doesn't merge with anything else.
        assert synth["ts"] == "3000.0"
        # Reaction metadata preserved for downstream introspection.
        assert synth["_hermes_reaction"]["name"] == "thumbsup"
        assert synth["_hermes_reaction"]["action"] == "added"
        assert synth["_hermes_reaction"]["reacted_to_ts"] == "2000.0"
        # Pre-authorized as addressed-to-the-bot (skips mention gate only).
        assert synth["_hermes_force_process"] is True


    @pytest.mark.asyncio
    async def test_target_channel_handoff_routes_top_level(self):
        """reaction_trigger_target=C_TRAIN routes the synthesized turn to the
        target channel as a new top-level message (#45265)."""
        adapter = _make_adapter()
        self._enable_triggers(adapter, ["task"])
        adapter.config.extra["reaction_trigger_target"] = "C_TRAIN"
        mock_client = adapter._team_clients["T1"]
        mock_client.conversations_replies = AsyncMock(return_value={
            "messages": [{"ts": "1000.0", "user": "U_OTHER", "text": "Handoff me"}]
        })
        forwarded: list[dict] = []

        async def _capture(event):
            forwarded.append(event)

        with patch.object(adapter, "_handle_slack_message", new=_capture):
            await adapter._handle_slack_reaction({
                "type": "reaction_added",
                "user": "U1",
                "reaction": "task",
                "item": {"type": "message", "channel": "C1", "ts": "1000.0"},
                "item_user": "U_OTHER",
                "event_ts": "3000.0",
            })

        assert len(forwarded) == 1
        synth = forwarded[0]
        assert synth["channel"] == "C_TRAIN"
        assert "thread_ts" not in synth
        assert synth["_hermes_no_thread_response"] is True
        assert synth["_hermes_reaction_source_channel"] == "C1"

    @pytest.mark.asyncio
    async def test_target_channel_with_thread_routes_to_thread(self):
        """A C123:<ts> target routes into that thread of the target channel."""
        adapter = _make_adapter()
        self._enable_triggers(adapter, ["task"])
        adapter.config.extra["reaction_trigger_target"] = "C_TRAIN:1719.0001"
        mock_client = adapter._team_clients["T1"]
        mock_client.conversations_replies = AsyncMock(return_value={
            "messages": [{"ts": "1000.0", "user": "U_OTHER", "text": "Handoff me"}]
        })
        forwarded: list[dict] = []

        async def _capture(event):
            forwarded.append(event)

        with patch.object(adapter, "_handle_slack_message", new=_capture):
            await adapter._handle_slack_reaction({
                "type": "reaction_added",
                "user": "U1",
                "reaction": "task",
                "item": {"type": "message", "channel": "C1", "ts": "1000.0"},
                "item_user": "U_OTHER",
                "event_ts": "3000.0",
            })

        assert len(forwarded) == 1
        assert forwarded[0]["channel"] == "C_TRAIN"
        assert forwarded[0]["thread_ts"] == "1719.0001"
        assert "_hermes_no_thread_response" not in forwarded[0]

    @pytest.mark.asyncio
    async def test_hook_fires_even_when_routing_disabled(self):
        """The gateway reaction handler fires for every human reaction on a
        message item, independent of the reaction_triggers opt-in."""
        adapter = _make_adapter()  # opt-in NOT set
        hook_events: list[dict] = []

        async def _hook(ctx):
            hook_events.append(ctx)

        adapter.set_reaction_handler(_hook)
        forwarded: list[dict] = []

        async def _capture(event):
            forwarded.append(event)

        with patch.object(adapter, "_handle_slack_message", new=_capture):
            await adapter._handle_slack_reaction({
                "type": "reaction_added",
                "user": "U1",
                "reaction": "thumbsup",
                "item": {"type": "message", "channel": "C1", "ts": "1000.0"},
                "item_user": "U_BOT",
                "event_ts": "3000.0",
            })
            await adapter._handle_slack_reaction(
                {
                    "type": "reaction_removed",
                    "user": "U1",
                    "reaction": "thumbsup",
                    "item": {"type": "message", "channel": "C1", "ts": "1000.0"},
                    "item_user": "U_BOT",
                    "event_ts": "3001.0",
                },
                removed=True,
            )

        assert forwarded == []  # routing stays off
        assert [e["event_name"] for e in hook_events] == [
            "reaction:added",
            "reaction:removed",
        ]
        assert hook_events[0]["platform"] == "slack"
        assert hook_events[0]["reaction"] == "thumbsup"
        assert hook_events[0]["user_id"] == "U1"
        assert hook_events[0]["channel_id"] == "C1"
        assert hook_events[0]["message_ts"] == "1000.0"


    def test_trigger_config_parsing(self):
        """reaction_triggers accepts bool / list / string forms."""
        adapter = _make_adapter()
        assert adapter._slack_reaction_triggers() is None  # default off
        adapter.config.extra["reaction_triggers"] = False
        assert adapter._slack_reaction_triggers() is None
        adapter.config.extra["reaction_triggers"] = True
        assert adapter._slack_reaction_triggers() == set()
        adapter.config.extra["reaction_triggers"] = ["thumbsup", ":task:"]
        assert adapter._slack_reaction_triggers() == {"thumbsup", "task"}
        adapter.config.extra["reaction_triggers"] = "thumbsup, task"
        assert adapter._slack_reaction_triggers() == {"thumbsup", "task"}
        adapter.config.extra["reaction_triggers"] = "all"
        assert adapter._slack_reaction_triggers() == set()
        adapter.config.extra["reaction_triggers"] = "false"
        assert adapter._slack_reaction_triggers() is None


class TestSlackReactionAuthorizationGate:
    """The synthesized reaction event must pass through the same early
    authorization rejection as typed messages — an unauthorized reactor
    cannot wake the agent."""

    @pytest.mark.asyncio
    async def test_unauthorized_reactor_rejected_in_pipeline(self):
        adapter = _make_adapter()
        adapter.config.extra["reaction_triggers"] = True
        mock_client = adapter._team_clients["T1"]
        mock_client.conversations_replies = AsyncMock(return_value={
            "messages": [{"ts": "1000.0", "user": "U_BOT", "text": "Proposal"}]
        })

        class _Runner:
            def __init__(self):
                self.handled = []
                self.auth_checked = []

            async def handle(self, event):
                self.handled.append(event)
                return None

            def _is_user_authorized(self, source):
                self.auth_checked.append(source.user_id)
                return False  # reject everyone

        runner = _Runner()
        adapter.set_message_handler(runner.handle)
        adapter.handle_message = AsyncMock()

        await adapter._handle_slack_reaction({
            "type": "reaction_added",
            "user": "U_RANDO",
            "reaction": "thumbsup",
            "item": {"type": "message", "channel": "C1", "ts": "1000.0"},
            "item_user": "U_BOT",
            "event_ts": "3000.0",
        })

        # The auth gate saw the reactor's user id and rejected it before
        # any MessageEvent reached the gateway.
        assert "U_RANDO" in runner.auth_checked
        assert runner.handled == []
        adapter.handle_message.assert_not_called()
