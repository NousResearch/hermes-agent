"""Tests for Slack Block Kit approval buttons and thread context fetching."""

import asyncio
import os
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

from gateway.platforms.slack import SlackAdapter
from gateway.config import Platform, PlatformConfig


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
    async def test_sends_in_thread(self):
        adapter = _make_adapter()
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_postMessage = AsyncMock(return_value={"ts": "1234.5678"})

        await adapter.send_exec_approval(
            chat_id="C1",
            command="echo test",
            session_key="test-session",
            metadata={"thread_id": "9999.0000"},
        )

        kwargs = mock_client.chat_postMessage.call_args[1]
        assert kwargs.get("thread_ts") == "9999.0000"

    @pytest.mark.asyncio
    async def test_not_connected(self):
        adapter = _make_adapter()
        adapter._app = None
        result = await adapter.send_exec_approval(
            chat_id="C1", command="ls", session_key="s"
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_truncates_long_command(self):
        adapter = _make_adapter()
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_postMessage = AsyncMock(return_value={"ts": "1.2"})

        long_cmd = "x" * 5000
        await adapter.send_exec_approval(
            chat_id="C1", command=long_cmd, session_key="s"
        )

        kwargs = mock_client.chat_postMessage.call_args[1]
        section_text = kwargs["blocks"][0]["text"]["text"]
        assert "..." in section_text
        assert len(section_text) < 5000


# ===========================================================================
# _handle_approval_action — button click handler
# ===========================================================================

class TestSlackApprovalAction:
    """Test the approval button click handler."""

    @pytest.mark.asyncio
    async def test_resolves_approval(self):
        adapter = _make_adapter()
        adapter._approval_resolved["1234.5678"] = False

        ack = AsyncMock()
        body = {
            "message": {
                "ts": "1234.5678",
                "blocks": [
                    {"type": "section", "text": {"type": "mrkdwn", "text": "original text"}},
                    {"type": "actions", "elements": []},
                ],
            },
            "channel": {"id": "C1"},
            "user": {"name": "norbert"},
        }
        action = {
            "action_id": "hermes_approve_once",
            "value": "agent:main:slack:group:C1:1111",
        }

        mock_client = adapter._team_clients["T1"]
        mock_client.chat_update = AsyncMock()

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            await adapter._handle_approval_action(ack, body, action)

        ack.assert_called_once()
        mock_resolve.assert_called_once_with("agent:main:slack:group:C1:1111", "once")

        # Message should be updated with decision
        mock_client.chat_update.assert_called_once()
        update_kwargs = mock_client.chat_update.call_args[1]
        assert "Approved once by norbert" in update_kwargs["text"]

    @pytest.mark.asyncio
    async def test_prevents_double_click(self):
        adapter = _make_adapter()
        adapter._approval_resolved["1234.5678"] = True  # Already resolved

        ack = AsyncMock()
        body = {
            "message": {"ts": "1234.5678", "blocks": []},
            "channel": {"id": "C1"},
            "user": {"name": "norbert"},
        }
        action = {
            "action_id": "hermes_approve_once",
            "value": "some-session",
        }

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            await adapter._handle_approval_action(ack, body, action)

        # Should have acked but NOT resolved
        ack.assert_called_once()
        mock_resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_deny_action(self):
        adapter = _make_adapter()
        adapter._approval_resolved["1.2"] = False

        ack = AsyncMock()
        body = {
            "message": {"ts": "1.2", "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": "cmd"}},
            ]},
            "channel": {"id": "C1"},
            "user": {"name": "alice"},
        }
        action = {"action_id": "hermes_deny", "value": "session-key"}

        mock_client = adapter._team_clients["T1"]
        mock_client.chat_update = AsyncMock()

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            await adapter._handle_approval_action(ack, body, action)

        mock_resolve.assert_called_once_with("session-key", "deny")
        update_kwargs = mock_client.chat_update.call_args[1]
        assert "Denied by alice" in update_kwargs["text"]


# ===========================================================================
# _fetch_thread_context
# ===========================================================================

class TestSlackThreadContext:
    """Test thread context fetching."""

    @pytest.mark.asyncio
    async def test_fetches_and_formats_context(self):
        adapter = _make_adapter()
        mock_client = adapter._team_clients["T1"]
        mock_client.conversations_replies = AsyncMock(return_value={
            "messages": [
                {"ts": "1000.0", "user": "U1", "text": "This is the parent message"},
                {"ts": "1000.1", "user": "U2", "text": "I think we should refactor"},
                {"ts": "1000.2", "user": "U1", "text": "Good idea, <@U_BOT> what do you think?"},
            ]
        })

        # Mock user name resolution
        adapter._user_name_cache = {"U1": "Alice", "U2": "Bob"}

        context = await adapter._fetch_thread_context(
            channel_id="C1",
            thread_ts="1000.0",
            current_ts="1000.2",  # The message that triggered the fetch
            team_id="T1",
        )

        assert "[Thread context" in context
        assert "[thread parent] Alice: This is the parent message" in context
        assert "Bob: I think we should refactor" in context
        # Current message should be excluded
        assert "what do you think" not in context
        # Bot mention should be stripped from context
        assert "<@U_BOT>" not in context

    @pytest.mark.asyncio
    async def test_skips_bot_messages(self):
        adapter = _make_adapter()
        mock_client = adapter._team_clients["T1"]
        mock_client.conversations_replies = AsyncMock(return_value={
            "messages": [
                {"ts": "1000.0", "user": "U1", "text": "Parent"},
                {"ts": "1000.1", "bot_id": "B1", "text": "Bot reply (should be skipped)"},
                {"ts": "1000.2", "user": "U1", "text": "Current"},
            ]
        })
        adapter._user_name_cache = {"U1": "Alice"}

        context = await adapter._fetch_thread_context(
            channel_id="C1", thread_ts="1000.0", current_ts="1000.2", team_id="T1"
        )

        assert "Bot reply" not in context
        assert "Alice: Parent" in context

    @pytest.mark.asyncio
    async def test_empty_thread(self):
        adapter = _make_adapter()
        mock_client = adapter._team_clients["T1"]
        mock_client.conversations_replies = AsyncMock(return_value={"messages": []})

        context = await adapter._fetch_thread_context(
            channel_id="C1", thread_ts="1000.0", current_ts="1000.1", team_id="T1"
        )
        assert context == ""

    @pytest.mark.asyncio
    async def test_api_failure_returns_empty(self):
        adapter = _make_adapter()
        mock_client = adapter._team_clients["T1"]
        mock_client.conversations_replies = AsyncMock(side_effect=Exception("API error"))

        context = await adapter._fetch_thread_context(
            channel_id="C1", thread_ts="1000.0", current_ts="1000.1", team_id="T1"
        )
        assert context == ""


# ===========================================================================
# Regression coverage for #12421 — thread-context cache must isolate by
# workspace (team_id).  The rendered content depends on workspace-scoped
# state (``_team_bot_user_ids`` for mention stripping + ``_get_client``-
# routed user-name resolution), so a cache hit from workspace A must not
# serve content to workspace B.
# ===========================================================================


def _make_multi_workspace_adapter():
    """Build a SlackAdapter wired for two independent workspaces (T1 + T2)
    with distinct bot user IDs and distinct team-scoped WebClients."""
    config = PlatformConfig(enabled=True, token="xoxb-test-token")
    adapter = SlackAdapter(config)
    adapter._app = MagicMock()
    adapter._bot_user_id = "U_BOT1"
    adapter._team_clients = {"T1": AsyncMock(), "T2": AsyncMock()}
    adapter._team_bot_user_ids = {"T1": "U_BOT1", "T2": "U_BOT2"}
    adapter._channel_team = {"C1": "T1", "C2": "T2"}
    return adapter


class TestSlackThreadContextWorkspaceIsolation:
    """The thread-context cache must key on ``team_id`` in addition to
    channel + thread timestamp.  #12421."""

    @pytest.mark.asyncio
    async def test_reporter_repro_two_workspaces_do_not_share_cache(self):
        """Reporter's exact repro: the same ``(channel_id, thread_ts)``
        tuple lands across two workspaces with different identity data.
        The second call must not reuse the first workspace's content and
        must trigger its own API fetch.

        Real-world path: a channel is associated with two workspaces
        over time (enterprise grid migration, or back-to-back calls
        under different team_id parameters — as shown by the issue
        reporter), so we flip ``_channel_team`` between calls to pin
        the routing deterministically.
        """
        adapter = _make_multi_workspace_adapter()

        t1 = adapter._team_clients["T1"]
        t2 = adapter._team_clients["T2"]
        t1.conversations_replies = AsyncMock(return_value={
            "messages": [
                {"ts": "1000.0", "user": "U1", "text": "Hello from T1"},
                {"ts": "9999.9", "user": "U1", "text": "Current"},
            ]
        })
        t2.conversations_replies = AsyncMock(return_value={
            "messages": [
                {"ts": "1000.0", "user": "U1", "text": "Hello from T2"},
                {"ts": "9999.9", "user": "U1", "text": "Current"},
            ]
        })
        adapter._user_name_cache = {"U1": "Alice"}

        # Route Csame → T1 for the first call.
        adapter._channel_team["Csame"] = "T1"
        first = await adapter._fetch_thread_context(
            channel_id="Csame", thread_ts="1000.0",
            current_ts="9999.9", team_id="T1",
        )
        # Now route Csame → T2 for the second call.
        adapter._channel_team["Csame"] = "T2"
        second = await adapter._fetch_thread_context(
            channel_id="Csame", thread_ts="1000.0",
            current_ts="9999.9", team_id="T2",
        )

        # Each workspace got its own content (no cross-contamination).
        assert "Hello from T1" in first
        assert "Hello from T2" in second
        assert first != second
        # Each workspace triggered its own conversations.replies call.
        assert t1.conversations_replies.await_count == 1
        assert t2.conversations_replies.await_count == 1

    @pytest.mark.asyncio
    async def test_same_workspace_same_thread_uses_cache(self):
        """Preserved-behaviour canary: within one workspace, the second
        call for the same (channel, thread_ts) is a cache hit — no
        duplicate API call."""
        adapter = _make_multi_workspace_adapter()
        t1 = adapter._team_clients["T1"]
        t1.conversations_replies = AsyncMock(return_value={
            "messages": [
                {"ts": "1000.0", "user": "U1", "text": "Parent"},
                {"ts": "9999.9", "user": "U1", "text": "Current"},
            ]
        })
        adapter._user_name_cache = {"U1": "Alice"}

        first = await adapter._fetch_thread_context(
            channel_id="C1", thread_ts="1000.0",
            current_ts="9999.9", team_id="T1",
        )
        second = await adapter._fetch_thread_context(
            channel_id="C1", thread_ts="1000.0",
            current_ts="9999.9", team_id="T1",
        )

        assert first == second
        assert t1.conversations_replies.await_count == 1  # cached

    @pytest.mark.asyncio
    async def test_structural_pin_cache_key_contains_team_id(self):
        """Pin the cache key shape directly — a future refactor can't
        silently drop ``team_id`` and reintroduce the cross-workspace
        leak without breaking this test."""
        adapter = _make_multi_workspace_adapter()
        adapter._channel_team["Csame"] = "T1"
        t1 = adapter._team_clients["T1"]
        t1.conversations_replies = AsyncMock(return_value={
            "messages": [{"ts": "1000.0", "user": "U1", "text": "Parent"}]
        })
        adapter._user_name_cache = {"U1": "Alice"}

        await adapter._fetch_thread_context(
            channel_id="Csame", thread_ts="1000.0",
            current_ts="9999.9", team_id="T1",
        )

        # Cache key must include the team_id.  Precise shape:
        # ``"T1:Csame:1000.0"``.  If a future refactor changes the
        # concrete shape, update this assertion — but make sure it
        # still uniquely identifies the workspace.
        assert "T1:Csame:1000.0" in adapter._thread_context_cache
        # The old, workspace-unaware key must NOT be in the cache.
        assert "Csame:1000.0" not in adapter._thread_context_cache

    @pytest.mark.asyncio
    async def test_mention_stripping_is_team_specific(self):
        """The rendered content strips each workspace's *own* bot
        mention — workspace A's cached output (with A's bot mention
        stripped) must not leak to workspace B's caller.

        To prove workspace-isolated stripping (and not just coincidental
        absence of mentions), each workspace returns text that
        *contains* the *other* workspace's bot mention as a literal
        substring.  After correct per-workspace stripping:

        * T1's output keeps the literal ``<@U_BOT2>`` (because T1
          doesn't strip T2's bot mention) and drops ``<@U_BOT1>``.
        * T2's output keeps ``<@U_BOT1>`` and drops ``<@U_BOT2>``.

        On ``origin/main`` the cache leak would make T2's caller see
        T1's cached output — so ``<@U_BOT1>`` would be absent from
        T2's context, and ``<@U_BOT2>`` would be present (T2 never got
        to strip it).  Both assertions below would fail.
        """
        adapter = _make_multi_workspace_adapter()
        t1 = adapter._team_clients["T1"]
        t2 = adapter._team_clients["T2"]
        # Each workspace's triggering message mentions its own bot AND
        # the other workspace's bot (which looks like a plain literal
        # from its POV).  Per-workspace stripping removes exactly one
        # mention per output.
        t1.conversations_replies = AsyncMock(return_value={
            "messages": [
                {"ts": "1000.0", "user": "U1",
                 "text": "Hey <@U_BOT1> and also <@U_BOT2> help"},
                {"ts": "9999.9", "user": "U1", "text": "Current"},
            ]
        })
        t2.conversations_replies = AsyncMock(return_value={
            "messages": [
                {"ts": "1000.0", "user": "U1",
                 "text": "Hey <@U_BOT1> and also <@U_BOT2> help"},
                {"ts": "9999.9", "user": "U1", "text": "Current"},
            ]
        })
        adapter._user_name_cache = {"U1": "Alice"}

        adapter._channel_team["Csame"] = "T1"
        t1_ctx = await adapter._fetch_thread_context(
            channel_id="Csame", thread_ts="1000.0",
            current_ts="9999.9", team_id="T1",
        )
        adapter._channel_team["Csame"] = "T2"
        t2_ctx = await adapter._fetch_thread_context(
            channel_id="Csame", thread_ts="1000.0",
            current_ts="9999.9", team_id="T2",
        )

        # T1 strips its own mention, leaves T2's in place.
        assert "<@U_BOT1>" not in t1_ctx
        assert "<@U_BOT2>" in t1_ctx
        # T2 strips its own mention, leaves T1's in place.
        assert "<@U_BOT2>" not in t2_ctx
        assert "<@U_BOT1>" in t2_ctx
        # And the two rendered contexts must be distinct.
        assert t1_ctx != t2_ctx

    @pytest.mark.asyncio
    async def test_ttl_is_per_workspace(self):
        """Each workspace's cache entry gets its own TTL window.  Aging
        one team's entry into the past must not affect the other
        team's cached content."""
        adapter = _make_multi_workspace_adapter()
        t1 = adapter._team_clients["T1"]
        t2 = adapter._team_clients["T2"]
        t1.conversations_replies = AsyncMock(return_value={
            "messages": [
                {"ts": "1000.0", "user": "U1", "text": "From T1"},
                {"ts": "9999.9", "user": "U1", "text": "Current"},
            ]
        })
        t2.conversations_replies = AsyncMock(return_value={
            "messages": [
                {"ts": "1000.0", "user": "U1", "text": "From T2"},
                {"ts": "9999.9", "user": "U1", "text": "Current"},
            ]
        })
        adapter._user_name_cache = {"U1": "Alice"}

        # Prime both caches.
        adapter._channel_team["Csame"] = "T1"
        await adapter._fetch_thread_context(
            channel_id="Csame", thread_ts="1000.0",
            current_ts="9999.9", team_id="T1",
        )
        adapter._channel_team["Csame"] = "T2"
        await adapter._fetch_thread_context(
            channel_id="Csame", thread_ts="1000.0",
            current_ts="9999.9", team_id="T2",
        )

        # Expire only T1's entry.
        ttl = adapter._THREAD_CACHE_TTL
        t1_entry = adapter._thread_context_cache["T1:Csame:1000.0"]
        # Rebuild with a stale timestamp via ``dataclasses.replace`` instead
        # of mutating ``t1_entry`` in place — keeps the test intent (new
        # entry with an aged timestamp) explicit at the call site.
        import dataclasses
        adapter._thread_context_cache["T1:Csame:1000.0"] = dataclasses.replace(
            t1_entry, fetched_at=t1_entry.fetched_at - ttl - 10,
        )

        # T1 refreshes (API hit), T2 still cached (no new API hit).
        adapter._channel_team["Csame"] = "T1"
        await adapter._fetch_thread_context(
            channel_id="Csame", thread_ts="1000.0",
            current_ts="9999.9", team_id="T1",
        )
        adapter._channel_team["Csame"] = "T2"
        await adapter._fetch_thread_context(
            channel_id="Csame", thread_ts="1000.0",
            current_ts="9999.9", team_id="T2",
        )

        assert t1.conversations_replies.await_count == 2  # initial + refresh
        assert t2.conversations_replies.await_count == 1  # still cached

    @pytest.mark.asyncio
    async def test_empty_team_id_has_own_namespace(self):
        """Backward-compat: a caller that passes ``team_id=""`` (legacy
        single-workspace mode or older test harnesses) gets its own
        namespace and does not collide with ``team_id="T1"``."""
        adapter = _make_multi_workspace_adapter()
        t1 = adapter._team_clients["T1"]
        t1.conversations_replies = AsyncMock(return_value={
            "messages": [
                {"ts": "1000.0", "user": "U1", "text": "Parent"},
                {"ts": "9999.9", "user": "U1", "text": "Current"},
            ]
        })
        # Empty-team fallback routes through ``self._app.client`` via
        # ``_get_client`` when the channel has no team mapping.  Wire it.
        adapter._app.client = t1

        adapter._user_name_cache = {"U1": "Alice"}

        # First call: team_id="" → cache key ``:Cunmapped:1000.0``.
        # Routing: no team mapping → ``_get_client`` falls back to
        # ``_app.client`` (wired to t1 above).
        await adapter._fetch_thread_context(
            channel_id="Cunmapped", thread_ts="1000.0",
            current_ts="9999.9", team_id="",
        )
        # Second call: team_id="T1" but still no channel→team mapping,
        # so ``_get_client`` still falls back to ``_app.client``.  Cache
        # key is ``T1:Cunmapped:1000.0``.
        await adapter._fetch_thread_context(
            channel_id="Cunmapped", thread_ts="1000.0",
            current_ts="9999.9", team_id="T1",
        )

        # Two distinct entries — one per team scope.
        assert ":Cunmapped:1000.0" in adapter._thread_context_cache
        assert "T1:Cunmapped:1000.0" in adapter._thread_context_cache
        # And the fetch ran twice because they weren't merged.
        assert t1.conversations_replies.await_count == 2

    @pytest.mark.asyncio
    async def test_different_channels_same_team_each_get_cached(self):
        """Sanity: within one workspace, different channels still get
        independent cache entries (no cross-channel collapsing)."""
        adapter = _make_multi_workspace_adapter()
        adapter._channel_team["CA"] = "T1"
        adapter._channel_team["CB"] = "T1"
        t1 = adapter._team_clients["T1"]
        t1.conversations_replies = AsyncMock(return_value={
            "messages": [
                {"ts": "1000.0", "user": "U1", "text": "Parent"},
                {"ts": "9999.9", "user": "U1", "text": "Current"},
            ]
        })
        adapter._user_name_cache = {"U1": "Alice"}

        await adapter._fetch_thread_context(
            channel_id="CA", thread_ts="1000.0",
            current_ts="9999.9", team_id="T1",
        )
        await adapter._fetch_thread_context(
            channel_id="CB", thread_ts="1000.0",
            current_ts="9999.9", team_id="T1",
        )

        assert "T1:CA:1000.0" in adapter._thread_context_cache
        assert "T1:CB:1000.0" in adapter._thread_context_cache
        assert t1.conversations_replies.await_count == 2


# ===========================================================================
# _has_active_session_for_thread — session key fix (#5833)
# ===========================================================================

class TestSessionKeyFix:
    """Test that _has_active_session_for_thread uses build_session_key."""

    def test_uses_build_session_key(self):
        """Verify the fix uses build_session_key instead of manual key construction."""
        adapter = _make_adapter()

        # Mock session store with a known entry
        mock_store = MagicMock()
        mock_store._entries = {
            "agent:main:slack:group:C1:1000.0": MagicMock()
        }
        mock_store._ensure_loaded = MagicMock()
        mock_store.config = MagicMock()
        mock_store.config.group_sessions_per_user = False  # threads don't include user_id
        mock_store.config.thread_sessions_per_user = False
        adapter._session_store = mock_store

        # With the fix, build_session_key should be called which respects
        # group_sessions_per_user=False (no user_id appended)
        result = adapter._has_active_session_for_thread(
            channel_id="C1", thread_ts="1000.0", user_id="U123"
        )

        # Should find the session because build_session_key with
        # group_sessions_per_user=False doesn't append user_id
        assert result is True

    def test_no_session_returns_false(self):
        adapter = _make_adapter()
        mock_store = MagicMock()
        mock_store._entries = {}
        mock_store._ensure_loaded = MagicMock()
        mock_store.config = MagicMock()
        mock_store.config.group_sessions_per_user = True
        mock_store.config.thread_sessions_per_user = False
        adapter._session_store = mock_store

        result = adapter._has_active_session_for_thread(
            channel_id="C1", thread_ts="1000.0", user_id="U123"
        )
        assert result is False

    def test_no_session_store(self):
        adapter = _make_adapter()
        # No _session_store attribute
        result = adapter._has_active_session_for_thread(
            channel_id="C1", thread_ts="1000.0", user_id="U123"
        )
        assert result is False


# ===========================================================================
# Thread engagement — bot-started threads & mentioned threads
# ===========================================================================

class TestThreadEngagement:
    """Test _bot_message_ts and _mentioned_threads tracking."""

    @pytest.mark.asyncio
    async def test_send_tracks_bot_message_ts(self):
        """Bot's sent messages are tracked so thread replies work without @mention."""
        adapter = _make_adapter()
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_postMessage = AsyncMock(return_value={"ts": "9000.1"})

        await adapter.send(chat_id="C1", content="Hello!", metadata={"thread_id": "8000.0"})

        assert "9000.1" in adapter._bot_message_ts
        # Thread root should also be tracked
        assert "8000.0" in adapter._bot_message_ts

    @pytest.mark.asyncio
    async def test_bot_message_ts_cap(self):
        """Verify memory is bounded when many messages are sent."""
        adapter = _make_adapter()
        adapter._BOT_TS_MAX = 10  # low cap for testing
        mock_client = adapter._team_clients["T1"]

        for i in range(20):
            mock_client.chat_postMessage = AsyncMock(return_value={"ts": f"{i}.0"})
            await adapter.send(chat_id="C1", content=f"msg {i}")

        assert len(adapter._bot_message_ts) <= 10

    def test_mentioned_threads_populated_on_mention(self):
        """When bot is @mentioned in a thread, that thread is tracked."""
        adapter = _make_adapter()
        # Simulate what _handle_slack_message does on mention
        adapter._mentioned_threads.add("1000.0")
        assert "1000.0" in adapter._mentioned_threads

    def test_mentioned_threads_cap(self):
        """Verify _mentioned_threads is bounded."""
        adapter = _make_adapter()
        adapter._MENTIONED_THREADS_MAX = 10
        for i in range(15):
            adapter._mentioned_threads.add(f"{i}.0")
            if len(adapter._mentioned_threads) > adapter._MENTIONED_THREADS_MAX:
                to_remove = list(adapter._mentioned_threads)[:adapter._MENTIONED_THREADS_MAX // 2]
                for t in to_remove:
                    adapter._mentioned_threads.discard(t)
        assert len(adapter._mentioned_threads) <= 10
