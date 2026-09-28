"""Mattermost plugin — slash-free approval matcher + reaction-based approval UX.

A-side (slash-free matcher):
    Inbound text starting with ``approve`` / ``yes`` / ``y`` / ``deny`` /
    ``no`` / ``n`` / ``approve session`` / ``approve always`` (case-insensitive)
    is reclassified as a COMMAND and rewritten to ``/<token>`` so the gateway
    runner's existing ``_PLAIN_COMMANDS`` dispatch handles it unchanged. Only
    fires in DM, in threads, or in free-response channels.

B-side (reactions):
    When ``_send_exec_approval_prompt`` posts the approval card, the bot seeds
    ``✅`` (once), ``♾️`` (session), and ``🚫`` (deny) reactions on its own
    message. Inbound ``reaction_added`` WS events on those reactions resolve
    the pending approval via ``tools.approval.resolve_gateway_approval`` —
    matching the Matrix / Telegram / Slack flow.

Both paths are multiplex-safe: ``session_key`` is the dict key, and the
approval lookup is per-adapter (no public HTTP callback URL is needed).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─── helpers ────────────────────────────────────────────────────────────────


def _make_adapter():
    """Build a MattermostAdapter with the live plugin module loaded.

    The plugin module imports ``gateway.config``, ``gateway.platforms.helpers``
    etc., so we need the project root on sys.path and a token+url in env so
    ``_is_connected`` returns True.

    v2: also allow the canonical test user_ids in the adapter's allowlist so the
    reaction-auth gate (default-empty in production) doesn't break tests that
    assume the old "any non-bot user can resolve" behavior. The dedicated
    unauthorized-reactor test sets ``_allowed_user_ids = set()`` explicitly.
    """
    os.environ.setdefault("MATTERMOST_URL", "https://mm.example.com")
    os.environ.setdefault("MATTERMOST_TOKEN", "test-token-xxx")
    from gateway.config import PlatformConfig
    from plugins.platforms.mattermost.adapter import MattermostAdapter
    cfg = PlatformConfig(enabled=True, token="test-token-xxx")
    adapter = MattermostAdapter(cfg)
    # v2: pin the allowlist to the canonical test users; tests that want to
    # exercise the "unauthorized reactor" branch clear this explicitly.
    adapter._allowed_user_ids = {"user-1", "user-99"}
    return adapter


def _ws_event_post(channel_id: str, sender_id: str, post_id: str, message: str,
                   channel_type: str = "O", root_id: str = ""):
    """Build a minimal ``posted`` WS event payload for ``_handle_ws_event``.

    The post string MUST be valid JSON (the adapter parses it with ``json.loads``);
    ``message`` is escaped via ``json.dumps`` so single/double quotes inside don't
    break parsing.
    """
    post_obj = {"id": post_id, "channel_id": channel_id, "user_id": sender_id,
                "message": message, "root_id": root_id, "type": ""}
    post_str = json.dumps(post_obj)  # valid JSON, escapes quotes properly
    return {
        "event": "posted",
        "data": {
            "post": post_str,
            "channel_type": channel_type,
            "sender_name": "alice",
        },
    }


def _ws_event_reaction_added(post_id: str, user_id: str, emoji_name: str):
    """Build a minimal ``reaction_added`` WS event payload.

    Mattermost's actual shape (verified against master
    ``app/reaction.go::sendReactionEvent``):
        data.reaction = JSON-encoded ``model.Reaction`` string
    """
    reaction = {"user_id": user_id, "post_id": post_id, "emoji_name": emoji_name}
    return {
        "event": "reaction_added",
        "data": {"reaction": json.dumps(reaction)},
    }


# ─── A-side: slash-free matcher ──────────────────────────────────────────────


class TestSlashFreeMatcherRemoved:
    """Bare ``approve`` / ``deny`` / ``yes`` text is NOT treated as a command.

    The slash-free matcher was tried first and removed 2026-09-26: bare text in
    chat is too ambiguous (``yes`` could be answering a question, not approving
    a pending command) and it broke user workflows when partial sentences
    happened to start with one of those words. Slash form (``/approve`` or
    ``<space>/approve`` to bypass Mattermost's slash-router) is reliable, so
    we stick with that. These tests pin the removal so the matcher can't
    quietly come back."""

    @pytest.mark.asyncio
    async def test_bare_approve_in_dm_is_plain_text(self, monkeypatch):
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        captured: list = []

        async def fake_handle_message(event):
            captured.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachments",
                            AsyncMock(return_value=([], [])))

        ev = _ws_event_post(channel_id="dm-ch", sender_id="user-1", post_id="p1",
                            message="approve", channel_type="D")
        await adapter._handle_ws_event(ev)

        assert len(captured) == 1
        evt = captured[0]
        # Bare text — no slash rewrite, no command classification.
        assert evt.text == "approve"
        from gateway.platforms.event import MessageType
        assert evt.message_type == MessageType.TEXT

    @pytest.mark.asyncio
    async def test_bare_deny_in_dm_is_plain_text(self, monkeypatch):
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        captured: list = []

        async def fake_handle_message(event):
            captured.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachments",
                            AsyncMock(return_value=([], [])))

        ev = _ws_event_post(channel_id="dm-ch", sender_id="user-1", post_id="p2",
                            message="deny", channel_type="D")
        await adapter._handle_ws_event(ev)

        assert captured[0].text == "deny"
        from gateway.platforms.event import MessageType
        assert captured[0].message_type == MessageType.TEXT

    @pytest.mark.asyncio
    async def test_bare_yes_in_thread_is_plain_text(self, monkeypatch):
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        adapter._bot_username = "mossoth"
        captured: list = []

        async def fake_handle_message(event):
            captured.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachments",
                            AsyncMock(return_value=([], [])))

        ev = _ws_event_post(channel_id="ch-x", sender_id="user-1", post_id="p3",
                            message="@mossoth yes", channel_type="O", root_id="parent-1")
        await adapter._handle_ws_event(ev)

        # The mention prefix is stripped upstream by the channel-gating pass;
        # the bare ``yes`` text is delivered as plain TEXT (not a command).
        assert captured[0].text == "yes"
        from gateway.platforms.event import MessageType
        assert captured[0].message_type == MessageType.TEXT

    @pytest.mark.asyncio
    async def test_bare_approve_session_is_plain_text(self, monkeypatch):
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        adapter._bot_username = "mossoth"
        captured: list = []

        async def fake_handle_message(event):
            captured.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachments",
                            AsyncMock(return_value=([], [])))

        ev = _ws_event_post(channel_id="ch-x", sender_id="user-1", post_id="p4",
                            message="@mossoth approve session", channel_type="O", root_id="parent-1")
        await adapter._handle_ws_event(ev)

        # Mention-stripped, plain TEXT — no slash rewrite.
        assert captured[0].text == "approve session"
        from gateway.platforms.event import MessageType
        assert captured[0].message_type == MessageType.TEXT

    @pytest.mark.asyncio
    async def test_slash_approve_still_routes_to_command(self, monkeypatch):
        """``/approve`` (slash form) is delivered to the runner as a COMMAND
        MessageType — the MessageType classification was restored 2026-09-27
        (regression in v1, see PR #124249 review). The runner's
        ``_PLAIN_COMMANDS`` dispatch handles the command classification.
        """
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        captured: list = []

        async def fake_handle_message(event):
            captured.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachments",
                            AsyncMock(return_value=([], [])))

        ev = _ws_event_post(channel_id="dm-ch", sender_id="user-1", post_id="p6",
                            message="/approve", channel_type="D")
        await adapter._handle_ws_event(ev)

        # Adapter classifies the slash form as COMMAND (restored 2026-09-27).
        assert captured[0].text == "/approve"
        from gateway.platforms.event import MessageType
        assert captured[0].message_type == MessageType.COMMAND

    @pytest.mark.asyncio
    async def test_spaced_slash_approve_still_routes_to_command(self, monkeypatch):
        """``<space>/approve`` (the Mattermost slash-router bypass) is delivered
        verbatim — leading space stripped (Mattermost emits this when the user
        types `` /approve`` to avoid the slash-router auto-popup), classified
        as COMMAND (classification restored 2026-09-27).
        """
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        captured: list = []

        async def fake_handle_message(event):
            captured.append(event)

        monkeypatch.setattr(adapter, "handle_message", fake_handle_message)
        monkeypatch.setattr(adapter, "_download_attachments",
                            AsyncMock(return_value=([], [])))

        ev = _ws_event_post(channel_id="dm-ch", sender_id="user-1", post_id="p7",
                            message=" /approve", channel_type="D")
        await adapter._handle_ws_event(ev)

        # Leading space stripped, slash command preserved, classified COMMAND.
        assert captured[0].text == "/approve"
        from gateway.platforms.event import MessageType
        assert captured[0].message_type == MessageType.COMMAND


# ─── B-side: reaction-based approval ─────────────────────────────────────────


class TestReactionApproval:
    """``✅ → once``, ``♾️ → session``, ``🚫 → deny`` reaction clicks resolve the approval."""

    @pytest.mark.asyncio
    async def test_resolve_removes_the_tapped_emoji_reaction(self, monkeypatch):
        """v2: the handler removes only the chosen-emoji reaction (the others are
        left in place until the user manually dismisses them — Mattermost will
        de-duplicate the bot's own reactions across taps via ``_approval_resolved``).
        Compared to the v1 design (delete all three), v2 is simpler and more
        correct: the user sees their tap registered, and the bot's other seeded
        emojis stay visible so the user knows the card was a multi-choice prompt.
        """
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        adapter._approval_prompts_by_event["post-1"] = {
            "session_key": "sess-X",
            "requester_user_id": "user-1",
            "chat_id": "dm-ch",
            "choices": ["once", "session", "always", "deny"],
        }
        deleted: list = []

        async def fake_delete(post_id, emoji_name):
            deleted.append((post_id, emoji_name))

        async def fake_edit(chat_id, post_id, text, **_):
            return None

        monkeypatch.setattr(adapter, "_delete_reaction", fake_delete)
        monkeypatch.setattr(adapter, "edit_message", fake_edit)

        with patch("tools.approval.resolve_gateway_approval", return_value=1):
            ev = _ws_event_reaction_added(post_id="post-1", user_id="user-1", emoji_name="white_check_mark")
            await adapter._handle_ws_event(ev)

        # Allow scheduled delete-tasks to run.
        await asyncio.sleep(0)

        # v2: only the chosen emoji (✅) gets deleted.
        assert deleted == [("post-1", "white_check_mark")]

    @pytest.mark.asyncio
    async def test_double_click_does_not_resolve_twice(self, monkeypatch):
        """v2: ``_approval_resolved`` is the double-click guard; second tap is a
        no-op even though the entry has been removed from the registry by the
        first call (and would otherwise re-create the entry on a subsequent
        re-add)."""
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        adapter._approval_prompts_by_event["post-2"] = {
            "session_key": "sess-Y",
            "requester_user_id": "user-1",
        }
        monkeypatch.setattr(adapter, "_delete_reaction", AsyncMock(return_value=True))
        monkeypatch.setattr(adapter, "edit_message", AsyncMock(return_value=None))

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            ev = _ws_event_reaction_added(post_id="post-2", user_id="user-1", emoji_name="white_check_mark")
            await adapter._handle_ws_event(ev)
            # Re-register the entry to simulate a race where the registry is
            # re-added before the second tap arrives — _approval_resolved must
            # still hold the post_id, so the second tap is rejected.
            adapter._approval_prompts_by_event["post-2"] = {
                "session_key": "sess-Y", "requester_user_id": "user-1",
            }
            await adapter._handle_ws_event(ev)

        # First call resolves; second call finds resolved=True (not in registry
        # but the guard survives).
        assert mock_resolve.call_count == 1

    @pytest.mark.asyncio
    async def test_check_mark_resolves_pending_approval_as_once(self, monkeypatch):
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        adapter._approval_prompts_by_event["post-1"] = {
            "session_key": "agent:main:mattermost:dm:user-1",
            "requester_user_id": "user-1",
            # v2: registry stores request_id; '' = no filter / use FIFO head.
            "request_id": "",
        }

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            ev = _ws_event_reaction_added(post_id="post-1", user_id="user-1", emoji_name="white_check_mark")
            await adapter._handle_ws_event(ev)

        # v2: request_id threaded through; ``None`` when the registry entry is empty.
        mock_resolve.assert_called_once_with(
            "agent:main:mattermost:dm:user-1", "once", request_id=None)

    @pytest.mark.asyncio
    async def test_request_id_filters_resolution_to_correct_card(self, monkeypatch):
        """v2: when 2+ approvals are pending in one session and each has a
        unique ``request_id``, a tap on card A must NOT resolve card B's
        command. ``tools/approval.py:153`` filters by ``request_id`` first;
        we just need to thread it through."""
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        # Two pending approvals in the same session, distinguished by request_id.
        adapter._approval_prompts_by_event["post-A"] = {
            "session_key": "sess-shared",
            "requester_user_id": "user-1",
            "request_id": "req-A",
        }
        adapter._approval_prompts_by_event["post-B"] = {
            "session_key": "sess-shared",
            "requester_user_id": "user-1",
            "request_id": "req-B",
        }

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            # Tap on card B (post-B)
            ev = _ws_event_reaction_added(post_id="post-B", user_id="user-1", emoji_name="white_check_mark")
            await adapter._handle_ws_event(ev)

        # The handler must have threaded req-B through, NOT req-A or ''.
        mock_resolve.assert_called_once_with(
            "sess-shared", "once", request_id="req-B")

    @pytest.mark.asyncio
    async def test_infinity_resolves_pending_approval_as_always(self, monkeypatch):
        """v2 matrix-parity: ♾️ (infinity) → always (mirrors matrix/adapter.py:902)."""
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        adapter._approval_prompts_by_event["post-2"] = {
            "session_key": "sess-X",
            "requester_user_id": "user-1",
        }

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            ev = _ws_event_reaction_added(post_id="post-2", user_id="user-1", emoji_name="infinity")
            await adapter._handle_ws_event(ev)

        mock_resolve.assert_called_once_with(
            "sess-X", "always", request_id=None)

    @pytest.mark.asyncio
    async def test_cyclone_resolves_pending_approval_as_session(self, monkeypatch):
        """v2 matrix-parity: 🌀 (cyclone) → session (mirrors matrix/adapter.py:902)."""
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        adapter._approval_prompts_by_event["post-cyclone"] = {
            "session_key": "sess-Cyclone",
            "requester_user_id": "user-1",
        }

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            ev = _ws_event_reaction_added(post_id="post-cyclone", user_id="user-1", emoji_name="cyclone")
            await adapter._handle_ws_event(ev)

        mock_resolve.assert_called_once_with(
            "sess-Cyclone", "session", request_id=None)

    @pytest.mark.asyncio
    async def test_cross_resolves_pending_approval_as_deny(self, monkeypatch):
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        adapter._approval_prompts_by_event["post-3"] = {
            "session_key": "sess-Y",
            "requester_user_id": "user-1",
        }

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            ev = _ws_event_reaction_added(post_id="post-3", user_id="user-1", emoji_name="no_entry_sign")
            await adapter._handle_ws_event(ev)

        mock_resolve.assert_called_once_with("sess-Y", "deny", request_id=None)

    @pytest.mark.asyncio
    async def test_any_allowed_non_bot_reaction_resolves(self):
        """When no requester is recorded (gateway runner does not carry it), any
        ALLOWLISTED non-bot reaction on the approval post resolves — by construction
        only the bot posts approval cards, so any allowlisted user reaction is a
        tap on the card.

        v2: the auth gate runs BEFORE the requester check. If the user is not in
        ``MATTERMOST_ALLOWED_USERS``, the reaction is ignored (tested separately).
        """
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        adapter._approval_prompts_by_event["post-4"] = {
            "session_key": "sess-Z",
            "requester_user_id": "",  # never populated by the gateway runner
        }

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            ev = _ws_event_reaction_added(post_id="post-4", user_id="user-99", emoji_name="white_check_mark")
            await adapter._handle_ws_event(ev)

        # v2: ``None`` for empty registry request_id (no filter / FIFO head).
        mock_resolve.assert_called_once_with("sess-Z", "once", request_id=None)

    @pytest.mark.asyncio
    async def test_reaction_from_unauthorized_user_is_rejected(self, monkeypatch):
        """v2: adapter-side allowlist gate (mirror of matrix's _is_authorized_user).
        A user NOT in ``_allowed_user_ids`` tapping a bot-seeded reaction is
        rejected — even on a bot-posted approval card."""
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        # Clear the default test allowlist; this test pins the rejection path.
        adapter._allowed_user_ids = {"user-1"}
        adapter._approval_prompts_by_event["post-X"] = {
            "session_key": "sess-Locked",
            "requester_user_id": "user-1",
            "chat_id": "dm-ch",
        }

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            ev = _ws_event_reaction_added(post_id="post-X", user_id="intruder-99",
                                          emoji_name="white_check_mark")
            await adapter._handle_ws_event(ev)

        mock_resolve.assert_not_called()
        # Entry should NOT be consumed (the unauthorized tap must leave the card open).
        assert "post-X" in adapter._approval_prompts_by_event

    @pytest.mark.asyncio
    async def test_reaction_on_non_approval_post_is_noop(self):
        """User reacts to some unrelated post — must not crash, must not call resolve."""
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        # no entry in _approval_prompts_by_event for post-99

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            ev = _ws_event_reaction_added(post_id="post-99", user_id="user-1", emoji_name="white_check_mark")
            await adapter._handle_ws_event(ev)

        mock_resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_emoji_on_approval_post_is_noop(self):
        """✅/♾️/🚫 only; a 👍 or ❤️ on the same post must not resolve."""
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        adapter._approval_prompts_by_event["post-5"] = {
            "session_key": "sess-W",
            "requester_user_id": "user-1",
        }

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            ev = _ws_event_reaction_added(post_id="post-5", user_id="user-1", emoji_name="👍")
            await adapter._handle_ws_event(ev)

        mock_resolve.assert_not_called()


class TestWSEventShape:
    """The Mattermost ``reaction_added`` WS event has a NESTED payload
    (``data.reaction`` is a JSON-encoded ``model.Reaction`` string), not the
    flat shape this code originally assumed. These tests pin the contract so a
    future Mattermost upgrade can't silently break approval resolution."""

    @pytest.mark.asyncio
    async def test_flat_shape_event_is_ignored(self):
        """A legacy/incorrect flat event must NOT resolve — that's the bug
        we hit in the 2026-09-26 session when this code looked for
        ``data.post_id`` instead of ``data.reaction.post_id``."""
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        adapter._approval_prompts_by_event["post-X"] = {
            "session_key": "sess-Z",
            "requester_user_id": "",
            "chat_id": "chat-1",
            "choices": ["once", "session", "deny"],
        }
        ev = {
            "event": "reaction_added",
            "data": {  # WRONG shape — flat
                "post_id": "post-X",
                "user_id": "user-1",
                "emoji_name": "white_check_mark",
            },
        }

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            await adapter._handle_ws_event(ev)

        mock_resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_nested_shape_with_string_payload(self):
        """The actual Mattermost shape: ``data.reaction`` is a JSON string."""
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        adapter._approval_prompts_by_event["post-X"] = {
            "session_key": "sess-Z",
            "requester_user_id": "",
            "chat_id": "chat-1",
            "choices": ["once", "session", "deny"],
        }
        reaction = json.dumps({
            "user_id": "user-1",
            "post_id": "post-X",
            "emoji_name": "white_check_mark",
        })
        ev = {"event": "reaction_added", "data": {"reaction": reaction}}

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            await adapter._handle_ws_event(ev)

        # v2: request_id threaded through; '' when the registry entry doesn't have one.
        mock_resolve.assert_called_once_with("sess-Z", "once", request_id=None)

    @pytest.mark.asyncio
    async def test_nested_shape_with_dict_payload(self):
        """Robustness: if the server ever sends ``data.reaction`` as an
        already-parsed dict (some clients/proxies do this), the handler
        should still work."""
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        adapter._approval_prompts_by_event["post-X"] = {
            "session_key": "sess-Z",
            "requester_user_id": "",
            "chat_id": "chat-1",
            "choices": ["once", "session", "deny"],
        }
        ev = {
            "event": "reaction_added",
            "data": {
                "reaction": {  # already a dict, not a string
                    "user_id": "user-1",
                    "post_id": "post-X",
                    "emoji_name": "cyclone",  # v2: 🌀 = session (was infinity in v1)
                },
            },
        }

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            await adapter._handle_ws_event(ev)

        # v2: request_id threaded through; '' when the registry entry doesn't have one.
        mock_resolve.assert_called_once_with("sess-Z", "session", request_id=None)

    @pytest.mark.asyncio
    async def test_malformed_reaction_payload_is_silently_ignored(self):
        """A garbage ``data.reaction`` value must not crash the WS loop."""
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        ev = {"event": "reaction_added", "data": {"reaction": "{not valid json"}}

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            # Must not raise
            await adapter._handle_ws_event(ev)

        mock_resolve.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_reaction_field_is_silently_ignored(self):
        """A reaction event without the ``reaction`` field must be ignored."""
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        ev = {"event": "reaction_added", "data": {}}  # no "reaction" key

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            await adapter._handle_ws_event(ev)

        mock_resolve.assert_not_called()


# ─── B-side: _send_exec_approval_prompt registers the prompt + seeds reactions ─


class TestApprovalPromptSendsReactions:

    @pytest.mark.asyncio
    async def test_send_exec_approval_registers_and_seeds_reactions(self):
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        # Stub the HTTP layer so we don't actually call Mattermost.
        adapter.send = AsyncMock(return_value=_ok_send("post-A"))
        adapter._send_reaction = AsyncMock(return_value=True)

        from gateway.platforms.base import ExecApprovalPrompt
        prompt = ExecApprovalPrompt(
            chat_id="ch-1",
            session_key="agent:main:mattermost:dm:user-1",
            text="⚠️ Hermes wants to run a command that needs your OK\n\n```\nls\n```\nWhy it was flagged: dangerous",
            # v2: matrix-parity choice set with `always` offered. Bot seeds
            # exactly one reaction per choice — ✅/🌀/♾️/🚫.
            actions=[("Allow Once", "once", "primary"),
                     ("Approve Session", "session", ""),
                     ("Approve Always", "always", ""),
                     ("Deny", "deny", "danger")],
            command="ls", description="dangerous", smart_denied=False,
            metadata={"requester_user_id": "user-1"},
            # v2: pass request_id through the dataclass directly (in production
            # ``send_exec_approval`` reads it from metadata; tests construct
            # the prompt explicitly).
            request_id="req-12345",
        )

        result = await adapter._send_exec_approval_prompt(prompt)

        assert result.success is True
        assert "post-A" in adapter._approval_prompts_by_event
        entry = adapter._approval_prompts_by_event["post-A"]
        assert entry["session_key"] == "agent:main:mattermost:dm:user-1"
        assert entry["requester_user_id"] == "user-1"
        # v2: request_id is threaded through from the prompt to the registry,
        # so a later tap can resolve only this specific approval.
        assert entry["request_id"] == "req-12345"
        # v2: matrix-parity emoji set — one reaction per offered choice.
        seeded = [call.args[1] for call in adapter._send_reaction.await_args_list]
        assert seeded == ["white_check_mark", "cyclone", "infinity", "no_entry_sign"]
        # v3: the same set is recorded on the registry entry so a tap can
        # retract the WHOLE approval card (not just the tapped emoji).
        assert entry["seeded_emojis"] == ["white_check_mark", "cyclone", "infinity", "no_entry_sign"]

    @pytest.mark.asyncio
    async def test_tap_retracts_all_seeded_emoji_not_just_tapped_one(self):
        """v3 regression: tapping one emoji must remove ALL seeded reactions
        (✅ 🌀 ♾️ ❌), not just the one the user tapped. Without this the
        approval card stays cluttered with 3 emoji after each tap.

        Mirrors the picker handler which already iterates ``picker.choices``.
        """
        adapter = _make_adapter()
        adapter._bot_user_id = "bot-id"
        adapter._allowed_user_ids = {"user-1"}
        adapter.send = AsyncMock(return_value=_ok_send("post-X"))
        adapter._send_reaction = AsyncMock(return_value=True)
        adapter._delete_reaction = AsyncMock(return_value=True)

        from gateway.platforms.base import ExecApprovalPrompt
        prompt = ExecApprovalPrompt(
            chat_id="ch-1",
            session_key="agent:main:mattermost:dm:user-1",
            text="⚠️ flagged",
            actions=[("Allow Once", "once", "primary"),
                     ("Approve Session", "session", ""),
                     ("Approve Always", "always", ""),
                     ("Deny", "deny", "danger")],
            command="ls", description="x", smart_denied=False,
            metadata={"requester_user_id": "user-1"},
            request_id="req-X",
        )
        await adapter._send_exec_approval_prompt(prompt)
        adapter._delete_reaction.reset_mock()

        # User taps ✅ — every other seeded emoji must also be retracted.
        with patch("tools.approval.resolve_gateway_approval", return_value=1):
            ev = _ws_event_reaction_added(
                post_id="post-X", user_id="user-1", emoji_name="white_check_mark")
            await adapter._handle_ws_event(ev)
        # _delete_reaction calls are scheduled as fire-and-forget tasks; drain
        # the loop so the mock records them before we inspect it.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        deleted = [call.args[1] for call in adapter._delete_reaction.await_args_list]
        assert sorted(deleted) == sorted(
            ["white_check_mark", "cyclone", "infinity", "no_entry_sign"]
        ), f"expected all 4 seeded emoji retracted, got {deleted!r}"


def _ok_send(post_id: str):
    from gateway.platforms.base import SendResult
    return SendResult(success=True, message_id=post_id)
