"""Room modes (free / mention / watch) and federation-invitation handling
for the Nextcloud Talk adapter.

Ported from chefboyrdave21/hermes-nextcloud-talk (offered on PR #11458):

* ``mention`` rooms only start a turn for messages that @mention the bot;
  ``watch`` rooms answer mentions immediately and buffer the rest into one
  periodic ``internal`` digest that the agent may answer with ``NO_REPLY``;
  ``free`` (default) keeps today's behaviour. DMs are always ``free`` and
  ``!``-commands always reach the bot.
* Mode gating runs AFTER the pre-effect authorization gate (PR #11458 P1)
  and BEFORE any ack / download / STT, so a gated message costs nothing and
  a watch buffer never holds an unauthorized actor's message.
* Federation invitations are auto-accepted only when PENDING and sent by an
  allowlisted FULL cloud id; everything else stays pending for a human.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.nextcloud_talk.adapter import (
    DEFAULT_ROOM_MODE,
    ROOM_MODE_FREE,
    ROOM_MODE_MENTION,
    ROOM_MODE_WATCH,
    WATCH_BUFFER_MAX_MESSAGES,
    NextcloudTalkAdapter,
    _env_enablement,
    _is_allowed_inviter,
    _is_mention,
    _parse_room_modes_csv,
)

TOKEN = "roomtok1"
WATCH = "watchtok"
BOT = "hermes"


def _make_adapter(monkeypatch, *, authorize=True, **extra_overrides) -> NextcloudTalkAdapter:
    monkeypatch.setenv("NEXTCLOUD_TALK_APP_PASSWORD", "app-pw")
    for var in (
        "NEXTCLOUD_TALK_ROOM_MODES", "NEXTCLOUD_TALK_DEFAULT_ROOM_MODE",
        "NEXTCLOUD_TALK_ALLOWED_INVITERS", "NEXTCLOUD_TALK_DISPLAY_NAME",
    ):
        monkeypatch.delenv(var, raising=False)
    extra = {
        "nextcloud_url": "https://nc.example.com",
        "username": BOT,
        "conversations": [{"token": TOKEN}],
        **extra_overrides,
    }
    adapter = NextcloudTalkAdapter(PlatformConfig(enabled=True, extra=extra))
    if authorize is not None:
        if authorize is True or authorize is False:
            _verdict = authorize
            adapter.set_authorization_check(lambda *a, **k: _verdict)
        else:
            adapter.set_authorization_check(authorize)
    return adapter


def _instrument(adapter):
    """Spy every side-effect path so a test can assert nothing fired."""
    client = MagicMock()
    client.send_message = AsyncMock(return_value=(True, 555, None))
    client.download_file = AsyncMock(return_value=(True, "/tmp/x", None))
    adapter._client = client
    adapter._download_attachment = AsyncMock(return_value="/tmp/x")
    adapter.send = AsyncMock()
    seen = []
    adapter.handle_message = AsyncMock(side_effect=lambda e: seen.append(e))
    return client, seen


def _talk_msg(text, *, msg_id=101, user="niko", token=TOKEN, params=None) -> dict:
    return {
        "id": msg_id,
        "token": token,
        "actorId": user,
        "actorDisplayName": user.title(),
        "message": text,
        "messageParameters": params or {},
        "systemMessage": "",
    }


# ── Pure helpers ───────────────────────────────────────────────────────


class TestMentionDetection:
    def test_rich_user_parameter(self):
        params = {"mention-user1": {"type": "user", "id": BOT, "name": "Hermes"}}
        assert _is_mention("{mention-user1} hi", params, BOT, "", TOKEN) is True

    def test_rich_user_parameter_other_user(self):
        params = {"mention-user1": {"type": "user", "id": "alice", "name": "Alice"}}
        assert _is_mention("{mention-user1} hi", params, BOT, "Hermes", TOKEN) is False

    def test_rich_call_parameter_is_room_wide_mention(self):
        params = {"mention-call1": {"type": "call", "id": TOKEN, "name": "Ops"}}
        assert _is_mention("{mention-call1} standup?", params, BOT, "", TOKEN) is True

    @pytest.mark.parametrize("text", [
        "@hermes can you check?",
        "hey @Hermes",
        '@"Hermes Bot" please',
        "@Hermes Bot please",
    ])
    def test_plain_text_fallback(self, text):
        assert _is_mention(text, {}, BOT, "Hermes Bot", TOKEN) is True

    @pytest.mark.parametrize("text", [
        "mail me at hermes@example.com",   # not an @-prefix mention
        "@hermesx is someone else",
        "nothing for the bot here",
    ])
    def test_plain_text_negatives(self, text):
        assert _is_mention(text, {}, BOT, "Hermes Bot", TOKEN) is False


class TestInviterAllowlist:
    def test_exact_full_cloud_id_matches_case_insensitively(self):
        assert _is_allowed_inviter("Alice@Cloud.Example.COM", ["alice@cloud.example.com"]) is True

    def test_scheme_and_prefix_are_normalized(self):
        assert _is_allowed_inviter(
            "federated_users/alice@https://cloud.example.com/", ["alice@cloud.example.com"],
        ) is True

    def test_local_part_alone_never_matches(self):
        # ``alice`` on the allowlist must not admit alice@attacker.example —
        # this is the deliberate divergence from the upstream plugin.
        assert _is_allowed_inviter("alice@attacker.example", ["alice"]) is False
        assert _is_allowed_inviter("alice@cloud.example.com", ["alice"]) is False

    def test_other_host_does_not_match(self):
        assert _is_allowed_inviter("alice@attacker.example", ["alice@cloud.example.com"]) is False

    def test_empty_allowlist_or_inviter_fails_closed(self):
        assert _is_allowed_inviter("alice@cloud.example.com", []) is False
        assert _is_allowed_inviter("", ["alice@cloud.example.com"]) is False
        assert _is_allowed_inviter(None, ["alice@cloud.example.com"]) is False


class TestRoomModeParsing:
    def test_csv_parsing_skips_malformed_and_unknown(self):
        parsed = _parse_room_modes_csv("a=mention, b=watch ,c=bogus,d,=free,e=FREE")
        assert parsed == {"a": "mention", "b": "watch", "e": "free"}

    def test_env_enablement_seeds_modes_and_federation(self, monkeypatch):
        monkeypatch.setenv("NEXTCLOUD_TALK_URL", "https://nc.example.com")
        monkeypatch.setenv("NEXTCLOUD_TALK_USERNAME", BOT)
        monkeypatch.setenv("NEXTCLOUD_TALK_APP_PASSWORD", "pw")
        monkeypatch.setenv("NEXTCLOUD_TALK_ROOM_MODES", "a=mention,b=watch")
        monkeypatch.setenv("NEXTCLOUD_TALK_DEFAULT_ROOM_MODE", "mention")
        monkeypatch.setenv("NEXTCLOUD_TALK_ALLOWED_INVITERS", "alice@cloud.example.com, bob@other.example")
        monkeypatch.setenv("NEXTCLOUD_TALK_FEDERATION_POLL_INTERVAL", "120")
        monkeypatch.setenv("NEXTCLOUD_TALK_DISPLAY_NAME", "Hermes Bot")
        seed = _env_enablement()
        assert seed["room_modes"] == {"a": "mention", "b": "watch"}
        assert seed["default_room_mode"] == "mention"
        assert seed["display_name"] == "Hermes Bot"
        assert seed["federation"] == {
            "allowed_inviters": ["alice@cloud.example.com", "bob@other.example"],
            "poll_interval": 120,
        }


# ── Mode resolution ────────────────────────────────────────────────────


class TestRoomModeResolution:
    def test_default_is_free_for_backwards_compatibility(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        assert DEFAULT_ROOM_MODE == ROOM_MODE_FREE
        assert adapter._room_mode(TOKEN) == ROOM_MODE_FREE

    def test_per_conversation_mode_wins_over_room_modes_map(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            conversations=[{"token": TOKEN, "mode": "watch"}, {"token": "b"}],
            room_modes={TOKEN: "mention", "b": "mention"},
            default_room_mode="free",
        )
        assert adapter._room_mode(TOKEN) == ROOM_MODE_WATCH
        assert adapter._room_mode("b") == ROOM_MODE_MENTION

    def test_unknown_mode_falls_back_to_default(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            conversations=[{"token": TOKEN, "mode": "shout"}],
            default_room_mode="mention",
        )
        assert adapter._room_mode(TOKEN) == ROOM_MODE_MENTION

    def test_dm_is_always_free(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, default_room_mode="mention")
        assert adapter._classify_chat("not-configured") == "dm"
        assert adapter._room_mode("not-configured") == ROOM_MODE_FREE

    def test_invalid_inviter_entries_are_dropped(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            federation={"allowed_inviters": ["alice", "Bob@Other.Example", ""]},
        )
        assert adapter._allowed_inviters == ["bob@other.example"]


# ── Mention mode ───────────────────────────────────────────────────────


class TestMentionMode:
    @pytest.mark.asyncio
    async def test_unmentioned_message_triggers_zero_effects(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, conversations=[{"token": TOKEN, "mode": "mention"}])
        client, seen = _instrument(adapter)
        msg = _talk_msg("just chatting")
        msg["messageParameters"] = {
            "file": {"type": "file", "name": "memo.mp3", "path": "memo.mp3",
                     "mimetype": "audio/mpeg", "size": "1"},
        }
        await adapter._on_poll_message(msg, TOKEN)
        client.send_message.assert_not_awaited()          # no "Thinking..." ack
        adapter._download_attachment.assert_not_awaited()  # no WebDAV egress
        assert adapter._pending_acks == {}
        assert adapter._watch_buffers == {}
        assert seen == []

    @pytest.mark.asyncio
    async def test_rich_mention_starts_a_turn(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, conversations=[{"token": TOKEN, "mode": "mention"}])
        _, seen = _instrument(adapter)
        params = {"mention-user1": {"type": "user", "id": BOT, "name": "Hermes"}}
        await adapter._on_poll_message(_talk_msg("{mention-user1} status?", params=params), TOKEN)
        assert len(seen) == 1
        assert seen[0].source.user_id == "niko"

    @pytest.mark.asyncio
    async def test_plain_text_mention_uses_display_name(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch, display_name="Hermes Bot",
            conversations=[{"token": TOKEN, "mode": "mention"}],
        )
        _, seen = _instrument(adapter)
        await adapter._on_poll_message(_talk_msg('@"Hermes Bot" status?'), TOKEN)
        assert len(seen) == 1

    @pytest.mark.asyncio
    async def test_bang_commands_bypass_mode_gating(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, conversations=[{"token": TOKEN, "mode": "mention"}])
        _, seen = _instrument(adapter)
        await adapter._on_poll_message(_talk_msg("!new"), TOKEN)
        assert len(seen) == 1
        assert seen[0].get_command() == "new"

    @pytest.mark.asyncio
    async def test_authorization_gate_precedes_mode_gating(self, monkeypatch):
        """An unauthorized actor's @mention must not reach the gateway."""
        adapter = _make_adapter(
            monkeypatch, authorize=lambda uid, *a, **k: uid == "niko",
            conversations=[{"token": TOKEN, "mode": "mention"}],
        )
        _, seen = _instrument(adapter)
        params = {"mention-user1": {"type": "user", "id": BOT, "name": "Hermes"}}
        await adapter._on_poll_message(
            _talk_msg("{mention-user1} hi", user="mallory", params=params), TOKEN,
        )
        assert seen == []


# ── Watch mode ─────────────────────────────────────────────────────────


class TestWatchMode:
    def _watch_adapter(self, monkeypatch, **kw):
        adapter = _make_adapter(
            monkeypatch,
            conversations=[{"token": WATCH, "alias": "ops", "mode": "watch"}, {"token": TOKEN}],
            **kw,
        )
        client, seen = _instrument(adapter)
        return adapter, client, seen

    @pytest.mark.asyncio
    async def test_unmentioned_messages_are_buffered_without_effects(self, monkeypatch):
        adapter, client, seen = self._watch_adapter(monkeypatch)
        await adapter._on_poll_message(_talk_msg("deploy went fine", msg_id=1, token=WATCH), WATCH)
        await adapter._on_poll_message(_talk_msg("ack", msg_id=2, user="alice", token=WATCH), WATCH)
        client.send_message.assert_not_awaited()
        assert seen == []
        assert [e["message_id"] for e in adapter._watch_buffers[WATCH]] == [1, 2]
        assert adapter._watch_buffers[WATCH][1]["user_id"] == "alice"

    @pytest.mark.asyncio
    async def test_direct_mention_in_watch_room_is_answered_immediately(self, monkeypatch):
        adapter, _, seen = self._watch_adapter(monkeypatch)
        params = {"mention-user1": {"type": "user", "id": BOT, "name": "Hermes"}}
        await adapter._on_poll_message(
            _talk_msg("{mention-user1} can you check?", token=WATCH, params=params), WATCH,
        )
        assert len(seen) == 1
        assert adapter._watch_buffers == {}

    @pytest.mark.asyncio
    async def test_unauthorized_actor_is_never_buffered(self, monkeypatch):
        """The digest bypasses the gateway allowlist as an internal event, so
        the pre-effect gate is the ONLY authority — it must run first."""
        adapter, _, seen = self._watch_adapter(
            monkeypatch, authorize=lambda uid, *a, **k: uid == "niko",
        )
        await adapter._on_poll_message(_talk_msg("psst", user="mallory", token=WATCH), WATCH)
        assert adapter._watch_buffers == {}
        assert seen == []

    @pytest.mark.asyncio
    async def test_flush_builds_one_internal_digest_and_clears_buffer(self, monkeypatch):
        adapter, _, seen = self._watch_adapter(monkeypatch)
        await adapter._on_poll_message(_talk_msg("deploy went fine", msg_id=1, token=WATCH), WATCH)
        await adapter._on_poll_message(
            _talk_msg("is the bot able to summarize this?", msg_id=2, user="alice", token=WATCH),
            WATCH,
        )
        await adapter._flush_watch_buffers()

        assert len(seen) == 1
        event = seen[0]
        assert event.internal is True
        assert event.source.chat_id == WATCH
        assert event.source.chat_type == "group"
        assert event.source.user_id == "alice"        # most recent authorized sender
        assert event.message_id == f"watch-digest-{WATCH}-2"
        assert "2 message(s)" in event.text and '"ops"' in event.text
        assert "- Niko (niko) #1: deploy went fine" in event.text
        assert "- Alice (alice) #2: is the bot able to summarize this?" in event.text
        assert event.text.rstrip().endswith("NO_REPLY.")
        assert adapter._watch_buffers == {}

    @pytest.mark.asyncio
    async def test_flush_with_empty_buffers_emits_nothing(self, monkeypatch):
        adapter, _, seen = self._watch_adapter(monkeypatch)
        await adapter._flush_watch_buffers()
        adapter._watch_buffers[WATCH] = []
        await adapter._flush_watch_buffers()
        assert seen == []

    @pytest.mark.asyncio
    async def test_buffer_is_bounded(self, monkeypatch):
        adapter, _, _ = self._watch_adapter(monkeypatch)
        for i in range(WATCH_BUFFER_MAX_MESSAGES + 5):
            await adapter._on_poll_message(_talk_msg(f"m{i}", msg_id=i + 1, token=WATCH), WATCH)
        buffer = adapter._watch_buffers[WATCH]
        assert len(buffer) == WATCH_BUFFER_MAX_MESSAGES
        assert buffer[0]["message_id"] == 6   # oldest dropped first

    @pytest.mark.asyncio
    async def test_free_room_in_same_adapter_is_unaffected(self, monkeypatch):
        adapter, _, seen = self._watch_adapter(monkeypatch)
        await adapter._on_poll_message(_talk_msg("hello", token=TOKEN), TOKEN)
        assert len(seen) == 1
        assert adapter._watch_buffers == {}


# ── Background loops: started only when needed ─────────────────────────


class TestBackgroundLoops:
    async def _connect(self, adapter):
        client = MagicMock()
        client.list_conversations = AsyncMock(return_value=(True, [], None))
        client.join_conversation = AsyncMock(return_value=(True, None))
        client.close = AsyncMock()
        adapter._client = client
        adapter._poll_loop = AsyncMock()
        assert await adapter.connect() is True
        return client

    @pytest.mark.asyncio
    async def test_default_config_starts_no_background_loops(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        await self._connect(adapter)
        try:
            assert adapter._watch_flush_task is None
            assert adapter._invitation_task is None
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_watch_room_and_allowlist_start_their_loops(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            conversations=[{"token": TOKEN, "mode": "watch"}],
            federation={"allowed_inviters": ["alice@cloud.example.com"]},
        )
        await self._connect(adapter)
        try:
            assert adapter._watch_flush_task is not None
            assert adapter._invitation_task is not None
        finally:
            await adapter.disconnect()
        assert adapter._watch_flush_task is None
        assert adapter._invitation_task is None


# ── Federation invitations ─────────────────────────────────────────────


def _invite(invite_id, inviter, *, state=0, room_name="Shared", local_token=None):
    invite = {
        "id": invite_id,
        "inviterCloudId": inviter,
        "inviterDisplayName": inviter.split("@")[0].title(),
        "roomName": room_name,
        "remoteToken": f"remote-{invite_id}",
    }
    if state is not None:
        invite["state"] = state
    if local_token:
        invite["localToken"] = local_token
    return invite


class TestFederationInvitations:
    def _adapter(self, monkeypatch, invitations, *, accept=(True, {"token": "fedtok", "displayName": "Shared"}, None)):
        adapter = _make_adapter(
            monkeypatch,
            federation={"allowed_inviters": ["alice@cloud.example.com"]},
        )
        client = MagicMock()
        client.list_federation_invitations = AsyncMock(return_value=(True, invitations, None, 200))
        client.accept_federation_invitation = AsyncMock(return_value=accept)
        adapter._client = client
        return adapter, client

    @pytest.mark.asyncio
    async def test_allowlisted_pending_invite_is_accepted_and_adopted(self, monkeypatch):
        adapter, client = self._adapter(monkeypatch, [_invite(7, "alice@cloud.example.com")])
        await adapter._process_federation_invitations()
        client.accept_federation_invitation.assert_awaited_once_with(7)
        assert adapter._invites_seen == {7: "accepted"}
        assert any(c["token"] == "fedtok" and c.get("federated") for c in adapter._conversations)
        assert adapter._classify_chat("fedtok") == "group"
        assert adapter._chat_name_cache["fedtok"] == "Shared"

    @pytest.mark.asyncio
    async def test_non_allowlisted_invite_stays_pending_and_is_logged_once(self, monkeypatch, caplog):
        adapter, client = self._adapter(monkeypatch, [_invite(8, "mallory@attacker.example")])
        with caplog.at_level("INFO", logger="plugins.platforms.nextcloud_talk.adapter"):
            await adapter._process_federation_invitations()
            await adapter._process_federation_invitations()
        client.accept_federation_invitation.assert_not_awaited()
        assert adapter._invites_seen == {8: "left-pending"}
        assert sum("left pending" in r.getMessage() for r in caplog.records) == 1
        assert len(adapter._conversations) == 1

    @pytest.mark.asyncio
    async def test_same_local_name_on_other_host_is_not_accepted(self, monkeypatch):
        adapter, client = self._adapter(monkeypatch, [_invite(9, "alice@attacker.example")])
        await adapter._process_federation_invitations()
        client.accept_federation_invitation.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_pending_invite_is_skipped(self, monkeypatch):
        adapter, client = self._adapter(monkeypatch, [_invite(10, "alice@cloud.example.com", state=1)])
        await adapter._process_federation_invitations()
        client.accept_federation_invitation.assert_not_awaited()
        assert adapter._invites_seen == {10: "not-pending"}

    @pytest.mark.asyncio
    async def test_missing_state_is_treated_as_pending(self, monkeypatch):
        adapter, client = self._adapter(monkeypatch, [_invite(11, "alice@cloud.example.com", state=None)])
        await adapter._process_federation_invitations()
        client.accept_federation_invitation.assert_awaited_once_with(11)

    @pytest.mark.asyncio
    async def test_failed_accept_is_retried_next_pass(self, monkeypatch):
        adapter, client = self._adapter(
            monkeypatch, [_invite(12, "alice@cloud.example.com")], accept=(False, None, "HTTP 500"),
        )
        await adapter._process_federation_invitations()
        await adapter._process_federation_invitations()
        assert client.accept_federation_invitation.await_count == 2
        assert 12 not in adapter._invites_seen
        assert len(adapter._conversations) == 1

    @pytest.mark.asyncio
    async def test_accepted_invite_is_decided_once(self, monkeypatch):
        adapter, client = self._adapter(monkeypatch, [_invite(13, "alice@cloud.example.com")])
        await adapter._process_federation_invitations()
        await adapter._process_federation_invitations()
        client.accept_federation_invitation.assert_awaited_once()
        assert sum(c["token"] == "fedtok" for c in adapter._conversations) == 1

    @pytest.mark.asyncio
    async def test_endpoint_404_is_not_an_error(self, monkeypatch):
        adapter, client = self._adapter(monkeypatch, [])
        client.list_federation_invitations = AsyncMock(return_value=(False, [], "HTTP 404", 404))
        await adapter._process_federation_invitations()
        await adapter._process_federation_invitations()
        client.accept_federation_invitation.assert_not_awaited()
        assert adapter._federation_unavailable_logged is True

    @pytest.mark.asyncio
    async def test_no_allowlist_means_no_invitation_requests(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        client = MagicMock()
        client.list_federation_invitations = AsyncMock()
        adapter._client = client
        await adapter._process_federation_invitations()
        client.list_federation_invitations.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_adopted_room_starts_polling_when_connected(self, monkeypatch):
        adapter, _client = self._adapter(monkeypatch, [_invite(14, "alice@cloud.example.com")])
        polled = []

        async def fake_poll(token):
            polled.append(token)

        adapter._poll_loop = fake_poll
        adapter._poll_tasks.append(asyncio.get_event_loop().create_task(fake_poll(TOKEN)))
        await adapter._process_federation_invitations()
        await asyncio.gather(*adapter._poll_tasks)
        assert polled == [TOKEN, "fedtok"]
