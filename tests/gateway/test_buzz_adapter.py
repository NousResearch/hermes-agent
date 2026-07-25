"""Tests for the Buzz platform adapter plugin."""

import asyncio
import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

# Load plugins/platforms/buzz/adapter.py under a unique module name
# (plugin_adapter_buzz) so it cannot collide with other plugin adapters
# loaded by sibling tests in the same xdist worker.
_buzz_mod = load_plugin_adapter("buzz")

BuzzAdapter = _buzz_mod.BuzzAdapter
hex_to_npub = _buzz_mod.hex_to_npub
npub_to_hex = _buzz_mod.npub_to_hex
_normalize_user_ref = _buzz_mod._normalize_user_ref
_cli_error_message = _buzz_mod._cli_error_message
_resolve_private_key = _buzz_mod._resolve_private_key
check_requirements = _buzz_mod.check_requirements
validate_config = _buzz_mod.validate_config
register = _buzz_mod.register
_env_enablement = _buzz_mod._env_enablement
_standalone_send = _buzz_mod._standalone_send

# Real key pair (Chip's public identity — public information, not a secret)
SELF_PUBKEY = "9fd5c7ba6d3ef224da78f541e0fcb9c50f72cc63edb19aae76ac6a0474dfa860"
SELF_NPUB = "npub1nl2u0wnd8mezfknc74q7pl9ec58h9nrrakce4tnk434qgaxl4psqe5twr6"
OTHER_PUBKEY = "a" * 64
CHANNEL = "ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd"
# Real DM conversation as materialized by a hosted relay: `dms list` returns
# [] for it (#68871) while `channels list` shows it as name "DM", empty
# description, indistinguishable from a channel except via message p-tags.
DM_CHANNEL = "6468cc16-a114-4f23-8b8c-02c1655cbf6b"

_ENV_VARS = (
    "BUZZ_RELAY_URL",
    "BUZZ_PRIVATE_KEY",
    "BUZZ_CHANNELS",
    "BUZZ_HOME_CHANNEL",
    "BUZZ_ALLOWED_USERS",
    "BUZZ_ALLOW_ALL_USERS",
    "BUZZ_POLL_INTERVAL",
    "BUZZ_CLI_PATH",
    "BUZZ_CREDENTIALS_FILE",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """Keep tests hermetic: no ambient Buzz env vars or real credentials."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(_buzz_mod, "_DEFAULT_CREDENTIALS_DIR", tmp_path / "no-creds")
    yield


def _event(event_id, pubkey=OTHER_PUBKEY, content="hello", created_at=1000, kind=9):
    return {
        "id": event_id,
        "pubkey": pubkey,
        "content": content,
        "created_at": created_at,
        "kind": kind,
        "tags": [["h", CHANNEL]],
    }


def _make_adapter(extra=None):
    from gateway.config import PlatformConfig

    cfg = PlatformConfig(enabled=True, extra={"relay_url": "https://test.relay", **(extra or {})})
    adapter = BuzzAdapter(cfg)
    adapter._self_pubkey = SELF_PUBKEY
    adapter._self_npub = SELF_NPUB
    adapter._display_name = "Chip"
    adapter._private_key = "nsec1test"
    return adapter


class _ScriptedCli:
    """Fake ``_run_cli`` that routes on the buzz subcommand and records calls."""

    def __init__(self):
        self.responses = {}  # (group, cmd) -> list of (code, stdout, stderr)
        self.calls = []

    def script(self, group, cmd, payload, code=0, stderr=""):
        stdout = payload if isinstance(payload, str) else json.dumps(payload)
        self.responses.setdefault((group, cmd), []).append((code, stdout, stderr))

    async def __call__(self, args, *, input_text=None):
        self.calls.append((list(args), input_text))
        queue = self.responses.get((args[0], args[1]), [])
        if len(queue) > 1:
            return queue.pop(0)
        if queue:
            return queue[0]
        return 0, "[]", ""


# ── bech32 / identity helpers ─────────────────────────────────────────────


class TestBech32Helpers:

    def test_hex_to_npub_known_pair(self):
        assert hex_to_npub(SELF_PUBKEY) == SELF_NPUB

    def test_npub_to_hex_known_pair(self):
        assert npub_to_hex(SELF_NPUB) == SELF_PUBKEY

    def test_round_trip(self):
        npub = hex_to_npub(OTHER_PUBKEY)
        assert npub is not None and npub.startswith("npub1")
        assert npub_to_hex(npub) == OTHER_PUBKEY

    def test_invalid_inputs(self):
        assert hex_to_npub("not-hex") is None
        assert hex_to_npub("ab") is None
        assert npub_to_hex("npub1invalidchecksum") is None
        assert npub_to_hex("nsec1notanpub") is None

    def test_normalize_user_ref(self):
        assert _normalize_user_ref(SELF_NPUB) == SELF_PUBKEY
        assert _normalize_user_ref(SELF_PUBKEY.upper()) == SELF_PUBKEY
        assert _normalize_user_ref("") is None
        assert _normalize_user_ref("bogus") is None


# ── Adapter init / config precedence ──────────────────────────────────────


class TestBuzzAdapterInit:

    def test_init_from_env(self, monkeypatch):
        monkeypatch.setenv("BUZZ_RELAY_URL", "https://env.relay")
        monkeypatch.setenv("BUZZ_CHANNELS", "aaa, bbb ,")
        monkeypatch.setenv("BUZZ_POLL_INTERVAL", "7.5")

        from gateway.config import PlatformConfig
        adapter = BuzzAdapter(PlatformConfig(enabled=True))

        assert adapter.relay_url == "https://env.relay"
        assert adapter.channels == ["aaa", "bbb"]
        assert adapter.poll_interval == 7.5

    def test_init_from_config_extra(self):
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(
            enabled=True,
            extra={
                "relay_url": "https://cfg.relay",
                "channels": ["ccc"],
                "poll_interval": 2,
                "home_channel": "ccc",
            },
        )
        adapter = BuzzAdapter(cfg)
        assert adapter.relay_url == "https://cfg.relay"
        assert adapter.channels == ["ccc"]
        assert adapter.poll_interval == 2.0
        assert adapter.home_channel == "ccc"

    def test_env_overrides_config(self, monkeypatch):
        monkeypatch.setenv("BUZZ_RELAY_URL", "https://env.relay")
        from gateway.config import PlatformConfig
        adapter = BuzzAdapter(PlatformConfig(enabled=True, extra={"relay_url": "https://cfg.relay"}))
        assert adapter.relay_url == "https://env.relay"

    def test_poll_interval_clamped_and_defaulted(self):
        adapter = _make_adapter({"poll_interval": 0.01})
        assert adapter.poll_interval == _buzz_mod._MIN_POLL_INTERVAL
        adapter = _make_adapter({"poll_interval": "garbage"})
        assert adapter.poll_interval == _buzz_mod._DEFAULT_POLL_INTERVAL

    def test_allowed_users_normalized_from_mixed_forms(self):
        adapter = _make_adapter({"allowed_users": [SELF_NPUB, OTHER_PUBKEY.upper(), "junk"]})
        assert adapter._allowed_pubkeys == {SELF_PUBKEY, OTHER_PUBKEY}


# ── CLI error contract ────────────────────────────────────────────────────


class TestCliErrorContract:

    def test_parses_json_error(self):
        msg = _cli_error_message('{"error":"relay_error","message":"boom","retryable":false}', 2)
        assert "relay_error" in msg and "boom" in msg and "exit 2" in msg

    def test_falls_back_to_raw_stderr(self):
        assert "garbage" in _cli_error_message("garbage output", 4)

    def test_empty_stderr(self):
        assert "exit code 3" in _cli_error_message("", 3)


# ── Seeding / high-water mark / de-dupe ───────────────────────────────────


class TestPollingDedupe:

    @pytest.fixture
    def adapter(self):
        a = _make_adapter()
        a._dispatched = []

        async def capture(**kwargs):
            a._dispatched.append(kwargs)

        a._dispatch_message = capture
        a._message_handler = AsyncMock()
        return a

    @pytest.mark.asyncio
    async def test_seed_sets_high_water_mark_without_dispatch(self, adapter):
        cli = _ScriptedCli()
        cli.script("messages", "get", [
            _event("e1", content="@Chip old history", created_at=100),
            _event("e2", content="@Chip newer history", created_at=200),
        ])
        adapter._run_cli = cli
        await adapter._seed_channel(CHANNEL, chat_type="group")

        state = adapter._channel_state[CHANNEL]
        assert state["last_ts"] == 200
        assert set(state["seen"]) == {"e1", "e2"}
        # Seeding must never replay history into the agent
        assert adapter._dispatched == []

    @pytest.mark.asyncio
    async def test_new_event_dispatched_once(self, adapter):
        cli = _ScriptedCli()
        cli.script("messages", "get", [_event("e1", content="@Chip hi", created_at=100)])
        adapter._run_cli = cli
        await adapter._seed_channel(CHANNEL, chat_type="group")

        # Poll 1: seeded event + a genuinely new mention
        cli.responses.clear()
        cli.script("messages", "get", [
            _event("e1", content="@Chip hi", created_at=100),
            _event("e2", content="hey @Chip, ping", created_at=150),
        ])
        await adapter._poll_channel(CHANNEL)
        assert [d["message_id"] for d in adapter._dispatched] == ["e2"]
        assert adapter._dispatched[0]["text"] == "hey @Chip, ping"
        assert adapter._channel_state[CHANNEL]["last_ts"] == 150

        # Poll 2: identical response — the seen-id set must de-dupe
        await adapter._poll_channel(CHANNEL)
        assert len(adapter._dispatched) == 1

    @pytest.mark.asyncio
    async def test_poll_uses_since_high_water_mark(self, adapter):
        adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 4242, "seen": {}}
        cli = _ScriptedCli()
        adapter._run_cli = cli
        await adapter._poll_channel(CHANNEL)
        args, _stdin = cli.calls[0]
        assert "--since" in args
        assert args[args.index("--since") + 1] == "4242"

    @pytest.mark.asyncio
    async def test_self_echo_suppressed(self, adapter):
        adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        cli = _ScriptedCli()
        cli.script("messages", "get", [
            _event("own", pubkey=SELF_PUBKEY, content="@Chip I mention myself", created_at=300),
        ])
        adapter._run_cli = cli
        await adapter._poll_channel(CHANNEL)
        assert adapter._dispatched == []
        # …but the event still advances the high-water mark
        assert adapter._channel_state[CHANNEL]["last_ts"] == 300

    @pytest.mark.asyncio
    async def test_non_chat_kinds_ignored(self, adapter):
        adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        cli = _ScriptedCli()
        cli.script("messages", "get", [
            _event("j1", content="@Chip join event", created_at=300, kind=40002),
        ])
        adapter._run_cli = cli
        await adapter._poll_channel(CHANNEL)
        assert adapter._dispatched == []

    @pytest.mark.asyncio
    async def test_seen_set_is_bounded(self, adapter):
        adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        from collections import OrderedDict
        adapter._channel_state[CHANNEL]["seen"] = OrderedDict()
        cli = _ScriptedCli()
        cli.script("messages", "get", [
            _event(f"e{i}", content="unrelated chatter", created_at=i)
            for i in range(_buzz_mod._SEEN_CAP + 100)
        ])
        adapter._run_cli = cli
        await adapter._poll_channel(CHANNEL)
        assert len(adapter._channel_state[CHANNEL]["seen"]) <= _buzz_mod._SEEN_CAP


# ── Mention gating / DMs / authorization ──────────────────────────────────


class TestMentionGating:

    @pytest.fixture
    def adapter(self):
        a = _make_adapter()
        a._dispatched = []

        async def capture(**kwargs):
            a._dispatched.append(kwargs)

        a._dispatch_message = capture
        a._message_handler = AsyncMock()
        a._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        return a

    async def _poll_with(self, adapter, *events):
        cli = _ScriptedCli()
        cli.script("messages", "get", list(events))
        adapter._run_cli = cli
        await adapter._poll_channel(CHANNEL)

    @pytest.mark.asyncio
    async def test_unaddressed_channel_message_ignored(self, adapter):
        await self._poll_with(adapter, _event("e1", content="just chatting", created_at=10))
        assert adapter._dispatched == []

    @pytest.mark.asyncio
    async def test_name_mention_dispatched(self, adapter):
        await self._poll_with(adapter, _event("e1", content="hey @Chip can you help?", created_at=10))
        assert len(adapter._dispatched) == 1

    @pytest.mark.asyncio
    async def test_bare_name_mention_dispatched_case_insensitive(self, adapter):
        await self._poll_with(adapter, _event("e1", content="chip: what do you think?", created_at=10))
        assert len(adapter._dispatched) == 1

    @pytest.mark.asyncio
    async def test_name_inside_word_not_a_mention(self, adapter):
        await self._poll_with(adapter, _event("e1", content="I love potato chips", created_at=10))
        assert adapter._dispatched == []

    @pytest.mark.asyncio
    async def test_npub_mention_dispatched(self, adapter):
        await self._poll_with(adapter, _event("e1", content=f"nostr:{SELF_NPUB} hello", created_at=10))
        assert len(adapter._dispatched) == 1

    @pytest.mark.asyncio
    async def test_dm_always_dispatched(self, adapter):
        adapter._channel_state[CHANNEL]["chat_type"] = "dm"
        await self._poll_with(adapter, _event("e1", content="no mention here", created_at=10))
        assert len(adapter._dispatched) == 1
        assert adapter._dispatched[0]["chat_type"] == "dm"

    @pytest.mark.asyncio
    async def test_allowlist_blocks_unauthorized(self, adapter):
        adapter._allowed_pubkeys = {"b" * 64}
        await self._poll_with(adapter, _event("e1", content="@Chip hello", created_at=10))
        assert adapter._dispatched == []

    @pytest.mark.asyncio
    async def test_allowlist_admits_npub_entry(self):
        a = _make_adapter({"allowed_users": [hex_to_npub(OTHER_PUBKEY)]})
        a._dispatched = []

        async def capture(**kwargs):
            a._dispatched.append(kwargs)

        a._dispatch_message = capture
        a._message_handler = AsyncMock()
        a._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        cli = _ScriptedCli()
        cli.script("messages", "get", [_event("e1", content="@Chip hello", created_at=10)])
        a._run_cli = cli
        await a._poll_channel(CHANNEL)
        assert len(a._dispatched) == 1


# ── DM classification via p-tags (issue #68871) ──────────────────────────
#
# `buzz dms list` returns [] on some hosted relays, so DM conversations leak
# in via `channels list` and get seeded chat_type="group".  The adapter must
# reclassify them from the Nostr tags of real traffic: DM messages are
# p-tagged to our own pubkey WITHOUT the text mentioning us, while channel
# messages only ever p-tag us when the text visibly @mentions us.


def _tagged_event(event_id, channel, *, content, pubkey=OTHER_PUBKEY,
                  created_at=1000, kind=9, p=None, reply_to=None):
    """Event with the tag shapes observed on a live relay (h/p/e tags)."""
    tags = [["h", channel]]
    if reply_to:
        tags.append(["e", reply_to, "", "reply"])
    if p:
        tags.append(["p", p])
    return {
        "id": event_id,
        "pubkey": pubkey,
        "content": content,
        "created_at": created_at,
        "kind": kind,
        "tags": tags,
    }


class TestDmClassification:

    @pytest.fixture
    def adapter(self):
        a = _make_adapter()
        a._dispatched = []

        async def capture(**kwargs):
            a._dispatched.append(kwargs)

        a._dispatch_message = capture
        a._message_handler = AsyncMock()
        # Metadata exactly as `channels list` returns it on the hosted relay.
        a._channel_meta = {
            DM_CHANNEL: {"channel_id": DM_CHANNEL, "name": "DM", "description": ""},
            CHANNEL: {
                "channel_id": CHANNEL,
                "name": "general",
                "description": "General conversation and community updates.",
            },
        }
        a._channel_names = {DM_CHANNEL: "DM", CHANNEL: "general"}
        # Both leaked in as group — the bug under test.
        a._channel_state[DM_CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        a._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        return a

    async def _poll_with(self, adapter, channel, *events):
        cli = _ScriptedCli()
        cli.script("messages", "get", list(events))
        adapter._run_cli = cli
        await adapter._poll_channel(channel)

    @pytest.mark.asyncio
    async def test_unmentioned_ptagged_dm_latches_and_dispatches(self, adapter):
        """The reported bug: a DM without an @mention must dispatch."""
        await self._poll_with(
            adapter, DM_CHANNEL,
            _tagged_event("e1", DM_CHANNEL, content="here's a test message", p=SELF_PUBKEY),
        )
        assert adapter._channel_state[DM_CHANNEL]["chat_type"] == "dm"
        assert [d["message_id"] for d in adapter._dispatched] == ["e1"]
        assert adapter._dispatched[0]["chat_type"] == "dm"

    @pytest.mark.asyncio
    async def test_dm_latch_persists_for_followup_messages(self, adapter):
        await self._poll_with(
            adapter, DM_CHANNEL,
            _tagged_event("e1", DM_CHANNEL, content="first direct message", p=SELF_PUBKEY),
        )
        # Follow-up carries no p-tag at all — the latched state must carry it.
        await self._poll_with(
            adapter, DM_CHANNEL,
            _tagged_event("e2", DM_CHANNEL, content="and a follow-up", created_at=1001),
        )
        assert [d["message_id"] for d in adapter._dispatched] == ["e1", "e2"]

    @pytest.mark.asyncio
    async def test_general_broadcast_without_mention_still_gated(self, adapter):
        await self._poll_with(
            adapter, CHANNEL,
            _tagged_event("e1", CHANNEL, content="just chatting away"),
        )
        assert adapter._dispatched == []
        assert adapter._channel_state[CHANNEL]["chat_type"] == "group"

    @pytest.mark.asyncio
    async def test_general_reply_ptagging_self_stays_channel(self, adapter):
        """A #general reply to us p-tags our pubkey (observed live) — that
        must NOT reclassify the channel; mention gating still applies."""
        await self._poll_with(
            adapter, CHANNEL,
            _tagged_event("e1", CHANNEL, content="@chip what's up?",
                          p=SELF_PUBKEY, reply_to="root-event"),
        )
        assert adapter._channel_state[CHANNEL]["chat_type"] == "group"
        # It carried a mention, so it dispatches — but as a group message.
        assert [d["chat_type"] for d in adapter._dispatched] == ["group"]

        # And once the mention is absent, the channel gate drops the message
        # even though the earlier reply p-tagged us.
        await self._poll_with(
            adapter, CHANNEL,
            _tagged_event("e2", CHANNEL, content="thanks everyone", created_at=1001),
        )
        assert len(adapter._dispatched) == 1

    @pytest.mark.asyncio
    async def test_general_nonreply_mention_ptag_stays_channel(self, adapter):
        """Typed channel mentions p-tag us WITHOUT a reply tag (observed
        live: '@Chip are you up and running?') — still not a DM: the p-tag
        is explained by the visible mention, and the metadata guard protects
        real channels regardless."""
        await self._poll_with(
            adapter, CHANNEL,
            _tagged_event("e1", CHANNEL, content="@Chip are you up and running?", p=SELF_PUBKEY),
        )
        assert adapter._channel_state[CHANNEL]["chat_type"] == "group"
        assert [d["chat_type"] for d in adapter._dispatched] == ["group"]

    @pytest.mark.asyncio
    async def test_channel_like_metadata_blocks_latch_even_without_mention(self, adapter):
        """Second guard on its own: even a p-tagged, un-mentioned message
        cannot reclassify a conversation whose metadata says real channel."""
        adapter._channel_meta[CHANNEL]["description"] = ""
        adapter._channel_meta[CHANNEL]["name"] = "announcements"
        await self._poll_with(
            adapter, CHANNEL,
            _tagged_event("e1", CHANNEL, content="fyi everyone", p=SELF_PUBKEY),
        )
        assert adapter._channel_state[CHANNEL]["chat_type"] == "group"
        assert adapter._dispatched == []

    @pytest.mark.asyncio
    async def test_configured_channel_without_metadata_not_latched(self):
        a = _make_adapter({"channels": ["private-chan"]})
        a._dispatched = []

        async def capture(**kwargs):
            a._dispatched.append(kwargs)

        a._dispatch_message = capture
        a._message_handler = AsyncMock()
        a._channel_state["private-chan"] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        cli = _ScriptedCli()
        cli.script("messages", "get", [
            _tagged_event("e1", "private-chan", content="no mention here", p=SELF_PUBKEY),
        ])
        a._run_cli = cli
        await a._poll_channel("private-chan")
        assert a._channel_state["private-chan"]["chat_type"] == "group"
        assert a._dispatched == []

    @pytest.mark.asyncio
    async def test_seed_latches_dm_from_history_without_dispatch(self, adapter):
        cli = _ScriptedCli()
        cli.script("messages", "get", [
            _tagged_event("h1", DM_CHANNEL, content="@Chip hello there", p=SELF_PUBKEY, created_at=100),
            _tagged_event("h2", DM_CHANNEL, content="here's a test message", p=SELF_PUBKEY, created_at=200),
            _tagged_event("h3", DM_CHANNEL, content="my own reply", pubkey=SELF_PUBKEY, created_at=300),
        ])
        adapter._run_cli = cli
        await adapter._seed_channel(DM_CHANNEL, chat_type="group")
        assert adapter._channel_state[DM_CHANNEL]["chat_type"] == "dm"
        # Seeding classifies but never replays history into the agent.
        assert adapter._dispatched == []

    @pytest.mark.asyncio
    async def test_non_chat_kind_ptag_does_not_latch(self, adapter):
        """Housekeeping events (joins, member adds) may p-tag members —
        only kind-9 chat traffic classifies."""
        cli = _ScriptedCli()
        cli.script("messages", "get", [
            _tagged_event("h1", DM_CHANNEL, content="member added", p=SELF_PUBKEY, kind=40002),
        ])
        adapter._run_cli = cli
        await adapter._seed_channel(DM_CHANNEL, chat_type="group")
        assert adapter._channel_state[DM_CHANNEL]["chat_type"] == "group"

    @pytest.mark.asyncio
    async def test_dm_leading_mention_stripped_for_commands(self, adapter):
        adapter._channel_state[DM_CHANNEL]["chat_type"] = "dm"
        await self._poll_with(
            adapter, DM_CHANNEL,
            _tagged_event("e1", DM_CHANNEL, content="@chip /whoami", p=SELF_PUBKEY),
        )
        assert adapter._dispatched[0]["text"] == "/whoami"

    @pytest.mark.asyncio
    async def test_dm_shaped_channel_discovered_when_dms_list_empty(self):
        """Fallback discovery: with `dms list` broken (returns []), a
        DM-shaped `channels list` entry gets watched; real channels not
        already watched are left alone."""
        a = _make_adapter()
        cli = _ScriptedCli()
        cli.script("dms", "list", [])
        cli.script("channels", "list", [
            {"channel_id": DM_CHANNEL, "name": "DM", "description": "", "created_at": 1},
            {"channel_id": CHANNEL, "name": "general",
             "description": "General conversation and community updates.", "created_at": 2},
        ])
        a._run_cli = cli
        await a._discover_dms(seed=False)
        # Watched as group; the p-tag latch flips it on the first real DM.
        assert a._channel_state[DM_CHANNEL]["chat_type"] == "group"
        assert a._may_reclassify_as_dm(DM_CHANNEL) is True
        assert CHANNEL not in a._channel_state
        assert a._may_reclassify_as_dm(CHANNEL) is False


# ── Sending ───────────────────────────────────────────────────────────────


class TestBuzzAdapterSend:

    @pytest.mark.asyncio
    async def test_send_success_via_stdin(self):
        adapter = _make_adapter()
        adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        cli = _ScriptedCli()
        cli.script("messages", "send", {"accepted": True, "event_id": "evt123", "message": ""})
        adapter._run_cli = cli

        result = await adapter.send(CHANNEL, "hello **markdown**")
        assert result.success is True
        assert result.message_id == "evt123"

        args, stdin_text = cli.calls[0]
        assert args[:2] == ["messages", "send"]
        assert args[args.index("--channel") + 1] == CHANNEL
        # Content travels via stdin (--content -), never argv
        assert args[args.index("--content") + 1] == "-"
        assert stdin_text == "hello **markdown**"
        # Our own event id is marked seen for echo suppression
        assert "evt123" in adapter._channel_state[CHANNEL]["seen"]

    @pytest.mark.asyncio
    async def test_send_reply_to(self):
        adapter = _make_adapter()
        cli = _ScriptedCli()
        cli.script("messages", "send", {"accepted": True, "event_id": "evt124", "message": ""})
        adapter._run_cli = cli
        await adapter.send(CHANNEL, "threaded", reply_to="root-event")
        args, _stdin = cli.calls[0]
        assert args[args.index("--reply-to") + 1] == "root-event"

    @pytest.mark.asyncio
    async def test_send_failure_maps_error_contract(self):
        adapter = _make_adapter()
        cli = _ScriptedCli()
        cli.script("messages", "send", "", code=2,
                   stderr='{"error":"relay_error","message":"relay unreachable"}')
        adapter._run_cli = cli
        result = await adapter.send(CHANNEL, "hi")
        assert result.success is False
        assert "relay unreachable" in result.error
        assert result.retryable is True

    @pytest.mark.asyncio
    async def test_send_empty_message_rejected(self):
        adapter = _make_adapter()
        result = await adapter.send(CHANNEL, "")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_send_image_url_falls_back_to_link(self):
        adapter = _make_adapter()
        cli = _ScriptedCli()
        cli.script("messages", "send", {"accepted": True, "event_id": "evt125", "message": ""})
        adapter._run_cli = cli
        result = await adapter.send_image(CHANNEL, "https://example.com/cat.png", caption="a cat")
        assert result.success is True
        _args, stdin_text = cli.calls[0]
        assert "https://example.com/cat.png" in stdin_text
        assert "a cat" in stdin_text

    @pytest.mark.asyncio
    async def test_send_image_local_file_uses_file_flag(self, tmp_path):
        img = tmp_path / "shot.png"
        img.write_bytes(b"\x89PNG fake")
        adapter = _make_adapter()
        cli = _ScriptedCli()
        cli.script("messages", "send", {"accepted": True, "event_id": "evt126", "message": ""})
        adapter._run_cli = cli
        result = await adapter.send_image(CHANNEL, str(img), caption="screenshot")
        assert result.success is True
        args, _stdin = cli.calls[0]
        assert args[args.index("--file") + 1] == str(img)


# ── Lifecycle ─────────────────────────────────────────────────────────────


class TestBuzzAdapterLifecycle:

    @pytest.mark.asyncio
    async def test_connect_fails_without_relay_url(self):
        from gateway.config import PlatformConfig
        adapter = BuzzAdapter(PlatformConfig(enabled=True, extra={}))
        assert await adapter.connect() is False

    @pytest.mark.asyncio
    async def test_disconnect_cancels_poll_task(self):
        adapter = _make_adapter()

        async def forever():
            while True:
                await asyncio.sleep(3600)

        adapter._poll_task = asyncio.create_task(forever())
        task = adapter._poll_task
        await adapter.disconnect()
        assert task.cancelled()
        assert adapter._poll_task is None
        assert adapter.is_connected is False

    @pytest.mark.asyncio
    async def test_get_chat_info_uses_cached_names(self):
        adapter = _make_adapter()
        adapter._channel_names[CHANNEL] = "general"
        adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        info = await adapter.get_chat_info(CHANNEL)
        assert info == {"name": "general", "type": "group", "chat_id": CHANNEL}

    @pytest.mark.asyncio
    async def test_dm_discovery_registers_conversation(self):
        adapter = _make_adapter()
        cli = _ScriptedCli()
        cli.script("dms", "list", [{"dm_id": "dm-1", "participants": [OTHER_PUBKEY], "created_at": 5}])
        adapter._run_cli = cli
        await adapter._discover_dms(seed=False)
        assert adapter._channel_state["dm-1"]["chat_type"] == "dm"


# ── Credentials / requirements ────────────────────────────────────────────


class TestCredentialResolution:

    def test_env_key_wins(self, monkeypatch):
        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1fromenv")
        assert _resolve_private_key() == "nsec1fromenv"

    def test_credentials_file_fallback(self, monkeypatch, tmp_path):
        creds = tmp_path / "agent_credentials.json"
        creds.write_text(json.dumps({"nsec": "nsec1fromfile", "npub": "npub1x"}), encoding="utf-8")
        monkeypatch.setenv("BUZZ_CREDENTIALS_FILE", str(creds))
        assert _resolve_private_key() == "nsec1fromfile"

    def test_default_dir_glob(self, monkeypatch, tmp_path):
        (tmp_path / "chip_nostr_credentials.json").write_text(
            json.dumps({"private_key_hex": "ab" * 32}), encoding="utf-8"
        )
        monkeypatch.setattr(_buzz_mod, "_DEFAULT_CREDENTIALS_DIR", tmp_path)
        assert _resolve_private_key() == "ab" * 32

    def test_missing_everywhere(self):
        assert _resolve_private_key() == ""

    def test_check_requirements(self, monkeypatch):
        assert check_requirements() is False
        monkeypatch.setenv("BUZZ_RELAY_URL", "https://r")
        assert check_requirements() is False  # still no key
        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1x")
        assert check_requirements() is True

    def test_validate_config_from_extra(self, monkeypatch):
        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1x")
        from gateway.config import PlatformConfig
        assert validate_config(PlatformConfig(extra={"relay_url": "https://r"})) is True
        assert validate_config(PlatformConfig(extra={})) is False


# ── Env enablement / registration / standalone send ──────────────────────


class TestEnvEnablement:

    def test_returns_none_when_unconfigured(self):
        assert _env_enablement() is None

    def test_seeds_extra_and_home_channel(self, monkeypatch):
        monkeypatch.setenv("BUZZ_RELAY_URL", "https://r")
        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1x")
        monkeypatch.setenv("BUZZ_CHANNELS", "chan-a,chan-b")
        seed = _env_enablement()
        assert seed["relay_url"] == "https://r"
        assert seed["channels"] == ["chan-a", "chan-b"]
        # Home channel defaults to the first watched channel
        assert seed["home_channel"]["chat_id"] == "chan-a"

    def test_explicit_home_channel(self, monkeypatch):
        monkeypatch.setenv("BUZZ_RELAY_URL", "https://r")
        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1x")
        monkeypatch.setenv("BUZZ_HOME_CHANNEL", "home-chan")
        assert _env_enablement()["home_channel"]["chat_id"] == "home-chan"


class TestBuzzPluginRegistration:

    def test_register_platform_contract(self):
        from gateway.platform_registry import platform_registry

        platform_registry.unregister("buzz")
        ctx = MagicMock()
        register(ctx)
        ctx.register_platform.assert_called_once()
        kwargs = ctx.register_platform.call_args.kwargs
        assert kwargs["name"] == "buzz"
        assert kwargs["cron_deliver_env_var"] == "BUZZ_HOME_CHANNEL"
        assert kwargs["allowed_users_env"] == "BUZZ_ALLOWED_USERS"
        assert kwargs["allow_all_env"] == "BUZZ_ALLOW_ALL_USERS"
        assert callable(kwargs["standalone_sender_fn"])
        assert callable(kwargs["env_enablement_fn"])
        assert set(kwargs["required_env"]) == {"BUZZ_RELAY_URL", "BUZZ_PRIVATE_KEY"}


class TestStandaloneSend:

    @pytest.mark.asyncio
    async def test_standalone_send_success(self, monkeypatch, tmp_path):
        from gateway.config import PlatformConfig

        fake_cli = tmp_path / "buzz"
        fake_cli.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.setenv("BUZZ_RELAY_URL", "https://r")
        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1x")
        monkeypatch.setenv("BUZZ_CLI_PATH", str(fake_cli))

        captured = {}

        async def fake_exec(cli_path, args, *, relay_url, private_key, input_text=None, timeout=30.0):
            captured.update(cli_path=cli_path, args=args, relay_url=relay_url, input_text=input_text)
            return 0, json.dumps({"accepted": True, "event_id": "evt-cron", "message": ""}), ""

        monkeypatch.setattr(_buzz_mod, "_exec_buzz", fake_exec)

        result = await _standalone_send(PlatformConfig(enabled=True, extra={}), CHANNEL, "cron says hi")
        assert result == {"success": True, "message_id": "evt-cron"}
        assert captured["args"][:2] == ["messages", "send"]
        assert captured["input_text"] == "cron says hi"
        # The private key must never be part of argv
        assert all("nsec1x" not in str(a) for a in captured["args"])

    @pytest.mark.asyncio
    async def test_standalone_send_unconfigured(self):
        from gateway.config import PlatformConfig
        result = await _standalone_send(PlatformConfig(enabled=True, extra={}), CHANNEL, "hi")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_standalone_send_falls_back_to_home_channel(self, monkeypatch, tmp_path):
        from gateway.config import PlatformConfig

        fake_cli = tmp_path / "buzz"
        fake_cli.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.setenv("BUZZ_RELAY_URL", "https://r")
        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1x")
        monkeypatch.setenv("BUZZ_CLI_PATH", str(fake_cli))
        monkeypatch.setenv("BUZZ_HOME_CHANNEL", "home-chan")

        captured = {}

        async def fake_exec(cli_path, args, *, relay_url, private_key, input_text=None, timeout=30.0):
            captured["args"] = args
            return 0, json.dumps({"accepted": True, "event_id": "e"}), ""

        monkeypatch.setattr(_buzz_mod, "_exec_buzz", fake_exec)
        result = await _standalone_send(PlatformConfig(enabled=True, extra={}), "", "hi")
        assert result["success"] is True
        assert captured["args"][captured["args"].index("--channel") + 1] == "home-chan"
