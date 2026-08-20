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


# ── Adapter init / config precedence ──────────────────────────────────────


class TestBuzzAdapterInit:


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

    def test_outbound_mention_pubkeys_reject_casefold_collisions(self):
        from gateway.config import PlatformConfig

        cfg = PlatformConfig(
            enabled=True,
            extra={
                "relay_url": "https://cfg.relay",
                "outbound_mention_pubkeys": {
                    "Agent": OTHER_PUBKEY,
                    "agent": SELF_PUBKEY,
                },
            },
        )

        with pytest.raises(ValueError, match="duplicate display name"):
            BuzzAdapter(cfg)

    def test_validate_config_rejects_invalid_outbound_mention_mapping(
        self, monkeypatch
    ):
        from gateway.config import PlatformConfig

        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1test")
        cfg = PlatformConfig(
            enabled=True,
            extra={
                "relay_url": "https://cfg.relay",
                "outbound_mention_pubkeys": "not-a-mapping",
            },
        )

        assert validate_config(cfg) is False

    def test_env_overrides_config(self, monkeypatch):
        monkeypatch.setenv("BUZZ_RELAY_URL", "https://env.relay")
        from gateway.config import PlatformConfig
        adapter = BuzzAdapter(PlatformConfig(enabled=True, extra={"relay_url": "https://cfg.relay"}))
        assert adapter.relay_url == "https://env.relay"


# ── CLI error contract ────────────────────────────────────────────────────


class TestCliErrorContract:

    def test_parses_json_error(self):
        msg = _cli_error_message('{"error":"relay_error","message":"boom","retryable":false}', 2)
        assert "relay_error" in msg and "boom" in msg and "exit 2" in msg


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
    async def test_allowlist_blocks_unauthorized(self, adapter):
        adapter._allowed_pubkeys = {"b" * 64}
        await self._poll_with(adapter, _event("e1", content="@Chip hello", created_at=10))
        assert adapter._dispatched == []


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
        assert len(cli.calls) == 1

        args, stdin_text = cli.calls[0]
        assert args[:2] == ["messages", "send"]
        assert args[args.index("--channel") + 1] == CHANNEL
        # Content travels via stdin (--content -), never argv
        assert args[args.index("--content") + 1] == "-"
        assert stdin_text == "hello **markdown**"
        # Our own event id is marked seen for echo suppression
        assert "evt123" in adapter._channel_state[CHANNEL]["seen"]


    @pytest.mark.asyncio
    async def test_send_configured_agent_handoff_uses_exact_pubkey(self):
        adapter = _make_adapter(
            {"outbound_mention_pubkeys": {"CodexTerraM": OTHER_PUBKEY}}
        )
        cli = _ScriptedCli()
        cli.script(
            "messages",
            "send",
            {"accepted": True, "event_id": "evt-agent-handoff", "message": ""},
        )
        adapter._run_cli = cli

        result = await adapter.send(
            CHANNEL,
            "@CodexTerraM — review the clean rebuild.",
            metadata={"thread_id": "thread-root"},
        )

        assert result.success is True
        assert len(cli.calls) == 1
        args, stdin_text = cli.calls[0]
        assert args[args.index("--mention") + 1] == OTHER_PUBKEY
        assert stdin_text == "@CodexTerraM — review the clean rebuild."

    @pytest.mark.asyncio
    async def test_send_uses_casefold_consistently_for_unicode_names(self):
        adapter = _make_adapter(
            {
                "outbound_mention_pubkeys": {
                    "i": OTHER_PUBKEY,
                    "ı": SELF_PUBKEY,
                }
            }
        )
        cli = _ScriptedCli()
        cli.script(
            "messages",
            "send",
            {"accepted": True, "event_id": "evt-unicode-name", "message": ""},
        )
        adapter._run_cli = cli

        result = await adapter.send(CHANNEL, "@ı — review this.")

        assert result.success is True
        args, stdin_text = cli.calls[0]
        mention_values = [
            args[index + 1]
            for index, value in enumerate(args)
            if value == "--mention"
        ]
        assert mention_values == [SELF_PUBKEY]
        assert stdin_text == "@ı — review this."

    @pytest.mark.asyncio
    async def test_send_never_downgrades_configured_agent_handoff(self):
        adapter = _make_adapter(
            {"outbound_mention_pubkeys": {"CodexTerraM": OTHER_PUBKEY}}
        )
        cli = _ScriptedCli()
        cli.script(
            "messages",
            "send",
            "",
            code=1,
            stderr=(
                "mention '@CodexTerraM' does not match a current channel member"
            ),
        )
        adapter._run_cli = cli

        result = await adapter.send(
            CHANNEL,
            "@CodexTerraM — review the clean rebuild.",
        )

        assert result.success is False
        assert len(cli.calls) == 1
        args, stdin_text = cli.calls[0]
        assert args[args.index("--mention") + 1] == OTHER_PUBKEY
        assert stdin_text == "@CodexTerraM — review the clean rebuild."

    @pytest.mark.asyncio
    async def test_send_configured_handoff_only_protects_its_own_marker(self):
        adapter = _make_adapter(
            {"outbound_mention_pubkeys": {"CodexTerraM": OTHER_PUBKEY}}
        )
        cli = _ScriptedCli()
        cli.script(
            "messages",
            "send",
            "",
            code=1,
            stderr="mention '@ghost' does not match a current channel member",
        )
        cli.script(
            "messages",
            "send",
            {"accepted": True, "event_id": "evt-mixed", "message": ""},
        )
        adapter._run_cli = cli

        result = await adapter.send(
            CHANNEL,
            "@CodexTerraM — ask @ghost to review.",
        )

        assert result.success is True
        assert len(cli.calls) == 2
        first_args, first_text = cli.calls[0]
        second_args, second_text = cli.calls[1]
        assert first_args == second_args
        assert first_args[first_args.index("--mention") + 1] == OTHER_PUBKEY
        assert first_text == "@CodexTerraM — ask @ghost to review."
        assert second_text == "@CodexTerraM — ask ghost to review."

    @pytest.mark.asyncio
    async def test_send_unconfigured_prefix_does_not_strip_configured_handoff(self):
        adapter = _make_adapter(
            {"outbound_mention_pubkeys": {"Ghostly": OTHER_PUBKEY}}
        )
        cli = _ScriptedCli()
        cli.script(
            "messages",
            "send",
            "",
            code=1,
            stderr="mention '@ghost' does not match a current channel member",
        )
        cli.script(
            "messages",
            "send",
            {"accepted": True, "event_id": "evt-prefix", "message": ""},
        )
        adapter._run_cli = cli

        result = await adapter.send(
            CHANNEL,
            "@Ghostly — ask @ghost to review.",
        )

        assert result.success is True
        assert [text for _args, text in cli.calls] == [
            "@Ghostly — ask @ghost to review.",
            "@Ghostly — ask ghost to review.",
        ]

    @pytest.mark.asyncio
    async def test_send_retries_unresolved_mentions_as_readable_text(self):
        adapter = _make_adapter()
        cli = _ScriptedCli()
        cli.script(
            "messages",
            "send",
            "",
            code=1,
            stderr=(
                "user_error: mention '@teknium1' does not match a current "
                "channel member; retry with --mention <pubkey>"
            ),
        )
        cli.script(
            "messages",
            "send",
            {"accepted": True, "event_id": "evt-fallback", "message": ""},
        )
        adapter._run_cli = cli

        result = await adapter.send(
            CHANNEL,
            "Ask @teknium1 and notify @Codex, but keep bob@example.com intact.",
            metadata={"thread_id": "thread-root"},
        )

        assert result.success is True
        assert result.message_id == "evt-fallback"
        assert len(cli.calls) == 2
        first_args, first_text = cli.calls[0]
        fallback_args, fallback_text = cli.calls[1]
        assert first_text == "Ask @teknium1 and notify @Codex, but keep bob@example.com intact."
        assert fallback_text == "Ask teknium1 and notify @Codex, but keep bob@example.com intact."
        assert fallback_args == first_args
        assert fallback_args[fallback_args.index("--reply-to") + 1] == "thread-root"

    @pytest.mark.asyncio
    async def test_send_retries_unicode_unresolved_mention(self):
        adapter = _make_adapter()
        cli = _ScriptedCli()
        cli.script(
            "messages",
            "send",
            "",
            code=1,
            stderr="mention '@Иван' does not match a current channel member",
        )
        cli.script(
            "messages",
            "send",
            {"accepted": True, "event_id": "evt-unicode", "message": ""},
        )
        adapter._run_cli = cli

        result = await adapter.send(CHANNEL, "Ask @Иван Петров to review")

        assert result.success is True
        assert [text for _args, text in cli.calls] == [
            "Ask @Иван Петров to review",
            "Ask Иван Петров to review",
        ]

    @pytest.mark.asyncio
    async def test_send_retries_apostrophe_name(self):
        adapter = _make_adapter()
        cli = _ScriptedCli()
        cli.script(
            "messages",
            "send",
            "",
            code=1,
            stderr="mention '@O'Brien' does not match a current channel member",
        )
        cli.script(
            "messages",
            "send",
            {"accepted": True, "event_id": "evt-apostrophe", "message": ""},
        )
        adapter._run_cli = cli

        result = await adapter.send(CHANNEL, "Ask @O'Brien to review")

        assert result.success is True
        assert [text for _args, text in cli.calls] == [
            "Ask @O'Brien to review",
            "Ask O'Brien to review",
        ]

    @pytest.mark.asyncio
    async def test_send_retries_each_new_unresolved_name_within_cap(self):
        adapter = _make_adapter()
        cli = _ScriptedCli()
        cli.script(
            "messages",
            "send",
            "",
            code=1,
            stderr="mention '@Ghost One' does not match a current channel member",
        )
        cli.script(
            "messages",
            "send",
            "",
            code=1,
            stderr="mention '@Ghost Two' does not match a current channel member",
        )
        cli.script(
            "messages",
            "send",
            {"accepted": True, "event_id": "evt-multiple", "message": ""},
        )
        adapter._run_cli = cli

        result = await adapter.send(CHANNEL, "ping @Ghost One and @Ghost Two")

        assert result.success is True
        assert [text for _args, text in cli.calls] == [
            "ping @Ghost One and @Ghost Two",
            "ping Ghost One and @Ghost Two",
            "ping Ghost One and Ghost Two",
        ]

    @pytest.mark.asyncio
    async def test_send_caps_sequential_mention_fallback_retries(self):
        adapter = _make_adapter()
        cli = _ScriptedCli()
        for name in ("One", "Two", "Three", "Four"):
            cli.script(
                "messages",
                "send",
                "",
                code=1,
                stderr=(
                    f"mention '@Ghost {name}' does not match a current channel member"
                ),
            )
        adapter._run_cli = cli

        result = await adapter.send(
            CHANNEL,
            "ping @Ghost One, @Ghost Two, @Ghost Three, and @Ghost Four",
        )

        assert result.success is False
        assert result.error == (
            "mention '@Ghost Four' does not match a current channel member"
        )
        assert len(cli.calls) == 4

    @pytest.mark.asyncio
    async def test_send_retries_ambiguous_mention_without_silencing_other_mentions(self):
        adapter = _make_adapter()
        cli = _ScriptedCli()
        cli.script(
            "messages",
            "send",
            "",
            code=1,
            stderr=(
                "mention '@Twin' is ambiguous; candidates: npub1a, npub1b. "
                "Retry with --mention <pubkey>"
            ),
        )
        cli.script(
            "messages",
            "send",
            {"accepted": True, "event_id": "evt-ambiguous", "message": ""},
        )
        adapter._run_cli = cli

        result = await adapter.send(CHANNEL, "Ask @Twin and notify @Codex")

        assert result.success is True
        assert cli.calls[1][1] == "Ask Twin and notify @Codex"

    @pytest.mark.asyncio
    async def test_send_retries_max_mentions_with_all_markers_neutralized(self):
        adapter = _make_adapter()
        cli = _ScriptedCli()
        cli.script(
            "messages",
            "send",
            "",
            code=1,
            stderr="too many unique message mentions (max 2)",
        )
        cli.script(
            "messages",
            "send",
            {"accepted": True, "event_id": "evt-max", "message": ""},
        )
        adapter._run_cli = cli

        result = await adapter.send(
            CHANNEL,
            "Ask @Иван, @李明, and @Codex; email bob@example.com",
        )

        assert result.success is True
        assert cli.calls[1][1] == (
            "Ask Иван, 李明, and Codex; email bob@example.com"
        )

    @pytest.mark.asyncio
    async def test_send_does_not_retry_unrelated_failure(self):
        adapter = _make_adapter()
        cli = _ScriptedCli()
        cli.script(
            "messages",
            "send",
            "",
            code=1,
            stderr="forbidden: sender is not authorized",
        )
        adapter._run_cli = cli

        result = await adapter.send(CHANNEL, "Ask @Codex")

        assert result.success is False
        assert len(cli.calls) == 1

    @pytest.mark.asyncio
    async def test_send_propagates_second_attempt_failure_without_third_attempt(self):
        adapter = _make_adapter()
        cli = _ScriptedCli()
        cli.script(
            "messages",
            "send",
            "",
            code=1,
            stderr="mention '@ghost' does not match a current channel member",
        )
        cli.script(
            "messages",
            "send",
            "",
            code=1,
            stderr="relay unavailable after fallback",
        )
        adapter._run_cli = cli

        result = await adapter.send(CHANNEL, "Ask @ghost")

        assert result.success is False
        assert result.error == "relay unavailable after fallback"
        assert len(cli.calls) == 2

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

    @pytest.mark.asyncio
    async def test_send_image_configured_handoff_uses_exact_pubkey(self, tmp_path):
        img = tmp_path / "shot.png"
        img.write_bytes(b"\x89PNG fake")
        adapter = _make_adapter(
            {"outbound_mention_pubkeys": {"CodexTerraM": OTHER_PUBKEY}}
        )
        cli = _ScriptedCli()
        cli.script(
            "messages",
            "send",
            {"accepted": True, "event_id": "evt-image-handoff", "message": ""},
        )
        adapter._run_cli = cli

        result = await adapter.send_image(
            CHANNEL,
            str(img),
            caption="@CodexTerraM — review this screenshot.",
        )

        assert result.success is True
        args, stdin_text = cli.calls[0]
        mention_values = [
            args[index + 1]
            for index, value in enumerate(args)
            if value == "--mention"
        ]
        assert mention_values == [OTHER_PUBKEY]
        assert stdin_text == "@CodexTerraM — review this screenshot."

    @pytest.mark.asyncio
    async def test_send_image_never_downgrades_configured_handoff(self, tmp_path):
        img = tmp_path / "shot.png"
        img.write_bytes(b"\x89PNG fake")
        adapter = _make_adapter(
            {"outbound_mention_pubkeys": {"CodexTerraM": OTHER_PUBKEY}}
        )
        cli = _ScriptedCli()
        cli.script(
            "messages",
            "send",
            "",
            code=1,
            stderr=(
                "mention '@CodexTerraM' does not match a current channel member"
            ),
        )
        adapter._run_cli = cli

        result = await adapter.send_image(
            CHANNEL,
            str(img),
            caption="@CodexTerraM — review this screenshot.",
        )

        assert result.success is False
        assert len(cli.calls) == 1
        args, stdin_text = cli.calls[0]
        assert args[args.index("--mention") + 1] == OTHER_PUBKEY
        assert stdin_text == "@CodexTerraM — review this screenshot."

    @pytest.mark.asyncio
    async def test_send_image_configured_handoff_only_protects_its_marker(
        self, tmp_path
    ):
        img = tmp_path / "shot.png"
        img.write_bytes(b"\x89PNG fake")
        adapter = _make_adapter(
            {"outbound_mention_pubkeys": {"CodexTerraM": OTHER_PUBKEY}}
        )
        cli = _ScriptedCli()
        cli.script(
            "messages",
            "send",
            "",
            code=1,
            stderr="mention '@ghost' does not match a current channel member",
        )
        cli.script(
            "messages",
            "send",
            {"accepted": True, "event_id": "evt-image-mixed", "message": ""},
        )
        adapter._run_cli = cli

        result = await adapter.send_image(
            CHANNEL,
            str(img),
            caption="@CodexTerraM — ask @ghost to review this screenshot.",
        )

        assert result.success is True
        assert len(cli.calls) == 2
        first_args, first_text = cli.calls[0]
        second_args, second_text = cli.calls[1]
        mention_values = [
            first_args[index + 1]
            for index, value in enumerate(first_args)
            if value == "--mention"
        ]
        assert mention_values == [OTHER_PUBKEY]
        assert second_args == first_args
        assert first_text == (
            "@CodexTerraM — ask @ghost to review this screenshot."
        )
        assert second_text == "@CodexTerraM — ask ghost to review this screenshot."

    @pytest.mark.asyncio
    async def test_send_image_retries_unresolved_caption_mention(self, tmp_path):
        img = tmp_path / "shot.png"
        img.write_bytes(b"\x89PNG fake")
        adapter = _make_adapter()
        cli = _ScriptedCli()
        cli.script(
            "messages",
            "send",
            "",
            code=1,
            stderr="mention '@ghost' does not match a current channel member",
        )
        cli.script(
            "messages",
            "send",
            {"accepted": True, "event_id": "evt-image-fallback", "message": ""},
        )
        adapter._run_cli = cli

        result = await adapter.send_image(
            CHANNEL,
            str(img),
            caption="For @ghost",
            reply_to="thread-root",
        )

        assert result.success is True
        assert [text for _args, text in cli.calls] == ["For @ghost", "For ghost"]
        assert cli.calls[1][0] == cli.calls[0][0]


# ── Lifecycle ─────────────────────────────────────────────────────────────


class TestBuzzAdapterLifecycle:


    @pytest.mark.asyncio
    async def test_disconnect_releases_scoped_lock(self, monkeypatch):
        """The identity lock taken in connect() must be released on disconnect."""
        import gateway.status as gateway_status

        released = []
        monkeypatch.setattr(
            gateway_status,
            "release_scoped_lock",
            lambda platform, key: released.append((platform, key)),
        )
        adapter = _make_adapter()
        adapter._lock_key = "wss://relay.example:" + SELF_PUBKEY
        await adapter.disconnect()
        assert released == [("buzz", "wss://relay.example:" + SELF_PUBKEY)]
        assert adapter._lock_key is None

    @pytest.mark.asyncio
    async def test_connect_fails_when_identity_lock_held(self, monkeypatch):
        """A second profile using the same relay+pubkey must fail fast."""
        import gateway.status as gateway_status

        monkeypatch.setattr(
            gateway_status, "acquire_scoped_lock", lambda platform, key: False
        )
        adapter = _make_adapter()
        adapter.cli_path = "/fake/buzz"
        monkeypatch.setattr(_buzz_mod, "_resolve_private_key", lambda extra=None: "nsec1test")
        cli = _ScriptedCli()
        cli.script(
            "users", "get",
            [{"pubkey": SELF_PUBKEY, "display_name": "Chip"}],
        )
        adapter._run_cli = cli
        assert await adapter.connect() is False
        assert adapter._lock_key is None


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


# ── Env enablement / registration / standalone send ──────────────────────


class TestEnvEnablement:

    def test_returns_none_when_unconfigured(self):
        assert _env_enablement() is None


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
    async def test_standalone_invalid_outbound_mapping_returns_error(
        self, monkeypatch, tmp_path
    ):
        from gateway.config import PlatformConfig

        fake_cli = tmp_path / "buzz"
        fake_cli.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.setenv("BUZZ_RELAY_URL", "https://r")
        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1x")
        monkeypatch.setenv("BUZZ_CLI_PATH", str(fake_cli))

        result = await _standalone_send(
            PlatformConfig(
                enabled=True,
                extra={"outbound_mention_pubkeys": "not-a-mapping"},
            ),
            CHANNEL,
            "hello",
        )

        assert result == {
            "error": "Buzz standalone send: invalid outbound_mention_pubkeys"
        }

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
    async def test_standalone_send_configured_handoff_uses_exact_pubkey(
        self, monkeypatch, tmp_path
    ):
        from gateway.config import PlatformConfig

        fake_cli = tmp_path / "buzz"
        fake_cli.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.setenv("BUZZ_RELAY_URL", "https://r")
        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1x")
        monkeypatch.setenv("BUZZ_CLI_PATH", str(fake_cli))
        calls = []

        async def fake_exec(
            cli_path, args, *, relay_url, private_key, input_text=None, timeout=30.0
        ):
            calls.append((list(args), input_text))
            return 0, json.dumps(
                {"accepted": True, "event_id": "evt-cron", "message": ""}
            ), ""

        monkeypatch.setattr(_buzz_mod, "_exec_buzz", fake_exec)

        result = await _standalone_send(
            PlatformConfig(
                enabled=True,
                extra={
                    "outbound_mention_pubkeys": {"CodexTerraM": OTHER_PUBKEY}
                },
            ),
            CHANNEL,
            "@CodexTerraM — review the scheduled result.",
        )

        assert result == {"success": True, "message_id": "evt-cron"}
        args, stdin_text = calls[0]
        assert args[args.index("--mention") + 1] == OTHER_PUBKEY
        assert stdin_text == "@CodexTerraM — review the scheduled result."

    @pytest.mark.asyncio
    async def test_standalone_send_retries_unresolved_mention(self, monkeypatch, tmp_path):
        from gateway.config import PlatformConfig

        fake_cli = tmp_path / "buzz"
        fake_cli.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.setenv("BUZZ_RELAY_URL", "https://r")
        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1x")
        monkeypatch.setenv("BUZZ_CLI_PATH", str(fake_cli))
        calls = []

        async def fake_exec(cli_path, args, *, relay_url, private_key, input_text=None, timeout=30.0):
            calls.append((list(args), input_text))
            if len(calls) == 1:
                return 1, "", "mention '@ghost' does not match a current channel member"
            return 0, json.dumps({"accepted": True, "event_id": "evt-cron", "message": ""}), ""

        monkeypatch.setattr(_buzz_mod, "_exec_buzz", fake_exec)

        result = await _standalone_send(
            PlatformConfig(enabled=True, extra={}),
            CHANNEL,
            "Cron asks @ghost",
            thread_id="thread-root",
        )

        assert result == {"success": True, "message_id": "evt-cron"}
        assert [text for _args, text in calls] == ["Cron asks @ghost", "Cron asks ghost"]
        assert calls[1][0] == calls[0][0]

    @pytest.mark.asyncio
    async def test_standalone_configured_handoff_only_protects_its_own_marker(
        self, monkeypatch, tmp_path
    ):
        from gateway.config import PlatformConfig

        fake_cli = tmp_path / "buzz"
        fake_cli.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.setenv("BUZZ_RELAY_URL", "https://r")
        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1x")
        monkeypatch.setenv("BUZZ_CLI_PATH", str(fake_cli))
        calls = []

        async def fake_exec(
            cli_path, args, *, relay_url, private_key, input_text=None, timeout=30.0
        ):
            calls.append((list(args), input_text))
            if len(calls) == 1:
                return 1, "", "mention '@ghost' does not match a current channel member"
            return 0, json.dumps(
                {"accepted": True, "event_id": "evt-cron", "message": ""}
            ), ""

        monkeypatch.setattr(_buzz_mod, "_exec_buzz", fake_exec)

        result = await _standalone_send(
            PlatformConfig(
                enabled=True,
                extra={
                    "outbound_mention_pubkeys": {"CodexTerraM": OTHER_PUBKEY}
                },
            ),
            CHANNEL,
            "@CodexTerraM asks @ghost to review",
        )

        assert result == {"success": True, "message_id": "evt-cron"}
        assert [text for _args, text in calls] == [
            "@CodexTerraM asks @ghost to review",
            "@CodexTerraM asks ghost to review",
        ]
        assert calls[1][0] == calls[0][0]
        assert calls[0][0][calls[0][0].index("--mention") + 1] == OTHER_PUBKEY
