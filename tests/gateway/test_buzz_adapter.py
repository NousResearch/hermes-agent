"""Tests for the Buzz platform adapter plugin."""

import asyncio
import json
import os

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
    "BUZZ_REQUIRE_MENTION",
    "BUZZ_THREAD_REQUIRE_MENTION",
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

        args, stdin_text = cli.calls[0]
        assert args[:2] == ["messages", "send"]
        assert args[args.index("--channel") + 1] == CHANNEL
        # Content travels via stdin (--content -), never argv
        assert args[args.index("--content") + 1] == "-"
        assert stdin_text == "hello **markdown**"
        # Our own event id is marked seen for echo suppression
        assert "evt123" in adapter._channel_state[CHANNEL]["seen"]


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




# ── Thread-scoped mention gating (thread_require_mention) ────────────────
#
# A thread the agent takes part in is rooted at a message it wrote or
# replied to. With thread_require_mention=False those threads stop needing
# an address on every turn, while the channel top level stays gated.


def _thread_event(event_id, *, content, root=None, parent=None,
                  pubkey=OTHER_PUBKEY, created_at=1000):
    """Chat event with NIP-10 markers exactly as the relay delivers them."""
    tags = [["h", CHANNEL]]
    if root:
        tags.append(["e", root, "", "root"])
    if parent:
        tags.append(["e", parent, "", "reply"])
    return {
        "id": event_id,
        "pubkey": pubkey,
        "content": content,
        "created_at": created_at,
        "kind": 9,
        "tags": tags,
    }


class TestEventThreadRoot:
    """The tag reading that everything below rests on."""

    def test_root_marker_wins_over_reply_marker(self):
        event = _thread_event("e1", content="x", root="rootid", parent="parentid")
        assert _buzz_mod._event_thread_root(event) == "rootid"

    def test_lone_reply_marker_is_the_root(self):
        """A reply to the thread's first message names only that message."""
        event = _thread_event("e1", content="x", parent="rootid")
        assert _buzz_mod._event_thread_root(event) == "rootid"

    def test_deprecated_positional_tags_list_root_first(self):
        event = {"id": "e1", "pubkey": OTHER_PUBKEY, "content": "x", "kind": 9,
                 "created_at": 1, "tags": [["h", CHANNEL], ["e", "rootid"], ["e", "parentid"]]}
        assert _buzz_mod._event_thread_root(event) == "rootid"

    def test_top_level_message_has_no_thread(self):
        assert _buzz_mod._event_thread_root(_thread_event("e1", content="x")) == ""

    def test_malformed_tags_are_ignored(self):
        for tags in (None, "nope", [["e"], ["e", ""], "junk", 42]):
            assert _buzz_mod._event_thread_root({"tags": tags}) == ""


class TestThreadRequireMentionConfig:

    def test_defaults_to_true(self):
        assert _make_adapter().thread_require_mention is True

    def test_config_extra_disables_it(self):
        assert _make_adapter({"thread_require_mention": False}).thread_require_mention is False

    def test_env_overrides_config(self, monkeypatch):
        monkeypatch.setenv("BUZZ_THREAD_REQUIRE_MENTION", "false")
        assert _make_adapter({"thread_require_mention": True}).thread_require_mention is False

    def test_yaml_config_bridges_to_env(self, monkeypatch):
        # Empty, not unset: the bridge treats it as unowned (falsy) and still
        # writes, while monkeypatch owns the var and unsets it afterwards —
        # _apply_yaml_config writes straight to os.environ.
        monkeypatch.setenv("BUZZ_THREAD_REQUIRE_MENTION", "")
        _buzz_mod._apply_yaml_config({}, {"extra": {"thread_require_mention": False}})
        assert os.environ["BUZZ_THREAD_REQUIRE_MENTION"] == "false"

    def test_yaml_config_does_not_override_explicit_env(self, monkeypatch):
        monkeypatch.setenv("BUZZ_THREAD_REQUIRE_MENTION", "false")
        _buzz_mod._apply_yaml_config({}, {"extra": {"thread_require_mention": True}})
        assert os.environ["BUZZ_THREAD_REQUIRE_MENTION"] == "false"


class TestThreadMentionGating:

    def _adapter(self, extra=None):
        a = _make_adapter(extra)
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

    async def _agent_sends(self, adapter, event_id, *, reply_to=None):
        cli = _ScriptedCli()
        cli.script("messages", "send", {"accepted": True, "event_id": event_id, "message": ""})
        adapter._run_cli = cli
        result = await adapter.send(CHANNEL, "on it", reply_to=reply_to)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_thread_reply_stays_gated_by_default(self):
        """Default true: nothing changes for anyone who does not opt in."""
        adapter = self._adapter()
        await self._agent_sends(adapter, "agent1")
        await self._poll_with(
            adapter, _thread_event("r1", content="and one more thing", root="agent1"),
        )
        assert adapter._dispatched == []

    @pytest.mark.asyncio
    async def test_reply_to_the_agents_own_message_dispatches(self):
        """The core case: someone replies to what the agent posted."""
        adapter = self._adapter({"thread_require_mention": False})
        await self._agent_sends(adapter, "agent1")
        await self._poll_with(
            adapter, _thread_event("r1", content="and one more thing", root="agent1"),
        )
        assert [d["message_id"] for d in adapter._dispatched] == ["r1"]

    @pytest.mark.asyncio
    async def test_agents_reply_joins_a_humans_thread(self):
        """The agent answering inside someone else's thread joins it, so the
        follow-up needs no further address."""
        adapter = self._adapter({"thread_require_mention": False})
        await self._poll_with(
            adapter, _thread_event("h1", content="@Chip take a look", created_at=10),
        )
        await self._agent_sends(adapter, "agent1", reply_to="h1")
        await self._poll_with(
            adapter,
            _thread_event("h2", content="thanks, and what about the rest?",
                          root="h1", parent="agent1", created_at=20),
        )
        assert [d["message_id"] for d in adapter._dispatched] == ["h1", "h2"]

    @pytest.mark.asyncio
    async def test_channel_top_level_stays_gated(self):
        """Waiving mentions in threads must not open the channel itself."""
        adapter = self._adapter({"thread_require_mention": False})
        await self._agent_sends(adapter, "agent1")
        await self._poll_with(adapter, _thread_event("t1", content="unrelated chatter"))
        assert adapter._dispatched == []

    @pytest.mark.asyncio
    async def test_foreign_thread_stays_gated(self):
        """Two humans talking in their own thread are not the agent's cue."""
        adapter = self._adapter({"thread_require_mention": False})
        await self._poll_with(
            adapter, _thread_event("r1", content="what do you think?", root="their-root"),
        )
        assert adapter._dispatched == []

    @pytest.mark.asyncio
    async def test_allowlist_still_applies_inside_the_thread(self):
        """Waiving the mention changes HOW to phrase a message, never WHO
        may send one."""
        adapter = self._adapter({"thread_require_mention": False})
        adapter._allowed_pubkeys = {"b" * 64}
        await self._agent_sends(adapter, "agent1")
        await self._poll_with(
            adapter, _thread_event("r1", content="do the thing", root="agent1"),
        )
        assert adapter._dispatched == []

    @pytest.mark.asyncio
    async def test_dispatch_still_strips_a_leading_mention(self):
        adapter = self._adapter({"thread_require_mention": False})
        await self._agent_sends(adapter, "agent1")
        await self._poll_with(
            adapter, _thread_event("r1", content="@Chip /whoami", root="agent1"),
        )
        assert [d["text"] for d in adapter._dispatched] == ["/whoami"]

    @pytest.mark.asyncio
    async def test_free_listening_install_is_unaffected(self):
        """require_mention=false dispatches everything, threaded or not —
        this setting only ever relaxes the gate, never tightens it."""
        adapter = self._adapter({"require_mention": False})
        await self._poll_with(
            adapter,
            _thread_event("t1", content="starting a thread", created_at=10),
            _thread_event("r1", content="the follow-up", root="t1", created_at=11),
        )
        assert [d["message_id"] for d in adapter._dispatched] == ["t1", "r1"]

    @pytest.mark.asyncio
    async def test_participation_survives_a_restart(self):
        """A fresh process re-learns its threads from channel history, so a
        gateway restart does not silently re-gate an ongoing conversation."""
        adapter = self._adapter({"thread_require_mention": False})
        cli = _ScriptedCli()
        cli.script("messages", "get", [
            _thread_event("h1", content="@Chip take a look", created_at=10),
            _thread_event("agent1", content="on it", root="h1",
                          pubkey=SELF_PUBKEY, created_at=11),
        ])
        adapter._run_cli = cli
        await adapter._seed_channel(CHANNEL, chat_type="group")
        assert adapter._dispatched == []  # history never dispatches

        await self._poll_with(
            adapter,
            _thread_event("h2", content="one more question", root="h1", created_at=20),
        )
        assert [d["message_id"] for d in adapter._dispatched] == ["h2"]

    @pytest.mark.asyncio
    async def test_thread_bookkeeping_is_bounded(self):
        """Both maps are capped like the seen-id set, so a busy channel
        cannot grow them without bound."""
        adapter = self._adapter({"thread_require_mention": False})
        state = adapter._channel_state[CHANNEL]
        for i in range(_buzz_mod._SEEN_CAP + 50):
            adapter._track_thread(
                state, _thread_event(f"e{i}", content="x", pubkey=SELF_PUBKEY),
            )
        assert len(state["thread_roots"]) == _buzz_mod._SEEN_CAP
        assert len(state["agent_threads"]) == _buzz_mod._SEEN_CAP
