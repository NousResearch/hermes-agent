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


# ── Thread-scoped sessions (NIP-10 thread roots) ─────────────────────────
#
# Buzz threads are Nostr kind-9 replies: the buzz CLI emits a lone
# ["e", <parent>, "", "reply"] tag (observed live), while NIP-10-conformant
# clients mark the canonical root with ["e", <root>, "", "root"].  The
# adapter must key sessions on the canonical thread ROOT — never the
# immediate parent — so sibling threads get separate Hermes contexts and a
# whole thread (replies-to-replies included) shares one.

THREAD_ROOT = "beef" * 16
THREAD_PARENT = "cafe" * 16
THREAD_CHILD = "face" * 16
LATE_SIBLING = "12" * 32
MIXED_DESCENDANT = "34" * 32
SIBLING_ROOT = "dead" * 16
AGENT_EVENT = "ab" * 32

parse_thread_root = _buzz_mod.parse_thread_root


def _thread_event(event_id, channel, *, content, root=None, reply=None,
                  tags=None, pubkey=OTHER_PUBKEY, created_at=1000):
    """Kind-9 event with NIP-10 style e-tags (root/reply markers)."""
    if tags is None:
        tags = [["h", channel]]
        if root:
            tags.append(["e", root, "", "root"])
        if reply:
            tags.append(["e", reply, "", "reply"])
    return {
        "id": event_id,
        "pubkey": pubkey,
        "content": content,
        "created_at": created_at,
        "kind": 9,
        "tags": tags,
    }


class TestParseThreadRoot:
    """Pure tag parsing: markers, legacy positional shapes, fail-closed."""

    def test_no_e_tags_is_top_level(self):
        assert parse_thread_root([["h", CHANNEL]]) is None
        assert parse_thread_root([]) is None
        assert parse_thread_root(None) is None

    def test_root_marker_wins_over_reply_marker(self):
        tags = [
            ["h", CHANNEL],
            ["e", THREAD_ROOT, "", "root"],
            ["e", THREAD_PARENT, "", "reply"],
        ]
        assert parse_thread_root(tags) == THREAD_ROOT

    def test_lone_reply_marker_references_parent(self):
        assert parse_thread_root([["e", THREAD_PARENT, "", "reply"]]) == THREAD_PARENT

    def test_legacy_positional_single_e_tag(self):
        assert parse_thread_root([["e", THREAD_ROOT]]) == THREAD_ROOT
        assert parse_thread_root([["e", THREAD_ROOT, "wss://relay"]]) == THREAD_ROOT
        # Author pubkey in the marker slot is a positional shape, not a marker.
        assert parse_thread_root([["e", THREAD_ROOT, "wss://relay", OTHER_PUBKEY]]) == THREAD_ROOT

    def test_legacy_positional_first_e_tag_is_root(self):
        tags = [["e", THREAD_ROOT, ""], ["e", THREAD_PARENT, ""]]
        assert parse_thread_root(tags) == THREAD_ROOT

    def test_mention_markers_do_not_create_threads(self):
        assert parse_thread_root([["e", THREAD_ROOT, "", "mention"]]) is None

    def test_non_hex_ids_are_ignored(self):
        assert parse_thread_root([["e", "not-an-event-id", "", "reply"]]) is None
        assert parse_thread_root([["e", "", "", "root"]]) is None

    def test_conflicting_root_markers_fail_closed(self):
        tags = [["e", THREAD_ROOT, "", "root"], ["e", SIBLING_ROOT, "", "root"]]
        assert parse_thread_root(tags) is None

    def test_malformed_tag_shapes_are_ignored(self):
        assert parse_thread_root(["e", THREAD_ROOT]) is None  # not a list of tags
        assert parse_thread_root([["e"], "junk", {"e": THREAD_ROOT}, 42]) is None
        # A malformed sibling entry must not poison a well-formed root marker.
        tags = [["e"], ["e", "bogus", "", "reply"], ["e", THREAD_ROOT, "", "root"]]
        assert parse_thread_root(tags) == THREAD_ROOT


class TestThreadScopedSessions:

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

    async def _poll_with(self, adapter, channel, *events):
        cli = _ScriptedCli()
        cli.script("messages", "get", list(events))
        adapter._run_cli = cli
        await adapter._poll_channel(channel)

    @pytest.mark.asyncio
    async def test_top_level_message_has_no_thread_id(self, adapter):
        await self._poll_with(
            adapter, CHANNEL, _event("e1", content="@Chip hello", created_at=10)
        )
        assert [d["message_id"] for d in adapter._dispatched] == ["e1"]
        assert adapter._dispatched[0]["thread_id"] is None

    @pytest.mark.asyncio
    async def test_direct_reply_uses_canonical_root(self, adapter):
        await self._poll_with(
            adapter, CHANNEL,
            _thread_event("r1", CHANNEL, content="@Chip in thread", root=THREAD_ROOT),
        )
        assert [d["thread_id"] for d in adapter._dispatched] == [THREAD_ROOT]

    @pytest.mark.asyncio
    async def test_nested_reply_with_markers_retains_root(self, adapter):
        """A reply-to-a-reply keys on the root marker, not the parent."""
        await self._poll_with(
            adapter, CHANNEL,
            _thread_event("r2", CHANNEL, content="@Chip deeper",
                          root=THREAD_ROOT, reply=THREAD_PARENT),
        )
        assert [d["thread_id"] for d in adapter._dispatched] == [THREAD_ROOT]

    @pytest.mark.asyncio
    async def test_reply_only_chain_resolves_to_root_via_observed_history(self, adapter):
        """buzz-CLI shape (lone "reply" marker per hop): the adapter chains
        observed events so nested replies still key on the canonical root."""
        await self._poll_with(
            adapter, CHANNEL,
            _thread_event(THREAD_ROOT, CHANNEL, content="thread starts here",
                          created_at=10),
            _thread_event(THREAD_PARENT, CHANNEL, content="@Chip first reply",
                          reply=THREAD_ROOT, created_at=11),
            _thread_event(THREAD_CHILD, CHANNEL, content="@Chip nested reply",
                          reply=THREAD_PARENT, created_at=12),
        )
        assert [d["thread_id"] for d in adapter._dispatched] == [THREAD_ROOT, THREAD_ROOT]

    @pytest.mark.asyncio
    async def test_batch_reply_chain_is_parent_first_with_timestamp_ties(self, adapter):
        """CLI batches may be reverse-ordered with second-level timestamps."""
        await self._poll_with(
            adapter, CHANNEL,
            _thread_event(THREAD_CHILD, CHANNEL, content="@Chip child",
                          reply=THREAD_PARENT, created_at=10),
            _thread_event(THREAD_PARENT, CHANNEL, content="@Chip parent",
                          reply=THREAD_ROOT, created_at=10),
            _thread_event(THREAD_ROOT, CHANNEL, content="@Chip root",
                          created_at=10),
        )
        assert [d["message_id"] for d in adapter._dispatched] == [
            THREAD_ROOT,
            THREAD_PARENT,
            THREAD_CHILD,
        ]
        assert [d["thread_id"] for d in adapter._dispatched] == [
            None,
            THREAD_ROOT,
            THREAD_ROOT,
        ]

    @pytest.mark.asyncio
    async def test_reply_only_out_of_order_latches_one_stable_scope(self, adapter):
        """A child can precede its parent on WebSocket delivery.

        The first child must not start under the immediate parent and then
        split later descendants onto a newly discovered root. Once the parent
        id becomes a provisional scope, the whole observed chain stays there.
        """
        events = [
            _thread_event(THREAD_CHILD, CHANNEL, content="@Chip child first",
                          reply=THREAD_PARENT, created_at=10),
            _thread_event(THREAD_PARENT, CHANNEL, content="@Chip parent late",
                          reply=THREAD_ROOT, created_at=11),
            _thread_event(LATE_SIBLING, CHANNEL, content="@Chip sibling later",
                          reply=THREAD_PARENT, created_at=12),
            _thread_event(THREAD_ROOT, CHANNEL, content="@Chip root latest",
                          created_at=13),
            _thread_event(MIXED_DESCENDANT, CHANNEL, content="@Chip explicit root later",
                          root=THREAD_ROOT, reply=THREAD_PARENT, created_at=14),
        ]
        state = adapter._channel_state[CHANNEL]
        # WebSocket events call _handle_event one at a time, without batch
        # dependency ordering.
        for event in events:
            await adapter._handle_event(CHANNEL, state, event)
        assert [d["thread_id"] for d in adapter._dispatched] == [
            THREAD_PARENT,
            THREAD_PARENT,
            THREAD_PARENT,
            None,
            THREAD_PARENT,
        ]
        roots = adapter._channel_state[CHANNEL]["roots"]
        assert (
            roots[THREAD_CHILD]
            == roots[THREAD_PARENT]
            == roots[LATE_SIBLING]
            == roots[MIXED_DESCENDANT]
        )

    @pytest.mark.asyncio
    async def test_agent_reply_bridges_the_thread_chain(self, adapter):
        """A user replying to the AGENT's reply must land in the same thread
        session: send() records its own event id against the root."""
        await self._poll_with(
            adapter, CHANNEL,
            _thread_event(THREAD_ROOT, CHANNEL, content="@Chip start", created_at=10),
        )
        cli = _ScriptedCli()
        cli.script("messages", "send", {"accepted": True, "event_id": AGENT_EVENT, "message": ""})
        adapter._run_cli = cli
        await adapter.send(CHANNEL, "agent answer", reply_to=THREAD_ROOT)

        await self._poll_with(
            adapter, CHANNEL,
            _thread_event(THREAD_CHILD, CHANNEL, content="@Chip follow-up",
                          reply=AGENT_EVENT, created_at=20),
        )
        assert adapter._dispatched[-1]["message_id"] == THREAD_CHILD
        assert adapter._dispatched[-1]["thread_id"] == THREAD_ROOT

    @pytest.mark.asyncio
    async def test_seed_history_builds_thread_map_without_dispatch(self, adapter):
        """Restart resilience: seeded history primes the event->root map so
        the first post-restart reply keys on the true root; seeding itself
        never dispatches."""
        cli = _ScriptedCli()
        cli.script("messages", "get", [
            _thread_event(THREAD_PARENT, CHANNEL, content="old reply",
                          reply=THREAD_ROOT, created_at=100),
            _thread_event(THREAD_ROOT, CHANNEL, content="old root", created_at=100),
        ])
        adapter._run_cli = cli
        await adapter._seed_channel(CHANNEL, chat_type="group")
        assert adapter._dispatched == []

        await self._poll_with(
            adapter, CHANNEL,
            _thread_event(THREAD_CHILD, CHANNEL, content="@Chip resuming",
                          reply=THREAD_PARENT, created_at=120),
        )
        assert [d["thread_id"] for d in adapter._dispatched] == [THREAD_ROOT]

    @pytest.mark.asyncio
    async def test_malformed_and_unrelated_tags_yield_no_thread(self, adapter):
        """Bogus e-tags (non-hex, self-referential, mention markers) must
        fail closed to the plain channel/user session."""
        await self._poll_with(
            adapter, CHANNEL,
            _thread_event("m1", CHANNEL, content="@Chip one",
                          tags=[["h", CHANNEL], ["e", "not-hex!", "", "reply"]],
                          created_at=10),
            _thread_event("m2", CHANNEL, content="@Chip two",
                          tags=[["h", CHANNEL], ["e", "m2", "", "root"]],
                          created_at=11),
            _thread_event(THREAD_CHILD, CHANNEL, content="@Chip three",
                          tags=[["h", CHANNEL], ["e", THREAD_CHILD, "", "root"]],
                          created_at=12),
            _thread_event("m3", CHANNEL, content="@Chip four",
                          tags=[["h", CHANNEL], ["e", THREAD_ROOT, "", "mention"],
                                ["p", SELF_PUBKEY]],
                          created_at=13),
        )
        assert [d["thread_id"] for d in adapter._dispatched] == [None, None, None, None]

    @pytest.mark.asyncio
    async def test_threaded_reply_still_respects_mention_gate(self, adapter):
        """Channel gating is unchanged: an un-mentioned threaded reply does
        not dispatch (require_mention default)."""
        await self._poll_with(
            adapter, CHANNEL,
            _thread_event("r1", CHANNEL, content="just thread chatter",
                          root=THREAD_ROOT),
        )
        assert adapter._dispatched == []

    @pytest.mark.asyncio
    async def test_dm_replies_thread_and_top_level_dms_do_not(self, adapter):
        adapter._channel_state[DM_CHANNEL] = {"chat_type": "dm", "last_ts": 0, "seen": {}}
        await self._poll_with(
            adapter, DM_CHANNEL,
            _thread_event("d1", DM_CHANNEL, content="plain dm", created_at=10),
            _thread_event("d2", DM_CHANNEL, content="threaded dm",
                          root=THREAD_ROOT, created_at=11),
        )
        assert [(d["chat_type"], d["thread_id"]) for d in adapter._dispatched] == [
            ("dm", None),
            ("dm", THREAD_ROOT),
        ]

    def test_session_keys_isolate_sibling_threads_and_share_a_thread(self):
        """The gateway invariant this feature exists for: sibling roots get
        distinct session keys; one root is one shared session for all
        participants; top-level traffic keeps the per-user channel session."""
        from gateway.config import Platform
        from gateway.session import SessionSource, build_session_key

        def src(thread_id=None, user=OTHER_PUBKEY):
            return SessionSource(
                platform=Platform("buzz"),
                chat_id=CHANNEL,
                chat_type="group",
                user_id=user,
                thread_id=thread_id,
            )

        thread_a = build_session_key(src(THREAD_ROOT))
        thread_a_other_user = build_session_key(src(THREAD_ROOT, user="b" * 64))
        thread_b = build_session_key(src(SIBLING_ROOT))
        top_level = build_session_key(src())

        assert thread_a != thread_b  # sibling threads are isolated
        assert thread_a == thread_a_other_user  # one thread, one shared session
        assert top_level not in (thread_a, thread_b)  # channel scope preserved


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


