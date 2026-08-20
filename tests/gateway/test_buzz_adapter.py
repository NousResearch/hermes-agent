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

    def test_parent_resolution_prefers_reply_then_root_then_positional(self):
        """_parse_thread_parent feeds both batch ordering and reply-context
        quoting: a reply marker names the parent outright; a root-only
        marked reply's parent IS the root; ambiguity fails closed."""
        _parse_thread_parent = _buzz_mod._parse_thread_parent
        # Nested marked reply: the reply marker is the parent, not the root.
        assert _parse_thread_parent(
            [["e", THREAD_ROOT, "", "root"], ["e", THREAD_PARENT, "", "reply"]]
        ) == THREAD_PARENT
        # Marked direct reply to the root: the root is the parent.
        assert _parse_thread_parent([["e", THREAD_ROOT, "", "root"]]) == THREAD_ROOT
        # Deprecated positional single ref is the parent.
        assert _parse_thread_parent([["e", THREAD_PARENT]]) == THREAD_PARENT
        # Conflicting marked roots are ambiguous and fail closed.
        assert _parse_thread_parent(
            [["e", THREAD_ROOT, "", "root"], ["e", SIBLING_ROOT, "", "root"]]
        ) is None
        # Deprecated positional NIP-10: first ref is the root, last is parent.
        assert _parse_thread_parent(
            [["e", THREAD_ROOT], ["e", THREAD_PARENT]]
        ) == THREAD_PARENT

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
            _thread_event(SIBLING_ROOT, CHANNEL, content="@Chip direct root reply later",
                          root=THREAD_ROOT, created_at=15),
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
            THREAD_PARENT,
        ]
        roots = adapter._channel_state[CHANNEL]["roots"]
        assert (
            roots[THREAD_CHILD]
            == roots[THREAD_PARENT]
            == roots[LATE_SIBLING]
            == roots[MIXED_DESCENDANT]
            == roots[SIBLING_ROOT]
            == roots[THREAD_ROOT]
        )

    @pytest.mark.asyncio
    async def test_provisional_scope_wins_when_root_precedes_late_parent(self, adapter):
        """The first child starts the stable scope even if root beats parent."""
        events = [
            _thread_event(THREAD_CHILD, CHANNEL, content="@Chip child first",
                          reply=THREAD_PARENT, created_at=10),
            _thread_event(THREAD_ROOT, CHANNEL, content="@Chip root second",
                          created_at=11),
            _thread_event(THREAD_PARENT, CHANNEL, content="@Chip parent late",
                          reply=THREAD_ROOT, created_at=12),
            _thread_event(SIBLING_ROOT, CHANNEL, content="@Chip direct root reply",
                          root=THREAD_ROOT, created_at=13),
        ]
        state = adapter._channel_state[CHANNEL]
        for event in events:
            await adapter._handle_event(CHANNEL, state, event)

        assert [d["thread_id"] for d in adapter._dispatched] == [
            THREAD_PARENT,
            None,
            THREAD_PARENT,
            THREAD_PARENT,
        ]
        roots = adapter._channel_state[CHANNEL]["roots"]
        assert roots[THREAD_ROOT] == THREAD_PARENT

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


# ── Thread reply-context seeding ──────────────────────────────────────────
#
# The root of a thread intentionally stays in the ordinary channel/user
# session, so the thread-scoped session created by the first reply would
# start blind to the message it hangs off (live repro: root says "the secret
# is X", first reply asks for the secret, agent can't answer).  The adapter
# therefore attaches each threaded event's ACTUAL immediate parent's author
# + text as ``reply_to_*`` context, which gateway.run folds into the
# triggering user message.  A first reply's parent IS the root, so the
# fresh thread session starts with root content on its very first turn;
# nested replies quote the message they actually answered — preserving the
# gateway's reply-disambiguation contract instead of re-injecting stale
# root text on every turn — all without rewriting history or injecting
# synthetic messages.  The canonical root is used ONLY for thread_id /
# session scoping.

ROOT_SECRET = "The secret in this root message is ROOT_SECRET_474"


class TestThreadReplyContextSeeding:

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
    async def test_first_reply_carries_unmentioned_root_context(self, adapter):
        """The live repro: the root never dispatches (no mention), yet the
        first reply must arrive quoting it — and only once per event even
        when the relay re-delivers the same batch."""
        batch = [
            _thread_event(THREAD_ROOT, CHANNEL, content=ROOT_SECRET, created_at=10),
            _thread_event(THREAD_CHILD, CHANNEL, content="@Chip what is the secret?",
                          reply=THREAD_ROOT, created_at=11),
        ]
        await self._poll_with(adapter, CHANNEL, *batch)
        assert [d["message_id"] for d in adapter._dispatched] == [THREAD_CHILD]
        reply = adapter._dispatched[0]
        assert reply["thread_id"] == THREAD_ROOT
        assert reply["reply_to_message_id"] == THREAD_ROOT
        assert reply["reply_to_text"] == ROOT_SECRET
        assert reply["reply_to_author_id"] == OTHER_PUBKEY
        assert reply["reply_to_is_own_message"] is False

        # Duplicate delivery of the identical batch must not re-dispatch
        # (and therefore cannot re-seed anything).
        await self._poll_with(adapter, CHANNEL, *batch)
        assert len(adapter._dispatched) == 1

    @pytest.mark.asyncio
    async def test_mentioned_root_stays_top_level_without_reply_context(self, adapter):
        """The root itself keeps the plain channel/user session shape: no
        thread_id, no reply context."""
        await self._poll_with(
            adapter, CHANNEL,
            _thread_event(THREAD_ROOT, CHANNEL, content=f"@Chip {ROOT_SECRET}", created_at=10),
        )
        root = adapter._dispatched[0]
        assert root["thread_id"] is None
        assert root["reply_to_message_id"] is None
        assert root["reply_to_text"] is None

    @pytest.mark.asyncio
    async def test_marked_direct_reply_to_root_quotes_root(self, adapter):
        """NIP-10 marked shape: a direct reply carrying only a ``root``
        marker has the root as its actual parent — it must quote it."""
        await self._poll_with(
            adapter, CHANNEL,
            _thread_event(THREAD_ROOT, CHANNEL, content=ROOT_SECRET, created_at=10),
            _thread_event(THREAD_CHILD, CHANNEL, content="@Chip in thread",
                          root=THREAD_ROOT, created_at=11),
        )
        reply = adapter._dispatched[0]
        assert reply["thread_id"] == THREAD_ROOT
        assert reply["reply_to_message_id"] == THREAD_ROOT
        assert reply["reply_to_text"] == ROOT_SECRET

    @pytest.mark.asyncio
    async def test_nested_reply_quotes_its_actual_parent(self, adapter):
        """Replies-to-replies quote the message they actually answered — the
        session still keys on the root (thread_id), but re-injecting stale
        root text every turn would break the gateway's reply-disambiguation
        contract.  Only the FIRST reply (parent == root) quotes the root."""
        await self._poll_with(
            adapter, CHANNEL,
            _thread_event(THREAD_ROOT, CHANNEL, content=ROOT_SECRET, created_at=10),
            _thread_event(THREAD_PARENT, CHANNEL, content="@Chip first reply",
                          reply=THREAD_ROOT, created_at=11),
            _thread_event(THREAD_CHILD, CHANNEL, content="@Chip nested reply",
                          reply=THREAD_PARENT, created_at=12),
        )
        assert [d["message_id"] for d in adapter._dispatched] == [THREAD_PARENT, THREAD_CHILD]
        # Both share the root-scoped session...
        assert [d["thread_id"] for d in adapter._dispatched] == [THREAD_ROOT, THREAD_ROOT]
        # ...but each quotes its own parent: root for the first reply, the
        # first reply (mention intact — it is quoted text, not a trigger)
        # for the nested one.
        assert [d["reply_to_message_id"] for d in adapter._dispatched] == [
            THREAD_ROOT,
            THREAD_PARENT,
        ]
        assert [d["reply_to_text"] for d in adapter._dispatched] == [
            ROOT_SECRET,
            "@Chip first reply",
        ]

    @pytest.mark.asyncio
    async def test_reply_to_agent_top_level_post_quotes_agent(self, adapter):
        """A user thread-replying to the AGENT's own top-level post gets that
        post as root context, flagged as the agent's own message (send()
        records its outbound text; inbound paths never see our sends)."""
        cli = _ScriptedCli()
        cli.script("messages", "send", {"accepted": True, "event_id": AGENT_EVENT, "message": ""})
        adapter._run_cli = cli
        await adapter.send(CHANNEL, "agent broadcast")

        await self._poll_with(
            adapter, CHANNEL,
            _thread_event(THREAD_CHILD, CHANNEL, content="@Chip tell me more",
                          reply=AGENT_EVENT, created_at=20),
        )
        reply = adapter._dispatched[-1]
        assert reply["thread_id"] == AGENT_EVENT
        assert reply["reply_to_text"] == "agent broadcast"
        assert reply["reply_to_author_id"] == SELF_PUBKEY
        assert reply["reply_to_is_own_message"] is True

    @pytest.mark.asyncio
    async def test_reply_to_agents_in_thread_reply_quotes_agent_keeps_root_scope(self, adapter):
        """A user replying to the AGENT's reply INSIDE a thread quotes that
        outbound reply (own-message flag set) while the session stays keyed
        on the canonical thread root."""
        await self._poll_with(
            adapter, CHANNEL,
            _thread_event(THREAD_ROOT, CHANNEL, content=ROOT_SECRET, created_at=10),
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
        reply = adapter._dispatched[-1]
        assert reply["thread_id"] == THREAD_ROOT  # canonical root still scopes
        assert reply["reply_to_message_id"] == AGENT_EVENT
        assert reply["reply_to_text"] == "agent answer"
        assert reply["reply_to_is_own_message"] is True

    @pytest.mark.asyncio
    async def test_restart_first_reply_to_root_quotes_root(self, adapter):
        """Restart resilience: seeded history primes the parent-text cache,
        so a first reply arriving post-restart still quotes a root that was
        posted before the gateway went down."""
        cli = _ScriptedCli()
        cli.script("messages", "get", [
            _thread_event(THREAD_ROOT, CHANNEL, content=ROOT_SECRET, created_at=100),
        ])
        adapter._run_cli = cli
        await adapter._seed_channel(CHANNEL, chat_type="group")
        assert adapter._dispatched == []

        await self._poll_with(
            adapter, CHANNEL,
            _thread_event(THREAD_CHILD, CHANNEL, content="@Chip resuming",
                          reply=THREAD_ROOT, created_at=120),
        )
        reply = adapter._dispatched[0]
        assert reply["thread_id"] == THREAD_ROOT
        assert reply["reply_to_message_id"] == THREAD_ROOT
        assert reply["reply_to_text"] == ROOT_SECRET

    @pytest.mark.asyncio
    async def test_restart_nested_reply_quotes_parent_from_history(self, adapter):
        """A nested reply arriving post-restart quotes its actual parent
        (fetched in seeded history), while still keying on the true root."""
        cli = _ScriptedCli()
        cli.script("messages", "get", [
            _thread_event(THREAD_PARENT, CHANNEL, content="old reply",
                          reply=THREAD_ROOT, created_at=110),
            _thread_event(THREAD_ROOT, CHANNEL, content=ROOT_SECRET, created_at=100),
        ])
        adapter._run_cli = cli
        await adapter._seed_channel(CHANNEL, chat_type="group")
        assert adapter._dispatched == []

        await self._poll_with(
            adapter, CHANNEL,
            _thread_event(THREAD_CHILD, CHANNEL, content="@Chip resuming",
                          reply=THREAD_PARENT, created_at=120),
        )
        reply = adapter._dispatched[0]
        assert reply["thread_id"] == THREAD_ROOT
        assert reply["reply_to_message_id"] == THREAD_PARENT
        assert reply["reply_to_text"] == "old reply"

    @pytest.mark.asyncio
    async def test_unknown_parent_dispatches_without_context(self, adapter):
        """An unobserved parent fails open: the reply still dispatches into
        its thread session, just without quoted context."""
        await self._poll_with(
            adapter, CHANNEL,
            _thread_event(THREAD_CHILD, CHANNEL, content="@Chip orphan reply",
                          reply=THREAD_ROOT, created_at=10),
        )
        reply = adapter._dispatched[0]
        assert reply["thread_id"] == THREAD_ROOT
        assert reply["reply_to_message_id"] is None
        assert reply["reply_to_text"] is None

    @pytest.mark.asyncio
    async def test_out_of_order_chain_fails_open_then_quotes_known_parents(self, adapter):
        """WebSocket delivery can hand children over before their parents.
        Unknown parents fail open; once a parent has been observed, later
        replies to it quote it — and duplicate re-delivery of an already
        seen event never re-dispatches (or re-seeds) anything."""
        state = adapter._channel_state[CHANNEL]
        events = [
            _thread_event(THREAD_CHILD, CHANNEL, content="@Chip child first",
                          reply=THREAD_PARENT, created_at=10),
            _thread_event(THREAD_PARENT, CHANNEL, content="@Chip parent late",
                          reply=THREAD_ROOT, created_at=11),
            _thread_event(LATE_SIBLING, CHANNEL, content="@Chip sibling later",
                          reply=THREAD_PARENT, created_at=12),
        ]
        for event in events:
            await adapter._handle_event(CHANNEL, state, event)

        child, parent, sibling = adapter._dispatched
        # Child dispatched before its parent existed: no context to quote.
        assert child["reply_to_text"] is None
        # The parent replies to the still-unseen root: fail open (and its
        # latched provisional scope — its own id — must never self-quote).
        assert parent["thread_id"] == THREAD_PARENT
        assert parent["reply_to_text"] is None
        # The sibling's parent is now observed, so it quotes it.
        assert sibling["thread_id"] == THREAD_PARENT
        assert sibling["reply_to_message_id"] == THREAD_PARENT
        assert sibling["reply_to_text"] == "@Chip parent late"

        # Duplicate delivery: same events again, nothing new dispatches.
        for event in events:
            await adapter._handle_event(CHANNEL, state, event)
        assert len(adapter._dispatched) == 3

    @pytest.mark.asyncio
    async def test_self_referential_reply_marker_never_self_quotes(self, adapter):
        """Tag garbage pointing a reply marker at the event's own id must not
        make the event quote itself, even though its text is cached."""
        await self._poll_with(
            adapter, CHANNEL,
            _thread_event(THREAD_ROOT, CHANNEL, content=ROOT_SECRET, created_at=10),
            _thread_event(THREAD_CHILD, CHANNEL, content="@Chip weird tags",
                          tags=[["h", CHANNEL],
                                ["e", THREAD_ROOT, "", "root"],
                                ["e", THREAD_CHILD, "", "reply"]],
                          created_at=11),
        )
        reply = adapter._dispatched[0]
        assert reply["thread_id"] == THREAD_ROOT
        assert reply["reply_to_text"] is None

    @pytest.mark.asyncio
    async def test_real_handoff_seeds_thread_session_with_root(self):
        """Strongest-level regression: run the REAL dispatch path — no
        _dispatch_message stub — through _poll_channel → _handle_event →
        _dispatch_message → BasePlatformAdapter.handle_message →
        build_session_key → message handler.

        Contract proven end-to-end: the root lands in the plain channel/user
        session; the first reply lands in a DIFFERENT, thread-scoped session
        whose triggering MessageEvent already carries the root's text —
        because the root IS the first reply's actual parent — so the thread
        session is seeded with root context before the agent processes the
        reply, with strict role alternation intact (the root text rides the
        reply's own turn, no synthetic messages)."""
        from gateway.session import build_session_key

        adapter = _make_adapter()
        received = []

        async def handler(event):
            received.append(event)
            return None

        adapter.set_message_handler(handler)
        adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}

        cli = _ScriptedCli()
        cli.script("messages", "get", [
            _thread_event(THREAD_ROOT, CHANNEL, content=f"@Chip {ROOT_SECRET}",
                          created_at=10),
            _thread_event(THREAD_CHILD, CHANNEL, content="@Chip what is the secret?",
                          reply=THREAD_ROOT, created_at=11),
        ])
        adapter._run_cli = cli
        await adapter._poll_channel(CHANNEL)
        # handle_message spawns one background task per session; drain them.
        for task in list(adapter._session_tasks.values()):
            await task

        by_id = {event.message_id: event for event in received}
        assert set(by_id) == {THREAD_ROOT, THREAD_CHILD}
        root_evt, reply_evt = by_id[THREAD_ROOT], by_id[THREAD_CHILD]

        # Root: ordinary channel/user session, untouched turn.
        assert root_evt.source.thread_id is None
        assert root_evt.reply_to_text is None
        # Reply: new thread-scoped session, seeded with the root's content.
        assert reply_evt.source.thread_id == THREAD_ROOT
        assert build_session_key(reply_evt.source) != build_session_key(root_evt.source)
        assert reply_evt.reply_to_message_id == THREAD_ROOT
        assert ROOT_SECRET in reply_evt.reply_to_text
        # Outbound replies still target the triggering leaf, not the root.
        from gateway.platforms.base import _reply_anchor_for_event

        assert _reply_anchor_for_event(reply_evt) == THREAD_CHILD

    @pytest.mark.asyncio
    async def test_gateway_runner_injects_root_quote_into_first_reply_turn(self):
        """The full contract, through the REAL GatewayRunner text pipeline:
        the first Buzz reply's MessageEvent (produced by the real adapter
        dispatch path) is handed to GatewayRunner._prepare_inbound_message_text
        with the thread session's (empty) history, and the resulting user
        turn opens with the root quote — root context and the reply share
        ONE user message, so role alternation cannot be violated."""
        from gateway.config import GatewayConfig, Platform, PlatformConfig
        from gateway.run import GatewayRunner
        from gateway.session import build_session_key

        adapter = _make_adapter()
        received = []

        async def handler(event):
            received.append(event)
            return None

        adapter.set_message_handler(handler)
        adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}

        cli = _ScriptedCli()
        cli.script("messages", "get", [
            _thread_event(THREAD_ROOT, CHANNEL, content=ROOT_SECRET, created_at=10),
            _thread_event(THREAD_CHILD, CHANNEL, content="@Chip what is the secret?",
                          reply=THREAD_ROOT, created_at=11),
        ])
        adapter._run_cli = cli
        await adapter._poll_channel(CHANNEL)
        for task in list(adapter._session_tasks.values()):
            await task

        (reply_evt,) = received  # the un-mentioned root never dispatches

        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            platforms={Platform("buzz"): PlatformConfig(enabled=True, extra={})},
        )
        runner.adapters = {}
        runner._model = "openai/gpt-4.1-mini"
        runner._base_url = None

        turn_text = await runner._prepare_inbound_message_text(
            event=reply_evt,
            source=reply_evt.source,
            history=[],  # a fresh thread session has no history yet
            session_key=build_session_key(reply_evt.source),
        )

        assert turn_text is not None
        assert turn_text.startswith(f'[Replying to: "{ROOT_SECRET}"]')
        assert turn_text.endswith("what is the secret?")


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
    async def test_send_uses_metadata_reply_to_message_id(self):
        """Gateway stream/progress pass reply anchors via metadata.

        Without honoring reply_to_message_id, mid-turn commentary posts as
        new top-level channel messages instead of thread replies.
        """
        adapter = _make_adapter()
        adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        cli = _ScriptedCli()
        cli.script("messages", "send", {"accepted": True, "event_id": "evt-reply", "message": ""})
        adapter._run_cli = cli

        result = await adapter.send(
            CHANNEL,
            "threaded reply",
            metadata={"reply_to_message_id": "root-event-abc"},
        )
        assert result.success is True
        args, _stdin = cli.calls[0]
        assert "--reply-to" in args
        assert args[args.index("--reply-to") + 1] == "root-event-abc"

    @pytest.mark.asyncio
    async def test_send_prefers_trigger_message_over_session_thread_root(self):
        """Fresh-final sends keep replying to the triggering leaf message.

        ``thread_id`` scopes the Hermes session; it is not the outbound reply
        anchor when ``reply_to_message_id`` identifies the actual trigger.
        """
        adapter = _make_adapter()
        adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        cli = _ScriptedCli()
        cli.script("messages", "send", {"accepted": True, "event_id": "evt-final"})
        adapter._run_cli = cli

        result = await adapter.send(
            CHANNEL,
            "fresh final",
            metadata={
                "thread_id": THREAD_ROOT,
                "reply_to_message_id": THREAD_CHILD,
            },
        )

        assert result.success is True
        args, _stdin = cli.calls[0]
        assert args[args.index("--reply-to") + 1] == THREAD_CHILD


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


# ── Editing and deleting (streaming) ──────────────────────────────────


class TestBuzzAdapterEdit:

    @pytest.mark.asyncio
    async def test_edit_targets_the_original_event_and_uses_stdin(self):
        adapter = _make_adapter()
        adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        cli = _ScriptedCli()
        cli.script("messages", "edit", {"accepted": True, "event_id": "edit1", "message": ""})
        adapter._run_cli = cli

        result = await adapter.edit_message(CHANNEL, "orig1", "partial answer")
        assert result.success is True

        args, stdin_text = cli.calls[0]
        assert args[:2] == ["messages", "edit"]
        assert args[args.index("--event") + 1] == "orig1"
        # Content travels via stdin (--content -), never argv, same as send
        assert args[args.index("--content") + 1] == "-"
        assert stdin_text == "partial answer"

    @pytest.mark.asyncio
    async def test_edit_returns_the_original_id_not_the_cli_event_id(self):
        """The stream consumer re-edits ONE message id for the whole stream.

        buzz-cli reports a fresh event id for each edit; returning that would
        make the second edit address a message that was never sent.
        """
        adapter = _make_adapter()
        adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        cli = _ScriptedCli()
        cli.script("messages", "edit", {"accepted": True, "event_id": "edit1"})
        adapter._run_cli = cli

        result = await adapter.edit_message(CHANNEL, "orig1", "text")
        assert result.message_id == "orig1"

    @pytest.mark.asyncio
    async def test_edit_marks_its_own_event_seen(self):
        adapter = _make_adapter()
        adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        cli = _ScriptedCli()
        cli.script("messages", "edit", {"accepted": True, "event_id": "edit1"})
        adapter._run_cli = cli

        await adapter.edit_message(CHANNEL, "orig1", "text")
        assert "edit1" in adapter._channel_state[CHANNEL]["seen"]

    @pytest.mark.asyncio
    async def test_edit_accepts_finalize_without_changing_behaviour(self):
        adapter = _make_adapter()
        adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        cli = _ScriptedCli()
        cli.script("messages", "edit", {"accepted": True, "event_id": "edit1"})
        adapter._run_cli = cli

        result = await adapter.edit_message(CHANNEL, "orig1", "text", finalize=True)
        assert result.success is True
        assert len(cli.calls) == 1

    @pytest.mark.asyncio
    async def test_edit_refreshes_reply_context_to_final_text(self):
        """A user replying after streaming must quote the completed answer."""
        adapter = _make_adapter()
        adapter._self_pubkey = SELF_PUBKEY
        adapter._channel_state[CHANNEL] = {
            "chat_type": "group",
            "last_ts": 0,
            "seen": {},
        }
        adapter._record_outbound_text(CHANNEL, AGENT_EVENT, "partial")
        cli = _ScriptedCli()
        cli.script("messages", "edit", {"accepted": True, "event_id": "edit1"})
        adapter._run_cli = cli

        result = await adapter.edit_message(
            CHANNEL, AGENT_EVENT, "complete answer", finalize=True
        )

        assert result.success is True
        assert adapter._thread_parent_context(
            adapter._channel_state[CHANNEL], AGENT_EVENT, "child1"
        ) == (SELF_PUBKEY, "complete answer")

    @pytest.mark.asyncio
    async def test_rapid_edits_wait_for_distinct_nostr_seconds(self, monkeypatch):
        """Edit precedence uses second-resolution Nostr timestamps.

        Successive edits for one target must therefore be signed in distinct
        seconds, or a client can keep an arbitrary partial instead of final.
        """
        adapter = _make_adapter()
        adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        cli = _ScriptedCli()
        cli.script("messages", "edit", {"accepted": True, "event_id": "edit1"})
        cli.script("messages", "edit", {"accepted": True, "event_id": "edit2"})
        adapter._run_cli = cli

        now = [100.90]
        sleeps = []
        monkeypatch.setattr(_buzz_mod.time, "time", lambda: now[0])

        async def advance_clock(delay):
            sleeps.append(delay)
            now[0] += delay

        monkeypatch.setattr(_buzz_mod.asyncio, "sleep", advance_clock)

        first = await adapter.edit_message(CHANNEL, "orig1", "partial")
        now[0] = 100.95
        final = await adapter.edit_message(CHANNEL, "orig1", "complete", finalize=True)

        assert first.success is True
        assert final.success is True
        assert len(cli.calls) == 2
        assert sleeps and now[0] >= 101.0

    @pytest.mark.asyncio
    async def test_edit_without_a_message_id_never_calls_the_cli(self):
        adapter = _make_adapter()
        cli = _ScriptedCli()
        adapter._run_cli = cli

        result = await adapter.edit_message(CHANNEL, "", "text")
        assert result.success is False
        assert cli.calls == []

    @pytest.mark.asyncio
    async def test_edit_with_empty_content_never_calls_the_cli(self):
        adapter = _make_adapter()
        cli = _ScriptedCli()
        adapter._run_cli = cli

        result = await adapter.edit_message(CHANNEL, "orig1", "")
        assert result.success is False
        assert cli.calls == []

    @pytest.mark.asyncio
    async def test_edit_relay_error_is_retryable_but_bad_input_is_not(self):
        adapter = _make_adapter()
        cli = _ScriptedCli()
        cli.script("messages", "edit", "", code=2, stderr="relay unreachable")
        adapter._run_cli = cli
        relay_failure = await adapter.edit_message(CHANNEL, "orig1", "text")
        assert relay_failure.success is False
        assert relay_failure.retryable is True

        adapter = _make_adapter()
        cli = _ScriptedCli()
        cli.script("messages", "edit", "", code=1, stderr="bad input")
        adapter._run_cli = cli
        input_failure = await adapter.edit_message(CHANNEL, "orig1", "text")
        assert input_failure.success is False
        assert input_failure.retryable is False

    @pytest.mark.asyncio
    async def test_delete_targets_the_event(self):
        adapter = _make_adapter()
        adapter._channel_state[CHANNEL] = {"chat_type": "group", "last_ts": 0, "seen": {}}
        cli = _ScriptedCli()
        cli.script("messages", "delete", {"accepted": True, "event_id": "del1"})
        adapter._run_cli = cli

        assert await adapter.delete_message(CHANNEL, "orig1") is True
        assert cli.calls[0][0] == ["messages", "delete", "--event", "orig1"]

    @pytest.mark.asyncio
    async def test_delete_failure_returns_false(self):
        adapter = _make_adapter()
        cli = _ScriptedCli()
        cli.script("messages", "delete", "", code=2, stderr="relay unreachable")
        adapter._run_cli = cli

        assert await adapter.delete_message(CHANNEL, "orig1") is False

    @pytest.mark.asyncio
    async def test_delete_without_a_message_id_never_calls_the_cli(self):
        adapter = _make_adapter()
        cli = _ScriptedCli()
        adapter._run_cli = cli

        assert await adapter.delete_message(CHANNEL, "") is False
        assert cli.calls == []
