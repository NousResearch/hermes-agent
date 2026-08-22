"""Tests for the Buzz platform adapter plugin."""

import asyncio
import hashlib
import json
import stat
from collections import OrderedDict
from pathlib import Path

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


    def test_unresolved_mention_fallback_uses_unicode_casefold_boundaries(self):
        fallback = getattr(_buzz_mod, "unresolved_mention_fallback", None)
        assert callable(fallback)
        assert fallback(
            "Ask @Straße now, not bob@example.com.",
            "mention '@STRASSE' is ambiguous",
        ) == "Ask Straße now, not bob@example.com."


@pytest.mark.asyncio
async def test_joined_channel_discovery_adds_and_seeds_authoritative_membership():
    adapter = _make_adapter()
    cli = _ScriptedCli()
    cli.script(
        "channels",
        "list",
        [
            {
                "channel_id": "joined-channel",
                "type": "community",
                "name": "Project",
            }
        ],
    )
    cli.script(
        "messages",
        "get",
        [
            {
                "id": "historical-event",
                "kind": 9,
                "pubkey": OTHER_PUBKEY,
                "content": "old",
                "created_at": 41,
                "tags": [],
            }
        ],
    )
    adapter._run_cli = cli

    changed = await adapter._discover_joined_channels()

    assert changed is True
    assert adapter._joined_channel_ids == {"joined-channel"}
    assert adapter._channel_state["joined-channel"]["last_ts"] == 41
    assert "historical-event" in adapter._channel_state["joined-channel"]["seen"]
    assert cli.calls[0][0] == ["channels", "list", "--member"]


@pytest.mark.asyncio
async def test_joined_channel_discovery_removes_departed_group():
    adapter = _make_adapter()
    adapter._joined_channel_ids = {"kept-channel", "departed-channel"}
    adapter._channel_state = {
        "kept-channel": {"chat_type": "group", "last_ts": 1, "seen": OrderedDict()},
        "departed-channel": {"chat_type": "group", "last_ts": 1, "seen": OrderedDict()},
    }
    adapter._channel_names = {
        "kept-channel": "Kept",
        "departed-channel": "Departed",
    }
    adapter._channel_meta = {
        "kept-channel": {"channel_id": "kept-channel", "name": "Kept"},
        "departed-channel": {"channel_id": "departed-channel", "name": "Departed"},
    }
    cli = _ScriptedCli()
    cli.script(
        "channels",
        "list",
        [{"channel_id": "kept-channel", "type": "community", "name": "Kept"}],
    )
    adapter._run_cli = cli

    changed = await adapter._discover_joined_channels()

    assert changed is True
    assert set(adapter._channel_state) == {"kept-channel"}
    assert "departed-channel" not in adapter._channel_names
    assert "departed-channel" not in adapter._channel_meta


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malformed_roster",
    [
        [{"type": "community", "name": "Missing id"}],
        [{"channel_id": "new-channel", "name": "Missing type"}],
        [{"channel_id": "new-channel", "type": "community"}, "not-an-object"],
    ],
)
async def test_malformed_joined_refresh_preserves_prior_snapshot(malformed_roster):
    adapter = _make_adapter()
    prior_state = {"chat_type": "group", "last_ts": 7, "seen": OrderedDict()}
    prior_meta = {
        "channel_id": "prior-channel",
        "type": "community",
        "name": "Prior",
    }
    adapter._joined_channel_ids = {"prior-channel"}
    adapter._channel_state = {"prior-channel": prior_state}
    adapter._channel_names = {"prior-channel": "Prior"}
    adapter._channel_meta = {"prior-channel": prior_meta}
    cli = _ScriptedCli()
    cli.script("channels", "list", malformed_roster)
    adapter._run_cli = cli

    assert await adapter._discover_joined_channels() is None
    assert adapter._joined_channel_ids == {"prior-channel"}
    assert adapter._channel_state == {"prior-channel": prior_state}
    assert adapter._channel_names == {"prior-channel": "Prior"}
    assert adapter._channel_meta == {"prior-channel": prior_meta}


@pytest.mark.asyncio
async def test_valid_empty_joined_refresh_clears_prior_group_snapshot():
    adapter = _make_adapter()
    adapter._joined_channel_ids = {"prior-channel"}
    adapter._channel_state = {
        "prior-channel": {"chat_type": "group", "last_ts": 7, "seen": OrderedDict()}
    }
    adapter._channel_names = {"prior-channel": "Prior"}
    adapter._channel_meta = {
        "prior-channel": {
            "channel_id": "prior-channel",
            "type": "community",
            "name": "Prior",
        }
    }
    cli = _ScriptedCli()
    cli.script("channels", "list", [])
    adapter._run_cli = cli

    assert await adapter._discover_joined_channels() is True
    assert adapter._joined_channel_ids == set()
    assert adapter._channel_state == {}
    assert adapter._channel_names == {}
    assert adapter._channel_meta == {}


@pytest.mark.asyncio
async def test_explicit_channels_restrict_dynamic_join_discovery():
    adapter = _make_adapter(extra={"channels": [CHANNEL]})
    cli = _ScriptedCli()
    cli.script(
        "channels",
        "list",
        [
            {"channel_id": CHANNEL, "type": "community", "name": "Configured"},
            {
                "channel_id": "unconfigured",
                "type": "community",
                "name": "Other",
            },
        ],
    )
    cli.script("messages", "get", [])
    adapter._run_cli = cli

    await adapter._discover_joined_channels()

    assert set(adapter._channel_state) == {CHANNEL}
    assert adapter._joined_channel_ids == {CHANNEL}


@pytest.mark.asyncio
async def test_irrelevant_targeted_refresh_preserves_joined_snapshot():
    adapter = _make_adapter()
    adapter._joined_channel_ids = {"prior-channel"}
    adapter._channel_state = {
        "prior-channel": {
            "chat_type": "group",
            "last_ts": 1,
            "seen": OrderedDict(),
        }
    }
    adapter._channel_names = {"prior-channel": "Prior"}
    adapter._channel_meta = {
        "prior-channel": {
            "channel_id": "prior-channel",
            "type": "community",
            "name": "Prior",
        }
    }
    cli = _ScriptedCli()
    cli.script(
        "channels",
        "list",
        [{"channel_id": "other-channel", "type": "community", "name": "Other"}],
    )
    adapter._run_cli = cli

    result = await adapter._discover_joined_channels(
        target_channel_id="unrelated-channel"
    )

    assert result is None
    assert adapter._joined_channel_ids == {"prior-channel"}
    assert set(adapter._channel_state) == {"prior-channel"}


@pytest.mark.asyncio
async def test_explicit_channel_filter_keeps_new_dm_discovery_independent():
    adapter = _make_adapter(extra={"channels": [CHANNEL]})
    adapter._joined_channel_ids = {CHANNEL}
    adapter._channel_state = {
        CHANNEL: {"chat_type": "group", "last_ts": 1, "seen": OrderedDict()}
    }
    direct_dm = "direct-dm"
    fallback_dm = "fallback-dm"
    unconfigured_community = "unconfigured-community"
    cli = _ScriptedCli()
    cli.script("dms", "list", [{"dm_id": direct_dm}])
    cli.script(
        "channels",
        "list",
        [
            {
                "channel_id": fallback_dm,
                "type": "community",
                "name": "DM",
                "description": "",
            },
            {
                "channel_id": unconfigured_community,
                "type": "community",
                "name": "Other",
                "description": "Community channel",
            },
        ],
    )
    adapter._run_cli = cli

    await adapter._discover_dms(seed=False)

    assert set(adapter._channel_state) == {CHANNEL, direct_dm, fallback_dm}
    assert adapter._channel_state[direct_dm]["chat_type"] == "dm"
    assert adapter._channel_state[fallback_dm]["chat_type"] == "group"
    assert unconfigured_community not in adapter._channel_state


@pytest.mark.asyncio
async def test_membership_subscription_and_reconciliation_cover_add_and_remove():
    adapter = _make_adapter()
    adapter._self_pubkey = SELF_PUBKEY
    adapter._channel_state = {
        "departed": {"chat_type": "group", "last_ts": 1, "seen": OrderedDict()}
    }

    class WebSocket:
        def __init__(self):
            self.frames = []

        async def send(self, frame):
            self.frames.append(json.loads(frame))

    websocket = WebSocket()
    subscriptions = await adapter._subscribe_websocket(websocket)
    membership_request = websocket.frames[-1]
    assert membership_request[2]["kinds"] == [
        _buzz_mod._WS_MEMBERSHIP_KIND,
        _buzz_mod._WS_MEMBERSHIP_REMOVED_KIND,
    ]

    async def discover_joined_channels(**_kwargs):
        adapter._channel_state.pop("departed")
        adapter._channel_state["joined"] = {
            "chat_type": "group",
            "last_ts": 50,
            "seen": OrderedDict(),
        }
        return True

    async def discover_dms(*, seed):
        assert "joined" in subscriptions.values()

    adapter._discover_joined_channels = discover_joined_channels
    adapter._discover_dms = discover_dms
    websocket.frames.clear()

    await adapter._handle_membership_event(
        websocket,
        subscriptions,
        {"created_at": 50, "kind": 44101, "tags": [["h", "departed"]]},
    )

    assert "departed" not in subscriptions.values()
    assert "joined" in subscriptions.values()
    assert any(frame[:2] == ["CLOSE", "hermes-buzz-0"] for frame in websocket.frames)
    assert any(frame[0] == "REQ" and frame[2]["#h"] == ["joined"] for frame in websocket.frames)


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

    @pytest.mark.parametrize(
        ("returncode", "retryable", "expected"),
        [(1, True, True), (2, False, False)],
    )
    def test_structured_retryable_field_overrides_exit_code(
        self, returncode, retryable, expected
    ):
        stderr = json.dumps(
            {"error": "relay_error", "message": "production failure", "retryable": retryable}
        )
        assert _buzz_mod._cli_failure_is_retryable(stderr, returncode) is expected

    @pytest.mark.parametrize(
        ("stderr", "returncode", "expected"),
        [("relay unavailable", 2, True), ("not-json", 1, False), ("{", 2, True)],
    )
    def test_legacy_or_malformed_stderr_uses_exit_code_fallback(
        self, stderr, returncode, expected
    ):
        assert _buzz_mod._cli_failure_is_retryable(stderr, returncode) is expected


@pytest.mark.asyncio
async def test_top_level_channel_message_becomes_its_stable_thread_root():
    adapter = _make_adapter()
    received = []

    async def capture(event):
        received.append(event)

    adapter.set_message_handler(capture)
    adapter.handle_message = capture

    await adapter._dispatch_message(
        text="hello",
        chat_id=CHANNEL,
        chat_type="group",
        user_id=OTHER_PUBKEY,
        user_name="Alice",
        message_id="root-event",
        created_at=1,
    )

    assert len(received) == 1
    assert received[0].source.thread_id == "root-event"


@pytest.mark.asyncio
async def test_channel_reply_keeps_the_explicit_thread_root():
    adapter = _make_adapter()
    received = []

    async def capture(event):
        received.append(event)

    adapter.set_message_handler(capture)
    adapter.handle_message = capture

    await adapter._dispatch_message(
        text="follow-up",
        chat_id=CHANNEL,
        chat_type="group",
        user_id=OTHER_PUBKEY,
        user_name="Alice",
        message_id="reply-event",
        created_at=2,
        thread_id="root-event",
    )

    assert len(received) == 1
    assert received[0].source.thread_id == "root-event"


@pytest.mark.asyncio
async def test_nip10_root_tag_reaches_plugin_dispatch_as_thread_root():
    adapter = _make_adapter(extra={"require_mention": False})
    adapter.require_mention = False
    adapter._user_names[OTHER_PUBKEY] = "Alice"
    received = []

    async def capture(event):
        received.append(event)

    adapter.set_message_handler(capture)
    adapter.handle_message = capture
    state = {"seen": OrderedDict(), "last_ts": 0, "chat_type": "group"}
    event = {
        "id": "reply-event",
        "kind": 9,
        "pubkey": OTHER_PUBKEY,
        "content": "follow-up",
        "created_at": 2,
        "tags": [
            ["h", CHANNEL],
            ["e", "root-event", "", "root"],
            ["e", "parent-event", "", "reply"],
        ],
    }

    await adapter._handle_event(CHANNEL, state, event)

    assert len(received) == 1
    assert received[0].source.thread_id == "root-event"


@pytest.mark.asyncio
async def test_legacy_two_unmarked_e_tags_use_first_as_stable_root():
    adapter = _make_adapter(extra={"require_mention": False})
    adapter.require_mention = False
    adapter._user_names[OTHER_PUBKEY] = "Alice"
    received = []

    async def capture(event):
        received.append(event)

    adapter.set_message_handler(capture)
    adapter.handle_message = capture
    state = {"seen": OrderedDict(), "last_ts": 0, "chat_type": "group"}

    await adapter._handle_event(
        CHANNEL,
        state,
        {
            "id": "nested-reply",
            "kind": 9,
            "pubkey": OTHER_PUBKEY,
            "content": "legacy positional reply",
            "created_at": 3,
            "tags": [["e", "root-event"], ["e", "parent-event"]],
        },
    )

    assert received[0].source.thread_id == "root-event"


@pytest.mark.asyncio
async def test_newest_first_seed_canonicalizes_legacy_positional_thread_root():
    adapter = _make_adapter()
    cli = _ScriptedCli()
    cli.script(
        "messages",
        "get",
        [
            {
                "id": "nested-reply",
                "created_at": 3,
                "kind": 9,
                "tags": [["e", "root-event"], ["e", "parent-event"]],
            },
            {
                "id": "parent-event",
                "created_at": 2,
                "kind": 9,
                "tags": [["e", "root-event"]],
            },
        ],
    )
    cli.script("messages", "send", {"accepted": True, "event_id": "answer"})
    adapter._run_cli = cli

    await adapter._seed_channel(CHANNEL, chat_type="group")
    result = await adapter.send(CHANNEL, "answer", reply_to="nested-reply")

    assert result.success is True
    assert adapter._channel_state[CHANNEL]["thread_roots"]["nested-reply"] == "root-event"
    send_args, _stdin = cli.calls[-1]
    assert send_args[send_args.index("--reply-to") + 1] == "root-event"


@pytest.mark.asyncio
async def test_nested_reply_send_targets_original_nip10_root():
    adapter = _make_adapter(extra={"require_mention": False})
    adapter.require_mention = False
    adapter._user_names[OTHER_PUBKEY] = "Alice"
    state = {"seen": OrderedDict(), "last_ts": 0, "chat_type": "group"}
    adapter._channel_state[CHANNEL] = state

    for event in (
        {
            "id": "first-reply",
            "kind": 9,
            "pubkey": OTHER_PUBKEY,
            "content": "first",
            "created_at": 2,
            "tags": [["e", "root-event", "", "root"]],
        },
        {
            "id": "nested-reply",
            "kind": 9,
            "pubkey": OTHER_PUBKEY,
            "content": "nested",
            "created_at": 3,
            "tags": [["e", "first-reply", "", "reply"]],
        },
    ):
        await adapter._handle_event(CHANNEL, state, event)

    cli = _ScriptedCli()
    cli.script("messages", "send", {"accepted": True, "event_id": "agent-reply"})
    adapter._run_cli = cli

    await adapter.send(CHANNEL, "answer", reply_to="nested-reply")

    assert cli.calls[-1][0][-2:] == ["--reply-to", "root-event"]


@pytest.mark.asyncio
async def test_seed_reconstructs_nested_reply_root_without_dispatch():
    adapter = _make_adapter(extra={"require_mention": False})
    adapter.require_mention = False
    received = []

    async def capture(event):
        received.append(event)

    adapter.set_message_handler(capture)
    adapter.handle_message = capture
    cli = _ScriptedCli()
    cli.script(
        "messages",
        "get",
        [
            {
                "id": "first-reply",
                "kind": 9,
                "pubkey": OTHER_PUBKEY,
                "content": "first",
                "created_at": 2,
                "tags": [["e", "root-event", "", "root"]],
            },
            {
                "id": "nested-reply",
                "kind": 9,
                "pubkey": OTHER_PUBKEY,
                "content": "nested",
                "created_at": 3,
                "tags": [["e", "first-reply", "", "reply"]],
            },
        ],
    )
    adapter._run_cli = cli

    await adapter._seed_channel(CHANNEL, chat_type="group")
    cli.script("messages", "send", {"accepted": True, "event_id": "agent-reply"})
    await adapter.send(CHANNEL, "answer", reply_to="nested-reply")

    assert received == []
    assert cli.calls[-1][0][-2:] == ["--reply-to", "root-event"]


def test_event_to_root_mapping_is_bounded():
    adapter = _make_adapter()
    state = {"seen": OrderedDict(), "last_ts": 0, "chat_type": "group"}

    for index in range(_buzz_mod._SEEN_CAP + 1):
        adapter._remember_thread_root(state, f"event-{index}", "root-event")

    assert len(state["thread_roots"]) == _buzz_mod._SEEN_CAP
    assert "event-0" not in state["thread_roots"]


@pytest.mark.asyncio
async def test_top_level_dm_uses_message_anchor_without_fragmenting_dm_session():
    adapter = _make_adapter()
    received = []

    async def capture(event):
        received.append(event)

    adapter.set_message_handler(capture)
    adapter.handle_message = capture

    await adapter._dispatch_message(
        text="hello",
        chat_id=DM_CHANNEL,
        chat_type="dm",
        user_id=OTHER_PUBKEY,
        user_name="Alice",
        message_id="dm-root",
        created_at=1,
    )

    assert len(received) == 1
    assert received[0].source.thread_id is None
    assert received[0].message_id == "dm-root"


@pytest.mark.asyncio
async def test_dm_follow_up_keeps_the_original_root():
    adapter = _make_adapter()
    adapter._user_names[OTHER_PUBKEY] = "Alice"
    received = []

    async def capture(event):
        received.append(event)

    adapter.set_message_handler(capture)
    adapter.handle_message = capture
    state = {"seen": OrderedDict(), "last_ts": 0, "chat_type": "dm"}
    event = {
        "id": "dm-follow-up",
        "kind": 9,
        "pubkey": OTHER_PUBKEY,
        "content": "more",
        "created_at": 2,
        "tags": [
            ["e", "dm-root", "", "root"],
            ["e", "dm-parent", "", "reply"],
        ],
    }

    await adapter._handle_event(DM_CHANNEL, state, event)

    assert len(received) == 1
    assert received[0].source.thread_id == "dm-root"


@pytest.mark.asyncio
async def test_concurrent_channel_messages_cannot_exchange_thread_roots():
    adapter = _make_adapter(extra={"require_mention": False})
    adapter.require_mention = False
    received = []
    both_resolving = asyncio.Event()
    resolver_count = 0

    async def resolve_name(_pubkey):
        nonlocal resolver_count
        resolver_count += 1
        if resolver_count == 2:
            both_resolving.set()
        await both_resolving.wait()
        return "Alice"

    async def capture(event):
        received.append(event)

    adapter._resolve_user_name = resolve_name
    adapter.set_message_handler(capture)
    adapter.handle_message = capture

    def inbound(message_id, root_id, sender):
        return {
            "id": message_id,
            "kind": 9,
            "pubkey": sender,
            "content": "follow-up",
            "created_at": 2,
            "tags": [["e", root_id, "", "root"]],
        }

    await asyncio.gather(
        adapter._handle_event(
            "channel-a",
            {"seen": OrderedDict(), "last_ts": 0, "chat_type": "group"},
            inbound("reply-a", "root-a", "a" * 64),
        ),
        adapter._handle_event(
            "channel-b",
            {"seen": OrderedDict(), "last_ts": 0, "chat_type": "group"},
            inbound("reply-b", "root-b", "c" * 64),
        ),
    )

    roots = {event.message_id: event.source.thread_id for event in received}
    assert roots == {"reply-a": "root-a", "reply-b": "root-b"}


@pytest.mark.asyncio
async def test_repository_path_containing_display_name_does_not_address_agent():
    adapter = _make_adapter(extra={"require_mention": True})
    adapter.require_mention = True
    adapter._user_names[OTHER_PUBKEY] = "Alice"
    received = []

    async def capture(event):
        received.append(event)

    adapter.set_message_handler(capture)
    adapter.handle_message = capture
    state = {"seen": OrderedDict(), "last_ts": 0, "chat_type": "group"}

    await adapter._handle_event(
        CHANNEL,
        state,
        {
            "id": "path-message",
            "kind": 9,
            "pubkey": OTHER_PUBKEY,
            "content": "Inspect /srv/example-project before release.",
            "created_at": 1,
            "tags": [],
        },
    )

    assert received == []

    await adapter._handle_event(
        CHANNEL,
        state,
        {
            "id": "explicit-mention",
            "kind": 9,
            "pubkey": OTHER_PUBKEY,
            "content": "@Chip inspect the release.",
            "created_at": 2,
            "tags": [],
        },
    )

    assert [event.message_id for event in received] == ["explicit-mention"]



@pytest.mark.asyncio
async def test_protected_relay_image_is_authenticated_cached_and_dispatched(
    monkeypatch, tmp_path
):
    image_bytes = b"\x89PNG\r\n\x1a\nprotected-image"
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    image_url = f"https://relay.invalid/media/{image_hash}.png"
    monkeypatch.setenv("IMAGE_CACHE_DIR", str(tmp_path / "images"))
    adapter = _make_adapter(
        extra={
            "relay_url": "https://relay.invalid",
            "allowed_users": [OTHER_PUBKEY],
            "require_mention": True,
        }
    )
    received = []
    calls = []

    async def run_cli(args, **_kwargs):
        calls.append(list(args))
        assert args[:3] == ["media", "get", image_url]
        Path(args[args.index("--output") + 1]).write_bytes(image_bytes)
        return 0, "", ""

    async def capture(event):
        received.append(event)

    adapter._run_cli = run_cli
    adapter._user_names[OTHER_PUBKEY] = "Alice"
    adapter.set_message_handler(capture)
    adapter.handle_message = capture
    state = {"chat_type": "group", "last_ts": 0, "seen": OrderedDict()}

    await adapter._handle_event(
        CHANNEL,
        state,
        {
            "id": "image-event",
            "kind": 9,
            "pubkey": OTHER_PUBKEY,
            "content": f"@Chip inspect this\n![image]({image_url})",
            "created_at": 1,
            "tags": [],
        },
    )

    assert len(received) == 1
    assert received[0].message_type is _buzz_mod.MessageType.PHOTO
    assert received[0].source.thread_id == "image-event"
    assert received[0].text == "inspect this"
    assert received[0].media_types == ["image/png"]
    assert len(received[0].media_urls) == 1
    cached_path = Path(received[0].media_urls[0])
    assert cached_path.read_bytes() == image_bytes
    assert stat.S_IMODE(cached_path.stat().st_mode) == 0o600
    media_calls = [call for call in calls if call[:2] == ["media", "get"]]
    assert media_calls == [
        ["media", "get", image_url, "--output", media_calls[0][-1]]
    ]
    assert not Path(calls[0][-1]).exists()


@pytest.mark.asyncio
async def test_chmod_failure_unlinks_uncommitted_protected_cache(monkeypatch, tmp_path):
    image_bytes = b"\x89PNG\r\n\x1a\nprotected-image"
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    image_url = f"https://relay.invalid/media/{image_hash}.png"
    adapter = _make_adapter(extra={"relay_url": "https://relay.invalid"})
    cached_path = tmp_path / "cached.png"
    cli_temp_paths = []

    async def run_cli(args, **_kwargs):
        temporary_path = Path(args[args.index("--output") + 1])
        cli_temp_paths.append(temporary_path)
        temporary_path.write_bytes(image_bytes)
        return 0, "", ""

    def cache_image(data, *, ext):
        assert data == image_bytes
        cached_path.write_bytes(data)
        return str(cached_path)

    def fail_chmod(path, mode):
        assert Path(path) == cached_path
        assert mode == 0o600
        raise OSError("chmod failed")

    monkeypatch.setattr(_buzz_mod, "cache_image_from_bytes", cache_image)
    monkeypatch.setattr(_buzz_mod.os, "chmod", fail_chmod)
    adapter._run_cli = run_cli

    original = f"inspect ![image]({image_url})"
    text, media_urls, media_types = await adapter._ingest_buzz_images(original)

    assert text == original
    assert media_urls == []
    assert media_types == []
    assert not cached_path.exists()
    assert cli_temp_paths and not cli_temp_paths[0].exists()


@pytest.mark.parametrize(
    "url",
    [
        "https://example.invalid/media/" + ("a" * 64) + ".png",
        "https://user@relay.invalid/media/" + ("a" * 64) + ".png",
        "https://relay.invalid/media/" + ("a" * 64) + ".png?token=secret",
        "https://relay.invalid/media/" + ("a" * 64) + ".png#fragment",
        "https://relay.invalid/media/" + ("a" * 64) + ".svg",
    ],
)
def test_only_exact_relay_image_urls_are_eligible_for_authenticated_fetch(url):
    adapter = _make_adapter(extra={"relay_url": "https://relay.invalid"})
    assert adapter._buzz_image_metadata(url) is None


def test_wss_relay_accepts_same_origin_https_protected_image():
    digest = "a" * 64
    adapter = _make_adapter(extra={"relay_url": "wss://relay.invalid"})

    assert adapter._buzz_image_metadata(
        f"https://relay.invalid/media/{digest}.png"
    ) == (digest, ".png", "image/png")


def test_buzz_docs_state_protected_images_require_authorized_senders():
    docs = (
        Path(__file__).parents[2]
        / "website/docs/user-guide/messaging/buzz.md"
    ).read_text()

    assert "Protected images are fetched only for authorized senders" in docs


@pytest.mark.asyncio
async def test_retryable_protected_image_failure_retries_before_dispatch(
    monkeypatch, tmp_path
):
    image_bytes = b"\x89PNG\r\n\x1a\nprotected-after-retry"
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    image_url = f"https://relay.invalid/media/{image_hash}.png"
    monkeypatch.setenv("IMAGE_CACHE_DIR", str(tmp_path / "images"))
    monkeypatch.setattr(_buzz_mod, "_BUZZ_MEDIA_RETRY_DELAYS", (0,))
    adapter = _make_adapter(extra={"relay_url": "https://relay.invalid"})
    attempts = 0

    async def run_cli(args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return (
                4,
                "",
                '{"error":"server_error","message":"unavailable","retryable":true}',
            )
        Path(args[args.index("--output") + 1]).write_bytes(image_bytes)
        return 0, "", ""

    adapter._run_cli = run_cli
    text, media_urls, media_types = await adapter._ingest_buzz_images(
        f"inspect\n![image]({image_url})"
    )

    assert attempts == 2
    assert text == "inspect"
    assert media_types == ["image/png"]
    assert Path(media_urls[0]).read_bytes() == image_bytes


@pytest.mark.asyncio
async def test_protected_media_ingest_has_one_bounded_message_deadline(monkeypatch):
    image_url = "https://relay.invalid/media/" + ("a" * 64) + ".png"
    monkeypatch.setattr(_buzz_mod, "_BUZZ_MEDIA_TOTAL_TIMEOUT", 0.01)
    adapter = _make_adapter(extra={"relay_url": "https://relay.invalid"})
    output_paths = []

    async def run_cli(args, **_kwargs):
        output_paths.append(Path(args[args.index("--output") + 1]))
        await asyncio.sleep(60)

    adapter._run_cli = run_cli
    original = f"inspect\n![image]({image_url})"
    text, media_urls, media_types = await adapter._ingest_buzz_images(original)

    assert text == original
    assert media_urls == []
    assert media_types == []
    assert output_paths and all(not path.exists() for path in output_paths)


@pytest.mark.asyncio
async def test_protected_image_downloads_are_bounded_per_message(monkeypatch, tmp_path):
    payloads = [
        b"\x89PNG\r\n\x1a\nimage-" + str(index).encode() for index in range(5)
    ]
    urls = [
        f"https://relay.invalid/media/{hashlib.sha256(data).hexdigest()}.png"
        for data in payloads
    ]
    payload_by_url = dict(zip(urls, payloads, strict=True))
    monkeypatch.setenv("IMAGE_CACHE_DIR", str(tmp_path / "images"))
    adapter = _make_adapter(extra={"relay_url": "https://relay.invalid"})
    calls = []

    async def run_cli(args, **_kwargs):
        url = args[2]
        calls.append(url)
        Path(args[args.index("--output") + 1]).write_bytes(payload_by_url[url])
        return 0, "", ""

    adapter._run_cli = run_cli
    original = "inspect\n" + "\n".join(f"![image]({url})" for url in urls)
    text, media_urls, media_types = await adapter._ingest_buzz_images(original)

    assert calls == urls[:4]
    assert len(media_urls) == 4
    assert media_types == ["image/png"] * 4
    assert urls[4] in text
    assert all(url not in text for url in urls[:4])


@pytest.mark.asyncio
async def test_hash_mismatch_preserves_protected_image_link(monkeypatch, tmp_path):
    expected_url = "https://relay.invalid/media/" + ("a" * 64) + ".png"
    wrong_bytes = b"\x89PNG\r\n\x1a\nwrong-image"
    monkeypatch.setenv("IMAGE_CACHE_DIR", str(tmp_path / "images"))
    adapter = _make_adapter(extra={"relay_url": "https://relay.invalid"})

    async def run_cli(args, **_kwargs):
        Path(args[args.index("--output") + 1]).write_bytes(wrong_bytes)
        return 0, "", ""

    adapter._run_cli = run_cli
    original = f"inspect\n![image]({expected_url})"
    text, media_urls, media_types = await adapter._ingest_buzz_images(original)

    assert text == original
    assert media_urls == []
    assert media_types == []


@pytest.mark.asyncio
async def test_url_extension_must_match_downloaded_image_family(monkeypatch, tmp_path):
    png_bytes = b"\x89PNG\r\n\x1a\nactual-png"
    image_hash = hashlib.sha256(png_bytes).hexdigest()
    image_url = f"https://relay.invalid/media/{image_hash}.gif"
    monkeypatch.setenv("IMAGE_CACHE_DIR", str(tmp_path / "images"))
    adapter = _make_adapter(extra={"relay_url": "https://relay.invalid"})

    async def run_cli(args, **_kwargs):
        Path(args[args.index("--output") + 1]).write_bytes(png_bytes)
        return 0, "", ""

    adapter._run_cli = run_cli
    original = f"inspect\n![image]({image_url})"
    text, media_urls, media_types = await adapter._ingest_buzz_images(original)

    assert text == original
    assert media_urls == []
    assert media_types == []


@pytest.mark.asyncio
async def test_jpg_extension_accepts_detected_jpeg_family(monkeypatch, tmp_path):
    jpeg_bytes = b"\xff\xd8\xff\xe0actual-jpeg"
    image_hash = hashlib.sha256(jpeg_bytes).hexdigest()
    image_url = f"https://relay.invalid/media/{image_hash}.jpg"
    monkeypatch.setenv("IMAGE_CACHE_DIR", str(tmp_path / "images"))
    adapter = _make_adapter(extra={"relay_url": "https://relay.invalid"})

    async def run_cli(args, **_kwargs):
        Path(args[args.index("--output") + 1]).write_bytes(jpeg_bytes)
        return 0, "", ""

    adapter._run_cli = run_cli
    text, media_urls, media_types = await adapter._ingest_buzz_images(
        f"inspect\n![image]({image_url})"
    )

    assert text == "inspect"
    assert len(media_urls) == 1
    assert Path(media_urls[0]).read_bytes() == jpeg_bytes
    assert media_types == ["image/jpeg"]


@pytest.mark.asyncio
async def test_oversized_protected_image_is_not_cached_or_removed(monkeypatch, tmp_path):
    image_bytes = b"\x89PNG\r\n\x1a\noversized"
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    image_url = f"https://relay.invalid/media/{image_hash}.png"
    monkeypatch.setenv("IMAGE_CACHE_DIR", str(tmp_path / "images"))
    monkeypatch.setattr(
        _buzz_mod,
        "validate_inbound_media_size",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("too large")),
    )
    adapter = _make_adapter(extra={"relay_url": "https://relay.invalid"})

    async def run_cli(args, **_kwargs):
        Path(args[args.index("--output") + 1]).write_bytes(image_bytes)
        return 0, "", ""

    adapter._run_cli = run_cli
    original = f"inspect\n![image]({image_url})"
    text, media_urls, media_types = await adapter._ingest_buzz_images(original)

    assert text == original
    assert media_urls == []
    assert media_types == []
    assert not (tmp_path / "images").exists()


@pytest.mark.asyncio
async def test_unauthorized_image_message_is_not_downloaded():
    sender = "a" * 64
    image_url = "https://relay.invalid/media/" + ("b" * 64) + ".png"
    adapter = _make_adapter(
        extra={
            "relay_url": "https://relay.invalid",
            "allowed_users": ["c" * 64],
            "require_mention": True,
        }
    )
    adapter._user_names[sender] = "Alice"
    received = []

    async def run_cli(args, **_kwargs):
        raise AssertionError(f"unauthorized media must not be fetched: {args}")

    async def capture(event):
        received.append(event)

    adapter._run_cli = run_cli
    adapter.set_message_handler(capture)
    adapter.handle_message = capture
    state = {"chat_type": "group", "last_ts": 0, "seen": OrderedDict()}

    await adapter._handle_event(
        CHANNEL,
        state,
        {
            "id": "unauthorized-image",
            "kind": 9,
            "pubkey": sender,
            "content": f"@Chip inspect\n![image]({image_url})",
            "created_at": 1,
            "tags": [],
        },
    )

    assert received == []

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

    def test_configured_agent_handoff_accepts_npub_identity(self):
        target_pubkey = "d" * 64
        adapter = _make_adapter(
            extra={
                "outbound_mention_pubkeys": {
                    "ClaudeCode": _buzz_mod.hex_to_npub(target_pubkey)
                }
            }
        )

        assert adapter.outbound_mention_pubkeys == {
            "claudecode": ("ClaudeCode", target_pubkey)
        }

    def test_configured_handoffs_reject_invalid_or_ambiguous_configuration(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            _make_adapter(extra={"outbound_mention_pubkeys": ["not", "a", "mapping"]})

        with pytest.raises(ValueError, match="duplicate display name"):
            _make_adapter(
                extra={
                    "outbound_mention_pubkeys": {
                        "ClaudeCode": "c" * 64,
                        "claudecode": "d" * 64,
                    }
                }
            )

    @pytest.mark.asyncio
    async def test_configured_agent_handoff_uses_exact_structural_pubkey(self):
        target_pubkey = "c" * 64
        adapter = _make_adapter(
            extra={"outbound_mention_pubkeys": {"ClaudeCode": target_pubkey}}
        )
        cli = _ScriptedCli()
        cli.script(
            "messages",
            "send",
            {"accepted": True, "event_id": "handoff-event"},
        )
        adapter._run_cli = cli

        result = await adapter.send(
            CHANNEL,
            "@ClaudeCode review this target.",
            metadata={"thread_id": "root-event"},
        )

        assert result.success is True
        assert len(cli.calls) == 1
        args, stdin_text = cli.calls[0]
        assert args[args.index("--mention") + 1] == target_pubkey
        assert args[args.index("--reply-to") + 1] == "root-event"
        assert stdin_text == "@ClaudeCode review this target."

    @pytest.mark.asyncio
    async def test_configured_handoff_is_never_downgraded_to_plain_text(self):
        target_pubkey = "c" * 64
        adapter = _make_adapter(
            extra={"outbound_mention_pubkeys": {"ClaudeCode": target_pubkey}}
        )
        calls = []

        async def run_cli(args, *, input_text=None):
            calls.append((list(args), input_text))
            return (
                1,
                "",
                "mention '@ClaudeCode' does not match a current channel member",
            )

        adapter._run_cli = run_cli

        result = await adapter.send(CHANNEL, "@ClaudeCode review this target.")

        assert result.success is False
        assert len(calls) == 1
        assert calls[0][0][calls[0][0].index("--mention") + 1] == target_pubkey
        assert calls[0][1] == "@ClaudeCode review this target."

    @pytest.mark.asyncio
    async def test_configured_handoff_protects_itself_while_falling_back_other_mentions(self):
        target_pubkey = "c" * 64
        adapter = _make_adapter(
            extra={"outbound_mention_pubkeys": {"ClaudeCode": target_pubkey}}
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
            {"accepted": True, "event_id": "mixed-event"},
        )
        adapter._run_cli = cli

        result = await adapter.send(
            CHANNEL,
            "@ClaudeCode ask @ghost to review.",
            reply_to="root-event",
        )

        assert result.success is True
        assert len(cli.calls) == 2
        assert cli.calls[0][0] == cli.calls[1][0]
        assert cli.calls[0][0][cli.calls[0][0].index("--mention") + 1] == target_pubkey
        assert cli.calls[0][1] == "@ClaudeCode ask @ghost to review."
        assert cli.calls[1][1] == "@ClaudeCode ask ghost to review."

    @pytest.mark.asyncio
    async def test_configured_handoff_requires_a_complete_display_name(self):
        adapter = _make_adapter(
            extra={"outbound_mention_pubkeys": {"Claude": "c" * 64}}
        )
        cli = _ScriptedCli()
        cli.script(
            "messages",
            "send",
            {"accepted": True, "event_id": "plain-event"},
        )
        adapter._run_cli = cli

        result = await adapter.send(CHANNEL, "@ClaudeCode review this target.")

        assert result.success is True
        assert "--mention" not in cli.calls[0][0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("returncode", "retryable"), [(1, True), (2, False)]
    )
    async def test_send_uses_structured_retryability_over_exit_code(
        self, returncode, retryable
    ):
        adapter = _make_adapter()

        async def run_cli(args, *, input_text=None):
            return (
                returncode,
                "",
                json.dumps(
                    {
                        "error": "relay_error",
                        "message": "production failure",
                        "retryable": retryable,
                    }
                ),
            )

        adapter._run_cli = run_cli

        result = await adapter.send(CHANNEL, "hello")

        assert result.success is False
        assert result.retryable is retryable

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
    async def test_unresolved_outbound_mention_retries_as_readable_text(self):
        adapter = _make_adapter()
        cli = _ScriptedCli()
        cli.script(
            "messages",
            "send",
            "",
            code=1,
            stderr="user_error: mention '@ghost' does not match a current channel member",
        )
        cli.script(
            "messages",
            "send",
            {"accepted": True, "event_id": "delivered"},
        )
        adapter._run_cli = cli

        result = await adapter.send(
            CHANNEL,
            "Ask @ghost to review, but preserve bob@example.com.",
            metadata={"thread_id": "root-event"},
        )

        assert result.success is True
        assert [text for _args, text in cli.calls] == [
            "Ask @ghost to review, but preserve bob@example.com.",
            "Ask ghost to review, but preserve bob@example.com.",
        ]
        assert cli.calls[0][0] == cli.calls[1][0]
        retry_args = cli.calls[1][0]
        assert retry_args[retry_args.index("--reply-to") + 1] == "root-event"

    @pytest.mark.asyncio
    async def test_sequential_unresolved_mentions_are_bounded_and_delivered(self):
        adapter = _make_adapter()
        cli = _ScriptedCli()
        cli.script(
            "messages",
            "send",
            "",
            code=1,
            stderr="mention '@first' does not match a current channel member",
        )
        cli.script(
            "messages",
            "send",
            "",
            code=1,
            stderr="mention '@second' is ambiguous",
        )
        cli.script(
            "messages",
            "send",
            {"accepted": True, "event_id": "delivered"},
        )
        adapter._run_cli = cli

        result = await adapter.send(CHANNEL, "Ask @first, then @second please.")

        assert result.success is True
        assert [text for _args, text in cli.calls] == [
            "Ask @first, then @second please.",
            "Ask first, then @second please.",
            "Ask first, then second please.",
        ]

    @pytest.mark.asyncio
    async def test_mention_limit_fallback_neutralizes_only_mention_markers(self):
        adapter = _make_adapter()
        cli = _ScriptedCli()
        cli.script(
            "messages",
            "send",
            "",
            code=1,
            stderr="user_error: too many unique message mentions",
        )
        cli.script(
            "messages",
            "send",
            {"accepted": True, "event_id": "delivered"},
        )
        adapter._run_cli = cli

        result = await adapter.send(
            CHANNEL,
            "Notify @Alice and @Bob; preserve bob@example.com.",
        )

        assert result.success is True
        assert [text for _args, text in cli.calls] == [
            "Notify @Alice and @Bob; preserve bob@example.com.",
            "Notify Alice and Bob; preserve bob@example.com.",
        ]

    @pytest.mark.asyncio
    async def test_local_image_caption_retries_unresolved_mention(self, tmp_path):
        image = tmp_path / "report.png"
        image.write_bytes(b"not-a-real-image")
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
            {"accepted": True, "event_id": "image-event"},
        )
        adapter._run_cli = cli

        result = await adapter.send_image_file(
            CHANNEL,
            str(image),
            caption="Evidence for @ghost to review",
            reply_to="root-event",
        )

        assert result.success is True
        assert [text for _args, text in cli.calls] == [
            "Evidence for @ghost to review",
            "Evidence for ghost to review",
        ]
        assert cli.calls[0][0] == cli.calls[1][0]
        assert cli.calls[0][0][cli.calls[0][0].index("--file") + 1] == str(image)

    @pytest.mark.asyncio
    async def test_missing_local_image_file_uses_safe_notice_without_path_leak(
        self, tmp_path
    ):
        missing = tmp_path / "private-render.png"
        adapter = _make_adapter()
        cli = _ScriptedCli()
        cli.script(
            "messages",
            "send",
            {"accepted": True, "event_id": "notice-event"},
        )
        adapter._run_cli = cli

        result = await adapter.send_image_file(
            CHANNEL,
            str(missing),
            caption="Evidence",
            metadata={"thread_id": "root-event"},
        )

        assert result.success is True
        assert len(cli.calls) == 1
        args, stdin_text = cli.calls[0]
        assert "--file" not in args
        assert args[args.index("--reply-to") + 1] == "root-event"
        assert stdin_text == "Evidence\n⚠️ Couldn't deliver the image attachment."
        assert str(missing) not in stdin_text

    @pytest.mark.asyncio
    async def test_image_batch_preserves_metadata_thread_root(self, tmp_path):
        image = tmp_path / "report.png"
        image.write_bytes(b"not-a-real-image")
        adapter = _make_adapter()
        cli = _ScriptedCli()
        cli.script(
            "messages",
            "send",
            {"accepted": True, "event_id": "image-event"},
        )
        adapter._run_cli = cli

        await adapter.send_multiple_images(
            CHANNEL,
            [(image.as_uri(), "Evidence")],
            metadata={"thread_id": "root-event"},
        )

        args, _stdin_text = cli.calls[0]
        assert args[args.index("--file") + 1] == str(image)
        assert args[args.index("--reply-to") + 1] == "root-event"


    @pytest.mark.asyncio
    async def test_local_image_caption_uses_configured_structural_mention(self, tmp_path):
        target_pubkey = "c" * 64
        image = tmp_path / "report.png"
        image.write_bytes(b"not-a-real-image")
        adapter = _make_adapter(
            extra={"outbound_mention_pubkeys": {"ClaudeCode": target_pubkey}}
        )
        cli = _ScriptedCli()
        cli.script("messages", "send", {"accepted": True, "event_id": "image-handoff"})
        adapter._run_cli = cli

        result = await adapter.send_image(
            CHANNEL,
            str(image),
            caption="@ClaudeCode review this image.",
        )

        assert result.success is True
        assert len(cli.calls) == 1
        args, stdin_text = cli.calls[0]
        assert args[args.index("--mention") + 1] == target_pubkey
        assert stdin_text == "@ClaudeCode review this image."

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
    @pytest.mark.parametrize(
        ("channels_result", "extra"),
        [
            ((2, "", '{"error":"relay_error","message":"down","retryable":true}'), {}),
            ((0, "{}", ""), {}),
            ((0, "[]", ""), {}),
            ((0, json.dumps([{"channel_id": "other", "type": "community"}]), ""), {"channels": [CHANNEL]}),
        ],
        ids=["cli-failure", "malformed-payload", "empty-roster", "nonmatching-roster"],
    )
    async def test_unsuccessful_post_lock_connect_releases_identity_lock(
        self, monkeypatch, channels_result, extra
    ):
        import gateway.status as gateway_status

        released = []
        monkeypatch.setattr(
            gateway_status, "acquire_scoped_lock", lambda platform, key: (True, None)
        )
        monkeypatch.setattr(
            gateway_status,
            "release_scoped_lock",
            lambda platform, key: released.append((platform, key)),
        )
        monkeypatch.setattr(_buzz_mod, "_resolve_private_key", lambda extra=None: "nsec1test")
        adapter = _make_adapter(extra=extra)
        adapter.cli_path = "/fake/buzz"

        async def run_cli(args, *, input_text=None):
            if args == ["users", "get"]:
                return 0, json.dumps([{"pubkey": SELF_PUBKEY, "display_name": "Chip"}]), ""
            if args == ["channels", "list", "--member"]:
                return channels_result
            if args in (["dms", "list"], ["channels", "list"]):
                return 0, "[]", ""
            raise AssertionError(args)

        adapter._run_cli = run_cli

        assert await adapter.connect() is False
        lock_key = f"{adapter.relay_url}:{SELF_PUBKEY}"
        assert released == [("buzz", lock_key)]
        assert adapter._lock_key is None

    @pytest.mark.asyncio
    async def test_cancelled_post_lock_connect_releases_identity_lock(self, monkeypatch):
        import gateway.status as gateway_status

        released = []
        monkeypatch.setattr(
            gateway_status, "acquire_scoped_lock", lambda platform, key: (True, None)
        )
        monkeypatch.setattr(
            gateway_status,
            "release_scoped_lock",
            lambda platform, key: released.append((platform, key)),
        )
        monkeypatch.setattr(_buzz_mod, "_resolve_private_key", lambda extra=None: "nsec1test")
        adapter = _make_adapter()
        adapter.cli_path = "/fake/buzz"

        async def run_cli(args, *, input_text=None):
            if args == ["users", "get"]:
                return 0, json.dumps([{"pubkey": SELF_PUBKEY}]), ""
            raise asyncio.CancelledError

        adapter._run_cli = run_cli

        with pytest.raises(asyncio.CancelledError):
            await adapter.connect()
        lock_key = f"{adapter.relay_url}:{SELF_PUBKEY}"
        assert released == [("buzz", lock_key)]
        assert adapter._lock_key is None

    @pytest.mark.asyncio
    async def test_raised_post_lock_seed_failure_releases_identity_lock(self, monkeypatch):
        import gateway.status as gateway_status

        released = []
        monkeypatch.setattr(
            gateway_status, "acquire_scoped_lock", lambda platform, key: (True, None)
        )
        monkeypatch.setattr(
            gateway_status,
            "release_scoped_lock",
            lambda platform, key: released.append((platform, key)),
        )
        monkeypatch.setattr(_buzz_mod, "_resolve_private_key", lambda extra=None: "nsec1test")
        adapter = _make_adapter()
        adapter.cli_path = "/fake/buzz"
        cli = _ScriptedCli()
        cli.script("users", "get", [{"pubkey": SELF_PUBKEY}])
        cli.script(
            "channels",
            "list",
            [{"channel_id": CHANNEL, "type": "community", "name": "General"}],
        )
        adapter._run_cli = cli
        adapter._seed_channel = AsyncMock(side_effect=OSError("seed failed"))

        with pytest.raises(OSError, match="seed failed"):
            await adapter.connect()
        lock_key = f"{adapter.relay_url}:{SELF_PUBKEY}"
        assert released == [("buzz", lock_key)]
        assert adapter._lock_key is None

    @pytest.mark.asyncio
    async def test_connect_accepts_direct_dm_only_account(self, monkeypatch):
        monkeypatch.setattr(_buzz_mod, "_resolve_private_key", lambda extra=None: "nsec1test")
        adapter = _make_adapter(extra={"transport": "poll"})
        adapter.cli_path = "/fake/buzz"
        dm_id = "direct-dm"
        cli = _ScriptedCli()
        cli.script("users", "get", [{"pubkey": SELF_PUBKEY}])
        cli.script("channels", "list", [])
        cli.script("dms", "list", [{"dm_id": dm_id}])
        cli.script("messages", "get", [])
        cli.script("channels", "list", [])
        adapter._run_cli = cli

        assert await adapter.connect() is True
        assert adapter._channel_state[dm_id]["chat_type"] == "dm"
        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_connect_accepts_dm_fallback_when_configured_community_is_absent(
        self, monkeypatch
    ):
        monkeypatch.setattr(_buzz_mod, "_resolve_private_key", lambda extra=None: "nsec1test")
        adapter = _make_adapter(
            extra={"transport": "poll", "channels": [CHANNEL]}
        )
        adapter.cli_path = "/fake/buzz"
        dm_id = "fallback-dm"
        cli = _ScriptedCli()
        cli.script("users", "get", [{"pubkey": SELF_PUBKEY}])
        cli.script(
            "channels",
            "list",
            [{"channel_id": "other", "type": "community", "name": "Other"}],
        )
        cli.script("dms", "list", [])
        cli.script(
            "channels",
            "list",
            [
                {
                    "channel_id": dm_id,
                    "type": "community",
                    "name": "DM",
                    "description": "",
                }
            ],
        )
        cli.script("messages", "get", [])
        adapter._run_cli = cli

        assert await adapter.connect() is True
        assert set(adapter._channel_state) == {dm_id}
        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_successful_connect_keeps_identity_lock_until_disconnect(self, monkeypatch):
        import gateway.status as gateway_status

        released = []
        monkeypatch.setattr(
            gateway_status, "acquire_scoped_lock", lambda platform, key: (True, None)
        )
        monkeypatch.setattr(
            gateway_status,
            "release_scoped_lock",
            lambda platform, key: released.append((platform, key)),
        )
        monkeypatch.setattr(_buzz_mod, "_resolve_private_key", lambda extra=None: "nsec1test")
        adapter = _make_adapter(extra={"transport": "poll"})
        adapter.cli_path = "/fake/buzz"
        cli = _ScriptedCli()
        cli.script("users", "get", [{"pubkey": SELF_PUBKEY, "display_name": "Chip"}])
        cli.script(
            "channels", "list", [{"channel_id": CHANNEL, "type": "community", "name": "General"}]
        )
        cli.script("messages", "get", [])
        cli.script("dms", "list", [])
        cli.script("channels", "list", [])
        adapter._run_cli = cli

        assert await adapter.connect() is True
        lock_key = f"{adapter.relay_url}:{SELF_PUBKEY}"
        assert adapter._lock_key == lock_key
        assert released == []

        await adapter.disconnect()
        assert released == [("buzz", lock_key)]

    @pytest.mark.asyncio
    async def test_connect_fails_when_identity_lock_held(self, monkeypatch):
        """A second profile using the same relay+pubkey must fail fast."""
        import gateway.status as gateway_status

        monkeypatch.setattr(
            gateway_status,
            "acquire_scoped_lock",
            lambda platform, key: (False, {"pid": 4242}),
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
        assert adapter.fatal_error_code == "lock_conflict"
        assert adapter.fatal_error_retryable is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("profile_output", "returncode", "expected_code", "retryable"),
        [
            ("", 0, "connect_failed", True),
            ("[]", 0, "connect_failed", True),
            (json.dumps([{"display_name": "Chip"}]), 0, "connect_failed", False),
            (json.dumps([{"pubkey": "abc"}]), 0, "connect_failed", False),
            (json.dumps([{"pubkey": "g" * 64}]), 0, "connect_failed", False),
            (json.dumps([{"pubkey": {"hex": SELF_PUBKEY}}]), 0, "connect_failed", False),
            (json.dumps(["not-an-object"]), 0, "connect_failed", False),
            ("not-json", 0, "connect_failed", False),
            ("{}", 0, "connect_failed", False),
            (
                '{"error":"auth_error","message":"denied","retryable":true}',
                1,
                "network_error",
                True,
            ),
            (
                '{"error":"relay_error","message":"down","retryable":false}',
                2,
                "connect_failed",
                False,
            ),
            ("relay unavailable", 2, "network_error", True),
            ("invalid credentials", 1, "connect_failed", False),
        ],
    )
    async def test_profile_failure_classification(
        self,
        monkeypatch,
        profile_output,
        returncode,
        expected_code,
        retryable,
    ):
        adapter = _make_adapter()
        adapter.cli_path = "/fake/buzz"
        monkeypatch.setattr(_buzz_mod, "_resolve_private_key", lambda extra=None: "nsec1test")

        async def run_cli(args, *, input_text=None):
            assert args == ["users", "get"]
            if returncode:
                return returncode, "", profile_output
            return 0, profile_output, ""

        adapter._run_cli = run_cli

        assert await adapter.connect() is False
        assert adapter.fatal_error_code == expected_code
        assert adapter.fatal_error_retryable is retryable

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("returncode", "stderr", "expected_code", "retryable"),
        [
            (
                1,
                '{"error":"relay_error","message":"down","retryable":true}',
                "network_error",
                True,
            ),
            (
                2,
                '{"error":"auth_error","message":"denied","retryable":false}',
                "connect_failed",
                False,
            ),
            (2, "relay unavailable", "network_error", True),
            (1, "invalid credentials", "connect_failed", False),
        ],
    )
    async def test_joined_channel_failure_classification(
        self,
        monkeypatch,
        returncode,
        stderr,
        expected_code,
        retryable,
    ):
        import gateway.status as gateway_status

        adapter = _make_adapter()
        adapter.cli_path = "/fake/buzz"
        monkeypatch.setattr(_buzz_mod, "_resolve_private_key", lambda extra=None: "nsec1test")
        monkeypatch.setattr(
            gateway_status,
            "acquire_scoped_lock",
            lambda platform, key: (True, None),
        )

        async def run_cli(args, *, input_text=None):
            if args == ["users", "get"]:
                return 0, json.dumps(
                    [{"pubkey": SELF_PUBKEY, "display_name": "Chip"}]
                ), ""
            assert args == ["channels", "list", "--member"]
            return returncode, "", stderr

        adapter._run_cli = run_cli

        assert await adapter.connect() is False
        assert adapter.fatal_error_code == expected_code
        assert adapter.fatal_error_retryable is retryable

    @pytest.mark.asyncio
    async def test_malformed_joined_channel_payload_is_non_retryable_connect_failure(
        self, monkeypatch
    ):
        import gateway.status as gateway_status

        adapter = _make_adapter()
        adapter.cli_path = "/fake/buzz"
        monkeypatch.setattr(_buzz_mod, "_resolve_private_key", lambda extra=None: "nsec1test")
        monkeypatch.setattr(
            gateway_status,
            "acquire_scoped_lock",
            lambda platform, key: (True, None),
        )

        async def run_cli(args, *, input_text=None):
            if args == ["users", "get"]:
                return 0, json.dumps(
                    [{"pubkey": SELF_PUBKEY, "display_name": "Chip"}]
                ), ""
            assert args == ["channels", "list", "--member"]
            return 0, "{}", ""

        adapter._run_cli = run_cli

        assert await adapter.connect() is False
        assert adapter.fatal_error_code == "connect_failed"
        assert adapter.fatal_error_retryable is False


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
        platform_hint = kwargs["platform_hint"]
        assert "existing local image" in platform_hint
        assert "MEDIA:/absolute/path" in platform_hint
        assert "do not use Computer Use" in platform_hint
        assert "document" not in platform_hint.lower()


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
    async def test_standalone_mention_fallback_keeps_thread_and_media(
        self, monkeypatch, tmp_path
    ):
        from gateway.config import PlatformConfig

        fake_cli = tmp_path / "buzz"
        fake_cli.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.setenv("BUZZ_RELAY_URL", "https://r")
        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1x")
        monkeypatch.setenv("BUZZ_CLI_PATH", str(fake_cli))
        calls = []
        outcomes = [
            (
                1,
                "",
                "mention '@ghost' does not match a current channel member",
            ),
            (0, json.dumps({"accepted": True, "event_id": "delivered"}), ""),
        ]

        async def fake_exec(
            cli_path,
            args,
            *,
            relay_url,
            private_key,
            input_text=None,
            timeout=30.0,
        ):
            calls.append((list(args), input_text))
            return outcomes.pop(0)

        monkeypatch.setattr(_buzz_mod, "_exec_buzz", fake_exec)

        result = await _standalone_send(
            PlatformConfig(enabled=True, extra={}),
            CHANNEL,
            "Ask @ghost to review.",
            thread_id="root-event",
            media_files=["report.png"],
            force_document=True,
        )

        assert result == {"success": True, "message_id": "delivered"}
        assert [call[1] for call in calls] == [
            "Ask @ghost to review.",
            "Ask ghost to review.",
        ]
        assert calls[0][0] == calls[1][0]
        assert calls[0][0][calls[0][0].index("--reply-to") + 1] == "root-event"
        assert calls[0][0][calls[0][0].index("--file") + 1] == "report.png"

    @pytest.mark.asyncio
    async def test_standalone_configured_handoff_keeps_exact_pubkey_and_media(
        self, monkeypatch, tmp_path
    ):
        from gateway.config import PlatformConfig

        target_pubkey = "c" * 64
        fake_cli = tmp_path / "buzz"
        fake_cli.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.setenv("BUZZ_RELAY_URL", "https://r")
        monkeypatch.setenv("BUZZ_PRIVATE_KEY", "nsec1x")
        monkeypatch.setenv("BUZZ_CLI_PATH", str(fake_cli))
        calls = []

        async def fake_exec(
            cli_path,
            args,
            *,
            relay_url,
            private_key,
            input_text=None,
            timeout=30.0,
        ):
            calls.append((list(args), input_text))
            return 0, json.dumps({"accepted": True, "event_id": "handoff"}), ""

        monkeypatch.setattr(_buzz_mod, "_exec_buzz", fake_exec)
        pconfig = PlatformConfig(
            enabled=True,
            extra={"outbound_mention_pubkeys": {"ClaudeCode": target_pubkey}},
        )

        result = await _standalone_send(
            pconfig,
            CHANNEL,
            "@ClaudeCode review this artifact.",
            thread_id="root-event",
            media_files=["report.png"],
            force_document=True,
        )

        assert result == {"success": True, "message_id": "handoff"}
        assert len(calls) == 1
        args, stdin_text = calls[0]
        assert args[args.index("--mention") + 1] == target_pubkey
        assert args[args.index("--reply-to") + 1] == "root-event"
        assert args[args.index("--file") + 1] == "report.png"
        assert stdin_text == "@ClaudeCode review this artifact."


