import asyncio
import json
from collections import OrderedDict
from unittest.mock import AsyncMock
import pytest
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_buzz_mod = load_plugin_adapter("buzz")
BuzzAdapter = _buzz_mod.BuzzAdapter
TEST_PRIVATE_KEY = "00" * 31 + "03"
SELF_PUBKEY = "f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9"
CHANNEL = "ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd"
OTHER_PUBKEY = "a" * 64


@pytest.fixture(autouse=True)
def _clean_transport_override(monkeypatch):
    monkeypatch.delenv("BUZZ_TRANSPORT", raising=False)
    monkeypatch.delenv("BUZZ_AUTH_TAG", raising=False)
    monkeypatch.delenv("BUZZ_CHANNELS", raising=False)


def _make_adapter(extra=None):
    from gateway.config import PlatformConfig

    cfg = PlatformConfig(enabled=True, extra={"relay_url": "https://test.relay", **(extra or {})})
    adapter = BuzzAdapter(cfg)
    adapter._self_pubkey = SELF_PUBKEY
    adapter._private_key = TEST_PRIVATE_KEY
    adapter._display_name = "Chip"
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
        [{"channel_id": "   ", "name": "Blank id"}],
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


@pytest.mark.asyncio
async def test_future_membership_timestamp_cannot_poison_reconnect_cursor(monkeypatch):
    adapter = _make_adapter()
    adapter._membership_since = 100
    monkeypatch.setattr(_buzz_mod.time, "time", lambda: 1_000)
    adapter._discover_joined_channels = AsyncMock(return_value=True)
    adapter._discover_dms = AsyncMock(return_value=None)

    class WebSocket:
        async def send(self, _frame):
            return None

    await adapter._handle_membership_event(
        WebSocket(),
        {},
        {"created_at": 9_999_999, "kind": 44100, "tags": []},
    )

    assert adapter._membership_since == 1_000
    adapter._discover_joined_channels.assert_awaited_once_with(
        since=1_000,
        target_channel_id="",
    )


@pytest.mark.asyncio
async def test_membership_reconciliation_applies_channel_rename():
    adapter = _make_adapter()
    adapter._channel_state = {
        CHANNEL: {"chat_type": "group", "last_ts": 0, "seen": {}}
    }
    adapter._joined_channel_ids = {CHANNEL}
    adapter._channel_names = {CHANNEL: "old name"}
    adapter._channel_meta = {
        CHANNEL: {"channel_id": CHANNEL, "name": "old name", "description": "group"}
    }
    adapter._run_cli = AsyncMock(
        return_value=(
            0,
            json.dumps(
                [
                    {
                        "channel_id": CHANNEL,
                        "name": "new name",
                        "description": "group",
                    }
                ]
            ),
            "",
        )
    )
    adapter._discover_dms = AsyncMock()
    websocket = AsyncMock()
    subscriptions = {"hermes-buzz-0": CHANNEL}

    await adapter._handle_ws_message(websocket, subscriptions, ["EVENT", _buzz_mod._WS_MEMBERSHIP_SUB_ID, {"created_at": 1234, "kind": 44100}])

    assert adapter._channel_names[CHANNEL] == "new name"
    assert adapter._channel_meta[CHANNEL]["name"] == "new name"


@pytest.mark.asyncio
async def test_membership_reconciliation_closes_removed_channel():
    adapter = _make_adapter()
    adapter._channel_state = {
        CHANNEL: {"chat_type": "group", "last_ts": 0, "seen": {}}
    }
    adapter._joined_channel_ids = {CHANNEL}
    adapter._channel_names = {CHANNEL: "general"}
    adapter._channel_meta = {
        CHANNEL: {"channel_id": CHANNEL, "name": "general", "description": "group"}
    }
    adapter._run_cli = AsyncMock(return_value=(0, "[]", ""))
    adapter._discover_dms = AsyncMock()
    websocket = AsyncMock()
    subscriptions = {"hermes-buzz-0": CHANNEL}

    await adapter._handle_ws_message(websocket, subscriptions, ["EVENT", _buzz_mod._WS_MEMBERSHIP_SUB_ID, {"created_at": 1234, "kind": 44101}])

    assert CHANNEL not in adapter._channel_state
    assert subscriptions == {}
    assert json.loads(websocket.send.await_args_list[0].args[0]) == [
        "CLOSE",
        "hermes-buzz-0",
    ]
    assert adapter._joined_channel_ids == set()



@pytest.mark.asyncio
async def test_membership_refresh_does_not_resurrect_departed_group_from_public_listing():
    adapter = _make_adapter()
    adapter._channel_state[CHANNEL] = adapter._new_channel_state("group")
    adapter._channel_meta[CHANNEL] = {"channel_id": CHANNEL, "name": "room"}
    adapter._joined_channel_ids = {CHANNEL}
    async def cli(args, **kwargs):
        if args == ["channels", "list", "--member"]:
            return 0, "[]", ""
        if args == ["channels", "list"]:
            return 0, json.dumps([{"channel_id": CHANNEL, "name": "room"}]), ""
        return 0, "[]", ""
    adapter._run_cli = cli
    await adapter._handle_ws_message(AsyncMock(), {"old": CHANNEL},
        ["EVENT", _buzz_mod._WS_MEMBERSHIP_SUB_ID, {"kind": 44101, "created_at": 1}])
    assert CHANNEL not in adapter._channel_state
    assert CHANNEL not in adapter._joined_channel_ids


@pytest.mark.asyncio
async def test_poll_cadence_reconciles_roster_before_dms(monkeypatch):
    adapter = _make_adapter()
    adapter._poll_count = 4
    calls = []
    async def sleep(_):
        if calls:
            raise asyncio.CancelledError
    monkeypatch.setattr(_buzz_mod.asyncio, "sleep", sleep)
    adapter._discover_joined_channels = AsyncMock(side_effect=lambda: calls.append("roster"))
    adapter._discover_dms = AsyncMock(side_effect=lambda **kwargs: calls.append("dms"))
    with pytest.raises(asyncio.CancelledError):
        await adapter._poll_loop()
    assert calls == ["roster", "dms"]


@pytest.mark.asyncio
async def test_connect_seeds_only_authoritative_joined_roster(monkeypatch):
    adapter = _make_adapter({"transport": "poll", "channels": [CHANNEL, "not-joined"], "cli_path": "/fake/buzz"})
    adapter.cli_path = "/fake/buzz"
    monkeypatch.setattr(_buzz_mod, "_resolve_private_key", lambda *_: TEST_PRIVATE_KEY)
    monkeypatch.setattr(_buzz_mod, "_resolve_auth_tag", lambda *_: "")
    cli = _ScriptedCli()
    cli.script("users", "get", [{"pubkey": SELF_PUBKEY, "name": "agent", "display_name": "Agent"}])
    cli.script("channels", "list", [{"channel_id": CHANNEL, "name": "joined"}])
    adapter._run_cli = cli
    adapter._discover_dms = AsyncMock()
    adapter._wire_plugin_handlers = lambda *_: None
    try:
        assert await adapter.connect()
        assert set(adapter._channel_state) == {CHANNEL}
        assert adapter._joined_channel_ids == {CHANNEL}
        assert ["channels", "list", "--member"] in [c[0] for c in cli.calls]
        assert "not-joined" not in adapter._channel_state
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_connect_accepts_dm_only_authoritative_roster(monkeypatch):
    adapter = _make_adapter({"transport": "poll"})
    adapter.cli_path = "/fake/buzz"
    monkeypatch.setattr(_buzz_mod, "_resolve_private_key", lambda *_: TEST_PRIVATE_KEY)
    monkeypatch.setattr(_buzz_mod, "_resolve_auth_tag", lambda *_: "")
    cli = _ScriptedCli()
    cli.script("users", "get", [{"pubkey": SELF_PUBKEY, "name": "dm-agent"}])
    cli.script("channels", "list", [])
    cli.script("dms", "list", [{"dm_id": "direct"}])
    adapter._run_cli = cli
    adapter._wire_plugin_handlers = lambda *_: None
    try:
        assert await adapter.connect()
        assert adapter._channel_state["direct"]["chat_type"] == "dm"
        assert adapter._joined_channel_ids == set()
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_connect_names_an_old_cli_that_lacks_member_flag(monkeypatch):
    adapter = _make_adapter({"transport": "poll"})
    adapter.cli_path = "/fake/buzz"
    monkeypatch.setattr(_buzz_mod, "_resolve_private_key", lambda *_: TEST_PRIVATE_KEY)
    monkeypatch.setattr(_buzz_mod, "_resolve_auth_tag", lambda *_: "")
    cli = _ScriptedCli()
    cli.script("users", "get", [{"pubkey": SELF_PUBKEY, "name": "agent"}])
    cli.script("channels", "list", "", code=2, stderr="error: unexpected argument '--member' found\n")
    adapter._run_cli = cli
    try:
        assert await adapter.connect() is False
        assert "upgrade the buzz binary" in (adapter._fatal_error_message or "")
        assert adapter._fatal_error_retryable is False
    finally:
        await adapter.disconnect()
