"""Offline catch-up contracts for #133265, using the existing fake mautrix."""
import asyncio
import json
import logging
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.platforms.matrix import adapter as matrix
from tests.gateway.test_matrix import _make_fake_mautrix

NOW = 1_800_000_000
USER = "@bot:example.org"
ROOM = "!active:example.org"
QUIET = "!quiet:example.org"


def event(index, age=60, kind="m.room.message"):
    return {"event_id": f"$event{index}", "type": kind,
            "origin_server_ts": (NOW - age) * 1000, "sender": "@alice:example.org",
            "content": {"msgtype": "m.text", "body": f"hello {index}"}}


def batch(events, token="s1"):
    return {"next_batch": token, "rooms": {"join": {
        ROOM: {"timeline": {"events": events[::2]}},
        "!other:example.org": {"timeline": {"events": events[1::2]}},
    }}}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setattr(matrix.time, "time", lambda: NOW)
    monkeypatch.setattr(matrix, "_create_matrix_session", lambda proxy: MagicMock())
    for name, module in _make_fake_mautrix().items():
        monkeypatch.setitem(__import__("sys").modules, name, module)
    from mautrix.client import Client
    clients = []

    def make(window=0, responses=None, home=None, check_checkpoint=True):
        adapter = matrix.MatrixAdapter(PlatformConfig(enabled=True, token="fixture",
            extra={"homeserver": "https://matrix.example.org", "user_id": USER,
                   "e2ee_mode": "off", "catch_up_window_seconds": window}))
        adapter._encryption = False
        adapter._connect_authenticate = AsyncMock(return_value=True)
        adapter._refresh_dm_cache = AsyncMock()
        adapter._sync_loop = AsyncMock()
        adapter._handle_text_message = AsyncMock()
        original_init = Client.__init__

        class TestClient(Client):
            def __init__(self, **kwargs):
                original_init(self, **kwargs)
                self.sync = AsyncMock(side_effect=list(responses or [batch([])]))
                self.get_joined_rooms = AsyncMock(return_value=[ROOM, QUIET])
                clients.append(self)

            def handle_sync(self, data):
                # Production _absorb_sync must advance durable state BEFORE any handler runs.
                path = adapter._store_dir / "sync_token.json"
                if window:
                    assert self.sync_store.next_batch == data["next_batch"]
                if window and check_checkpoint and path.exists():
                    assert json.loads(path.read_text())["next_batch"] == data["next_batch"]
                return [asyncio.create_task(adapter._on_room_message(SimpleNamespace(
                    **raw, room_id=room, timestamp=raw["origin_server_ts"])))
                    for room, info in data.get("rooms", {}).get("join", {}).items()
                    for raw in info.get("timeline", {}).get("events", [])]

        monkeypatch.setattr(__import__("mautrix.client", fromlist=["Client"]), "Client", TestClient)
        scope = set_hermes_home_override(str(home or tmp_path))
        try:
            store = adapter._resolve_store_dir()
        finally:
            reset_hermes_home_override(scope)
        # connect resolves in this same profile scope; do not replace the path resolver.
        return adapter, store, home or tmp_path

    return make, clients


async def connect(adapter, home):
    scope = set_hermes_home_override(str(home))
    try:
        assert await adapter.connect()
    finally:
        reset_hermes_home_override(scope)
    await adapter.disconnect()


def seed(store, user=USER, token="s0", saved_at=NOW - 120):
    store.mkdir(parents=True, exist_ok=True)
    (store / "sync_token.json").write_text(json.dumps({
        "user_id": user, "next_batch": token, "saved_at": saved_at}))


def delivered(adapter):
    return {call.args[2] for call in adapter._handle_text_message.await_args_list}


@pytest.mark.asyncio
async def test_default_cold_boot_unchanged(harness):
    make, clients = harness
    adapter, store, home = make(responses=[batch([event(0)])])
    await connect(adapter, home)
    assert delivered(adapter) == set()
    clients[-1].sync.assert_awaited_once_with(timeout=10000, full_state=True)
    assert not (store / "sync_token.json").exists()


@pytest.mark.asyncio
async def test_since_window_and_restart_never_replays_first_batch(harness):
    make, clients = harness
    first, store, home = make(3600, [batch([event(0), event(1, 3601), event(2)])])
    seed(store)
    await connect(first, home)
    assert delivered(first) == {"$event0", "$event2"}
    clients[-1].sync.assert_awaited_once_with(since="s0", timeout=10000, full_state=True)
    assert QUIET in first._joined_rooms  # absent from incremental join map
    assert json.loads((store / "sync_token.json").read_text())["next_batch"] == "s1"

    second, _, home = make(3600)
    await connect(second, home)
    clients[-1].sync.assert_awaited_once_with(since="s1", timeout=10000, full_state=True)
    assert delivered(second) == set()


@pytest.mark.asyncio
async def test_per_room_cap_preserves_dm(harness, caplog):
    make, _ = harness
    data = {"next_batch": "s1", "rooms": {"join": {
        ROOM: {"timeline": {"events": [event(i, 50 - i) for i in range(25)]}},
        QUIET: {"timeline": {"events": [event("dm", 100)]}},
    }}}
    adapter, store, home = make(3600, [data])
    seed(store)
    with caplog.at_level(logging.INFO):
        await connect(adapter, home)
    assert delivered(adapter) == {"$eventdm"} | {f"$event{i}" for i in range(5, 25)}
    assert len([r for r in caplog.records if "overflow" in r.message]) == 1


@pytest.mark.asyncio
async def test_rejected_cursor_falls_back_without_catch_up(harness, caplog):
    make, clients = harness
    adapter, store, home = make(3600, [RuntimeError("M_UNKNOWN_POS"), batch([event(0)])])
    seed(store)
    with caplog.at_level(logging.WARNING):
        await connect(adapter, home)
    assert clients[-1].sync.await_args_list[0].kwargs["since"] == "s0"
    assert clients[-1].sync.await_args_list[1].kwargs == {"timeout": 10000, "full_state": True}
    assert delivered(adapter) == set()
    assert len([r for r in caplog.records if "catch-up" in r.message and r.levelno == logging.WARNING]) == 1


@pytest.mark.asyncio
async def test_later_old_incremental_event_still_dropped(harness):
    make, clients = harness
    adapter, store, home = make(3600, [batch([event(0)])])
    seed(store)
    await connect(adapter, home)
    adapter._client = clients[-1]
    await adapter._absorb_sync(clients[-1], batch([event(1)], "s2"))
    await adapter._on_room_message(SimpleNamespace(**event(0), room_id=ROOM,
                                                  timestamp=(NOW - 60) * 1000))
    assert delivered(adapter) == {"$event0"}
    assert adapter._handle_text_message.await_count == 1


@pytest.mark.asyncio
async def test_foreign_identity_ignored_and_profiles_isolated(harness, tmp_path):
    make, clients = harness
    paths = []
    for name in ("one", "two"):
        adapter, store, home = make(3600, [batch([event(0)])], tmp_path / name)
        seed(store, user="@other:example.org")
        await connect(adapter, home)
        clients[-1].sync.assert_awaited_once_with(timeout=10000, full_state=True)
        assert delivered(adapter) == set()
        assert json.loads((store / "sync_token.json").read_text())["user_id"] == USER
        paths.append(store)
    assert paths[0] != paths[1]


@pytest.mark.parametrize("raw,expected", [(None, 0), ("bad", 0), ([], 0),
    (-1, 0), ("3600", 3600), (999999, 86400), (float("inf"), 0)])
def test_window_resolution(raw, expected, caplog):
    from plugins.platforms.matrix.catch_up import resolve_catch_up_window
    assert resolve_catch_up_window({"catch_up_window_seconds": raw}) == expected
    assert len(caplog.records) == (1 if raw is None or isinstance(raw, list)
                                  or raw == "bad" or raw == float("inf") else 0)


def test_selection_includes_encrypted_and_logs_limited(caplog):
    from plugins.platforms.matrix.catch_up import select_catch_up_ids
    data = batch([event(0, kind="m.room.encrypted"), event(1, 3601)])
    data["rooms"]["join"][ROOM]["timeline"]["limited"] = True
    with caplog.at_level(logging.INFO):
        assert select_catch_up_ids(data, NOW, 3600, saved_at=NOW - 120, user_id=USER) == {"$event0"}
    assert len([r for r in caplog.records if "limited" in r.message and ROOM in r.message]) == 1


@pytest.mark.asyncio
async def test_persistent_store_skips_unchanged_and_bad_files(harness, tmp_path):
    from plugins.platforms.matrix.catch_up import load_since, persistent_sync_store
    store = persistent_sync_store(tmp_path, USER)
    await store.put_next_batch("s1")
    path = tmp_path / "sync_token.json"
    before = path.stat()
    await store.put_next_batch("s1")
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert await store.get_next_batch() == "s1"
    assert load_since(tmp_path, USER) == ("s1", NOW)
    assert load_since(tmp_path, "@other:example.org") is None
    for raw in ("invalid", "[]", "{}", '{"user_id": "@bot:example.org", "next_batch": 3}'):
        path.write_text(raw)
        assert load_since(tmp_path, USER) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["write_text", "replace"])
async def test_checkpoint_failure_advances_and_dispatches_once_per_streak(harness, monkeypatch, caplog, failure):
    from plugins.platforms.matrix import catch_up
    make, clients = harness
    adapter, store, home = make(3600, check_checkpoint=False)
    seed(store)
    await connect(adapter, home)
    client = clients[-1]
    adapter._client = client
    target = Path if failure == "write_text" else catch_up
    original = getattr(target, failure, None)
    with caplog.at_level(logging.WARNING):
        monkeypatch.setattr(target, failure, MagicMock(side_effect=OSError("read-only fixture")), raising=False)
        for i in (2, 3):
            assert await adapter._absorb_sync(client, batch([event(i, 1)], f"s{i}")) == f"s{i}"
            assert await client.sync_store.get_next_batch() == f"s{i}"
        assert delivered(adapter) == {"$event2", "$event3"}
        warnings = [r for r in caplog.records if r.name == catch_up.logger.name
                    and "could not save catch-up checkpoint" in r.message]
        assert len(warnings) == 1
        assert str(store / "sync_token.json") in warnings[0].message
        assert "restart is skipped until a checkpoint is saved again" in warnings[0].message
        monkeypatch.setattr(target, failure, original)
        await adapter._absorb_sync(client, batch([event(4, 1)], "s4"))
        monkeypatch.setattr(target, failure, MagicMock(side_effect=OSError("read-only fixture")))
        await adapter._absorb_sync(client, batch([event(5, 1)], "s5"))
    assert delivered(adapter) == {f"$event{i}" for i in (2, 3, 4, 5)}
    assert len([r for r in caplog.records if r.name == catch_up.logger.name
                and "could not save catch-up checkpoint" in r.message]) == 2


@pytest.mark.asyncio
async def test_own_join_room_history_not_exempt(harness):
    make, _ = harness
    data = batch([event(0), event(1)])
    data["rooms"]["join"][ROOM]["timeline"]["events"].append({
        "event_id": "$join", "type": "m.room.member", "state_key": USER,
        "sender": USER, "origin_server_ts": (NOW - 70) * 1000,
        "content": {"membership": "join"}})
    adapter, store, home = make(3600, [data])
    seed(store)
    await connect(adapter, home)
    assert delivered(adapter) == {"$event1"}


@pytest.mark.asyncio
async def test_checkpoint_time_excludes_already_synced_history(harness):
    make, _ = harness
    adapter, store, home = make(3600, [batch([event(0, 180), event(1), event(2, 125), event(3, 126)])])
    seed(store)
    await connect(adapter, home)
    assert delivered(adapter) == {"$event1", "$event2"}  # checkpoint - 5s is inclusive


@pytest.mark.asyncio
@pytest.mark.parametrize("saved_at", [None, "missing", "bad", True])
async def test_checkpoint_without_numeric_time_falls_back(harness, saved_at):
    make, clients = harness
    adapter, store, home = make(3600, [batch([event(0)])])
    seed(store, saved_at=saved_at)
    if saved_at == "missing":
        (store / "sync_token.json").write_text(json.dumps({"user_id": USER, "next_batch": "s0"}))
    await connect(adapter, home)
    clients[-1].sync.assert_awaited_once_with(timeout=10000, full_state=True)
    assert delivered(adapter) == set()


@pytest.mark.asyncio
async def test_since_reconnect_replaces_stale_joined_rooms(harness):
    make, _ = harness
    adapter, store, home = make(3600)
    seed(store)
    adapter._joined_rooms.add("!left:example.org")
    await connect(adapter, home)
    assert "!left:example.org" not in adapter._joined_rooms
    assert {ROOM, QUIET} <= adapter._joined_rooms


@pytest.mark.asyncio
async def test_overall_cap_keeps_newest_hundred(harness, caplog):
    make, _ = harness
    data = {"next_batch": "s1", "rooms": {"join": {
        f"!room{r}:example.org": {"timeline": {"events": [
            event(r * 20 + i, 130 - (r * 20 + i)) for i in range(20)]}}
        for r in range(6)}}}
    adapter, store, home = make(3600, [data])
    seed(store, saved_at=NOW - 200)
    with caplog.at_level(logging.INFO):
        await connect(adapter, home)
    assert adapter._handle_text_message.await_count == 100
    assert delivered(adapter) == {f"$event{i}" for i in range(20, 120)}
    assert len([r for r in caplog.records if "overflow" in r.message]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("sender", [USER, "@BOT:example.org"])
async def test_own_messages_do_not_consume_room_cap(harness, sender):
    make, _ = harness
    own = [dict(event(i, 50 - i), sender=sender) for i in range(25)]
    data = {"next_batch": "s1", "rooms": {"join": {
        ROOM: {"timeline": {"events": [event("human", 100), *own]}}}}}
    adapter, store, home = make(3600, [data])
    seed(store)
    await connect(adapter, home)
    assert delivered(adapter) == {"$eventhuman"}


@pytest.mark.asyncio
async def test_failed_checkpoint_removed_before_restart(harness, monkeypatch):
    make, clients = harness
    first, store, home = make(3600, [batch([event(0)])], check_checkpoint=False)
    seed(store)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", MagicMock(side_effect=OSError("full disk fixture")))
        await connect(first, home)
    assert delivered(first) == {"$event0"}
    assert await clients[-1].sync_store.get_next_batch() == "s1"
    assert not (store / "sync_token.json").exists()

    second, _, home = make(3600, [batch([event(0)])])
    await connect(second, home)
    clients[-1].sync.assert_awaited_once_with(timeout=10000, full_state=True)
    assert delivered(second) == set()


@pytest.mark.asyncio
async def test_failed_checkpoint_unlink_logs_one_replay_error(harness, monkeypatch, caplog):
    from plugins.platforms.matrix import catch_up
    make, clients = harness
    adapter, store, home = make(3600, check_checkpoint=False)
    seed(store)
    await connect(adapter, home)
    path = store / "sync_token.json"
    original_unlink = Path.unlink

    def fail_checkpoint_unlink(target, *args, **kwargs):
        if target == path:
            raise OSError("read-only fixture")
        return original_unlink(target, *args, **kwargs)

    adapter._client = clients[-1]
    with caplog.at_level(logging.WARNING):
        with monkeypatch.context() as patch:
            patch.setattr(Path, "write_text", MagicMock(side_effect=OSError("full disk fixture")))
            patch.setattr(Path, "unlink", fail_checkpoint_unlink)
            for i in (2, 3):
                await adapter._absorb_sync(clients[-1], batch([event(i, 1)], f"s{i}"))
    assert delivered(adapter) == {"$event2", "$event3"}
    assert await clients[-1].sync_store.get_next_batch() == "s3"
    assert json.loads(path.read_text())["next_batch"] == "s1"
    errors = [r for r in caplog.records if r.name == catch_up.logger.name and r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert str(path) in errors[0].message
    assert "restart may repeat up to 100 messages" in errors[0].message


@pytest.mark.asyncio
@pytest.mark.parametrize("join_age,expected", [(170, set()), (400, {"$event0"})])
async def test_state_only_rejoin_with_host_clock_behind(harness, join_age, expected):
    make, _ = harness
    # Checkpoint host time is 100s behind the homeserver; pre-checkpoint history
    # can pass the event bound, so the recent own join must exclude this room.
    data = batch([event(0, 180)])
    data["rooms"]["join"][ROOM].update({
        "timeline": {"limited": True, "events": [event(0, 180)]},
        "state": {"events": [{"type": "m.room.member", "state_key": USER,
            "origin_server_ts": (NOW - join_age) * 1000, "content": {"membership": "join"}}]}})
    adapter, store, home = make(3600, [data])
    seed(store, saved_at=NOW - 220)
    await connect(adapter, home)
    assert delivered(adapter) == expected


@pytest.mark.asyncio
async def test_state_only_rejoin_matches_own_id_case_insensitively(harness):
    make, _ = harness
    data = batch([event(0, 180)])
    data["rooms"]["join"][ROOM].update({
        "timeline": {"limited": True, "events": [event(0, 180)]},
        "state": {"events": [{"type": "m.room.member", "state_key": USER.upper(),
            "origin_server_ts": (NOW - 170) * 1000, "content": {"membership": "join"}}]}})
    adapter, store, home = make(3600, [data])
    seed(store, saved_at=NOW - 220)
    await connect(adapter, home)
    assert delivered(adapter) == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["timeline", "state"])
@pytest.mark.parametrize("previous", ["join", "leave", None])
async def test_own_member_profile_change_is_not_rejoin(harness, source, previous):
    make, _ = harness
    data = batch([event(0)])
    member = {"type": "m.room.member", "state_key": USER, "sender": USER,
              "origin_server_ts": (NOW - 70) * 1000,
              "content": {"membership": "join", "displayname": "New name"}}
    if previous is not None:
        member["unsigned"] = {"prev_content": {"membership": previous}}
    data["rooms"]["join"][ROOM].setdefault(source, {"events": []})["events"].append(member)
    adapter, store, home = make(3600, [data])
    seed(store)
    await connect(adapter, home)
    assert delivered(adapter) == ({"$event0"} if previous == "join" else set())


@pytest.mark.asyncio
async def test_disabled_window_removes_checkpoint_before_reenable(harness):
    make, clients = harness
    disabled, store, home = make(responses=[batch([event(0, 1)])])
    seed(store)
    await connect(disabled, home)
    assert delivered(disabled) == {"$event0"}
    assert not (store / "sync_token.json").exists()
    clients[-1].sync.assert_awaited_once_with(timeout=10000, full_state=True)

    restarted, _, home = make(3600, [batch([event(0)])])
    await connect(restarted, home)
    clients[-1].sync.assert_awaited_once_with(timeout=10000, full_state=True)
    assert delivered(restarted) == set()
