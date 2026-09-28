"""Canonical cron destinations must use the same owner and session as replies."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cron.scheduler_delivery import _deliver_result
from gateway.config import Platform, PlatformConfig, load_gateway_config
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.session import SessionSource, SessionStore
from plugins.platforms.matrix.adapter import MatrixAdapter


class _MissingEncryption(Exception):
    errcode = "M_NOT_FOUND"


def _schedule(coro, loop):
    future = Future()
    try:
        future.set_result(asyncio.run(coro))
    except Exception as exc:
        future.set_exception(exc)
    return future


@pytest.fixture
def destinations(tmp_path, monkeypatch):
    import hermes_state
    from agent.secret_scope import set_multiplex_active

    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    home = tmp_path / ".hermes"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("asyncio.run_coroutine_threadsafe", _schedule)
    monkeypatch.setattr(
        "plugins.platforms.matrix.adapter.time.time", lambda: 1_800_000_000
    )
    monkeypatch.setattr(
        "tools.send_message_tool._send_to_platform",
        AsyncMock(return_value={"error": "standalone unavailable"}),
    )
    runner = object.__new__(GatewayRunner)
    runner._primary_profile_name = "default"
    runner._profile_adapters = {}
    owners = {}
    for profile in ("default", "secondary"):
        target_home = home if profile == "default" else home / "profiles" / profile
        target_home.mkdir(parents=True)
        (target_home / "config.yaml").write_text(
            json.dumps({
                "cron": {"wrap_response": False},
                "platforms": {
                    "matrix": {"enabled": True, "extra": {"e2ee_mode": "off"}}
                },
            }),
            encoding="utf-8",
        )
        with _profile_runtime_scope(target_home, {}):
            config = load_gateway_config()
            config.multiplex_profiles = True
            adapter = MatrixAdapter(
                PlatformConfig(enabled=True, extra={"e2ee_mode": "off"})
            )
            room = f"!{profile}:remote.test"
            adapter._client = SimpleNamespace(
                resolve_room_alias=AsyncMock(
                    return_value=SimpleNamespace(room_id=room, servers=["route.test"])
                ),
                join_room=AsyncMock(return_value=room),
                get_state_event=AsyncMock(side_effect=_MissingEncryption()),
                send_message_event=AsyncMock(
                    side_effect=["$opened", "$sent", "$opened-again", "$sent-again"]
                ),
                crypto=None,
            )
            adapter._refresh_dm_cache = AsyncMock()
            adapter._is_dm_room = AsyncMock(return_value=False)
            adapter.gateway_runner = runner
            adapter.set_owner_profile(profile)
            adapter.set_session_store(SessionStore(target_home / "sessions", config))
            owners[profile] = (target_home, adapter, room)
    runner.adapters = {Platform.MATRIX: owners["default"][1]}
    runner._profile_adapters["secondary"] = {Platform.MATRIX: owners["secondary"][1]}
    runner.config = config
    set_multiplex_active(True)
    return runner, owners


@pytest.mark.parametrize("thread", [None, "$existing"])
@pytest.mark.parametrize("chat_type", ["group", "dm"])
def test_alias_delivery_seeds_the_owning_reply_session(destinations, thread, chat_type):
    runner, owners = destinations
    loop = SimpleNamespace(is_running=lambda: True)
    for index, profile in enumerate(("default", "secondary", "default")):
        home, adapter, room = owners[profile]
        adapter._is_dm_room.return_value = chat_type == "dm"
        brief = f"Scheduled brief {index} for {profile}"
        alias = "#scheduled:remote.test"
        target = f"matrix:{alias}" + (f"/{thread}" if thread else "")
        with _profile_runtime_scope(home, {}):
            error = _deliver_result(
                {"id": f"brief-{index}", "deliver": target, "attach_to_session": True},
                brief,
                runner.adapters,
                loop,
            )
            assert error is None, error
            args = adapter._client.send_message_event.await_args.args
            delivered_room = str(args[0])
            delivered_thread = args[2]["m.relates_to"]["event_id"]
            source = SessionSource(
                platform=Platform.MATRIX,
                chat_id=delivered_room,
                thread_id=delivered_thread,
                chat_type=chat_type,
                user_id="@alice:remote.test",
                profile=profile,
            )
            store = adapter._session_store
            reply_key = store._generate_session_key(source)
            assert delivered_room == room
            assert reply_key in store._entries, list(store._entries)
            entry = store.get_or_create_session(source)
            assert brief in json.dumps(store.load_transcript(entry.session_id))
            assert all(alias not in key for key in store._entries)
    assert [
        owner[1]._client.resolve_room_alias.await_count for owner in owners.values()
    ] == [2, 1]


@pytest.mark.parametrize(
    "failure", ["lookup", "join", "encryption", "send", "media", "timeout"]
)
def test_failed_alias_delivery_creates_no_continuation(
    destinations, failure, tmp_path, monkeypatch
):
    from gateway.platforms.base import SendResult

    runner, owners = destinations
    home, adapter, _ = owners["secondary"]
    client = adapter._client
    if failure == "lookup":
        client.resolve_room_alias.side_effect = ValueError("directory forbidden")
    if failure == "join":
        client.join_room.side_effect = ValueError("join forbidden")
    if failure == "encryption":
        client.get_state_event.side_effect = None
        client.get_state_event.return_value = {"algorithm": "m.megolm.v1.aes-sha2"}
    if failure == "send":
        client.send_message_event.side_effect = ValueError("send forbidden")
    brief = "Unseen cron brief"
    if failure == "media":
        image = tmp_path / "brief.png"
        image.write_bytes(b"image")
        brief = f"MEDIA:{image}"
        adapter.send_image_file = AsyncMock(
            return_value=SendResult(success=False, error="upload forbidden")
        )
    if failure == "timeout":
        monkeypatch.setattr(
            "cron.scheduler_delivery._live_send_text",
            lambda *a, **kw: (True, True, None),
        )
    alias = "#scheduled:remote.test"
    with _profile_runtime_scope(home, {}):
        error = _deliver_result(
            {
                "id": "failed-brief",
                "deliver": f"matrix:{alias}/$existing",
                "attach_to_session": True,
            },
            brief,
            runner.adapters,
            SimpleNamespace(is_running=lambda: True),
        )
        if failure != "timeout":
            assert error and alias in error
        assert adapter._session_store._entries == {}
    runner.adapters[Platform.MATRIX]._client.send_message_event.assert_not_awaited()


@pytest.mark.parametrize("destination", ["room", "alias"])
@pytest.mark.parametrize("has_origin", [True, False])
def test_per_user_thread_delivery_preserves_the_originating_participant(
    destinations, destination, has_origin
):
    runner, owners = destinations
    for index, profile in enumerate(("default", "secondary", "default")):
        home, adapter, room = owners[profile]
        store = adapter._session_store
        store.config.thread_sessions_per_user = True
        source = SessionSource(
            platform=Platform.MATRIX,
            chat_id=room,
            thread_id="$existing",
            chat_type="group",
            user_id="@alice:remote.test",
            profile=profile,
        )
        alice = store.get_or_create_session(source)
        bob = store.get_or_create_session(
            SessionSource(
                platform=Platform.MATRIX,
                chat_id=room,
                thread_id="$existing",
                chat_type="group",
                user_id="@bob:remote.test",
                profile=profile,
            )
        )
        original_entries = dict(store._entries)
        with _profile_runtime_scope(home, {}):
            store.append_to_transcript(
                alice.session_id,
                {
                    "role": "assistant",
                    "content": f"Alice's cached turn {index}",
                },
            )
            before = deepcopy(store.load_transcript(alice.session_id))
            target = room if destination == "room" else "#scheduled:remote.test"
            brief = f"Alice's scheduled brief {index}"
            error = _deliver_result(
                {
                    "id": f"alice-brief-{index}",
                    "deliver": f"matrix:{target}/$existing",
                    "attach_to_session": True,
                    "origin": source.to_dict() if has_origin else None,
                },
                brief,
                runner.adapters,
                SimpleNamespace(is_running=lambda: True),
            )
            assert error is None, error
            assert store._entries == original_entries
            transcript = store.load_transcript(alice.session_id)
            if not has_origin:
                assert transcript == before
                assert store.load_transcript(bob.session_id) == []
                continue
            assert transcript[:-1] == before
            assert (transcript[-1]["role"], transcript[-1]["content"]) == (
                "user",
                f"[Cron delivery: alice-brief-{index}]\n{brief}",
            )
            assert store.load_transcript(bob.session_id) == []
