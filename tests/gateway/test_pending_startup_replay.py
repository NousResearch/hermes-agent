"""Startup restores pending originals before newer input without spending their bodies."""

import asyncio
from copy import deepcopy
import json
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import unquote

import hermes_state
import pytest
from mautrix.api import HTTPAPI
from mautrix.client.api import ClientAPI
from mautrix.types import UserID

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.shutdown_pending import PendingQueueSnapshot
from hermes_constants import get_hermes_home
from plugins.platforms.matrix.adapter import MatrixAdapter
from utils import atomic_json_write


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "native_state",
    [
        "current",
        "deleted",
        "wrong-author",
        "unreadable",
        "reset-during-read",
        "receipt-during-read",
        "conflicting-copy",
        "identical-copy",
        "partial-withdraw-during-read",
    ],
)
async def test_startup_replays_only_current_native_input_and_retains_its_record(
    monkeypatch, native_state
):
    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    monkeypatch.setenv("MATRIX_ALLOW_ALL_USERS", "true")
    monkeypatch.setenv("MATRIX_REQUIRE_MENTION", "false")
    monkeypatch.setenv("MATRIX_AUTO_THREAD", "false")
    runner = GatewayRunner(GatewayConfig())
    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token="test",
            extra={
                "homeserver": "https://matrix.example.org",
                "user_id": "@hermes:example.org",
            },
        )
    )
    runner.adapters[Platform.MATRIX] = adapter
    adapter.gateway_runner = runner
    adapter._mark_connected()
    api = HTTPAPI(
        "https://matrix.example.org", token="test", client_session=MagicMock()
    )

    async def request(method, path, **kwargs):
        url = str(path)
        if "/state/m.room.member/" in url:
            return {"membership": "join", "displayname": "Alice"}
        if "/joined_members" in url:
            return {
                "joined": {
                    "@hermes:example.org": {},
                    "@alice:example.org": {},
                    "@third:example.org": {},
                }
            }
        if "/event/" not in url:
            return {}
        native_id = unquote(url.split("/event/", 1)[1])
        if native_state == "partial-withdraw-during-read" and native_id == "$old":

            def failed_write(*args, **kwargs):
                raise OSError("controlled unavailable spool write")

            monkeypatch.setattr("utils.atomic_json_write", failed_write)
            assert adapter.withdraw_pending_message(
                "$old", chat_id=source.chat_id, sender_id="@alice:example.org"
            )
        if native_state == "unreadable":
            raise OSError("controlled disconnected homeserver")
        if native_state == "reset-during-read":
            runner.session_store.reset_session(entry.session_key, source=source)
        if native_state == "receipt-during-read":
            db = runner.session_store._db_for_session_id(entry.session_id)
            db.append_message(
                entry.session_id,
                "user",
                "executed original",
                display_metadata={
                    "gateway_input_owner": snapshot.events[0]["input_owner"]["owner"]
                },
            )
        event = {
            "event_id": native_id,
            "room_id": "!room:example.org",
            "type": "m.room.message",
            "sender": "@other:example.org"
            if native_state == "wrong-author"
            else "@alice:example.org",
            "origin_server_ts": 1000,
            "content": {
                "msgtype": "m.text",
                "body": "second" if native_id == "$second" else "older",
            },
        }
        if native_state == "deleted":
            event["content"] = {}
            event["unsigned"] = {"redacted_because": {"event_id": "$redaction"}}
        return event

    monkeypatch.setattr(api, "request", request)
    adapter._client = ClientAPI(UserID("@hermes:example.org"), api=api)
    source = SessionSource(
        Platform.MATRIX,
        "!room:example.org",
        user_id="@alice:example.org",
        chat_type="group",
    )
    adapter._canonicalize(source)
    entry = runner.session_store.get_or_create_session(source)
    old = MessageEvent(text="older", source=source, message_id="$old")
    if native_state == "partial-withdraw-during-read":
        from gateway.platforms.base_pending import merge_recorded
        from gateway.platforms.base_pending_merge import _absorb_pending_text
        from gateway.pending_native import PendingNativeInput

        old._pending_native_input = PendingNativeInput.capture(
            old, {"msgtype": "m.text", "body": "older"}
        )
        secondary = MessageEvent(text="second", source=source, message_id="$second")
        secondary._pending_native_input = PendingNativeInput.capture(
            secondary, {"msgtype": "m.text", "body": "second"}
        )
        merge_recorded(old, secondary, _absorb_pending_text)
    snapshot = PendingQueueSnapshot.capture(entry.session_key, [old])
    directory = get_hermes_home() / "pending_messages"
    directory.mkdir(exist_ok=True)
    path = directory / "shutdown-input.json"
    atomic_json_write(path, snapshot.to_payload(), mode=0o600)
    if native_state in {"conflicting-copy", "identical-copy"}:
        duplicate = deepcopy(snapshot.to_payload())
        if native_state == "conflicting-copy":
            duplicate["events"][0]["event"]["text"] = "different original body"
        atomic_json_write(directory / "zz-duplicate-input.json", duplicate, mode=0o600)
    seen = []

    async def receive(event):
        seen.append((event.text, getattr(event, "_pending_snapshot_uid", None)))
        return None

    runner._wire_adapter_handlers(adapter, message_handler=receive)
    runner._startup_restore_in_progress = True
    runner._startup_restore_queue = [
        MessageEvent(text="newer", source=source, message_id="$new")
    ]
    monkeypatch.setattr(runner, "_await_startup_warmup", AsyncMock())
    await runner._finish_startup_restore()
    tasks = list(adapter._background_tasks)
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
    expected = (
        [("older", snapshot.events[0]["uid"])]
        if native_state in {"current", "identical-copy"}
        else []
    )
    expected.append(("newer", None))
    assert (
        seen,
        json.loads(path.read_text())["events"],
        runner._startup_restore_in_progress,
    ) == (
        expected,
        snapshot.events,
        False,
    )


@pytest.mark.asyncio
async def test_restoration_validates_media_and_receipts_in_the_routed_profile_home(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from gateway.input_owner import gateway_input_owner
    from gateway.pending_execution import consume_pending_execution
    from gateway.profile_routing import ProfileRoute
    from gateway.run import _profile_runtime_scope

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    launch = tmp_path / ".hermes"
    launch.mkdir()
    (launch / ".env").write_text("GATEWAY_ALLOWED_USERS=@alice:example.org\n")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    homes = {profile: launch / "profiles" / profile for profile in ("a", "b")}
    for home in homes.values():
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("{}\n")
        (home / ".env").write_text(
            "MATRIX_ALLOW_ALL_USERS=true\nMATRIX_REQUIRE_MENTION=false\nMATRIX_AUTO_THREAD=false\n"
        )
    adapter = MatrixAdapter(
        PlatformConfig(
            enabled=True,
            token="test",
            extra={
                "homeserver": "https://matrix.example.org",
                "user_id": "@hermes:example.org",
                "require_mention": False,
                "auto_thread": False,
                "allowed_users": ["@alice:example.org"],
            },
        )
    )
    runner = GatewayRunner(
        GatewayConfig(
            multiplex_profiles=True,
            profile_routes=[
                ProfileRoute(
                    name=profile, platform="matrix", profile=profile, chat_id=profile
                )
                for profile in homes
            ],
        )
    )
    runner.adapters[Platform.MATRIX] = adapter
    runner._profile_adapters = {profile: {} for profile in homes}
    adapter.gateway_runner = runner
    adapter._mark_connected()
    current = {}
    reads = []
    api = HTTPAPI(
        "https://matrix.example.org", token="test", client_session=MagicMock()
    )

    async def request(method, path, **kwargs):
        url = str(path)
        if "/state/m.room.member/" in url:
            return {"membership": "join", "displayname": "Alice"}
        if "/joined_members" in url:
            return {
                "joined": {
                    "@hermes:example.org": {},
                    "@alice:example.org": {},
                    "@third:example.org": {},
                }
            }
        if "/event/" in url:
            event_id = unquote(url.split("/event/", 1)[1])
            reads.append((event_id, get_hermes_home()))
            return current[event_id]
        return {}

    monkeypatch.setattr(api, "request", request)
    adapter._client = ClientAPI(UserID("@hermes:example.org"), api=api)
    observed = []

    async def receive(event):
        key = runner._session_key_for_source(event.source)
        entry = runner.session_store.lookup_by_session_key(key)
        assert entry is not None
        observed.append((
            event.message_id,
            get_hermes_home(),
            Path(event.media_urls[0]).read_bytes(),
        ))
        db = runner.session_store._db_for_session_id(entry.session_id)
        db.append_message(
            entry.session_id,
            "user",
            event.text,
            display_metadata={
                "gateway_input_owner": gateway_input_owner(event, event.source)
            },
        )
        consume_pending_execution(runner, event)
        return None

    runner._wire_adapter_handlers(adapter, message_handler=receive)
    monkeypatch.setattr(runner, "_await_startup_warmup", AsyncMock())
    active = is_multiplex_active()
    set_multiplex_active(True)
    expected = []
    try:
        for index, profile in enumerate(("a", "b", "a")):
            event_id = f"$input-{index}"
            content = {
                "msgtype": "m.image",
                "body": f"image-{index}",
                "url": f"mxc://example.org/{index}",
            }
            current[event_id] = {
                "type": "m.room.message",
                "room_id": profile,
                "event_id": event_id,
                "sender": "@alice:example.org",
                "content": content,
            }
            with _profile_runtime_scope(homes[profile]):
                cache = homes[profile] / "cache" / "same-name.png"
                cache.parent.mkdir(exist_ok=True)
                cache.write_bytes(f"native-{index}".encode())
                source = adapter.build_source(
                    chat_id=profile, user_id="@alice:example.org", chat_type="group"
                )
                assert adapter._canonicalize(source) is not None
                event = await adapter._build_inbound_event(
                    profile,
                    "@alice:example.org",
                    event_id,
                    content["body"],
                    content,
                    {},
                    ctx=(content["body"], False, "group", None, "Alice", False, source),
                    media_msgtype="m.image",
                    media_urls=[str(cache)],
                    media_types=["image/png"],
                )
                assert event is not None
                entry = runner.session_store.get_or_create_session(source)
                snapshot = PendingQueueSnapshot.capture(entry.session_key, [event])
                path = homes[profile] / "pending_messages" / f"pending-{index}.json"
                path.parent.mkdir(exist_ok=True)
                atomic_json_write(path, snapshot.to_payload(), mode=0o600)
            runner._startup_restore_in_progress = True
            await runner._finish_startup_restore()
            if tasks := list(adapter._background_tasks):
                await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
            expected.append((event_id, homes[profile], f"native-{index}".encode()))
            assert (observed, reads, path.exists(), get_hermes_home()) == (
                expected,
                [(item[0], item[1]) for item in expected],
                False,
                launch,
            )
    finally:
        set_multiplex_active(active)
