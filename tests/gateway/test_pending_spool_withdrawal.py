"""Authenticated deletion removes spooled constituents without changing their ownership."""

import json

import hermes_state
import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.native_message_deletion import NativeMessageDeletion
from gateway.platforms.base_pending import merge_recorded
from gateway.platforms.base_pending_merge import _absorb_pending_text
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.shutdown_pending import PendingQueueSnapshot
from hermes_constants import get_hermes_home
from utils import atomic_json_write


@pytest.mark.parametrize(
    "platform,origin,scope",
    [
        (Platform.MATRIX, False, "guild"),
        (Platform.DISCORD, False, "guild"),
        (Platform.DISCORD, True, "guild"),
        (Platform.SLACK, False, "workspace"),
        (Platform.SLACK, False, "wrong-workspace"),
    ],
)
def test_deletion_withdraws_only_its_spooled_contribution(
    monkeypatch, platform, origin, scope
):
    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    if platform == Platform.MATRIX:
        from plugins.platforms.matrix.adapter import MatrixAdapter

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
    elif platform == Platform.DISCORD:
        pytest.importorskip("discord")
        from plugins.platforms.discord.adapter import DiscordAdapter

        adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test"))
    else:
        pytest.importorskip("slack_bolt")
        from plugins.platforms.slack.adapter import SlackAdapter

        adapter = SlackAdapter(
            PlatformConfig(enabled=True, token="test", extra={"app_token": "test"})
        )
    runner = GatewayRunner(GatewayConfig())
    runner.adapters[platform] = adapter
    adapter.gateway_runner = runner
    runner._wire_adapter_handlers(adapter)
    adapter._mark_connected()
    source = adapter.build_source(
        chat_id="thread" if origin else "channel",
        user_id="author",
        chat_type="group",
        scope_id="workspace" if platform == Platform.SLACK else "guild",
    )
    if origin:
        source.auto_thread_created = True
        source.parent_chat_id = "channel"
    first = MessageEvent(
        text="first", source=source, message_id="first", channel_context="first context"
    )
    second = MessageEvent(
        text="second",
        source=source,
        message_id="second",
        channel_context="second context",
    )
    merge_recorded(first, second, _absorb_pending_text)
    entry = runner.session_store.get_or_create_session(source)
    snapshot = PendingQueueSnapshot.capture(entry.session_key, [first])
    directory = get_hermes_home() / "pending_messages"
    directory.mkdir(exist_ok=True)
    path = directory / "spooled.json"
    atomic_json_write(path, snapshot.to_payload(), mode=0o600)
    if platform == Platform.MATRIX:
        found = adapter.withdraw_pending_message(
            "second", chat_id=source.chat_id, sender_id="author"
        )
    else:
        found = adapter.withdraw_native_messages(
            NativeMessageDeletion(platform, scope, "channel", ("second",))
        )
    records = json.loads(path.read_text())["events"]
    actual = [
        (
            record["uid"],
            record["input_owner"],
            record["event"]["text"],
            record["event"]["channel_context"],
            record["event"]["merged_message_ids"],
        )
        for record in records
    ]
    expected = [
        (
            snapshot.events[0]["uid"],
            snapshot.events[0]["input_owner"],
            "first",
            "first context",
            [],
        )
    ]
    if scope == "wrong-workspace":
        expected = [
            (
                record["uid"],
                record["input_owner"],
                record["event"]["text"],
                record["event"]["channel_context"],
                record["event"]["merged_message_ids"],
            )
            for record in snapshot.events
        ]
    assert (found, actual) == (scope != "wrong-workspace", expected)
    if scope != "wrong-workspace":
        if platform == Platform.MATRIX:
            terminal = adapter.withdraw_pending_message(
                "first", chat_id=source.chat_id, sender_id="author"
            )
        else:
            terminal = adapter.withdraw_native_messages(
                NativeMessageDeletion(platform, scope, "channel", ("first",))
            )
        assert (terminal, json.loads(path.read_text())["events"]) == (
            True,
            [{**records[0], "withdrawn": True}],
        )


def test_spooled_withdrawal_uses_live_routes_across_two_profile_homes(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from gateway.profile_routing import ProfileRoute
    from gateway.run import _profile_runtime_scope
    from plugins.platforms.matrix.adapter import MatrixAdapter

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    launch = tmp_path / ".hermes"
    launch.mkdir()
    homes = {profile: launch / "profiles" / profile for profile in ("a", "b")}
    for home in homes.values():
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
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
    runner._wire_adapter_handlers(adapter)
    active = is_multiplex_active()
    set_multiplex_active(True)
    observations = []
    try:
        for index, profile in enumerate(("a", "b", "a")):
            source = adapter.build_source(
                chat_id=profile, user_id="author", chat_type="group"
            )
            assert adapter._canonicalize(source) is not None
            message_id = f"input-{index}"
            event = MessageEvent(text=message_id, source=source, message_id=message_id)
            with _profile_runtime_scope(homes[profile]):
                entry = runner.session_store.get_or_create_session(source)
                snapshot = PendingQueueSnapshot.capture(entry.session_key, [event])
                path = homes[profile] / "pending_messages" / f"input-{index}.json"
                path.parent.mkdir(exist_ok=True)
                atomic_json_write(path, snapshot.to_payload(), mode=0o600)
            found = adapter.withdraw_pending_message(
                message_id, chat_id=profile, sender_id="author"
            )
            observations.append((
                profile,
                found,
                json.loads(path.read_text())["events"],
                get_hermes_home(),
            ))
            assert observations[-1] == (
                profile,
                True,
                [{**snapshot.events[0], "withdrawn": True}],
                launch,
            )
        assert [
            (profile, found) for profile, found, _records, _home in observations
        ] == [("a", True), ("b", True), ("a", True)]
    finally:
        set_multiplex_active(active)
