"""Room threading commands persist on the receiving bot, not a routed runtime."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import hermes_yaml as yaml

from agent.secret_scope import set_multiplex_active
from gateway.config import GatewayConfig, Platform, load_gateway_config
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.session_identity import resolve_identity
from hermes_cli.commands import resolve_command
from hermes_cli.plugins import discover_plugins
from plugins.platforms.matrix.adapter import MatrixAdapter


ROOM = "!continuous:example.org"
OTHER_ROOM = "!tasks:example.org"


def _setup(tmp_path, monkeypatch, layout):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    launch = tmp_path / ".hermes"
    secondary = launch / "profiles" / "beta"
    routed = launch / "profiles" / "routed"
    for home in (launch, secondary, routed):
        home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    settings = {
        "enabled": True, "user_id": "@hermes:example.org", "require_mention": False,
        "session_scope": "room", "dm_mention_threads": True,
        "group_allow_admin_from": ["@alice:example.org"],
        "allow_admin_from": ["@alice:example.org"],
        "room_overrides": {ROOM: {"auto_thread": False}, OTHER_ROOM: {"auto_thread": True}},
    }
    config = {"matrix": settings}
    if layout == "platforms":
        config = {"platforms": config}
    elif layout == "gateway":
        config = {"gateway": config}
    elif layout == "extra":
        config = {"matrix": settings, "platforms": {"matrix": {"extra": {
            "room_overrides": {ROOM: {"auto_thread": False}, OTHER_ROOM: {"auto_thread": True}},
        }}}}
    for home in (launch, secondary, routed):
        (home / "config.yaml").write_text(
            "# Preserve this comment\n" + json.dumps({
                **config, "terminal": {"cwd": "${KEEP_THIS_TEMPLATE}"},
            }), encoding="utf-8")
    discover_plugins()
    runner = object.__new__(GatewayRunner)
    runner._primary_profile_name = "default"
    runner._resolve_profile_home_for_source = lambda _source: routed
    runner._session_key_for_source = lambda _source: "k"
    runner._hm_command_hooks = AsyncMock(return_value=(False, None, None))
    adapters, configs = {}, {}
    for name, home in (("default", launch), ("beta", secondary)):
        with _profile_runtime_scope(home, prepared_secret_scope={}):
            configs[name] = load_gateway_config()
            adapter = MatrixAdapter(configs[name].platforms[Platform.MATRIX])
        adapter.set_owner_profile(name)
        adapter._background_read_receipt = MagicMock()
        adapter._get_display_name = AsyncMock(return_value="Alice")
        adapter._get_room_member_count = AsyncMock(return_value=2)
        adapters[name] = adapter
    runner.config = configs["default"]
    runner.config.multiplex_profiles = True
    runner.adapters = {Platform.MATRIX: adapters["default"]}
    runner._profile_adapters = {"beta": {Platform.MATRIX: adapters["beta"]}}
    runner._profile_configs = configs
    return runner, adapters, {"default": launch, "beta": secondary, "routed": routed}


def _event(runner, adapter, args, user_id="@alice:example.org", chat_type="dm"):
    source = adapter.build_source(
        chat_id=ROOM, chat_type=chat_type, user_id=user_id, thread_id="$existing")
    source.profile = "routed"
    resolve_identity(source, runner=runner, adapter=adapter)
    return MessageEvent(text=f"/threading {args}".strip(), source=source)


async def _dispatch(runner, event, busy):
    handled, reply, _command, canonical = await runner._hm_resolve_command(
        event, event.source, "k")
    if handled:
        return reply
    assert canonical == "threading"
    if busy:
        return await runner._dispatch_busy_slash_command(
            event, resolve_command(canonical), "k", event.source)
    handled, reply = await runner._hm_dispatch_canonical_command(
        event, event.source, "k", canonical)
    assert handled
    return reply


@pytest.mark.asyncio
@pytest.mark.parametrize("layout", ["matrix", "platforms", "gateway", "extra"])
@pytest.mark.parametrize("busy", [False, True])
@pytest.mark.parametrize("is_dm", [False, True])
async def test_threading_command_updates_only_receiving_room_and_survives_restart(
        tmp_path, monkeypatch, layout, busy, is_dm):
    runner, adapters, homes = _setup(tmp_path, monkeypatch, layout)
    set_multiplex_active(True)
    try:
        for name in ("default", "beta", "default"):
            adapter, home = adapters[name], homes[name]
            adapter._get_room_member_count.return_value = 2 if is_dm else 3
            chat_type = "dm" if is_dm else "group"
            untouched = {key: (path / "config.yaml").read_bytes()
                         for key, path in homes.items() if key != name}
            for mode, override in (("on", True), ("status", True), ("off", False), ("default", None)):
                reply = await _dispatch(runner, _event(runner, adapter, mode, chat_type=chat_type), busy)
                assert ROOM in reply
                assert adapter._room_auto_thread_overrides.get(ROOM) is override
                assert adapter._room_auto_thread_overrides[OTHER_ROOM] is True
                with _profile_runtime_scope(home, prepared_secret_scope={}):
                    restarted = MatrixAdapter(load_gateway_config().platforms[Platform.MATRIX])
                assert restarted._room_auto_thread_overrides.get(ROOM) is override
                assert restarted._room_auto_thread_overrides[OTHER_ROOM] is True
                ctx = await adapter._resolve_message_context(
                    ROOM, "@alice:example.org", "$next", "Hello", {"body": "Hello"}, {})
                assert ctx[-1].thread_id == ("$next" if override else None)
                real = await adapter._resolve_message_context(
                    ROOM, "@alice:example.org", "$reply", "Hello", {"body": "Hello"},
                    {"rel_type": "m.thread", "event_id": "$existing"})
                assert real[-1].thread_id == "$existing"
            status = await _dispatch(runner, _event(runner, adapter, "", chat_type=chat_type), busy)
            assert ("mentions" if is_dm else "off") in status.lower()
            text = (home / "config.yaml").read_text(encoding="utf-8")
            assert "# Preserve this comment" in text
            assert yaml.safe_load(text)["terminal"]["cwd"] == "${KEEP_THIS_TEMPLATE}"
            assert all((homes[key] / "config.yaml").read_bytes() == before
                       for key, before in untouched.items())

        adapter = adapters["default"]
        event = _event(runner, adapter, "on")
        reply = AsyncMock(return_value="handled")
        adapter._message_handler = reply
        adapter.send = AsyncMock()
        key = adapter._source_session_key(event.source)
        adapter._active_sessions[key] = asyncio.Event()
        await adapter.handle_message(event)
        reply.assert_awaited_once()
        assert key not in adapter._pending_messages
    finally:
        set_multiplex_active(False)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [
    "invalid", "denied", "managed", "managed-other", "locked",
    "write", "unreadable", "unsupported", "unavailable",
])
@pytest.mark.parametrize("busy", [False, True])
async def test_threading_refusals_never_change_live_or_persisted_policy(
        tmp_path, monkeypatch, failure, busy):
    runner, adapters, homes = _setup(tmp_path, monkeypatch, "matrix")
    adapter, home = adapters["default"], homes["default"]
    event = _event(runner, adapter, "on")
    def configure_managed():
        managed = tmp_path / "managed"
        managed.mkdir()
        (managed / "config.yaml").write_text(json.dumps({
            "matrix": {"room_overrides": {
                ROOM if failure == "managed" else OTHER_ROOM: {"auto_thread": False},
            }},
        }), encoding="utf-8")
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))

    def fail_write(*_args, **_kwargs):
        raise OSError("disk full")

    def remove_adapters():
        runner.adapters = {}
        runner._profile_adapters = {}

    setups = {
        "invalid": lambda: setattr(event, "text", "/threading on off"),
        "denied": lambda: setattr(event.source, "user_id", "@mallory:example.org"),
        "managed": configure_managed,
        "managed-other": configure_managed,
        "locked": lambda: monkeypatch.setenv("HERMES_MANAGED", "nixos"),
        "write": lambda: monkeypatch.setattr("hermes_cli.config.atomic_config_replace", fail_write),
        "unreadable": lambda: (home / "config.yaml").write_text("matrix: [invalid", encoding="utf-8"),
        "unsupported": lambda: monkeypatch.setattr(
            adapter, "configure_room_threading",
            BasePlatformAdapter.configure_room_threading.__get__(adapter)),
        "unavailable": remove_adapters,
    }
    setups[failure]()
    before = (home / "config.yaml").read_bytes()
    policy = dict(adapter._room_auto_thread_overrides)
    set_multiplex_active(True)
    try:
        reply = await _dispatch(runner, event, busy)
        assert reply
        assert adapter._room_auto_thread_overrides == policy
        assert (home / "config.yaml").read_bytes() == before
    finally:
        set_multiplex_active(False)
