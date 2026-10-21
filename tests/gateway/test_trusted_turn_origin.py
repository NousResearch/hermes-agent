"""An admitted plugin origin survives native recovery and follows the current turn."""

import asyncio
from dataclasses import replace
from pathlib import Path
import weakref

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.profile_routing import parse_profile_routes
from gateway.run_runtime import initialize_gateway_runtime
from gateway.runtime_ownership import process_ownership
from gateway.session import SessionSource
from gateway.session_envelope import prepare_native, restore_native, snapshot_native
from gateway.session_identity import RoutingIdentity
from gateway.trusted_turn_origin import (capture_origin_claim, current_origin, origin_for_event,
                                         restore_origin_claim)
from hermes_cli.plugins import PluginContext, PluginManager
from hermes_cli.plugins_manifest import PluginManifest
from hermes_state_runtime import RuntimeStoreError


def _setup(tmp_path, monkeypatch):
    manager = PluginManager(scope_key=str(tmp_path))
    context = PluginContext(PluginManifest(name="sample", key="sample", source="user"), manager)
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    adapter = object()
    monkeypatch.setattr(runner, "_adapter_for_source", lambda source: adapter)
    monkeypatch.setattr(runner, "_is_user_authorized_for_source", lambda source, **kwargs: True)

    def capture(event):
        raw = event.raw_message
        if not isinstance(raw, dict) or raw.get("user") != event.source.user_id:
            raise ValueError("invalid original event")
        return {"user": raw["user"], "channel": raw["channel"], "team": raw["team"],
                "nested": {"markers": ["verified"]}}

    def validate(claim, source):
        return (claim.get("user") == source.user_id
                and claim.get("channel") == source.chat_id
                and claim.get("team") == source.scope_id)

    registration = context.register_inbound_origin_projector(
        platform="slack", capture=capture, validate=validate)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", user_id="U1", scope_id="T1")
    event = MessageEvent(text="hello", source=source, message_id="m1",
                         raw_message={"user": "U1", "channel": "C1", "team": "T1", "unused": "secret"})
    return runner, event, registration, context


@pytest.mark.asyncio
async def test_native_origin_projection_freezes_before_await_and_rechecks_source(tmp_path, monkeypatch):
    runner, event, registration, _ = _setup(tmp_path, monkeypatch)
    from gateway import session_ingress_context

    async def delayed_roles(*args):
        event.raw_message["user"] = "ATTACKER"
        return False

    monkeypatch.setattr(session_ingress_context, "reauthorize_roles", delayed_roles)
    payload = await prepare_native(runner, event)
    assert "raw_message" not in str(payload)
    assert "secret" not in str(payload)
    assert payload["native_text_v1"]["origin_claim_v1"]["claim"]["user"] == "U1"
    assert payload["native_text_v1"]["origin_claim_v1"]["transport_profile"] == "default"
    restored = restore_native(payload, runner)
    assert restored._trusted_origin_claim["claim"]["user"] == "U1"
    assert restored._trusted_origin_claim["transport_profile"] == "default"

    forged = dict(payload)
    forged["native_text_v1"] = dict(payload["native_text_v1"])
    forged["native_text_v1"]["source"] = dict(payload["native_text_v1"]["source"], user_id="U2")
    with pytest.raises(RuntimeStoreError):
        restore_native(forged, runner)
    other_bot = dict(payload)
    other_bot["native_text_v1"] = dict(payload["native_text_v1"])
    other_bot["native_text_v1"]["origin_claim_v1"] = dict(
        payload["native_text_v1"]["origin_claim_v1"], transport_profile="other-bot")
    with pytest.raises(RuntimeStoreError):
        restore_native(other_bot, runner)
    registration.dispose()
    with pytest.raises(RuntimeStoreError):
        restore_native(payload, runner)


@pytest.mark.asyncio
async def test_origin_scopes_first_queued_concurrent_missing_and_internal_turns(tmp_path, monkeypatch):
    runner, event, registration, _ = _setup(tmp_path, monkeypatch)
    restored = restore_native(snapshot_native(runner, event), runner)
    restored._gateway_accepted = True
    event._gateway_accepted = True
    second = replace(event, message_id="m2", source=replace(event.source, user_id="U2"),
                     raw_message={"user": "U2", "channel": "C1", "team": "T1"})
    second._gateway_accepted = True
    internal = replace(event, internal=True)
    internal._gateway_accepted = True
    seen = []

    async def inner(*args, **kwargs):
        origin = current_origin()
        seen.append(origin)
        await asyncio.sleep(0)
        assert current_origin() is origin
        assert await asyncio.to_thread(current_origin) is origin
        return {"final_response": "ok"}

    monkeypatch.setattr(runner, "_run_agent_inner", inner)

    async def run(msg, source, origin_event):
        return await runner._run_agent(msg, "", [], source, "s", turn_event=origin_event)

    assert current_origin() is None
    await asyncio.gather(run("first", restored.source, restored),
                         run("concurrent", second.source, second))
    assert {(origin.user_id, origin.message_ts) for origin in seen} == {("U1", "m1"), ("U2", "m2")}
    assert {origin.transport_profile for origin in seen} == {"default", None}
    assert seen[0].claims["sample"]["user"] == "U1" or seen[1].claims["sample"]["user"] == "U1"
    with pytest.raises(TypeError):
        seen[0].claims["new"] = "spoof"
    with pytest.raises(TypeError):
        seen[0].claims["sample"]["nested"]["markers"] = ("spoof",)
    assert current_origin() is None

    seen.clear()
    # A queued message cannot borrow the currently executing native admission's
    # acceptance if that queued message itself failed ingress.
    from gateway.session_ingress import executing_admission
    unaccepted = replace(second, message_id="m3")
    token = executing_admission.set(True)
    try:
        await run("unaccepted", unaccepted.source, unaccepted)
    finally:
        executing_admission.reset(token)
    assert seen == [None]
    seen.clear()
    async with _recursive_probe(runner, event, second, internal, seen):
        pass
    assert [(o.user_id if o else None) for o in seen] == ["U1", "U2", None, None, "U1"]
    assert current_origin() is None
    registration.dispose()


def test_receiving_bot_remains_distinct_from_routed_runtime_on_queued_and_restore(tmp_path, monkeypatch):
    runner, event, registration, _ = _setup(tmp_path, monkeypatch)
    source = replace(event.source, profile="brain")
    source._identity = RoutingIdentity(
        transport_profile="peppy", runtime_profile="brain", authorization_home=tmp_path,
        runtime_home=tmp_path, multiplexed=True)
    queued = replace(event, source=source)
    queued._gateway_accepted = True
    projected = capture_origin_claim(runner, queued, native=True,
                                     provenance={"transport_profile": "peppy"})
    assert projected["transport_profile"] == "peppy"
    assert projected["binding"][1] == "brain"

    restored_source = replace(source)
    restored_source._identity = source._identity
    restored = replace(queued, source=restored_source)
    restore_origin_claim(runner, restored, projected, provenance={"transport_profile": "peppy"})
    restored._native_origin_restored = True
    restored._gateway_accepted = True
    origin = origin_for_event(runner, restored, restored.source)
    assert (origin.profile, origin.transport_profile) == ("brain", "peppy")

    # Runtime route remains the same, but a later different receiving bot cannot
    # borrow the first bot's projection or its plugin privileges.
    restored.source._identity = replace(source._identity, transport_profile="other-bot")
    assert origin_for_event(runner, restored, restored.source) is None
    restored.source._identity = replace(source._identity, runtime_profile="wrong-runtime")
    assert origin_for_event(runner, restored, restored.source) is None
    with pytest.raises(RuntimeStoreError):
        restore_origin_claim(runner, restored, projected,
                             provenance={"transport_profile": "other-bot"})
    registration.dispose()


@pytest.mark.asyncio
async def test_native_restore_pins_secondary_receiving_bot_across_runtime_route(tmp_path, monkeypatch):
    _, _, registration, _ = _setup(tmp_path, monkeypatch)
    root = tmp_path / ".hermes"
    root.mkdir(mode=0o700)
    homes = [("default", root)]
    for name in ("peppy", "brain"):
        home = root / "profiles" / name
        home.mkdir(parents=True, mode=0o700)
        (home / "config.yaml").write_text("{}")
        homes.append((name, home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    config = GatewayConfig(multiplex_profiles=True)
    config._runtime_profile_homes = tuple(homes)
    config.profile_routes = parse_profile_routes([
        {"name": "test-channel", "platform": "slack", "chat_id": "C1",
         "bot_profile": "peppy", "profile": "brain"}])
    runner = GatewayRunner(config)
    process_ownership.reserve([home for _, home in homes])
    try:
        await initialize_gateway_runtime(runner)
        from gateway.session_ingress_context import native_callback, register_transport_home

        class Adapter:
            pass

        adapter = Adapter()
        runner._profile_adapters["peppy"] = {Platform.SLACK: adapter}
        runner._is_user_authorized_for_source = lambda *args, **kwargs: True
        register_transport_home(runner, "peppy", dict(homes)["peppy"])
        source = SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="channel",
                               user_id="U1", scope_id="T1")
        source._transport_adapter_ref = weakref.ref(adapter)
        source.profile = runner._profile_name_for_source(source)
        assert source.profile == "brain"
        event = MessageEvent(text="hello", source=source, message_id="m1",
                             raw_message={"user": "U1", "channel": "C1", "team": "T1"})
        with native_callback(runner, event, dict(homes)["peppy"], "peppy"):
            payload = snapshot_native(runner, event)
        claim = payload["native_text_v1"]["origin_claim_v1"]
        assert (claim["binding"][1], claim["transport_profile"]) == ("brain", "peppy")
        restored = restore_native(payload, runner)
        restored._gateway_accepted = True
        origin = origin_for_event(runner, restored, restored.source)
        assert (origin.profile, origin.transport_profile) == ("brain", "peppy")

        forged = dict(payload)
        forged["native_text_v1"] = dict(payload["native_text_v1"])
        forged["native_text_v1"]["origin_claim_v1"] = dict(claim, transport_profile="default")
        with pytest.raises(RuntimeStoreError):
            restore_native(forged, runner)
    finally:
        registration.dispose()
        for authority in list(getattr(runner, "session_authorities", []) or []):
            authority.db.close()
        for _, home in homes:
            process_ownership.release(home)


@pytest.mark.asyncio
async def test_recursive_queued_turn_uses_its_own_receiving_bot(tmp_path, monkeypatch):
    runner, event, registration, _ = _setup(tmp_path, monkeypatch)

    def routed(message_id, transport):
        source = replace(event.source, profile="brain")
        source._identity = RoutingIdentity(
            transport_profile=transport, runtime_profile="brain", authorization_home=tmp_path,
            runtime_home=tmp_path, multiplexed=True)
        turn = replace(event, message_id=message_id, source=source)
        turn._gateway_accepted = True
        return turn

    first, queued = routed("m1", "peppy"), routed("m2", "other-bot")
    seen = []

    async def inner(*args, **kwargs):
        seen.append(current_origin())
        if len(seen) == 1:
            await runner._run_agent("queued", "", [], queued.source, "s", turn_event=queued)
            assert current_origin() is seen[0]
        return {"final_response": "ok"}

    monkeypatch.setattr(runner, "_run_agent_inner", inner)
    await runner._run_agent("first", "", [], first.source, "s", turn_event=first)
    assert [(origin.profile, origin.transport_profile, origin.message_ts) for origin in seen] == [
        ("brain", "peppy", "m1"), ("brain", "other-bot", "m2")]
    assert current_origin() is None
    registration.dispose()


@pytest.mark.asyncio
async def test_legacy_native_row_cannot_recapture_missing_claim(tmp_path, monkeypatch):
    runner, event, registration, _ = _setup(tmp_path, monkeypatch)
    payload = snapshot_native(runner, event)
    payload["native_text_v1"].pop("origin_claim_v1")
    old = restore_native(payload, runner)
    old._gateway_accepted = True
    old.raw_message = {"user": "U1", "channel": "C1", "team": "T1"}  # even if later injected
    captured = []

    async def inner(*args, **kwargs):
        captured.append(current_origin())
        return {"final_response": "ok"}

    monkeypatch.setattr(runner, "_run_agent_inner", inner)
    await runner._run_agent("old", "", [], old.source, "s", turn_event=old)
    assert captured[0] is not None
    assert captured[0].claims == {}
    registration.dispose()


@pytest.mark.asyncio
async def test_restored_claim_cannot_follow_mutated_sender(tmp_path, monkeypatch):
    runner, event, registration, _ = _setup(tmp_path, monkeypatch)
    restored = restore_native(snapshot_native(runner, event), runner)
    restored._gateway_accepted = True
    restored.source.user_id = "U2"
    seen = []

    async def inner(*args, **kwargs):
        seen.append(current_origin())
        return {"final_response": "ok"}

    monkeypatch.setattr(runner, "_run_agent_inner", inner)
    await runner._run_agent("changed", "", [], restored.source, "s", turn_event=restored)
    assert seen == [None]
    registration.dispose()


class _recursive_probe:
    def __init__(self, runner, first, queued, internal, seen):
        self.runner, self.first, self.queued, self.internal, self.seen = runner, first, queued, internal, seen

    async def __aenter__(self):
        async def inner(*args, **kwargs):
            self.seen.append(current_origin())
            if len(self.seen) == 1:
                await self.runner._run_agent("queued", "", [], self.queued.source, "s", turn_event=self.queued)
                await self.runner._run_agent("internal", "", [], self.internal.source, "s", turn_event=self.internal)
                await self.runner._run_agent("steer", "", [], self.first.source, "s")
                self.seen.append(current_origin())
            return {"final_response": "ok"}
        self.runner._run_agent_inner = inner
        await self.runner._run_agent("first", "", [], self.first.source, "s", turn_event=self.first)

    async def __aexit__(self, *args):
        return False


def test_projection_rejects_oversize_nonjson_and_duplicate_registration(tmp_path, monkeypatch):
    runner, event, registration, context = _setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        context.register_inbound_origin_projector(platform="slack", capture=lambda event: {},
                                                  validate=lambda claim, source: True)
    registration.dispose()
    context.register_inbound_origin_projector(platform="slack", capture=lambda event: {"data": "x" * 20000},
                                              validate=lambda claim, source: True)
    with pytest.raises(RuntimeStoreError):
        snapshot_native(runner, event)

    context._manager._inbound_origin_projectors["slack"] = (
        "sample", lambda event: {"bad": {1, 2}}, lambda claim, source: True)
    with pytest.raises(RuntimeStoreError):
        snapshot_native(runner, event)

    context._manager._inbound_origin_projectors["slack"] = (
        "sample", lambda event: (_ for _ in ()).throw(ValueError("bad")), lambda claim, source: True)
    with pytest.raises(RuntimeStoreError, match="permission_denied"):
        snapshot_native(runner, event)
