"""An admitted plugin origin survives native recovery and follows the current turn."""

import asyncio
from dataclasses import replace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.session_envelope import prepare_native, restore_native, snapshot_native
from gateway.trusted_turn_origin import current_origin
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
    restored = restore_native(payload, runner)
    assert restored._trusted_origin_claim["claim"]["user"] == "U1"

    forged = dict(payload)
    forged["native_text_v1"] = dict(payload["native_text_v1"])
    forged["native_text_v1"]["source"] = dict(payload["native_text_v1"]["source"], user_id="U2")
    with pytest.raises(RuntimeStoreError):
        restore_native(forged, runner)
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
    assert seen[0].claims["sample"]["user"] == "U1" or seen[1].claims["sample"]["user"] == "U1"
    with pytest.raises(TypeError):
        seen[0].claims["new"] = "spoof"
    with pytest.raises(TypeError):
        seen[0].claims["sample"]["nested"]["markers"] = ("spoof",)
    assert current_origin() is None

    seen.clear()
    async with _recursive_probe(runner, event, second, internal, seen):
        pass
    assert [(o.user_id if o else None) for o in seen] == ["U1", "U2", None, None, "U1"]
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
