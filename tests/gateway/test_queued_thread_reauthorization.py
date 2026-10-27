"""Regression: a queued native turn on a shared-bot satellite's Discord THREAD must re-authorize.

Background: a unified (``multiplex_profiles``) gateway serves several profiles through ONE
platform bot; each routed profile has no adapter of its own and drains through the primary's
map (``authz_mixin._is_shared_bot_satellite``). Ingress admits such a message fine, because
``capture_provenance`` only needs the *primary* adapter (``transport_profile: null``) and the
route match resolves ``runtime_home`` to the served profile.

A message that arrives while a turn is still running is committed ``queued`` and re-validated
before it runs (``session_envelope.check_native_route`` -> ``restore_native`` ->
``restore_provenance`` -> ``_binding``, then ``_validate_native``). That preflight resolves the
adapter from ``source.profile`` — and for a shared-bot satellite that resolution is the one
path the capture never exercised. When it cannot name a single registered adapter, the row is
refused ``not_found`` and the whole session FIFO stays paused forever: every later message in
the thread lands behind it and is answered with a "could not be re-authorized" notice.
"""

from __future__ import annotations

import weakref
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.profile_routing import parse_profile_routes
from gateway.run import GatewayRunner
from gateway.runtime_ownership import process_ownership
from gateway.run_runtime import initialize_gateway_runtime
from gateway.session import SessionSource
from gateway.session_envelope import restore_native, snapshot_native, _validate_native
from gateway.session_ingress_context import native_callback, register_transport_home

GUILD = "1546589088034070698"
CTO_CHANNEL = "1561271563339169855"
CTO_THREAD = "1564423227227246715"
USER = "1546583770310909952"


class _FakeDiscordAdapter:
    """Weakref-able stand-in for the live bot; ``token`` feeds the connector fingerprint."""

    token = "fake-discord-bot-token"


def _reserve_homes(tmp_path, monkeypatch, names=("cto",)):
    root = tmp_path / ".hermes"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    import hermes_state
    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    homes = [("default", root)]
    for name in names:
        home = root / "profiles" / name
        home.mkdir(parents=True, mode=0o700)
        (home / "config.yaml").write_text("{}\n")
        homes.append((name, home))
    return root, homes


def _runner(root, homes):
    config = GatewayConfig()
    config.multiplex_profiles = True
    config._runtime_profile_homes = tuple(homes)
    config.profile_routes = parse_profile_routes([
        # The deployed shape: routes name the parent CHANNEL; the traffic is in its threads.
        {"name": "cto-channel", "platform": "discord", "guild_id": GUILD,
         "chat_id": CTO_CHANNEL, "profile": "cto"},
    ])
    return GatewayRunner(config)


def _thread_source():
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id=CTO_THREAD,
        chat_name="Clear Choice Group / #cto / design",
        chat_type="thread",
        user_id=USER,
        user_name="clearchoicegroup",
        scope_id=GUILD,
        guild_id=GUILD,
        thread_id=CTO_THREAD,
        parent_chat_id=CTO_CHANNEL,
    )


async def _admitted_queued_payload(tmp_path, monkeypatch):
    """Snapshot one in-thread Discord message the way the live ingress does."""
    root, homes = _reserve_homes(tmp_path, monkeypatch)
    runner = _runner(root, homes)
    process_ownership.reserve([home for _, home in homes])
    try:
        await initialize_gateway_runtime(runner)
        register_transport_home(runner, None, root)  # one bot, the primary's transport home
        adapter = _FakeDiscordAdapter()
        runner.adapters = {Platform.DISCORD: adapter}
        runner._is_user_authorized_for_source = lambda *a, **k: True

        source = _thread_source()
        # Ingress stamps the routed profile onto the primary's source and keeps the receiving
        # adapter as its transport provenance (``session_identity``).
        source.profile = runner._profile_name_for_source(source)
        source._transport_adapter_ref = weakref.ref(adapter)

        event = MessageEvent(
            text="What did you find out?", message_type=MessageType.TEXT,
            source=source, message_id="probe-msg")
        with native_callback(runner, event, root, None):
            # Exactly what ``admit_native`` persists for a fresh native turn.
            payload = snapshot_native(runner, event)
        provenance = payload["native_text_v1"]["provenance"]
        return runner, payload, provenance, source, adapter
    finally:
        for authority in list(getattr(runner, "session_authorities", []) or []):
            authority.db.close()
        for _, home in homes:
            process_ownership.release(home)


@pytest.mark.asyncio
async def test_thread_message_on_shared_bot_satellite_keeps_primary_provenance(tmp_path, monkeypatch):
    """The capture half: a routed thread stamps ``cto`` but binds the primary bot."""
    _, _, provenance, source, _ = await _admitted_queued_payload(tmp_path, monkeypatch)

    assert source.profile == "cto"
    assert provenance["platform"] == "discord"
    assert provenance["transport_profile"] is None, "the bot belongs to the launch profile"
    assert provenance["runtime_home"].endswith("/profiles/cto")


@pytest.mark.asyncio
async def test_queued_thread_message_reauthorizes_at_claim_time(tmp_path, monkeypatch):
    """The claim-time half: the same row must re-authorize, not refuse ``not_found``."""
    runner, payload, provenance, _, adapter = await _admitted_queued_payload(tmp_path, monkeypatch)

    restored = restore_native(payload, runner)
    assert restored.source.profile == "cto"
    # ``_validate_native`` is the exact gate ``check_native_route`` runs before a claim.
    _validate_native(runner, restored, provenance)
    assert runner._adapter_for_source(restored.source) is adapter


@pytest.mark.asyncio
async def test_restored_satellite_lane_still_reauthorizes_its_queued_row(tmp_path, monkeypatch):
    """The lane is reaped while idle; the next row must re-authorize against the restored source.

    ``SessionEntry.transport_profile`` is what pins the receiving bot after a restore. A
    shared-bot satellite's row is written with ``None``, so ``restore_identity`` pins nothing
    and the restored source has no transport ref — the state the claim-time preflight sees.
    """
    from gateway.session_envelope import check_native_route

    runner, payload, provenance, source, adapter = await _admitted_queued_payload(tmp_path, monkeypatch)
    entry = runner.session_store.get_or_create_session(source)
    sid = entry.session_id
    provenance["connector"] = runner._adapter_credential_fingerprint(adapter) or provenance["connector"]

    restored_source = runner._restored_source(entry)   # exactly what authorize() mounts
    assert restored_source is not None
    live_adapter = runner._adapter_for_source(restored_source)
    assert live_adapter is adapter, "a shared-bot satellite answers through the primary bot"

    await check_native_route(runner, payload, sid, restored_source, live_adapter)



@pytest.mark.asyncio
async def test_queued_row_survives_a_bot_reconnect(tmp_path, monkeypatch):
    """The bot reconnects (a new adapter object) between ingest and claim.

    ``gateway.log`` shows exactly this: ``Discord WebSocket unhealthy`` -> forced reconnect ->
    ``Connected as Hermes Choice#5759``, and the refusal follows minutes later. The queued row's
    provenance still names the pre-reconnect adapter while the registry now holds its replacement.
    """
    from gateway.session_envelope import check_native_route

    runner, payload, provenance, source, old_adapter = await _admitted_queued_payload(tmp_path, monkeypatch)
    entry = runner.session_store.get_or_create_session(source)
    sid = entry.session_id

    # Reconnect: same credential, brand-new adapter object in the registry.
    new_adapter = _FakeDiscordAdapter()
    assert new_adapter is not old_adapter
    runner.adapters[Platform.DISCORD] = new_adapter
    provenance["connector"] = runner._adapter_credential_fingerprint(new_adapter) or provenance["connector"]

    # The live lane still holds the pre-reconnect transport reference, as a reaped lane does.
    live_source = runner._restored_source(entry)
    live_source._transport_adapter_ref = weakref.ref(old_adapter)
    live_adapter = runner._adapter_for_source(live_source)

    await check_native_route(runner, payload, sid, live_source, live_adapter)
