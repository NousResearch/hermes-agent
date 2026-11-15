"""Regression: a profile route declared by ``user_id`` must resolve inside admission.

Background: ``gateway/run.py::_route_profile_for_source`` (ingress stamping) matches
``profile_routes`` with the sender's ``user_id``, so a route written as
``user_id: "+15550000001"`` stamps its profile onto the source. The admission path
matches the same routes again in ``gateway/session_ingress_context.py::_binding`` — but
it never passed ``user_id``, and ``ProfileRoute.matches`` returns ``False`` for a route
that declares a ``user_id`` when none is supplied. The ``user_id`` route therefore
cannot match there, a catch-all route wins, and the final
``(source.profile or primary) != runtime_profile`` check raises
``RuntimeStoreError('profile_mismatch')`` — every normal DM from that sender is refused
while the same gateway's inbound filter has just accepted it.

Without the fix ``capture_provenance`` raises ``profile_mismatch``; with it, the same
route the ingress stamping used resolves the runtime profile.
"""
import weakref
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.profile_routing import parse_profile_routes
from gateway.run import GatewayRunner
from gateway.run_runtime import initialize_gateway_runtime
from gateway.runtime_ownership import process_ownership
from gateway.session import SessionSource
from gateway.session_ingress_context import (
    capture_provenance, native_callback, register_transport_home)

from hermes_state_runtime import RuntimeStoreError


class _FakeAdapter:
    """Weakref-able stand-in for a live platform adapter (SimpleNamespace cannot weakref)."""


ALICE = "+15550000001"
BOB = "+15550000002"


def _reserve_homes(tmp_path, monkeypatch, names=("guest",)):
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
        (home / "config.yaml").write_text("{}")
        homes.append((name, home))
    return root, homes


def _runner(root, homes):
    config = GatewayConfig()
    config.multiplex_profiles = True
    config._runtime_profile_homes = tuple(homes)
    config.profile_routes = parse_profile_routes([
        {"name": "alice-dm", "platform": "bluebubbles", "user_id": ALICE, "profile": "guest"},
        {"name": "everyone-else", "platform": "bluebubbles", "profile": "default"},
    ])
    return GatewayRunner(config)


async def _admit_and_read_provenance(tmp_path, monkeypatch, user_id):
    """Admit one BlueBubbles DM the way a live adapter does and return its provenance."""
    root, homes = _reserve_homes(tmp_path, monkeypatch)
    runner = _runner(root, homes)
    process_ownership.reserve([home for _, home in homes])
    try:
        await initialize_gateway_runtime(runner)
        register_transport_home(runner, None, root)
        adapter = _FakeAdapter()
        runner.adapters = {Platform.BLUEBUBBLES: adapter}
        runner._is_user_authorized_for_source = lambda *a, **k: True

        stamped = None
        source = SessionSource(
            platform=Platform.BLUEBUBBLES, chat_id=f"any;-;{user_id}", chat_type="dm",
            user_id=user_id, user_name=user_id)
        # session_identity.identity_of stamps the routed profile onto the primary's source and
        # keeps the receiving adapter as provenance; hand-built sources need both, exactly as
        # build_source leaves them.
        stamped = runner._profile_name_for_source(source)
        source.profile = stamped
        source._transport_adapter_ref = weakref.ref(adapter)

        event = MessageEvent(
            text="hello", message_type=MessageType.TEXT, source=source, message_id="probe-msg")
        with native_callback(runner, event, root, None):
            provenance = capture_provenance(runner, event)
        return provenance, homes, stamped
    finally:
        for authority in list(getattr(runner, "session_authorities", []) or []):
            authority.db.close()
        for _, home in homes:
            process_ownership.release(home)


@pytest.mark.asyncio
async def test_user_id_route_resolves_the_profile_ingress_stamped(tmp_path, monkeypatch):
    """A DM from the routed sender is admitted on the route's profile, not refused."""
    provenance, homes, stamped = await _admit_and_read_provenance(tmp_path, monkeypatch, ALICE)

    assert stamped == "guest", "the ingress stamping is the reference for this route"
    assert provenance is not None
    assert provenance["runtime_home"] == str(dict(homes)["guest"])


@pytest.mark.asyncio
async def test_a_sender_outside_the_user_id_route_still_follows_the_catch_all(
        tmp_path, monkeypatch):
    """The sender's own ``user_id`` is matched — the route does not apply to everybody."""
    provenance, homes, stamped = await _admit_and_read_provenance(tmp_path, monkeypatch, BOB)

    assert stamped == "default"
    assert provenance is not None
    assert provenance["runtime_home"] == str(dict(homes)["default"])
