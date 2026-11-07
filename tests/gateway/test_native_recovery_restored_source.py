"""Boot recovery binds a queued chat to the adapter its persisted transport names, and admission persists it.

A satellite profile routed through the shared bot that also connects a bot-less platform (a2a) is a
credential boundary, so a source with no live provenance resolves to no adapter for it. Boot recovery
must therefore revive the routing entry's origin through ``_restored_source`` (the persisted
``transport_profile`` re-pins the receiving bot), as every other revival from durable state does.
Read from the bare origin, the binding's adapter is ``None`` and the queued rows fail
``admission_conflict`` on every start (``check_automation_route``), so the chat's FIFO never runs.
"""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.pairing import PairingStore
from gateway.platforms.base import BasePlatformAdapter
from gateway.profile_routing import parse_profile_routes
from gateway.session import SessionEntry, SessionSource


class _Stub(BasePlatformAdapter):
    pass


_Stub.__abstractmethods__ = frozenset()

A2A = Platform("a2a")
CHAT = "72719239"


def _stub(platform, runner):
    adapter = _Stub.__new__(_Stub)
    adapter.platform, adapter.gateway_runner = platform, runner
    adapter.config = PlatformConfig(enabled=True, extra={})
    adapter._pending_messages, adapter._active_sessions = {}, {}
    return adapter


@pytest.fixture
def rig(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    home = tmp_path / "hh"
    (home / "profiles" / "ops").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    served = [("default", home), ("ops", home / "profiles" / "ops")]
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True, extra={})}
    runner.config.profile_routes = parse_profile_routes(
        [{"name": "ops-dm", "platform": "telegram", "profile": "ops", "chat_id": CHAT}])
    runner.pairing_store = PairingStore(profile="default")
    runner.pairing_stores = {}
    runner._primary_profile_name = "default"
    primary = _stub(Platform.TELEGRAM, runner)
    runner.adapters = {Platform.TELEGRAM: primary}
    # ``ops`` shares the default bot for Telegram but has connected its own a2a adapter.
    runner._profile_adapters = {"ops": {A2A: _stub(A2A, runner)}}
    runner._profile_bot_platforms = {"ops": {A2A}}
    with patch("hermes_cli.profiles.profiles_to_serve", return_value=served), \
            patch("hermes_cli.profiles.get_profile_dir",
                  side_effect=lambda n: home if n == "default" else home / "profiles" / n), \
            patch("hermes_cli.profiles.profile_exists", return_value=True):
        yield SimpleNamespace(runner=runner, primary=primary)


def _entry(transport_profile):
    origin = SessionSource(platform=Platform.TELEGRAM, chat_id=CHAT, chat_type="dm", user_id=CHAT, profile="ops")
    now = datetime.now()
    return SessionEntry(session_key=f"agent:ops:telegram:dm:{CHAT}", session_id="sid-1", created_at=now,
                        updated_at=now, origin=origin, transport_profile=transport_profile)


@pytest.mark.asyncio
async def test_recovery_binds_the_persisted_receiving_bot_for_a_profile_with_its_own_a2a(rig):
    from gateway.run_runtime import recover_profile_native_sessions

    runner, entry = rig.runner, _entry("default")
    # The bare origin carries no provenance: the a2a-owning profile has no Telegram adapter to deliver by.
    assert runner._adapter_for_source(entry.origin) is None

    bound = []

    class _Db:
        def _read_all(self, sql, *args):
            return [{"target_session_id": "sid-1"}]

    class _Authority:
        profile_id = "ops"
        db = _Db()
        runner = rig.runner

        def logical_owner(self, session_id):
            return session_id

        async def recover_native_sessions(self, bindings):
            bound.extend(bindings)
            return {sid: "ready" for sid, _, _ in bindings}

    runner.session_store = SimpleNamespace(list_sessions=lambda: [entry])
    with patch("gateway.session_authorities.owner_scope", lambda authority: __import__("contextlib").nullcontext()), \
            patch("gateway.session_detached.recover_detached_sessions", lambda authority, pending: []):
        outcome = await recover_profile_native_sessions(runner, _Authority())

    assert outcome == {"sid-1": "ready"}
    [(sid, source, adapter)] = bound
    assert sid == "sid-1" and adapter is rig.primary
    assert runner._session_key_for_source(source) == entry.session_key


def test_admitted_source_is_pinned_so_its_entry_persists_the_receiving_bot(rig):
    """The source an admission rebuilds from its payload is the one the route's session entry is created
    from. Left unpinned it persists no ``transport_profile``, which is exactly what boot recovery needs."""
    import weakref

    from gateway.session_envelope import restore_native_source
    from gateway.session_identity import transport_profile_of

    origin = SessionSource(platform=Platform.TELEGRAM, chat_id=CHAT, chat_type="dm", user_id=CHAT, profile="ops")
    payload = {"native_text_v1": {"source": {**origin.to_dict(), "is_bot": False}, "provenance": {}}}

    def restore_provenance(runner, source, provenance):  # what the real one installs: the connector's adapter
        source._transport_adapter_ref = weakref.ref(rig.primary)

    with patch("gateway.session_ingress_context.restore_provenance", restore_provenance), \
            patch("gateway.session_ingress_context.callback_runner", lambda: rig.runner):
        source = restore_native_source(payload)

    assert transport_profile_of(source) == "default"
