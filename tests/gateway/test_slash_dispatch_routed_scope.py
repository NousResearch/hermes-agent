"""Slash dispatch binds the ROUTED profile's runtime scope around every table handler (#119915).

Under multiplex the message handler runs under the RECEIVING bot's scope so authorization can read
its ``.env``; a chat that bot serves for ANOTHER profile (``profile_routes`` / ``bot_profile``) has a
different runtime home. Handlers reading home-relative state (``/memory`` and ``/skills`` pending
records, the on-disk memory store, config write-back) must see the runtime home, so the dispatcher
binds it once for the whole table instead of each handler re-entering the scope.
"""

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.session import SessionSource
from hermes_constants import get_hermes_home
from tools import write_approval as wa


@pytest.fixture
def homes(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    routed = tmp_path / "profiles" / "beta"
    (launch / "pending" / "memory").mkdir(parents=True)
    (routed / "pending" / "memory").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    return launch, routed


def _runner(routed_home):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._resolve_profile_home_for_source = lambda _source: routed_home
    runner._session_key_for_source = lambda _source: "k"
    runner._evict_cached_agent = lambda _key: None
    return runner


def _event(text):
    source = SessionSource(platform=Platform.TELEGRAM, user_id="u", user_name="u",
                           chat_id="c", chat_type="dm", profile="beta")
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source, message_id="m")


@pytest.mark.asyncio
async def test_idle_dispatch_runs_table_handlers_under_the_routed_runtime(homes):
    launch, routed = homes
    with _profile_runtime_scope(routed):
        staged = wa.stage_write(wa.MEMORY, {"action": "add", "target": "memory", "content": "beta"},
                                summary="beta-only", origin="foreground")
    runner = _runner(routed)
    event = _event("/memory pending")

    # Ambient scope is the LAUNCH home (what the receiving bot's handler binds); the dispatcher
    # must still resolve the routed runtime for the handler.
    with _profile_runtime_scope(launch):
        handled, reply = await runner._hm_dispatch_canonical_command(event, event.source, "k", "memory")
        assert get_hermes_home() == launch  # scope restored after dispatch

    assert handled and staged["id"] in reply
    assert not list((launch / "pending" / "memory").glob("*.json"))


@pytest.mark.asyncio
async def test_busy_dispatch_binds_the_same_runtime_scope(homes):
    launch, routed = homes
    from hermes_cli.commands import resolve_command
    seen = {}

    async def _probe(_event):
        seen["home"] = get_hermes_home()
        return "ok"

    runner = _runner(routed)
    runner._handle_restart_command = _probe
    event = _event("/restart")
    with _profile_runtime_scope(launch):
        assert await runner._dispatch_busy_slash_command(
            event, resolve_command("restart"), "k", event.source) == "ok"
    assert seen["home"] == routed
