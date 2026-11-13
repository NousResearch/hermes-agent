"""Ensure repairs a ready owner's dead Windows launcher (#150448, found by @kvnloo on #152232).

Ensure returns a ready owner before discovery, so the discovery-path repair in
``test_gateway_windows_dead_launcher`` never runs while a gateway is up.
"""
import pytest

from tests.hermes_cli import test_gateway_windows_dead_launcher as dead_launcher
from tests.hermes_cli.test_gateway_windows_dead_launcher import _interpreter

store_update = dead_launcher.store_update  # the Store-update fixture, on a default and a custom home


@pytest.mark.platforms("windows")  # ``refresh_ready_owner_launcher`` acts on native Windows only
def test_ensure_repairs_a_ready_owners_launcher(store_update, monkeypatch):
    """A gateway can be ready on the new payload (``hermes gateway start``, or the old payload cleaned up
    behind a gateway that outlived the update) while its logon launcher names the deleted interpreter.
    Ensure returns before discovery then, so it repairs the owner's launcher itself: here the root's,
    for a named profile the root multiplexer serves."""
    from hermes_agent.hermes_cli import gateway_runtime as runtime

    profile = store_update.home / "profiles" / "work"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    endpoint = runtime.GatewayEndpoint(profile_id=str(profile), instance_id="running", authority_epoch=1,
                                       runtime_protocol=1, api_origin="http://127.0.0.1:9999", supervisor="windows",
                                       capabilities=frozenset(), control_home=str(store_update.home))
    monkeypatch.setattr(runtime, "_update_fenced", lambda *_: False)
    monkeypatch.setattr(runtime, "_observe_replacing_stale", lambda *_: runtime.GatewayDiscovery("ready", endpoint=endpoint))

    assert runtime.ensure_gateway_runtime(profile, timeout=1).state == "ready"
    assert _interpreter(store_update.vbs) == str(store_update.new)
