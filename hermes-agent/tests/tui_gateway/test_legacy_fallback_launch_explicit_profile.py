"""A launch-profile socket that names another profile runs that verb in the named profile.

``chat_ws.gateway_ws`` runs a launch-profile Webapp socket in the launch profile's scope, so a legacy
verb sent with an explicit ``profile`` was answered in that scope unless its handler binds
``profile`` itself (``_profile_scoped``): one that ignores it read the launch profile's secrets
instead of refusing the unscoped read. ``dispatch_legacy`` binds the named profile, as it already
did for a socket whose ticket names a non-launch profile.
"""
import threading
import types

import pytest

LAUNCH_PORTAL = "https://launch-portal.invalid"
BEE_PORTAL = "https://bee-portal.invalid"


@pytest.fixture
def homes(tmp_path, monkeypatch):
    from agent import secret_scope
    from tui_gateway import launch_profile_policy, server
    launch = tmp_path / "root"
    bee = launch / "profiles" / "bee"
    bee.mkdir(parents=True)
    (launch / ".env").write_text(f"HERMES_PORTAL_BASE_URL={LAUNCH_PORTAL}\n")
    (bee / ".env").write_text(f"HERMES_PORTAL_BASE_URL={BEE_PORTAL}\n")
    for home in (launch, bee):
        (home / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("HERMES_PORTAL_BASE_URL", LAUNCH_PORTAL)  # the launch process loaded its own .env
    monkeypatch.setenv("LAUNCH_ENV_ONLY_KEY", "from-process-env")  # systemd / Docker / ``op run`` only
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setattr(server, "_served_profile_homes", set())
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    previous = secret_scope.is_multiplex_active()
    launch_profile_policy.activate_multi_profile_hosting()
    seen = []

    def unwrapped_secret_read(rid, params):
        from agent.secret_scope import get_secret
        from hermes_constants import get_hermes_home
        seen.append((get_hermes_home().resolve(), get_secret("HERMES_PORTAL_BASE_URL")))
        return server._ok(rid, {})
    monkeypatch.setitem(server._methods, "probe.portal", unwrapped_secret_read)
    yield server, launch, bee, seen
    secret_scope.set_multiplex_active(previous)


def _call_on_launch_socket(server, launch, params):
    """One legacy fallback request on a socket whose ticket is the launch profile's, in the scope
    ``gateway_ws`` binds for it."""
    from gateway.runtime_bootstrap import _PURPOSE_CAPABILITIES
    from tui_gateway.launch_profile_policy import launch_profile_scope_if_multiplexed
    from tui_gateway.ws_legacy_fallback import dispatch_legacy
    actor = types.SimpleNamespace(profile_id=str(launch), capabilities=frozenset(_PURPOSE_CAPABILITIES["interactive"]))
    written, done = [], threading.Event()
    transport = types.SimpleNamespace(write=lambda frame: written.append(frame) or done.set() or True)
    with launch_profile_scope_if_multiplexed():
        reply = dispatch_legacy(server, {"jsonrpc": "2.0", "id": 1, "method": "probe.portal", "params": params},
                                transport, actor)
    return reply if reply is not None else (done.wait(10) and written[0])


def test_explicit_profile_on_a_launch_socket_reads_that_profiles_secrets(homes):
    server, launch, bee, seen = homes

    for params in ({"profile": "bee"}, {}, {"profile": "bee"}):
        assert "result" in _call_on_launch_socket(server, launch, params)

    assert seen == [(bee.resolve(), BEE_PORTAL), (launch.resolve(), LAUNCH_PORTAL), (bee.resolve(), BEE_PORTAL)]


def test_explicit_profile_that_does_not_resolve_is_refused_before_the_handler(homes):
    server, launch, _bee, seen = homes

    reply = _call_on_launch_socket(server, launch, {"profile": "missing"})

    assert reply["error"]["code"] == 4064, reply
    assert seen == []


def test_a_secondary_socket_naming_the_launch_profile_runs_in_the_launch_scope(homes, monkeypatch):
    """Reverse direction: a profile-bound socket runs in that profile's scope, so a verb naming the
    launch profile must not keep the secondary's secrets and home."""
    from gateway.runtime_bootstrap import _PURPOSE_CAPABILITIES
    from tui_gateway.ws_legacy_fallback import dispatch_legacy
    server, launch, bee, seen = homes
    env_only = []

    def probe(rid, params):
        from agent.secret_scope import get_secret
        env_only.append(get_secret("LAUNCH_ENV_ONLY_KEY"))
        return server._methods["probe.portal"](rid, params)
    monkeypatch.setitem(server._methods, "probe.env", probe)
    actor = types.SimpleNamespace(profile_id=str(bee), capabilities=frozenset(_PURPOSE_CAPABILITIES["interactive"]))
    written, done = [], threading.Event()
    transport = types.SimpleNamespace(write=lambda frame: written.append(frame) or done.set() or True)
    with server._session_profile_runtime_scope({"profile_home": str(bee)}):
        reply = dispatch_legacy(server, {"jsonrpc": "2.0", "id": 1, "method": "probe.env",
                                         "params": {"profile": "default"}}, transport, actor)
    assert "result" in (reply if reply is not None else (done.wait(10) and written[0]))
    assert seen == [(launch.resolve(), LAUNCH_PORTAL)]
    assert env_only == ["from-process-env"]
