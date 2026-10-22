"""Who counts as the Desktop app on this machine, on the unified gateway.

The welcome trial and first-run onboarding are offered only to the local Desktop. The proof is per
connection: the Desktop says it is local (``client.capabilities
{desktop_local}``), and that counts only on a connection that redeemed a native ticket.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.waiting import async_wait_until


class _Transport:
    def __init__(self, identity):
        self.frames = []
        self.auth_identity = identity
        self.desktop_local = False

    def write(self, frame):
        self.frames.append(frame)
        return True

    def close(self):
        pass


async def _connect(tmp_path, monkeypatch, *, native=True, config=None):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from tui_gateway import server

    home = tmp_path / ".hermes"
    home.mkdir()
    if config:
        (home / "config.yaml").write_text(config, encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_hermes_home", str(home))
    monkeypatch.setattr(server, "_db", None)
    monkeypatch.setattr("gateway.run._hermes_home", home)
    for name in ("HERMES_DESKTOP", "HERMES_DASHBOARD_SESSION_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    store = SessionStore(home / "sessions", GatewayConfig())
    runner = SimpleNamespace(_session_db=store._db, session_store=store, _draining=False, adapters={},
                             _evict_cached_agent=lambda route: None, _cached_agent_for=lambda route: None,
                             _peek_session_state=lambda key: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id="owner")
    identity: dict = {"user_id": "uid:1", "provider": "local", "profile_id": str(home), "instance_id": "owner"}
    if native:
        identity["native_bootstrap"] = True
    transport = _Transport(identity)
    connection = AuthorityConnection(authority, transport, identity)
    from gateway.session_local_migration import bind_native_transport
    bind_native_transport(authority, connection.actor, identity)
    return SimpleNamespace(authority=authority, connection=connection, transport=transport, runner=runner, home=home)


async def _request(conn, rid, method, params):
    from tui_gateway.ws import _dispatch_request
    reply = await _dispatch_request(conn.connection, {"jsonrpc": "2.0", "id": rid, "method": method,
                                                      "params": params}, method, conn.transport)
    if reply is None:
        def written():
            return next((f for f in conn.transport.frames if isinstance(f, dict) and f.get("id") == rid), None)
        await async_wait_until(written, timeout=20)
        reply = written()
    return reply


@pytest.mark.asyncio
async def test_a_fresh_install_is_offered_onboarding_once_the_desktop_says_it_is_local(tmp_path, monkeypatch):
    conn = await _connect(tmp_path, monkeypatch)
    try:
        before = await _request(conn, 1, "onboarding.state", {"assume_fresh": True})
        assert before["result"]["eligible"] is False

        from tui_gateway import server_requests
        await _request(conn, 2, "client.capabilities", {"server_requests": True})
        declared = await _request(conn, 3, "client.capabilities", {"desktop_local": True})
        assert "result" in declared, declared
        assert server_requests.answers_requests(conn.transport)

        after = await _request(conn, 4, "onboarding.state", {"assume_fresh": True})
        assert after["result"]["run"] is True and after["result"]["eligible"] is True, after
    finally:
        await conn.connection.close()


@pytest.mark.asyncio
async def test_the_declaration_counts_only_on_a_native_ticket(tmp_path, monkeypatch):
    # A client that reached the gateway any other way (a dashboard token, a remote credential) is not
    # the Desktop on this machine, whatever it says.
    conn = await _connect(tmp_path, monkeypatch, native=False)
    try:
        await _request(conn, 1, "client.capabilities", {"server_requests": True, "desktop_local": True})
        state = await _request(conn, 2, "onboarding.state", {"assume_fresh": True})
        assert state["result"]["eligible"] is False
    finally:
        await conn.connection.close()


@pytest.mark.asyncio
async def test_a_desktop_chat_runs_its_turns_as_work_for_the_local_desktop(tmp_path, monkeypatch):
    """The trial's free inference is resolved inside the turn, away from any request: the chat carries
    the fact from its creation, and only a Desktop chat created on a declared connection does."""
    from contextlib import nullcontext
    from gateway.session_local_recovery import local_source
    from gateway.session_policy import local_desktop_turn
    from hermes_cli.anon_auth import on_local_desktop_backend

    conn = await _connect(tmp_path, monkeypatch,
                          config="model:\n  provider: openrouter\n  default: anthropic/claude-sonnet-4\n")
    try:
        await _request(conn, 1, "client.capabilities", {"server_requests": True, "desktop_local": True})
        desktop = (await _request(conn, 2, "session.create", {"source": "gui", "request_id": "a"}))["result"]
        conn.transport.desktop_local = False
        tui = (await _request(conn, 3, "session.create", {"source": "tui", "request_id": "b"}))["result"]

        def inside_turn(stored):
            source = local_source(conn.authority, stored, conn.connection.actor.subject)
            with local_desktop_turn(conn.runner, source, nullcontext()):
                return on_local_desktop_backend()

        assert inside_turn(desktop["stored_session_id"]) is True
        assert inside_turn(tui["stored_session_id"]) is False
        assert on_local_desktop_backend() is False
    finally:
        await conn.connection.close()



@pytest.mark.asyncio
async def test_a_trial_users_desktop_chat_runs_on_the_free_model(tmp_path, monkeypatch):
    """A claimed welcome trial and no configured model: the chat freezes the free model instead of
    no model at all (a launch policy with no model can never be restored)."""
    from hermes_cli import anon_auth
    guest = {"auth_method": anon_auth.ANON_AUTH_METHOD, anon_auth.WELCOME_CLAIM_FLAG: True}
    monkeypatch.setattr(anon_auth, "current_nous_state", lambda *a, **k: guest)
    conn = await _connect(tmp_path, monkeypatch)
    try:
        await _request(conn, 1, "client.capabilities", {"desktop_local": True})
        created = await _request(conn, 2, "session.create", {"source": "gui", "request_id": "a"})
        assert "result" in created, created
        from hermes_state_local import local_receipt
        receipt = local_receipt(conn.authority.db, created["result"]["stored_session_id"])
        assert receipt["policy"]["model"] == anon_auth.GUEST_MODEL

        conn.transport.desktop_local = False
        other = await _request(conn, 3, "session.create", {"source": "gui", "request_id": "b"})
        assert "result" in other, other
        assert local_receipt(conn.authority.db, other["result"]["stored_session_id"])["policy"]["model"] == anon_auth.GUEST_MODEL
    finally:
        await conn.connection.close()


@pytest.mark.asyncio
async def test_a_fresh_launch_claims_the_trial_and_opens_onboarding_on_it(tmp_path, monkeypatch):
    """The whole first launch: the claim mints the guest, and the chat it opens runs on the free model."""
    from hermes_cli import anon_auth, welcome_trial
    state = {}
    guest = {"auth_method": anon_auth.ANON_AUTH_METHOD, anon_auth.WELCOME_CLAIM_FLAG: True}

    def claim(**kwargs):
        if welcome_trial.claim_blocker(assume_fresh=kwargs.get("assume_fresh", False)):
            return None
        state.update(guest)
        return dict(guest)

    monkeypatch.setattr(anon_auth, "current_nous_state", lambda *a, **k: dict(state) if state else None)
    monkeypatch.setattr(welcome_trial, "claim_desktop_free_tier", claim)
    monkeypatch.setattr(welcome_trial, "_has_nous_identity", lambda: bool(state))
    conn = await _connect(tmp_path, monkeypatch)
    try:
        await _request(conn, 1, "client.capabilities", {"desktop_local": True})
        started = await _request(conn, 2, "onboarding.start",
                                 {"fresh_launch": True, "assume_fresh": True, "facts_source": "local", "locale": "en"})
        assert "result" in started, started
        resumed = await _request(conn, 3, "session.resume", {"session_id": started["result"]["stored_session_id"]})
        assert "error" not in resumed, resumed
    finally:
        await conn.connection.close()


@pytest.mark.asyncio
async def test_the_first_local_desktop_connection_runs_the_desktop_boot_work(tmp_path, monkeypatch):
    from hermes_cli import anon_auth, free_tier_bootstrap, onboarding_run
    from tui_gateway import server
    seen = []
    monkeypatch.setattr(onboarding_run, "latch_first_install", lambda: seen.append(anon_auth.on_local_desktop_backend()))
    monkeypatch.setattr(server, "_local_desktop_boot_started", False)
    free_tier_bootstrap.reset_for_tests()
    conn = await _connect(tmp_path, monkeypatch)
    try:
        await _request(conn, 1, "client.capabilities", {"desktop_local": True})
        await _request(conn, 2, "client.capabilities", {"desktop_local": True})
        assert free_tier_bootstrap.wait_for_record(timeout=20) is not None
        assert seen == [True]
    finally:
        await conn.connection.close()
        free_tier_bootstrap.reset_for_tests()


def test_the_desktop_boot_work_runs_for_the_launch_profile_whichever_profile_connects_first(tmp_path, monkeypatch):
    """The setup record and the first-install latch belong to the launch profile. On a multiplexing
    gateway a Desktop connection runs in its own profile's scope, so a first connection for another
    profile must not build the launch record from that profile's home."""
    from agent import secret_scope
    from hermes_cli import free_tier_bootstrap, onboarding_run
    from hermes_constants import get_hermes_home, get_process_hermes_home, set_hermes_home_override
    from tui_gateway import server
    launch = get_process_hermes_home()
    secondary = tmp_path / "profiles" / "work"
    secondary.mkdir(parents=True)
    seen = []
    monkeypatch.setattr(onboarding_run, "latch_first_install", lambda: seen.append(get_hermes_home()))
    monkeypatch.setattr(server, "_local_desktop_boot_started", False)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    free_tier_bootstrap.reset_for_tests()
    token = set_hermes_home_override(secondary)
    try:
        server.begin_local_desktop_boot()
        assert free_tier_bootstrap.wait_for_record(timeout=20) is not None
    finally:
        from hermes_constants import reset_hermes_home_override
        reset_hermes_home_override(token)
        free_tier_bootstrap.reset_for_tests()
    assert seen == [launch]
