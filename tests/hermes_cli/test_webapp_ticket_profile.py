"""A Webapp tab's socket answers for the profile its gateway ticket names.

A tab opened on ``?profile=<name>`` mints its ``/api/ws`` ticket with that profile. The socket must
answer a request that names no profile (the sidebar's ``session.active_list``, the silence probe)
for that profile, as a native ticket does, not for the gateway's launch profile: otherwise the tab
cannot see its own running turn and reads it as ended. Its host scope survives, so a request that
names another served profile still reaches it.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

_TOKEN = "b" * 43


@pytest.fixture
def two_profiles(monkeypatch, tmp_path):
    from gateway.session_authorities import SessionAuthorities
    from gateway.session_authority import SessionAuthority
    from hermes_cli import web_server as server
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch

    launch = tmp_path / ".hermes"
    beta = launch / "profiles" / "beta"
    beta.mkdir(parents=True)
    for home in (launch, beta):
        (home / "config.yaml").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    registry = SessionAuthorities(launch, multiplexed=True)
    runner = SimpleNamespace(_draining=False, session_authorities=registry)
    authorities, dbs = {}, {}
    for name, home in (("default", launch), ("beta", beta)):
        db = dbs[name] = SessionDB(home / "state.db")
        authorities[name] = SessionAuthority(runner, profile_id=str(home), instance_id="ticket-profile", db=db,
                                             epoch=begin_runtime_epoch(db, instance_id="ticket-profile"))
        registry.add(home, authorities[name], name=name)
    runner.session_authority = authorities["default"]
    monkeypatch.setattr(server.app.state, "session_authority", authorities["default"], raising=False)
    monkeypatch.setattr(server.app.state, "ui_surface", "webapp", raising=False)
    monkeypatch.setattr(server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(server.app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(server, "_SESSION_TOKEN", _TOKEN)
    monkeypatch.setattr(server, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
    try:
        yield TestClient(server.app, base_url="http://127.0.0.1"), {"default": launch, "beta": beta}
    finally:
        for db in dbs.values():
            db.close()


def _ask(client, method, params, profile=None):
    """*method*'s reply on a tab socket opened for *profile* (``?profile=`` on the page)."""
    query = f"&profile={profile}" if profile else ""
    minted = client.post(f"/api/webapp/socket-ticket?socket=gateway{query}", headers={"X-Hermes-Session-Token": _TOKEN})
    assert minted.status_code == 200, minted.text
    ticket = minted.json()["ticket"]
    socket_query = f"&profile={profile}" if profile else ""
    # TestClient's WS default Host is "testserver", which the upgrade gate refuses.
    with client.websocket_connect(f"/api/ws?native_dial=browser&ticket={ticket}{socket_query}",
                                  headers={"host": "127.0.0.1"}) as ws:
        ws.send_json({"jsonrpc": "2.0", "id": "ask", "method": method, "params": params})
        while True:
            frame = ws.receive_json()
            if frame.get("id") == "ask":
                return frame


def _answering_home(client, profile=None):
    reply = _ask(client, "runtime.describe", {}, profile)
    assert "result" in reply, reply
    return Path(reply["result"]["profile_id"])


def test_a_tab_opened_on_a_profile_is_answered_for_it_and_keeps_its_host_scope(two_profiles):
    client, homes = two_profiles
    assert _answering_home(client, profile="beta") == homes["beta"]
    # Still the dashboard gate's host scope: naming another served profile reaches it, where a
    # profile-bound grant would be refused with ``profile_mismatch``.
    reply = _ask(client, "session.active_list", {"profile": "default"}, profile="beta")
    assert reply.get("result") == {"sessions": []}, reply


def test_a_tab_opened_without_a_profile_is_answered_for_the_launch_profile(two_profiles):
    client, homes = two_profiles
    assert _answering_home(client) == homes["default"]
