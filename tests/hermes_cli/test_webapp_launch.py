"""The private launch ticket crosses a browser launcher's argv, so only the server user may spend it.

Minting the link and opening the browser belong to whatever launches the Webapp; these tests cover
the handoff route the ticket is spent at (``/webapp/window-session``).
"""
import os

import pytest


@pytest.mark.platforms("linux")
def test_launch_ticket_is_spent_only_over_the_server_users_own_connection(monkeypatch):
    """Linux publishes every process's argv, so another local user can read a launch ticket
    out of the browser launcher before the browser spends it. That user's connection is
    refused, and the refusal does not burn the ticket for the operator's browser."""
    from fastapi.testclient import TestClient
    from hermes_cli import web_server as server
    from hermes_cli.web_routers import webapp

    monkeypatch.setattr(server.app.state, "ui_surface", "webapp", raising=False)
    monkeypatch.setattr(server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(server.app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(server.app.state, "webapp_window_tickets", {}, raising=False)
    monkeypatch.setattr(server, "_SESSION_TOKEN", "a" * 43)
    client = TestClient(server.app, base_url="http://127.0.0.1")
    headers = {"Origin": "http://127.0.0.1", "X-Hermes-Window-Ticket": webapp.mint_launch_ticket(server.app.state)}

    own_uid = os.getuid()  # windows-footgun: ok — platforms("linux") test
    monkeypatch.setattr(webapp, "_loopback_peer_uid", lambda client, server: own_uid + 1)
    stolen = client.post("/webapp/window-session", headers=headers)
    assert stolen.status_code == 403 and "a" * 43 not in stolen.text
    monkeypatch.setattr(webapp, "_loopback_peer_uid", lambda client, server: own_uid)
    granted = client.post("/webapp/window-session", headers=headers)
    assert granted.status_code == 200 and granted.json()["token"] == "a" * 43


@pytest.mark.platforms("macos")
def test_macos_launch_handoff_fails_closed_without_consuming_child_window_authority(monkeypatch):
    """An argv launch ticket is unusable until macOS has an authenticated OS handoff.

    This native platform contract makes no claim about cross-account argv visibility.
    Authenticated child-window tickets never pass through a launcher and still work.
    """
    from fastapi.testclient import TestClient
    from hermes_cli import web_server as server
    from hermes_cli.web_routers import webapp

    monkeypatch.setattr(server.app.state, "ui_surface", "webapp", raising=False)
    monkeypatch.setattr(server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(server.app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(server.app.state, "webapp_window_tickets", {}, raising=False)
    monkeypatch.setattr(server, "_SESSION_TOKEN", "a" * 43)
    client = TestClient(server.app, base_url="http://127.0.0.1")
    try:
        launch = webapp.mint_launch_ticket(server.app.state)
        headers = {"Origin": "http://127.0.0.1", "X-Hermes-Window-Ticket": launch}
        for _ in range(2):
            denied = client.post("/webapp/window-session", headers=headers)
            assert denied.status_code == 403 and server._SESSION_TOKEN not in denied.text
            assert launch in server.app.state.webapp_window_tickets

        issued = client.post("/api/webapp/window-ticket", headers={
            server._SESSION_HEADER_NAME: server._SESSION_TOKEN,
        })
        assert issued.status_code == 200
        headers["X-Hermes-Window-Ticket"] = issued.json()["ticket"]
        assert client.post("/webapp/window-session", headers=headers).json() == {"token": server._SESSION_TOKEN}
        assert client.post("/webapp/window-session", headers=headers).status_code == 403
    finally:
        client.close()


def test_a_link_for_the_user_to_open_by_hand_spends_once_without_a_peer_check(monkeypatch):
    """Where nothing identifies the connection's owner (macOS, the BSDs), ``hermes webapp`` prints a
    link for the user to open by hand. Its ticket spends without the peer check, once and within 60
    seconds, as a window ticket does; an argv launch ticket still never spends there."""
    from types import SimpleNamespace

    from fastapi.testclient import TestClient
    from hermes_cli import web_server as server
    from hermes_cli.web_routers import webapp

    monkeypatch.setattr(server.app.state, "ui_surface", "webapp", raising=False)
    monkeypatch.setattr(server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(server.app.state, "bound_host", "127.0.0.1", raising=False)
    monkeypatch.setattr(server.app.state, "webapp_window_tickets", {}, raising=False)
    monkeypatch.setattr(server, "_SESSION_TOKEN", "a" * 43)
    monkeypatch.setattr(webapp, "_peer_is_server_user", lambda request: False)  # macOS
    client = TestClient(server.app, base_url="http://127.0.0.1")

    def redeem(ticket):
        return client.post("/webapp/window-session", headers={"Origin": "http://127.0.0.1",
                                                              "X-Hermes-Window-Ticket": ticket})
    try:
        printed = webapp.mint_printed_launch_ticket(server.app.state)
        assert redeem(printed).json() == {"token": "a" * 43}
        assert redeem(printed).status_code == 403
        assert redeem(webapp.mint_launch_ticket(server.app.state)).status_code == 403
        late = webapp.mint_printed_launch_ticket(server.app.state)
        now = webapp.time.monotonic()
        monkeypatch.setattr(webapp, "time", SimpleNamespace(monotonic=lambda: now + webapp.LAUNCH_TICKET_TTL_SECONDS + 1))
        assert redeem(late).status_code == 403
    finally:
        client.close()


def _loopback_request():
    from types import SimpleNamespace
    return SimpleNamespace(client=SimpleNamespace(host="127.0.0.1", port=50123),
                           scope={"server": ("127.0.0.1", 9120)})


@pytest.mark.parametrize("platform", ["freebsd14", "openbsd7", "sunos5"])
def test_where_no_peer_check_exists_a_launch_ticket_is_never_spent(monkeypatch, platform):
    """Other Unixes show every process's argv to all local users by default and Hermes cannot tell
    who owns a loopback connection there. As on macOS, the launcher prints a link for the user to open
    by hand rather than passing a launch ticket to a browser launcher, and a launch ticket is never
    spent."""
    import sys

    from gateway import webapp_listener
    from hermes_cli.web_routers import webapp

    monkeypatch.setattr(sys, "platform", platform)
    assert webapp._peer_is_server_user(_loopback_request()) is False
    assert webapp_listener._one_use_links_work() is False


def test_windows_spends_a_launch_ticket_without_a_peer_check(monkeypatch):
    """Windows does not show a process's command line to other standard users (reading it needs a
    handle the process's DACL grants only its own user, SYSTEM and elevated administrators), so the
    ticket in the browser launcher's argv stays the user's own, as the stored credential does."""
    import sys

    from gateway import webapp_listener
    from hermes_cli.web_routers import webapp

    monkeypatch.setattr(sys, "platform", "win32")
    assert webapp._peer_is_server_user(_loopback_request()) is True
    assert webapp_listener._one_use_links_work() is True
