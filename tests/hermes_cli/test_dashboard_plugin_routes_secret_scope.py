"""Dashboard agent-plugin mutations run in the request's profile scope, off the event loop.

Regression for #115256: install/update reached ``git_credentials._github_token`` →
``get_secret("GITHUB_TOKEN")`` with no profile secret scope bound. Once the process serves more
than its launch profile (any cross-profile request flips multi-profile hosting on) that read fails
closed with ``UnscopedSecretError``, so both routes answered a bare 500 for every repo-backed
plugin; both also cloned/pulled and rescanned on the event loop.

Driven through the real route handlers over the real app, with the real scope helpers and the real
``get_secret``; the plugin action itself is stubbed so nothing touches the network.
"""

from __future__ import annotations

import asyncio

import pytest


@pytest.fixture
def client(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    # The launch profile's own credential source: what a scoped read must return.
    (get_hermes_home() / ".env").write_text("GITHUB_TOKEN=ghp_launch_token\n", encoding="utf-8")
    return TestClient(app, raise_server_exceptions=False,
                      headers={_SESSION_HEADER_NAME: _SESSION_TOKEN})


@pytest.fixture
def multiplexed():
    """The dashboard hosting a profile other than its launch one (the reported trigger)."""
    from agent.secret_scope import set_multiplex_active

    set_multiplex_active(True)
    try:
        yield
    finally:
        set_multiplex_active(False)


def _stub_action(monkeypatch, name: str, seen: dict, result: dict | None = None):
    """Stand in for the plugin action: read the credential like the git path does, note the thread."""
    import hermes_cli.plugins_cmd as plugins_cmd
    from agent.secret_scope import get_secret

    def _stub(*_args, **_kwargs):
        try:
            asyncio.get_running_loop()
            seen["on_event_loop"] = True
        except RuntimeError:
            seen["on_event_loop"] = False
        seen["token"] = get_secret("GITHUB_TOKEN")
        return result if result is not None else {"ok": True, "name": name}

    monkeypatch.setattr(plugins_cmd, name, _stub)


def test_install_route_reads_the_git_credential_in_profile_scope(client, multiplexed, monkeypatch):
    seen: dict = {}
    _stub_action(monkeypatch, "dashboard_install_plugin", seen)

    resp = client.post("/api/dashboard/agent-plugins/install", json={"identifier": "owner/repo"})

    assert resp.status_code == 200, resp.text
    assert seen["token"] == "ghp_launch_token"
    assert seen["on_event_loop"] is False  # the clone/pull body runs off the event loop


def test_update_route_reads_the_git_credential_in_profile_scope(client, multiplexed, monkeypatch):
    seen: dict = {}
    _stub_action(monkeypatch, "dashboard_update_user_plugin", seen)

    resp = client.post("/api/dashboard/agent-plugins/probe/update")

    assert resp.status_code == 200, resp.text
    assert seen["token"] == "ghp_launch_token"


def test_failure_and_request_shape_errors_are_unchanged(client, multiplexed, monkeypatch):
    """Backward compat: a failed action is still a 400 carrying its own message, and the
    identifier/catalog_name check still precedes the scoped body."""
    seen: dict = {}
    _stub_action(monkeypatch, "dashboard_install_plugin", seen,
                 result={"ok": False, "error": "Install failed: no such plugin."})

    failed = client.post("/api/dashboard/agent-plugins/install", json={"identifier": "owner/repo"})
    assert failed.status_code == 400
    assert "no such plugin" in failed.json()["detail"]

    malformed = client.post("/api/dashboard/agent-plugins/install", json={"identifier": ""})
    assert malformed.status_code == 400
    assert "catalog_name" in malformed.json()["detail"]
