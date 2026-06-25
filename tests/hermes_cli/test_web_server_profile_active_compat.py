"""Regression tests for the [CN-fork] P-008 /api/profiles/active compat layer.

The hermes-agent-cn desktop reads the sticky active profile from
``GET /api/profiles/active`` as ``.name`` and switches it via
``PUT /api/profiles/active``. Upstream only ships ``{active, current}`` and a
``POST`` verb. The fork's P-008 compat layer adds the ``name`` field and a
``PUT`` alias.

A v0.17.0 upstream sync silently reverted both halves: the GET response lost
``name`` (so the desktop's ``ActiveProfileResponse`` Zod schema rejected the
payload with ``path:["name"], received: undefined`` and the whole profile screen
failed to load) and the ``PUT`` verb 405'd (so profile switching broke). See
Eynzof/Hermes-CN-Desktop#301. These tests fail loudly if either half regresses
again.
"""

from starlette.testclient import TestClient

from hermes_cli import profiles as profiles_mod
from hermes_cli import web_server


def _client():
    web_server.app.state.auth_required = False
    web_server.app.state.bound_host = None
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return client


def test_get_active_profile_includes_name_compat(monkeypatch):
    monkeypatch.setattr(profiles_mod, "get_active_profile", lambda: "worker")
    monkeypatch.setattr(profiles_mod, "get_active_profile_name", lambda: "default")
    client = _client()
    try:
        resp = client.get("/api/profiles/active")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # P-008 compat: the desktop reads `.name`; it must be present and mirror
        # the sticky `active`. The upstream `active`/`current` pair stays too.
        assert body["name"] == "worker"
        assert body["active"] == "worker"
        assert body["current"] == "default"
    finally:
        client.close()


def test_put_active_profile_alias_sets_sticky(monkeypatch):
    calls: dict[str, str] = {}
    monkeypatch.setattr(
        profiles_mod, "set_active_profile", lambda name: calls.__setitem__("name", name)
    )
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda name: name)
    client = _client()
    try:
        # The desktop switches profiles via PUT (fork P-008 alias), not POST.
        resp = client.put("/api/profiles/active", json={"name": "worker"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["ok"] is True
        assert calls["name"] == "worker"
    finally:
        client.close()


def test_post_active_profile_still_supported(monkeypatch):
    """Upstream's POST verb keeps working alongside the PUT alias."""
    calls: dict[str, str] = {}
    monkeypatch.setattr(
        profiles_mod, "set_active_profile", lambda name: calls.__setitem__("name", name)
    )
    monkeypatch.setattr(profiles_mod, "normalize_profile_name", lambda name: name)
    client = _client()
    try:
        resp = client.post("/api/profiles/active", json={"name": "worker"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["ok"] is True
        assert calls["name"] == "worker"
    finally:
        client.close()
