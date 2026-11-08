"""A profile created through ``POST /api/profiles`` is served by the live multiplexer right away.

Creation asks the multiplexer to rescan over its control socket. On the gateway's own listener (the
Webapp) that socket answers on the loop serving the route, so a create run on the loop waited out the
socket's timeout instead, the multiplexer only served the profile at its periodic rescan, and the
Webapp refused to switch to it ("This gateway does not serve profile ...") until then.
"""
import asyncio

import pytest


@pytest.fixture
def client(monkeypatch, _isolate_hermes_home):
    from starlette.testclient import TestClient

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli import profiles
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: home / "profiles")
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def test_a_created_profile_is_served_without_waiting_for_the_rescan(client, monkeypatch):
    from hermes_cli import gateway_multiplex_served

    served = []

    def multiplexer(name, **_kwargs):
        # The control socket answers on the route's loop: called from that loop, it never does.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            served.append(name)
            return ["default", name]
        return None

    monkeypatch.setattr(gateway_multiplex_served, "notify_multiplexer_profiles_changed", multiplexer)

    response = client.post("/api/profiles", json={"name": "testing-profile", "no_skills": True})

    assert response.status_code == 200, response.text
    assert served == ["testing-profile"]
