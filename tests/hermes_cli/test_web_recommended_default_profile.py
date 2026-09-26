"""``GET /api/model/recommended-default?provider=nous`` answers for the requested profile.

The Nous branch returned before the route entered the profile scope, so ``?profile=b`` got the
tier read, Portal URL and caches of the profile that launched the dashboard, and an unknown
profile answered 200 instead of the scope's 404. Only the Portal account read is stubbed.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli import profiles as profiles_mod

    freebie = profiles_mod.get_profile_dir("freebie")
    freebie.mkdir(parents=True, exist_ok=True)
    (freebie / "config.yaml").write_text("model: {}\n", encoding="utf-8")

    from hermes_constants import get_hermes_home
    import hermes_cli.nous_account as nous_account

    # The launch profile's account is paid, the "freebie" profile's is free tier.
    monkeypatch.setattr(nous_account, "get_nous_portal_account_info", lambda **_k: SimpleNamespace(
        is_free_tier=get_hermes_home().name == "freebie"))

    from hermes_cli import web_server

    with TestClient(web_server.app, raise_server_exceptions=False) as c:
        c.headers["Authorization"] = f"Bearer {web_server._SESSION_TOKEN}"
        yield c


def test_nous_recommendation_reads_the_requested_profile(client):
    launch = client.get("/api/model/recommended-default", params={"provider": "nous"})
    named = client.get("/api/model/recommended-default",
                       params={"provider": "nous", "profile": "freebie"})

    assert launch.status_code == named.status_code == 200
    assert launch.json()["free_tier"] is False
    assert named.json()["free_tier"] is True


def test_nous_recommendation_unknown_profile_is_404(client):
    resp = client.get("/api/model/recommended-default",
                      params={"provider": "nous", "profile": "no-such-profile"})

    assert resp.status_code == 404, resp.text
    assert "no-such-profile" in resp.json()["detail"]
