"""``GET /api/model/recommended-default?provider=nous`` answers for the requested profile.

The Nous branch returned before the route entered the profile scope, so ``?profile=b`` got the
tier read, Portal URL and caches of the profile that launched the dashboard, and an unknown
profile answered 200 instead of the scope's 404. Only the Portal account read is stubbed.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


@pytest.fixture()
def client(_isolate_hermes_home, monkeypatch):
    from hermes_cli import profiles as profiles_mod

    freebie = profiles_mod.get_profile_dir("freebie")
    freebie.mkdir(parents=True, exist_ok=True)
    (freebie / "config.yaml").write_text("model: {}\n", encoding="utf-8")

    from hermes_constants import get_hermes_home
    from hermes_cli import nous_account

    # The launch profile's account is paid, the "freebie" profile's is free tier.
    monkeypatch.setattr(nous_account, "get_nous_portal_account_info", lambda **_k: SimpleNamespace(
        is_free_tier=get_hermes_home().name == "freebie"))

    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    c = TestClient(app, raise_server_exceptions=False)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def test_nous_recommendation_reads_the_requested_profile(client):
    launch = client.get("/api/model/recommended-default", params={"provider": "nous"})
    named = client.get("/api/model/recommended-default",
                       params={"provider": "nous", "profile": "freebie"})
    unknown = client.get("/api/model/recommended-default",
                         params={"provider": "nous", "profile": "no-such-profile"})

    assert launch.status_code == named.status_code == 200
    assert launch.json()["free_tier"] is False
    assert named.json()["free_tier"] is True
    assert unknown.status_code == 404, unknown.text
    assert "no-such-profile" in unknown.json()["detail"]
