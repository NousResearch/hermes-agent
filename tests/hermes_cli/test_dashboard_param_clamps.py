"""Dashboard query-param clamps (#39200 + #74778 salvage).

FastAPI Query bounds reject out-of-range values at the validation layer
(422) instead of letting them reach SQL/insights code: an unbounded
``limit`` drags every session row out of SQLite in one hit (multiplied
across every profile's state.db on the fan-out endpoint), and an
unbounded/inverted ``days`` forces full-history InsightsEngine work.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "clamp-test-token")
    from hermes_cli import web_server

    with TestClient(web_server.app, raise_server_exceptions=False) as c:
        c.headers["Authorization"] = "Bearer clamp-test-token"
        yield c


class TestSessionPaginationClamps:
    def test_oversized_limit_rejected(self, client):
        r = client.get("/api/sessions", params={"limit": 10_000})
        assert r.status_code == 422

    def test_negative_limit_rejected(self, client):
        r = client.get("/api/sessions", params={"limit": -1})
        assert r.status_code == 422

    def test_profile_fanout_limit_clamped(self, client):
        r = client.get("/api/profiles/sessions", params={"limit": 10_000})
        assert r.status_code == 422

    def test_in_range_limit_accepted(self, client):
        r = client.get("/api/sessions", params={"limit": 50})
        assert r.status_code == 200


class TestAnalyticsDaysClamps:
    @pytest.mark.parametrize("days", [0, -5, 100_000])
    def test_out_of_range_days_rejected(self, client, days):
        r = client.get("/api/analytics/usage", params={"days": days})
        assert r.status_code == 422

    def test_in_range_days_accepted(self, client):
        r = client.get("/api/analytics/usage", params={"days": 30})
        assert r.status_code == 200
