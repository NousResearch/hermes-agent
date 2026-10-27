"""All-time (``days=all``) range for the analytics endpoints.

The Analytics page's range selector gained an "All" preset that must show the
full recorded history instead of the 7d/30d/90d windows. The backend models it
as the literal ``days=all`` → no date cutoff (epoch cutoff ``0.0``), while
numeric days keep the 1-365 bound.
"""
import time

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from hermes_cli.web_routers import analytics
from hermes_state import SessionDB

# Well outside every preset window (90d / 365d) so "all" vs windowed differ.
OLD_STARTED_AT = time.time() - 400 * 86400
OLD_DAY = time.strftime("%Y-%m-%d", time.localtime(OLD_STARTED_AT))


def _seed(db):
    db.create_session("old", source="cli")
    db.update_token_counts(
        "old", input_tokens=500, output_tokens=50,
        model="legacy-model", billing_provider="acme", api_call_count=1,
    )
    with db._lock:
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?", (OLD_STARTED_AT, "old"),
        )
        db._conn.commit()
    db.create_session("new", source="cli")
    db.update_token_counts(
        "new", input_tokens=100, output_tokens=10,
        model="current-model", billing_provider="acme", api_call_count=1,
    )


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    _seed(db)
    db.close()
    return path


def _patch_db(monkeypatch, db_path):
    # Fresh connection per call: the route closes the DB in a finally block.
    monkeypatch.setattr(analytics, "_open_session_db_for_profile",
                        lambda *a, **k: SessionDB(db_path))
    monkeypatch.setattr(analytics, "_model_capabilities", lambda *a, **k: {})


class TestResolveWindow:
    def test_all_forms_map_to_no_cutoff(self):
        assert analytics._resolve_window("all") is None
        assert analytics._resolve_window("ALL") is None
        assert analytics._resolve_window(" all ") is None

    def test_numeric_forms(self):
        assert analytics._resolve_window(7) == 7
        assert analytics._resolve_window("30") == 30

    @pytest.mark.parametrize("bad", [0, -1, 366, "abc", ""])
    def test_invalid_windows_rejected(self, bad):
        with pytest.raises(HTTPException) as info:
            analytics._resolve_window(bad)
        assert info.value.status_code == 422


class TestAllTimeBackend:
    def test_usage_all_includes_pre_window_history(self, db_path, monkeypatch):
        _patch_db(monkeypatch, db_path)

        windowed = analytics._get_usage_analytics(days=90)
        assert windowed["period_days"] == 90
        assert windowed["totals"]["total_sessions"] == 1
        assert OLD_DAY not in {day["day"] for day in windowed["daily"]}

        all_time = analytics._get_usage_analytics(days="all")
        assert all_time["period_days"] is None
        assert all_time["totals"]["total_sessions"] == 2
        assert OLD_DAY in {day["day"] for day in all_time["daily"]}
        assert {m["model"] for m in all_time["by_model"]} >= {"legacy-model", "current-model"}

    def test_models_all_includes_pre_window_history(self, db_path, monkeypatch):
        _patch_db(monkeypatch, db_path)

        windowed = analytics._get_models_analytics(days=90)
        assert {m["model"] for m in windowed["models"]} == {"current-model"}

        all_time = analytics._get_models_analytics(days="all")
        assert all_time["period_days"] is None
        assert {m["model"] for m in all_time["models"]} == {"current-model", "legacy-model"}
        assert all_time["totals"]["total_sessions"] == 2

    def test_empty_store_all_range_is_graceful(self, tmp_path, monkeypatch):
        empty = tmp_path / "empty.db"
        SessionDB(empty).close()
        _patch_db(monkeypatch, empty)

        payload = analytics._get_usage_analytics(days="all")
        assert payload["daily"] == []
        assert payload["totals"]["total_sessions"] == 0
        assert analytics._get_models_analytics(days="all")["models"] == []


class TestAllTimeEndpoint:
    def _client(self, monkeypatch, db_path):
        _patch_db(monkeypatch, db_path)
        monkeypatch.setattr(analytics, "_session_db_path_for_profile", lambda profile: db_path)
        app = FastAPI()
        app.include_router(analytics.router)
        return TestClient(app)

    def test_usage_endpoint_accepts_all(self, db_path, monkeypatch):
        client = self._client(monkeypatch, db_path)
        resp = client.get("/api/analytics/usage?days=all")
        assert resp.status_code == 200
        body = resp.json()
        assert body["period_days"] is None
        assert body["totals"]["total_sessions"] == 2
        assert OLD_DAY in {day["day"] for day in body["daily"]}

    def test_models_endpoint_accepts_all(self, db_path, monkeypatch):
        client = self._client(monkeypatch, db_path)
        resp = client.get("/api/analytics/models?days=all")
        assert resp.status_code == 200
        assert {m["model"] for m in resp.json()["models"]} == {"current-model", "legacy-model"}

    def test_invalid_days_rejected(self, db_path, monkeypatch):
        client = self._client(monkeypatch, db_path)
        assert client.get("/api/analytics/usage?days=bogus").status_code == 422
        assert client.get("/api/analytics/models?days=0").status_code == 422
