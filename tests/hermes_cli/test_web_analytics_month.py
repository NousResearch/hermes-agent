"""/api/analytics/month and /api/analytics/budgets: month-to-date usage per provider and its budgets."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import config as config_module
from hermes_cli.config import load_config
from hermes_cli.usage_budget import read_budgets
from hermes_cli.web_routers import analytics
from hermes_state import SessionDB


def _reload_config():
    config_module._LOAD_CONFIG_CACHE.clear()
    config_module._RAW_CONFIG_CACHE.clear()
    return load_config()


def _client(tmp_path, monkeypatch) -> TestClient:
    """A store holding one unpriced xiaomi session from right now, and a config with an unrelated key."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_state

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("s", "cli")
    db.update_token_counts("s", input_tokens=1000, output_tokens=200, model="mimo-v2.5-pro",
                           billing_provider="xiaomi", cost_status="unknown", api_call_count=1)
    db.close()
    monkeypatch.setattr(hermes_state, "_default_db_path", lambda: db_path)
    (tmp_path / "config.yaml").write_text("display:\n  skin: herald-os\n", encoding="utf-8")
    _reload_config()
    app = FastAPI()
    app.include_router(analytics.router)
    return TestClient(app)


def test_month_route_reports_this_months_tokens_and_unpriced_sessions(tmp_path, monkeypatch):
    (row,) = _client(tmp_path, monkeypatch).get("/api/analytics/month").json()["providers"]
    assert (row["provider"], row["tokens"], row["unpriced_sessions"], row["budget"]) == ("xiaomi", 1200, 1, None)


def test_setting_a_budget_persists_it_and_keeps_other_settings(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    resp = client.put("/api/analytics/budgets", json={"provider": "xiaomi", "monthly_tokens": 500_000_000})
    assert resp.status_code == 200
    config = _reload_config()
    assert read_budgets(config) == {"xiaomi": ("tokens", 500_000_000.0)}
    assert config["display"]["skin"] == "herald-os"
    assert client.get("/api/analytics/month").json()["providers"][0]["budget"]["kind"] == "tokens"


def test_a_budget_in_both_units_or_below_zero_is_rejected(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    both = client.put("/api/analytics/budgets", json={"provider": "x", "monthly_tokens": 1, "monthly_usd": 1})
    negative = client.put("/api/analytics/budgets", json={"provider": "x", "monthly_usd": -5})
    assert (both.status_code, negative.status_code) == (400, 400)


def test_clearing_a_budget_removes_it(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    client.put("/api/analytics/budgets", json={"provider": "xiaomi", "monthly_usd": 5})
    assert client.put("/api/analytics/budgets", json={"provider": "xiaomi"}).status_code == 200
    assert read_budgets(_reload_config()) == {}
