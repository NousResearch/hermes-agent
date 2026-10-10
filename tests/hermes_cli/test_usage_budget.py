"""Month-to-date usage per billing provider against ``usage.budgets`` (hermes_cli/usage_budget.py).

Most sessions on a token plan carry no price (``cost_status='unknown'``), so tokens are the unit every
provider can be budgeted in; money is counted only where it is known, and never shown as $0.
"""
from datetime import UTC, datetime

from hermes_cli import config as config_module
from hermes_cli.config import load_config, validate_config_structure
from hermes_cli.usage_budget import month_usage_rows, month_window, read_budgets, summarize_month
from hermes_state import SessionDB

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)  # day 10 of a 31-day month, 9.5 days elapsed


def _write_config(home, text):
    (home / "config.yaml").write_text(text, encoding="utf-8")
    config_module._LOAD_CONFIG_CACHE.clear()
    config_module._RAW_CONFIG_CACHE.clear()


def test_budgets_round_trip_through_the_real_loader(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path, (
        "usage:\n  budgets:\n"
        "    xiaomi: {monthly_tokens: 500000000}\n"
        "    openrouter: {monthly_usd: 20}\n"
        "    deepseek: {monthly_tokens: null, monthly_usd: null}\n"))
    assert read_budgets(load_config()) == {"xiaomi": ("tokens", 500000000.0), "openrouter": ("usd", 20.0)}


def test_no_usage_section_means_no_budgets():
    assert read_budgets({}) == {}


def test_the_usage_root_is_a_known_config_key():
    issues = validate_config_structure({"usage": {"budgets": {"xiaomi": {"monthly_tokens": 1}}}})
    assert not [i for i in issues if "usage" in i.message]


def test_month_window_counts_partial_days():
    start, days, elapsed = month_window(NOW)
    assert (start.day, start.hour, days) == (1, 0, 31)
    assert round(elapsed, 2) == 9.5


def test_token_budget_projects_the_month_and_dates_the_overrun():
    rows = [{"provider": "xiaomi", "tokens": 190_000_000, "estimated_cost": 0.0, "actual_cost": 0.0,
             "sessions": 300, "unpriced_sessions": 300}]
    budget = summarize_month(rows, {"xiaomi": ("tokens", 500_000_000.0)}, NOW)["providers"][0]["budget"]
    assert round(budget["used_ratio"], 2) == 0.38
    assert round(budget["projected_ratio"], 2) == 1.24  # 20M/day over 31 days against 500M
    assert budget["runs_out_on"] == "2026-10-26"  # 500M at 20M/day from Oct 1


def test_a_pace_inside_the_budget_has_no_overrun_date():
    rows = [{"provider": "openrouter", "tokens": 1, "estimated_cost": 3.0, "actual_cost": 0.0,
             "sessions": 5, "unpriced_sessions": 0}]
    budget = summarize_month(rows, {"openrouter": ("usd", 20.0)}, NOW)["providers"][0]["budget"]
    assert (budget["used"], budget["runs_out_on"]) == (3.0, None)


def test_unpriced_usage_is_counted_never_priced_at_zero():
    rows = [{"provider": "xiaomi", "tokens": 1000, "estimated_cost": 0.0, "actual_cost": 0.0,
             "sessions": 3, "unpriced_sessions": 3}]
    row = summarize_month(rows, {}, NOW)["providers"][0]
    assert (row["unpriced_sessions"], row["budget"]) == (3, None)


def test_a_budgeted_provider_with_no_usage_still_shows():
    month = summarize_month([], {"openrouter": ("usd", 20.0)}, NOW)
    assert month["providers"][0]["provider"] == "openrouter"
    assert month["providers"][0]["budget"]["used"] == 0


def test_month_rows_count_this_month_plus_auxiliary_calls_only(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    start = month_window(NOW)[0].timestamp()
    conn = db._conn
    conn.execute("INSERT INTO sessions (id, source, started_at, billing_provider, input_tokens, cache_read_tokens,"
                 " output_tokens, cost_status) VALUES ('now', 'cli', ?, 'xiaomi', 100, 50, 10, 'unknown')",
                 (start + 3600,))
    conn.execute("INSERT INTO sessions (id, source, started_at, billing_provider, input_tokens, output_tokens,"
                 " cost_status) VALUES ('old', 'cli', ?, 'xiaomi', 9999, 9999, 'unknown')", (start - 3600,))
    conn.execute("INSERT INTO session_model_usage (session_id, model, billing_provider, task, input_tokens,"
                 " output_tokens) VALUES ('now', 'mimo-v2.5', 'xiaomi', 'compression', 20, 5)")
    conn.commit()
    (row,) = month_usage_rows(conn, start)
    assert (row["provider"], row["tokens"], row["sessions"], row["unpriced_sessions"]) == ("xiaomi", 185, 1, 1)
