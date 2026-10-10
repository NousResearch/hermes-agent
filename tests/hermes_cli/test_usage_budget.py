"""Month-to-date usage per billing provider against ``usage.budgets`` (hermes_cli/usage_budget.py).

Most sessions on a token plan carry no price (``cost_status='unknown'``), so tokens are the unit every
provider can be budgeted in; money is counted only where it is known, and never shown as $0.
"""
from hermes_cli import config as config_module
from hermes_cli.config import load_config, validate_config_structure
from hermes_cli.usage_budget import read_budgets


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
