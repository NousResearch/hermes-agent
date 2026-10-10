"""``usage.subscriptions``: a provider the user pays for as a flat plan bills as "included", not "unknown"."""

import pytest

from agent.usage_pricing import CanonicalUsage, estimate_usage_cost, resolve_billing_route
from hermes_cli import config as config_module

_USAGE = CanonicalUsage(input_tokens=1000, output_tokens=200)


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def write_config(text: str) -> None:
        (tmp_path / "config.yaml").write_text(text, encoding="utf-8")
        config_module._LOAD_CONFIG_CACHE.clear()
        config_module._RAW_CONFIG_CACHE.clear()

    write_config("")
    return write_config


def test_a_subscription_provider_costs_nothing_extra(home):
    home("usage:\n  subscriptions: [Xiaomi]\n")

    cost = estimate_usage_cost("mimo-v2.5-pro", _USAGE, provider="xiaomi")

    assert (cost.status, cost.amount_usd, cost.label) == ("included", 0, "included")


def test_only_the_listed_providers_are_included(home):
    assert resolve_billing_route("mimo-v2.5-pro", provider="xiaomi").billing_mode != "subscription_included"

    home("usage:\n  subscriptions: [xiaomi]\n")

    assert resolve_billing_route("deepseek-v4-pro", provider="deepseek").billing_mode != "subscription_included"
    assert resolve_billing_route("mimo-v2.5-pro", provider="xiaomi").billing_mode == "subscription_included"


def test_a_malformed_setting_marks_nothing(home):
    home("usage:\n  subscriptions: xiaomi\n")

    assert resolve_billing_route("mimo-v2.5-pro", provider="xiaomi").billing_mode != "subscription_included"
