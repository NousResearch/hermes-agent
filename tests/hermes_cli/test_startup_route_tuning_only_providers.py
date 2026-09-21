"""Regression tests: settings-only ``providers.<key>`` blocks must not hijack routing (#118153)."""

import pytest

from hermes_cli import model_switch


@pytest.fixture(autouse=True)
def _no_direct_aliases(monkeypatch):
    monkeypatch.setattr(model_switch, "DIRECT_ALIASES", {})


def test_tuning_only_block_does_not_hijack_vendor_prefixed_default():
    """A tuning-only providers.<vendor> block (timeouts/caps only, no routing fields)
    must not turn ``<vendor>/<model>`` into a provider switch onto that vendor."""
    route = model_switch.resolve_startup_model_route(
        "deepseek/deepseek-v4-flash-0731",
        current_provider="nous",
        user_providers={"deepseek": {"stale_timeout_seconds": 45, "request_timeout_seconds": 90}},
    )
    assert route is None


def test_routing_declaring_block_still_resolves_vendor_prefixed_default():
    """Behavior contract preserved: a providers.<vendor> entry that DOES declare
    routing (base_url/key/...) must keep resolving ``<vendor>/<model>`` defaults."""
    route = model_switch.resolve_startup_model_route(
        "deepseek/deepseek-v4-pro",
        user_providers={"deepseek": {"base_url": "https://api.deepseek.example/v1"}},
    )
    assert route == model_switch.StartupModelRoute("deepseek-v4-pro", "deepseek", "")
