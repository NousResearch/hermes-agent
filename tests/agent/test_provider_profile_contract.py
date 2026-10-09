"""Contract tests for provider identity declarations."""

from providers.base import ProviderProfile


def test_provider_identity_defaults_are_non_aggregating() -> None:
    profile = ProviderProfile(name="example")

    assert profile.aliases == ()
    assert profile.base_url_env_var == ""
    assert profile.is_aggregator is False
    assert profile.is_routing_aggregator is None


def test_provider_identity_fields_are_declarative() -> None:
    profile = ProviderProfile(
        name="example-router",
        aliases=("example", "router-example"),
        base_url_env_var="EXAMPLE_BASE_URL",
        is_aggregator=True,
        is_routing_aggregator=False,
    )

    assert profile.aliases == ("example", "router-example")
    assert profile.base_url_env_var == "EXAMPLE_BASE_URL"
    assert profile.is_aggregator is True
    assert profile.is_routing_aggregator is False
