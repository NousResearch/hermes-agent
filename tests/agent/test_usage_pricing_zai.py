from agent.usage_pricing import (
    CanonicalUsage,
    estimate_usage_cost,
    resolve_billing_route,
)


def test_zai_coding_route_is_subscription_included():
    route = resolve_billing_route(
        "glm-5.3",
        provider="zai-coding",
        base_url="https://api.z.ai/api/coding/paas/v4",
    )
    assert route.billing_mode == "subscription_included"
    assert route.provider == "zai-coding"


def test_zai_coding_base_url_routes_to_subscription():
    route = resolve_billing_route(
        "glm-5.3",
        provider="zai",
        base_url="https://api.z.ai/api/coding/paas/v4",
    )
    assert route.billing_mode == "subscription_included"
    assert route.provider == "zai-coding"


def test_zai_metered_api_stays_off_subscription():
    route = resolve_billing_route(
        "glm-5.3",
        provider="zai",
        base_url="https://api.z.ai/api/paas/v4",
    )
    assert route.billing_mode != "subscription_included"


def test_zai_coding_estimate_usage_cost_is_included():
    result = estimate_usage_cost(
        "glm-5.3",
        CanonicalUsage(input_tokens=1000, output_tokens=500),
        provider="zai-coding",
        base_url="https://api.z.ai/api/coding/paas/v4",
    )
    assert result.status == "included"
    assert result.amount_usd is not None
    assert float(result.amount_usd) == 0.0


def test_zai_unconfirmed_routes_are_not_subscription_included():
    for provider, base_url in (
        ("zai-coding", None),
        ("zai", "https://open.bigmodel.cn/api/coding/paas/v4"),
        ("zai", "https://lookalike.example/api/coding/paas/v4"),
    ):
        route = resolve_billing_route("glm-5.3", provider=provider, base_url=base_url)
        assert route.billing_mode != "subscription_included"
