"""Auto-routed auxiliary calls on xAI OAuth refresh the grant after a 403 bad-credentials (#84845).

A plugin/hook ``call_llm`` with no provider override inherits the main ``xai-oauth`` route as
``resolved_provider == "auto"``; the refresh rung must still resolve the concrete backend from the
client's host so the expired bearer is refreshed and retried instead of benching the only grant.
An auto client tagged with the API-key ``xai`` backend must NOT borrow that row (its tag, not the host,
names the provider): an XAI_API_KEY 401 would otherwise spend a stale ``xai-oauth`` grant's refresh
rotation and silently switch routes.
"""

import pytest

from agent.auxiliary_client import _auth_refresh_provider_for_route


@pytest.mark.parametrize(
    ("effective_provider", "expected"),
    [("", "xai-oauth"), ("auto", "xai-oauth"), ("xai-oauth", "xai-oauth"), ("xai", "xai")],
)
def test_auto_route_on_xai_host_refreshes_xai_oauth_unless_api_key_backend(effective_provider, expected):
    assert _auth_refresh_provider_for_route("auto", "https://api.x.ai/v1", effective_provider) == expected


@pytest.mark.parametrize("chain_label", ["api-key", "local/custom", "actual"])
def test_a_discovery_chain_label_is_not_a_provider(chain_label):
    """The discovery chain tags clients with its labels; an Anthropic 401 must still refresh ``anthropic``."""
    assert _auth_refresh_provider_for_route("auto", "https://api.anthropic.com", chain_label) == "anthropic"


def test_unknown_host_on_auto_route_stays_auto():
    assert _auth_refresh_provider_for_route("auto", "https://unknown.example.com/v1") == "auto"


@pytest.mark.parametrize("alias", ["x-ai", "grok"])
def test_an_alias_tag_names_its_canonical_provider(alias):
    """An alias of the API-key ``xai`` backend must not borrow the ``xai-oauth`` grant's refresh."""
    assert _auth_refresh_provider_for_route("auto", "https://api.x.ai/v1", alias) == "xai"
