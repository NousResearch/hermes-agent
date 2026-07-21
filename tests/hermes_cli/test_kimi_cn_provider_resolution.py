"""Regression tests for explicit Kimi China model switching."""

import pytest

from hermes_cli.providers import resolve_provider_full


@pytest.mark.parametrize("slug", ["kimi-coding-cn", "kimi-cn", "moonshot-cn"])
def test_kimi_cn_provider_resolves_to_domestic_endpoint(slug):
    provider = resolve_provider_full(slug)

    assert provider is not None
    assert provider.id == "kimi-coding-cn"
    assert provider.api_key_env_vars == ("KIMI_CN_API_KEY",)
    assert provider.base_url == "https://api.moonshot.cn/v1"
