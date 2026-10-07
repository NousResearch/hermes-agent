"""``ProviderProfile.resolve_base_url``: the default and each bundled override."""

from __future__ import annotations

import pytest

from providers import get_provider_profile
from providers.base import ProviderProfile

DEFAULT = "https://api.example.test/v1"


def _profile(name: str) -> ProviderProfile:
    profile = get_provider_profile(name)
    assert profile is not None, name
    return profile


class TestDefaultHook:
    def test_env_override_wins_and_loses_trailing_slash(self):
        profile = ProviderProfile(name="acme")
        url = profile.resolve_base_url(api_key="k", default_url=DEFAULT, env_url="https://proxy.test/v1/")
        assert url == "https://proxy.test/v1"

    def test_no_override_returns_default(self):
        profile = ProviderProfile(name="acme")
        assert profile.resolve_base_url(api_key="k", default_url=DEFAULT, env_url="") == DEFAULT

    def test_status_variant_returns_override_as_written(self):
        profile = ProviderProfile(name="acme")
        url = profile.resolve_base_url(api_key="", default_url=DEFAULT, env_url="https://proxy.test/v1/", probe=False)
        assert url == "https://proxy.test/v1/"
        assert profile.resolve_base_url(api_key="", default_url=DEFAULT, env_url="", probe=False) == DEFAULT

    def test_profile_without_override_uses_default(self):
        profile = _profile("gmi")
        assert type(profile).resolve_base_url is ProviderProfile.resolve_base_url


class TestKimiHook:
    MOONSHOT = "https://api.moonshot.ai/v1"

    @pytest.mark.parametrize("name", ["kimi-coding", "kimi-coding-cn"])
    @pytest.mark.parametrize("probe", [True, False])
    def test_key_prefix_routes_to_kimi_code(self, name, probe):
        from hermes_cli.auth import KIMI_CODE_BASE_URL

        profile = _profile(name)
        kwargs = {"default_url": self.MOONSHOT, "env_url": "", "probe": probe}
        assert profile.resolve_base_url(api_key="sk-kimi-abc", **kwargs) == KIMI_CODE_BASE_URL
        assert profile.resolve_base_url(api_key="sk-legacy", **kwargs) == self.MOONSHOT
        assert profile.resolve_base_url(api_key="", **kwargs) == self.MOONSHOT

    def test_env_override_wins_verbatim(self):
        profile = _profile("kimi-coding")
        url = profile.resolve_base_url(api_key="sk-kimi-abc", default_url=self.MOONSHOT, env_url="https://o.test/v1/")
        assert url == "https://o.test/v1/"
