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


class TestZaiHook:
    DEFAULT = "https://api.z.ai/api/paas/v4"

    def test_probe_detects_and_caches(self, monkeypatch):
        from hermes_cli import auth_zai_kimi

        monkeypatch.setattr(auth_zai_kimi, "_zai_probe_failed_until", {})
        calls = []

        def detect(api_key, timeout=8.0):
            calls.append(api_key)
            return {"id": "cn", "base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-5", "label": "China"}

        monkeypatch.setattr("hermes_cli.auth.detect_zai_endpoint", detect)
        profile = _profile("zai")
        for _ in range(2):
            url = profile.resolve_base_url(api_key="glm-key", default_url=self.DEFAULT, env_url="")
            assert url == "https://open.bigmodel.cn/api/paas/v4"
        assert calls == ["glm-key"]  # second call answered from the auth.json cache

    def test_status_variant_never_probes(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.auth.detect_zai_endpoint", lambda *a, **k: pytest.fail("probed"))
        profile = _profile("zai")
        assert profile.resolve_base_url(api_key="glm-key", default_url=self.DEFAULT, env_url="", probe=False) == self.DEFAULT
        assert profile.resolve_base_url(api_key="glm-key", default_url=self.DEFAULT, env_url="https://o.test/", probe=False) == "https://o.test/"

    def test_env_override_and_missing_key_skip_probe(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.auth.detect_zai_endpoint", lambda *a, **k: pytest.fail("probed"))
        profile = _profile("zai")
        assert profile.resolve_base_url(api_key="glm-key", default_url=self.DEFAULT, env_url="https://o.test/v4") == "https://o.test/v4"
        assert profile.resolve_base_url(api_key="", default_url=self.DEFAULT, env_url="") == self.DEFAULT


class TestCopilotHook:
    DEFAULT = "https://api.githubcopilot.com"

    def test_exchange_endpoint_wins(self, monkeypatch):
        import hermes_cli.copilot_auth as copilot_auth

        monkeypatch.setattr(copilot_auth, "resolve_copilot_token", lambda: ("ghu_x", "env"))
        monkeypatch.setattr(copilot_auth, "get_copilot_api_token", lambda raw: ("tok", "https://api.ent.test"))
        profile = _profile("copilot")
        assert profile.resolve_base_url(api_key="tok", default_url=self.DEFAULT, env_url="") == "https://api.ent.test"

    def test_no_exchange_endpoint_keeps_env_override_and_status_skips_exchange(self, monkeypatch):
        import hermes_cli.copilot_auth as copilot_auth

        monkeypatch.setattr(copilot_auth, "resolve_copilot_token", lambda: ("ghu_x", "env"))
        monkeypatch.setattr(copilot_auth, "get_copilot_api_token", lambda raw: ("tok", None))
        profile = _profile("copilot")
        assert profile.resolve_base_url(api_key="tok", default_url=self.DEFAULT, env_url="https://o.test/") == "https://o.test"
        monkeypatch.setattr(copilot_auth, "resolve_copilot_token", lambda: pytest.fail("exchanged"))
        assert profile.resolve_base_url(api_key="tok", default_url=self.DEFAULT, env_url="", probe=False) == self.DEFAULT


class TestLMStudioHook:
    def test_runtime_normalises_to_v1_status_keeps_as_written(self):
        profile = _profile("lmstudio")
        kwargs = {"api_key": "", "default_url": "http://127.0.0.1:1234/v1", "env_url": "http://host:1234/api/v1"}
        assert profile.resolve_base_url(**kwargs) == "http://host:1234/v1"
        assert profile.resolve_base_url(**kwargs, probe=False) == "http://host:1234/api/v1"


class TestActualHook:
    @pytest.mark.parametrize("probe", [True, False])
    def test_root_urls_gain_v1(self, probe):
        profile = _profile("actual")
        default = "https://api.actual.inc/v1"
        assert profile.resolve_base_url(api_key="", default_url=default, env_url="http://127.0.0.1:8080", probe=probe) == "http://127.0.0.1:8080/v1"
        assert profile.resolve_base_url(api_key="", default_url=default, env_url="https://api.actual.inc/", probe=probe) == default
        assert profile.resolve_base_url(api_key="", default_url=default, env_url="", probe=probe) == default


def test_user_override_without_the_hook_keeps_bundled_routing(monkeypatch):
    """A ``$HERMES_HOME`` plugin re-registering ``kimi-coding`` as a plain ProviderProfile keeps the
    bundled key-prefix routing: an ``sk-kimi-`` key still reaches the Kimi Code endpoint."""
    import providers
    from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider_base_url
    from hermes_cli.auth_zai_kimi import KIMI_CODE_BASE_URL

    plain = ProviderProfile(name="kimi-coding", api_mode="chat_completions")
    monkeypatch.setattr(providers, "get_provider_profile", lambda name: plain)
    url = resolve_provider_base_url(PROVIDER_REGISTRY["kimi-coding"], api_key="sk-kimi-abc", env_url="")
    assert url == KIMI_CODE_BASE_URL


def test_runtime_reregistration_without_the_hook_keeps_bundled_routing(monkeypatch):
    """A plain ``kimi-coding`` registered after discovery (legacy module / direct call) replaces the
    process-wide entry; the bundled profile's routing still applies."""
    import providers
    from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider_base_url
    from hermes_cli.auth_zai_kimi import KIMI_CODE_BASE_URL

    providers.get_provider_profile("kimi-coding")  # discovery
    plain = ProviderProfile(name="kimi-coding", api_mode="chat_completions")
    monkeypatch.setitem(providers._REGISTRY, "kimi-coding", plain)
    assert providers.get_provider_profile("kimi-coding") is plain
    url = resolve_provider_base_url(PROVIDER_REGISTRY["kimi-coding"], api_key="sk-kimi-abc", env_url="")
    assert url == KIMI_CODE_BASE_URL
