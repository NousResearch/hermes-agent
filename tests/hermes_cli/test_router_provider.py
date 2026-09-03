"""Behavior contracts for the Ramp Router (api.router.com) provider.

Router is Responses-native: the host implements GET /v1/models and
POST /v1/responses, and /v1/chat/completions is only a minimal
compatibility shim translated onto Responses.
These tests pin the host mandate, the runtime URL detection that mirrors
it, and the profile/auth registry wiring — same contract suite shape as
tests/hermes_cli/test_meta_prompt_cache.py.
"""

import pytest

from hermes_cli.providers import determine_api_mode, host_mandated_api_mode
from hermes_cli import runtime_provider as rp


class TestHostMandatedRouterResponses:
    @pytest.mark.parametrize(
        "url",
        [
            "https://api.router.com/v1",
            "https://api.router.com/v1/",
            "https://api.router.com/v1/chat/completions",
            "https://API.ROUTER.COM/v1",
            "https://api.router.com",
            "https://api.router.com:443/v1",
            "https://attacker.test@api.router.com/v1",
        ],
    )
    def test_host_mandated_router_returns_codex_responses(self, url):
        assert host_mandated_api_mode(url) == "codex_responses"

    @pytest.mark.parametrize(
        "url",
        [
            "https://api.router.com.attacker.test/v1",
            "https://proxy.test/api.router.com/v1",
            "https://api.router.com.evil/v1",
            "https://router.com/v1",
            "https://www.router.com/v1",
            "https://app.router.com/v1",
            "https://docs.router.com/v1",
            "https://generic.example.com/v1",
            "",
        ],
    )
    def test_host_mandated_router_rejects_spoofs(self, url):
        assert host_mandated_api_mode(url) != "codex_responses"
        # Generic/unrelated hosts must stay None (contract: no clobber of an
        # explicitly configured api_mode on endpoints we don't recognize).
        if url in (
            "https://api.router.com.attacker.test/v1",
            "https://proxy.test/api.router.com/v1",
            "https://generic.example.com/v1",
            "https://app.router.com/v1",
            "https://docs.router.com/v1",
            "",
        ):
            assert host_mandated_api_mode(url) is None

    def test_determine_api_mode_router_via_named_custom(self):
        assert determine_api_mode("router", "https://api.router.com/v1") == "codex_responses"
        assert determine_api_mode("custom", "https://api.router.com/v1") == "codex_responses"

    def test_runtime_detect_router(self):
        assert rp._detect_api_mode_for_url("https://api.router.com/v1") == "codex_responses"
        assert rp._detect_api_mode_for_url("https://api.router.com/v1/chat/completions") == "codex_responses"
        assert rp._detect_api_mode_for_url("https://API.ROUTER.COM/v1") == "codex_responses"

    def test_runtime_detect_router_rejects_spoofs(self):
        assert rp._detect_api_mode_for_url("https://api.router.com.attacker.test/v1") is None
        assert rp._detect_api_mode_for_url("https://proxy.test/api.router.com/v1") is None
        assert rp._detect_api_mode_for_url("https://router.com/v1") is None
        assert rp._detect_api_mode_for_url("https://app.router.com/v1") is None

    def test_fallback_api_mode_router(self):
        assert rp._fallback_api_mode("router", "https://api.router.com/v1", "gpt-5.4-mini") == "codex_responses"
        assert rp._fallback_api_mode("custom", "https://api.router.com/v1", "gpt-5.4-mini") == "codex_responses"
        # generic endpoints stay chat_completions
        assert rp._fallback_api_mode("custom", "https://generic.example.com/v1", "gpt-5.4-mini") == "chat_completions"


class TestRouterProfileRegistration:
    def test_profile_registered_with_responses_mode(self):
        from providers import get_provider_profile

        profile = get_provider_profile("router")
        assert profile is not None
        assert profile.api_mode == "codex_responses"
        assert profile.auth_type == "api_key"
        assert profile.base_url.startswith("https://api.router.com")

    def test_profile_aliases_resolve(self):
        from providers import get_provider_profile

        canonical = get_provider_profile("router")
        for alias in ("ramp-router", "ramp", "router.com"):
            assert get_provider_profile(alias) is canonical, alias

    def test_documented_env_var_is_primary(self):
        from providers import get_provider_profile

        profile = get_provider_profile("router")
        # RAMP_ROUTER_API_KEY is the variable Router's docs tell users to
        # set; it must stay first so key resolution prefers it.
        assert profile.env_vars[0] == "RAMP_ROUTER_API_KEY"

    def test_no_hardcoded_fallback_models(self):
        # Router model IDs are account-scoped (BYOK accounts see extra
        # entries) and the vendor docs say to read the catalog at runtime —
        # an offline fallback list would advertise IDs a key may not have.
        from providers import get_provider_profile

        profile = get_provider_profile("router")
        assert profile.fallback_models == ()

    def test_auth_registry_autowired(self):
        from hermes_cli.auth import PROVIDER_REGISTRY

        config = PROVIDER_REGISTRY.get("router")
        assert config is not None
        assert config.auth_type == "api_key"
        # Key vars must not contain the base-url override var, which is
        # split out into base_url_env_var by the auto-registry.
        assert "RAMP_ROUTER_API_KEY" in config.api_key_env_vars
        assert "RAMP_ROUTER_BASE_URL" not in config.api_key_env_vars
        assert config.base_url_env_var == "RAMP_ROUTER_BASE_URL"
        assert config.inference_base_url.startswith("https://api.router.com")
