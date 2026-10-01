"""Behavior contracts for the Ramp Router (api.router.com) provider.

Router is Responses-native: the host implements GET /v1/models and
POST /v1/responses, and /v1/chat/completions is only a minimal
compatibility shim translated onto Responses.
These tests pin the host mandate, the runtime URL detection that mirrors
it, and the profile/auth registry wiring — same contract suite shape as
tests/hermes_cli/test_meta_prompt_cache.py.
"""
from hermes_cli.provider_auth import get_provider_config

import pytest

from providers.routing import InvocationRequest, endpoint_api_mode, resolve_invocation_route

def _mode(provider: str, base_url: str, model: str = "") -> str:
    return resolve_invocation_route(InvocationRequest(provider=provider, base_url=base_url, model=model)).api_mode


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
        assert endpoint_api_mode(url) == "codex_responses"

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
        assert endpoint_api_mode(url) != "codex_responses"
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
            assert endpoint_api_mode(url) is None

    def test_determine_api_mode_router_via_named_custom(self):
        assert _mode("router", "https://api.router.com/v1") == "codex_responses"
        assert _mode("custom", "https://api.router.com/v1") == "codex_responses"

    def test_runtime_detect_router(self):
        assert endpoint_api_mode("https://api.router.com/v1") == "codex_responses"
        assert endpoint_api_mode("https://api.router.com/v1/chat/completions") == "codex_responses"
        assert endpoint_api_mode("https://API.ROUTER.COM/v1") == "codex_responses"

    def test_runtime_detect_router_rejects_spoofs(self):
        assert endpoint_api_mode("https://api.router.com.attacker.test/v1") is None
        assert endpoint_api_mode("https://proxy.test/api.router.com/v1") is None
        assert endpoint_api_mode("https://router.com/v1") is None
        assert endpoint_api_mode("https://app.router.com/v1") is None

    def test_fallback_api_mode_router(self):
        assert _mode("router", "https://api.router.com/v1", "gpt-5.4-mini") == "codex_responses"
        assert _mode("custom", "https://api.router.com/v1", "gpt-5.4-mini") == "codex_responses"
        # generic endpoints stay chat_completions
        assert _mode("custom", "https://generic.example.com/v1", "gpt-5.4-mini") == "chat_completions"


class TestRouterProfileRegistration:


    def test_auth_registry_autowired(self):

        config = get_provider_config("router")
        assert config is not None
        assert config.auth_type == "api_key"
        # Key vars must not contain the base-url override var, which is
        # split out into base_url_env_var by the auto-registry.
        assert "RAMP_ROUTER_API_KEY" in config.api_key_env_vars
        assert "RAMP_ROUTER_BASE_URL" not in config.api_key_env_vars
        assert config.base_url_env_var == "RAMP_ROUTER_BASE_URL"
        assert config.inference_base_url.startswith("https://api.router.com")
