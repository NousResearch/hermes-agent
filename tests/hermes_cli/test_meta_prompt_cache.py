"""Behavior contracts for Meta Muse prompt-caching host mandate."""

import pytest

from providers.routing import InvocationRequest, endpoint_api_mode, resolve_invocation_route

def _mode(provider: str, base_url: str, model: str = "") -> str:
    return resolve_invocation_route(InvocationRequest(provider=provider, base_url=base_url, model=model)).api_mode

class TestHostMandatedMetaResponses:
    @pytest.mark.parametrize(
        "url",
        [
            "https://api.meta.ai/v1",
            "https://api.meta.ai/v1/",
            "https://api.meta.ai/v1/chat/completions",
            "https://API.META.AI/v1",
            "https://api.meta.ai",
            "https://api.meta.ai:443/v1",
            "https://api.meta.ai./v1",
            "https://attacker.test@api.meta.ai/v1",
        ],
    )
    def test_host_mandated_meta_returns_codex_responses(self, url):
        assert endpoint_api_mode(url) == "codex_responses"

    @pytest.mark.parametrize(
        "url",
        [
            "https://api.meta.ai.attacker.test/v1",
            "https://proxy.test/api.meta.ai/v1",
            "https://api.meta.ai.evil/v1",
            "https://meta.ai/v1",
            "https://www.meta.ai/v1",
            "https://api.meta.com/v1",
            "https://[::1]/v1",
            "https://generic.example.com/v1",
            "",
        ],
    )
    def test_host_mandated_meta_rejects_spoofs(self, url):
        assert endpoint_api_mode(url) != "codex_responses"
        # Must be None for generic/unrelated hosts (contract: no clobber)
        if url in (
            "https://generic.example.com/v1",
            "https://[::1]/v1",
            "",
            "https://meta.ai/v1",
            "https://api.meta.ai.attacker.test/v1",
            "https://proxy.test/api.meta.ai/v1",
        ):
            assert endpoint_api_mode(url) is None

    def test_determine_api_mode_meta_via_named_custom(self):
        assert _mode("meta", "https://api.meta.ai/v1") == "codex_responses"
        assert _mode("custom", "https://api.meta.ai/v1") == "codex_responses"
        assert _mode("generic", "https://generic.example.com/v1") == "chat_completions"

    def test_determine_api_mode_meta_with_trailing_slash(self):
        assert _mode("meta", "https://api.meta.ai/v1/") == "codex_responses"

    def test_runtime_detect_meta(self):
        assert endpoint_api_mode("https://api.meta.ai/v1") == "codex_responses"
        assert endpoint_api_mode("https://api.meta.ai/v1/chat/completions") == "codex_responses"
        assert endpoint_api_mode("https://API.META.AI/v1") == "codex_responses"

    def test_runtime_detect_meta_rejects_spoofs(self):
        assert endpoint_api_mode("https://api.meta.ai.attacker.test/v1") is None
        assert endpoint_api_mode("https://proxy.test/api.meta.ai/v1") is None
        assert endpoint_api_mode("https://meta.ai/v1") is None
        assert endpoint_api_mode("https://generic.example.com/v1") is None

    def test_fallback_api_mode_meta(self):
        assert _mode("meta", "https://api.meta.ai/v1", "muse-spark-1.2") == "codex_responses"
        assert _mode("custom", "https://api.meta.ai/v1", "muse-spark-1.2") == "codex_responses"
        # generic still chat
        assert _mode("custom", "https://generic.example.com/v1", "muse-spark-1.2") == "chat_completions"

class TestMetaConfigRoundtrip:
    def test_providers_meta_api_mode_roundtrip(self):
        from hermes_cli.config import _normalize_custom_provider_entry

        entry = {"name": "Meta", "base_url": "https://api.meta.ai/v1", "api_mode": "codex_responses"}
        normalized = _normalize_custom_provider_entry(entry)
        assert normalized.get("api_mode") == "codex_responses"

        entry2 = {"name": "Meta", "base_url": "https://api.meta.ai/v1", "transport": "codex_responses"}
        normalized2 = _normalize_custom_provider_entry(entry2)
        # transport is lifted to api_mode via _normalize path or at least preserved
        assert normalized2.get("api_mode") == "codex_responses" or normalized2.get("transport") == "codex_responses"
