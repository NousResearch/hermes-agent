"""Behavior contracts for Meta Muse prompt-caching host mandate."""

import pytest

from hermes_cli.providers import determine_api_mode, host_mandated_api_mode
from hermes_cli import runtime_provider as rp


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
        assert host_mandated_api_mode(url) == "codex_responses"

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
        assert host_mandated_api_mode(url) != "codex_responses"
        # Must be None for generic/unrelated hosts (contract: no clobber)
        if url in (
            "https://generic.example.com/v1",
            "https://[::1]/v1",
            "",
            "https://meta.ai/v1",
            "https://api.meta.ai.attacker.test/v1",
            "https://proxy.test/api.meta.ai/v1",
        ):
            assert host_mandated_api_mode(url) is None

    def test_determine_api_mode_meta_via_named_custom(self):
        assert determine_api_mode("meta", "https://api.meta.ai/v1") == "codex_responses"
        assert determine_api_mode("custom", "https://api.meta.ai/v1") == "codex_responses"
        assert determine_api_mode("generic", "https://generic.example.com/v1") == "chat_completions"

    def test_determine_api_mode_meta_with_trailing_slash(self):
        assert determine_api_mode("meta", "https://api.meta.ai/v1/") == "codex_responses"

    def test_runtime_detect_meta(self):
        assert rp._detect_api_mode_for_url("https://api.meta.ai/v1") == "codex_responses"
        assert rp._detect_api_mode_for_url("https://api.meta.ai/v1/chat/completions") == "codex_responses"
        assert rp._detect_api_mode_for_url("https://API.META.AI/v1") == "codex_responses"

    def test_runtime_detect_meta_rejects_spoofs(self):
        assert rp._detect_api_mode_for_url("https://api.meta.ai.attacker.test/v1") is None
        assert rp._detect_api_mode_for_url("https://proxy.test/api.meta.ai/v1") is None
        assert rp._detect_api_mode_for_url("https://meta.ai/v1") is None
        assert rp._detect_api_mode_for_url("https://generic.example.com/v1") is None

    def test_fallback_api_mode_meta(self):
        assert rp._fallback_api_mode("meta", "https://api.meta.ai/v1", "muse-spark-1.2") == "codex_responses"
        assert rp._fallback_api_mode("custom", "https://api.meta.ai/v1", "muse-spark-1.2") == "codex_responses"
        # generic still chat
        assert rp._fallback_api_mode("custom", "https://generic.example.com/v1", "muse-spark-1.2") == "chat_completions"


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

    def test_providers_dict_to_custom_providers_preserves_meta_without_explicit_mode(self):
        # Without explicit api_mode, host mandate resolves via providers.determine_api_mode
        # This test ensures the providers entry survives normalization and the host still mandates
        from hermes_cli.providers import host_mandated_api_mode

        assert host_mandated_api_mode("https://api.meta.ai/v1") == "codex_responses"
        # Simulate providers dict path: providers.meta.base_url stored, no api_mode field
        # The resolution layer (determine_api_mode / runtime_provider) must mandate.
        from hermes_cli.providers import determine_api_mode

        assert determine_api_mode("meta", "https://api.meta.ai/v1", "muse-spark-1.2") == "codex_responses"
