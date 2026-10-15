"""Tests for the bundled OCI Generative AI provider plugin."""

from providers import get_provider_profile
from providers.base import ProviderProfile


def _profile() -> ProviderProfile:
    p = get_provider_profile("oci")
    assert p is not None
    return p


class TestOCIProfile:
    def test_aliases_resolve(self) -> None:
        for alias in ("oci-genai", "oracle", "oracle-cloud"):
            assert get_provider_profile(alias) is _profile()

    def test_openai_compatible_bearer_key(self) -> None:
        p = _profile()
        assert p.api_mode == "chat_completions"
        assert p.auth_type == "api_key"
        assert p.env_vars[0] == "OCI_GENAI_API_KEY"
        assert p.base_url.endswith("/openai/v1")

    def test_no_model_listing_network_call(self) -> None:
        """The OpenAI-compatible route 404s on GET /models; setup must use the curated list."""
        p = _profile()
        assert p.supports_model_listing is False
        assert p.fetch_models() is None
        assert "openai.gpt-oss-120b" in p.fallback_models

    def test_avoids_zstd_stream_encoding(self) -> None:
        assert "zstd" not in _profile().default_headers["Accept-Encoding"]

    def test_curated_list_skips_llama(self) -> None:
        """OCI 400s Llama tool calls whose first message is not a user turn; Hermes always opens with a system prompt."""
        assert not any(m.startswith("meta.") for m in _profile().fallback_models)
