"""Tests for the bundled OCI Generative AI provider plugin."""

import pytest

from agent.models_dev import lookup_models_dev_context
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

    def test_context_window_comes_from_models_dev_oci_catalog(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the models.dev mapping, gpt-oss on OCI fell back to the generic 256K window (it's 128K)."""
        catalog = {
            "oci": {
                "models": {
                    "openai.gpt-oss-120b": {
                        "limit": {"context": 128000, "output": 16384}
                    }
                }
            }
        }
        monkeypatch.setattr(
            "agent.models_dev.fetch_models_dev", lambda *a, **k: catalog
        )
        assert lookup_models_dev_context("oci", "openai.gpt-oss-120b") == 128000


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }
]


def _main_loop_kwargs(model: str, reasoning_config: dict | None = None) -> dict:
    from agent.transports.chat_completions import ChatCompletionsTransport

    return ChatCompletionsTransport().build_kwargs(
        model=model,
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        tools=_TOOLS,
        provider_profile=_profile(),
        reasoning_config=reasoning_config,
    )


class TestOCIReasoningWire:
    @pytest.mark.parametrize("model", ["openai.gpt-6-sol", "openai.gpt-6-luna", "openai.gpt-5.6-sol"])
    def test_tool_turns_turn_reasoning_off_where_oci_requires_it(self, model: str) -> None:
        """OCI 400s tools on these models unless reasoning_effort is "none" (the default is not)."""
        assert _main_loop_kwargs(model)["reasoning_effort"] == "none"
        assert _main_loop_kwargs(model, {"enabled": True, "effort": "high"})["reasoning_effort"] == "none"

    @pytest.mark.parametrize("model", ["openai.gpt-5.5", "openai.gpt-5", "openai.o3", "openai.gpt-oss-120b"])
    def test_other_models_keep_their_default(self, model: str) -> None:
        """gpt-5, o3 and o4-mini reject "none"; models that take tools as-is get no reasoning field."""
        kw = _main_loop_kwargs(model, {"enabled": True, "effort": "high"})
        assert "reasoning_effort" not in kw
        assert "reasoning" not in kw.get("extra_body", {})

    def test_auxiliary_calls_skip_generic_reasoning_body(self) -> None:
        """OpenAI models on OCI 400 the generic extra_body.reasoning fallback ("Unknown parameter")."""
        from agent.auxiliary_client import _build_call_kwargs

        kw = _build_call_kwargs(
            "oci",
            "openai.gpt-5.5",
            [{"role": "user", "content": "hi"}],
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url=_profile().base_url,
        )
        assert "reasoning" not in kw.get("extra_body", {})
