"""Regression tests for the openai-api reasoning gate (Issue #76255).

Direct api.openai.com (openai-api provider) returns HTTP 400
"Unsupported parameter: 'reasoning.effort'" for non-reasoning models
(gpt-4o-mini, gpt-4.1-mini, gpt-4o, gpt-4.1, ...). Only o-series (o1, o3, o4)
and GPT-5 models accept reasoning controls.

Tests verify:
1. Standard GPT-4 / GPT-3.5 models do NOT support reasoning.
2. o-series AND GPT-5 family models DO support reasoning.
3. Fine-tuned variants (ft:o4-mini:..., ft:gpt-5:...) are handled correctly.
4. Main Responses transport (agent/transports/codex.py) gates reasoning.
5. Auxiliary Responses adapter (agent/auxiliary_client.py) gates reasoning.
"""

from unittest.mock import MagicMock

import pytest

from agent.auxiliary_client import _CodexCompletionsAdapter
from agent.model_metadata import openai_model_supports_reasoning
from agent.transports.codex import ResponsesApiTransport


# ---------------------------------------------------------------------------
# Unit tests for the helper function
# ---------------------------------------------------------------------------

class TestOpenaiModelSupportsReasoning:
    """Tests for openai_model_supports_reasoning()."""

    @pytest.mark.parametrize("model", [
        # o-series
        "o1",
        "o1-mini",
        "o1-preview",
        "o3",
        "o3-mini",
        "o4",
        "o4-mini",
        "o4-mini-2025-04-16",
        "openai/o4-mini",        # vendor-prefixed
        "openai/o1-preview",
        "ft:o4-mini:org:name",   # fine-tuned o-series
        "ft:o3:myorg:v1",
        # GPT-5 family
        "gpt-5",
        "gpt-5.6-sol",
        "gpt-5.1-codex",
        "gpt-5.5-pro",
        "openai/gpt-5",
        "ft:gpt-5:org:v1",
    ])
    def test_reasoning_models_return_true(self, model):
        assert openai_model_supports_reasoning(model) is True, (
            f"Expected {model!r} to support reasoning"
        )

    @pytest.mark.parametrize("model", [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "",
        None,
    ])
    def test_non_reasoning_models_return_false(self, model):
        assert openai_model_supports_reasoning(model) is False, (
            f"Expected {model!r} NOT to support reasoning"
        )


# ---------------------------------------------------------------------------
# Integration tests via main transport build_kwargs
# ---------------------------------------------------------------------------

def _make_messages():
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]


def _base_params(model: str, base_url: str, **extra):
    return {
        "base_url": base_url,
        "model_lower": model.lower(),
        "is_xai_responses": False,
        "is_github_responses": False,
        "is_codex_backend": False,
        "replay_encrypted_reasoning": False,
        **extra,
    }


OPENAI_DIRECT_URL = "https://api.openai.com/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"


class TestReasoningGateIntegration:
    """Verify build_kwargs behaviour for openai-api provider."""

    def test_gpt4o_mini_does_not_get_reasoning_field(self):
        """GPT-4o-mini on api.openai.com must NOT include reasoning (fixes #76255)."""
        transport = ResponsesApiTransport()
        kwargs = transport.build_kwargs(
            "gpt-4o-mini",
            _make_messages(),
            **_base_params("gpt-4o-mini", OPENAI_DIRECT_URL),
        )
        assert "reasoning" not in kwargs, (
            "gpt-4o-mini on api.openai.com must not receive reasoning field"
        )

    def test_gpt41_mini_does_not_get_reasoning_field(self):
        """gpt-4.1-mini on api.openai.com must NOT include reasoning (fixes #76255)."""
        transport = ResponsesApiTransport()
        kwargs = transport.build_kwargs(
            "gpt-4.1-mini",
            _make_messages(),
            **_base_params("gpt-4.1-mini", OPENAI_DIRECT_URL),
        )
        assert "reasoning" not in kwargs, (
            "gpt-4.1-mini on api.openai.com must not receive reasoning field"
        )

    def test_o4_mini_gets_reasoning_field(self):
        """o4-mini on api.openai.com SHOULD receive reasoning."""
        transport = ResponsesApiTransport()
        kwargs = transport.build_kwargs(
            "o4-mini",
            _make_messages(),
            **_base_params("o4-mini", OPENAI_DIRECT_URL),
        )
        assert "reasoning" in kwargs, (
            "o4-mini on api.openai.com must receive reasoning field"
        )
        assert "effort" in kwargs["reasoning"]

    def test_gpt5_gets_reasoning_field(self):
        """GPT-5 family on api.openai.com SHOULD receive reasoning."""
        transport = ResponsesApiTransport()
        kwargs = transport.build_kwargs(
            "gpt-5.6-sol",
            _make_messages(),
            **_base_params("gpt-5.6-sol", OPENAI_DIRECT_URL),
        )
        assert "reasoning" in kwargs, (
            "gpt-5.6-sol on api.openai.com must receive reasoning field"
        )

    def test_non_openai_direct_gpt_model_still_gets_reasoning(self):
        """GPT-4o-mini via OpenRouter must still get reasoning (not affected by guard)."""
        transport = ResponsesApiTransport()
        kwargs = transport.build_kwargs(
            "gpt-4o-mini",
            _make_messages(),
            **_base_params("gpt-4o-mini", OPENROUTER_URL),
        )
        assert "reasoning" in kwargs, (
            "gpt-4o-mini via OpenRouter should still receive reasoning field"
        )


# ---------------------------------------------------------------------------
# Integration tests via auxiliary client adapter (_CodexCompletionsAdapter)
# ---------------------------------------------------------------------------

class TestAuxiliaryReasoningGateIntegration:
    """Verify auxiliary client adapter gates reasoning for direct OpenAI calls."""

    def test_auxiliary_gpt4o_mini_omits_reasoning_on_openai_direct(self):
        mock_client = MagicMock()
        mock_client.base_url = OPENAI_DIRECT_URL
        mock_client.responses.create.side_effect = RuntimeError("Stop after create")
        adapter = _CodexCompletionsAdapter(mock_client, "gpt-4o-mini")

        with pytest.raises(RuntimeError, match="Stop after create"):
            adapter.create(
                messages=_make_messages(),
                extra_body={"reasoning": {"effort": "medium"}},
            )

        assert mock_client.responses.create.called
        call_kwargs = mock_client.responses.create.call_args[1]
        assert "reasoning" not in call_kwargs, (
            "Auxiliary gpt-4o-mini on api.openai.com must not send reasoning"
        )

    def test_auxiliary_o4_mini_includes_reasoning_on_openai_direct(self):
        mock_client = MagicMock()
        mock_client.base_url = OPENAI_DIRECT_URL
        mock_client.responses.create.side_effect = RuntimeError("Stop after create")
        adapter = _CodexCompletionsAdapter(mock_client, "o4-mini")

        with pytest.raises(RuntimeError, match="Stop after create"):
            adapter.create(
                messages=_make_messages(),
                extra_body={"reasoning": {"effort": "medium"}},
            )

        assert mock_client.responses.create.called
        call_kwargs = mock_client.responses.create.call_args[1]
        assert "reasoning" in call_kwargs, (
            "Auxiliary o4-mini on api.openai.com must send reasoning"
        )

    def test_auxiliary_gpt5_includes_reasoning_on_openai_direct(self):
        mock_client = MagicMock()
        mock_client.base_url = OPENAI_DIRECT_URL
        mock_client.responses.create.side_effect = RuntimeError("Stop after create")
        adapter = _CodexCompletionsAdapter(mock_client, "gpt-5.6-sol")

        with pytest.raises(RuntimeError, match="Stop after create"):
            adapter.create(
                messages=_make_messages(),
                extra_body={"reasoning": {"effort": "medium"}},
            )

        assert mock_client.responses.create.called
        call_kwargs = mock_client.responses.create.call_args[1]
        assert "reasoning" in call_kwargs, (
            "Auxiliary gpt-5.6-sol on api.openai.com must send reasoning"
        )

    def test_auxiliary_non_openai_direct_preserves_reasoning(self):
        mock_client = MagicMock()
        mock_client.base_url = OPENROUTER_URL
        mock_client.responses.create.side_effect = RuntimeError("Stop after create")
        adapter = _CodexCompletionsAdapter(mock_client, "gpt-4o-mini")

        with pytest.raises(RuntimeError, match="Stop after create"):
            adapter.create(
                messages=_make_messages(),
                extra_body={"reasoning": {"effort": "medium"}},
            )

        assert mock_client.responses.create.called
        call_kwargs = mock_client.responses.create.call_args[1]
        assert "reasoning" in call_kwargs, (
            "Auxiliary gpt-4o-mini via OpenRouter must preserve reasoning"
        )
