"""GLM / Z.AI thinking behavior on the Anthropic-Messages wire.

Contract (docs.z.ai/devpack/latest-model, mirrored from the ZCode builtin config):

- GLM Coding Plan Anthropic routes (``api.z.ai/api/anthropic``, ``open.bigmodel.cn/api/anthropic``)
  receive ``thinking.type`` + ``output_config.effort`` — never manual ``budget_tokens``.
- GLM-5.3 always reasons: ``disabled`` is converted to ``enabled`` + ``low`` (the server
  rejects the disable); the documented effort trio is low/high/max, so ``medium`` clamps to
  ``high`` locally instead of trusting a server-side auto-convert.
- GLM-5.2 accepts a real ``thinking.type=disabled`` (its "nothink") and the high/max pair.
- Older GLM families (4.5–5.1) get a plain thinking toggle.
- Keys are Bearer (sent as ANTHROPIC_AUTH_TOKEN), not x-api-key.

GLM on the chat_completions route handles ``thinking`` via ``extra_body`` in
``ChatCompletionsTransport`` (the ``zai`` profile) — not covered here.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestGlmCodingAnthropicThinking:
    """GLM-family thinking on the Anthropic wire (GLM Coding Plan)."""

    def test_glm_53_effort_maps_to_output_config(self) -> None:
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="glm-5.3",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "max"},
            base_url="https://api.z.ai/api/anthropic",
        )
        assert kwargs["thinking"] == {"type": "enabled"}
        assert kwargs["output_config"] == {"effort": "max"}
        assert "budget_tokens" not in kwargs.get("thinking", {})
        assert "temperature" not in kwargs
        assert kwargs["max_tokens"] == 4096

    def test_glm_53_medium_clamps_to_high(self) -> None:
        """``medium`` is not a documented level on the Anthropic route — clamp to ``high``."""
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="glm-5.3",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url="https://api.z.ai/api/anthropic",
        )
        assert kwargs["output_config"] == {"effort": "high"}

    def test_glm_53_disabled_converts_to_low(self) -> None:
        """GLM-5.3 rejects ``thinking.type=disabled``; disabled -> enabled + low."""
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="glm-5.3",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": False},
            base_url="https://api.z.ai/api/anthropic",
        )
        assert kwargs["thinking"] == {"type": "enabled"}
        assert kwargs["output_config"] == {"effort": "low"}

    def test_glm_53_unset_effort_leaves_server_default(self) -> None:
        """No explicit effort -> thinking enabled only; the server defaults to max."""
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="glm-5.3",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True},
            base_url="https://api.z.ai/api/anthropic",
        )
        assert kwargs["thinking"] == {"type": "enabled"}
        assert "output_config" not in kwargs

    def test_glm_52_disabled_sends_real_disable(self) -> None:
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="glm-5.2",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": False},
            base_url="https://api.z.ai/api/anthropic",
        )
        assert kwargs["thinking"] == {"type": "disabled"}
        assert "output_config" not in kwargs

    def test_glm_52_xhigh_maps_to_max_and_low_clamps_to_high(self) -> None:
        from agent.anthropic_adapter import build_anthropic_kwargs

        xhigh = build_anthropic_kwargs(
            model="glm-5.2",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "xhigh"},
            base_url="https://api.z.ai/api/anthropic",
        )
        assert xhigh["output_config"] == {"effort": "max"}
        low = build_anthropic_kwargs(
            model="glm-5.2",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "low"},
            base_url="https://api.z.ai/api/anthropic",
        )
        assert low["output_config"] == {"effort": "high"}  # GLM-5.2 floor is high

    def test_glm_older_family_gets_plain_toggle(self) -> None:
        from agent.anthropic_adapter import build_anthropic_kwargs

        enabled = build_anthropic_kwargs(
            model="glm-4.6",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "high"},
            base_url="https://api.z.ai/api/anthropic",
        )
        assert enabled["thinking"] == {"type": "enabled"}
        assert "output_config" not in enabled
        disabled = build_anthropic_kwargs(
            model="glm-4.6",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": False},
            base_url="https://api.z.ai/api/anthropic",
        )
        assert disabled["thinking"] == {"type": "disabled"}

    @pytest.mark.parametrize(
        "model,expected_effort",
        [
            # Relay spellings must hit the GLM wire, not the legacy manual path.
            ("z-ai/glm-5.3", "high"),
            ("GLM-5.3", "high"),
            ("glm-5p3", "high"),
            ("zai-org-glm-5-2", "high"),
            ("glm-5.2[1m]", "high"),
        ],
    )
    def test_glm_alias_spellings_use_glm_wire(self, model: str, expected_effort: str) -> None:
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=4096,
            reasoning_config={"enabled": True, "effort": "high"},
            base_url="https://api.z.ai/api/anthropic",
        )
        assert kwargs.get("thinking", {}).get("type") == "enabled", kwargs.get("thinking")
        assert "budget_tokens" not in kwargs.get("thinking", {})
        assert kwargs["output_config"] == {"effort": expected_effort}

    def test_glm_models_get_coding_plan_output_ceiling(self) -> None:
        from agent.anthropic_adapter import _get_anthropic_max_output

        assert _get_anthropic_max_output("glm-5.3") == 128_000
        assert _get_anthropic_max_output("glm-5.3-flash") == 128_000
        assert _get_anthropic_max_output("glm-5.2[1m]") == 128_000
        assert _get_anthropic_max_output("glm-5") == 64_000


class TestGlmAnthropicEndpointClassification:
    """GLM Coding Plan endpoints: bearer auth + model-family detection."""

    @pytest.mark.parametrize(
        "base_url,expected",
        [
            ("https://api.z.ai/api/anthropic", True),
            ("https://api.z.ai/api/anthropic/v1", True),
            ("https://open.bigmodel.cn/api/anthropic", True),
            ("https://api.z.ai/api/paas/v4", False),       # OpenAI-compatible route
            ("https://api.z.ai/api/coding/paas/v4", False),  # OpenAI-compatible route
            ("https://api.deepseek.com/anthropic", False),
            ("https://api.anthropic.com", False),
        ],
    )
    def test_is_glm_anthropic_endpoint(self, base_url: str, expected: bool) -> None:
        from agent.anthropic_endpoints import _is_glm_anthropic_endpoint

        assert _is_glm_anthropic_endpoint(base_url) is expected

    def test_glm_endpoint_uses_bearer_auth_for_regular_api_keys(self) -> None:
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            from agent.anthropic_adapter import build_anthropic_client

            build_anthropic_client(
                "zai-plan-key-123",
                base_url="https://api.z.ai/api/anthropic",
            )
            kwargs = mock_sdk.Anthropic.call_args[1]
            assert kwargs["auth_token"] == "zai-plan-key-123"
            assert "api_key" not in kwargs
            # bearer-only construction omits x-api-key so an env ANTHROPIC_API_KEY never rides along
            assert kwargs["default_headers"]["X-Api-Key"] is mock_sdk.Omit.return_value

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("glm-5.3", True),
            ("glm-5.2[1m]", True),
            ("z-ai/glm-5.3", True),
            ("glm-4.7", True),
            ("glm_4.6", True),
            ("GLM-5.3-Flash", True),
            ("zai-org-glm-5-2", True),
            ("claude-opus-4-6", False),
            ("kimi-k2.5", False),
            ("qwen3-max", False),
            ("", False),
            (None, False),
        ],
    )
    def test_model_name_is_glm_family(self, model, expected: bool) -> None:
        from agent.anthropic_endpoints import _model_name_is_glm_family

        assert _model_name_is_glm_family(model) is expected
