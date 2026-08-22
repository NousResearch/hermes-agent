"""Parity tests: pin the exact current transport behavior per provider.

These tests document the flag-based contract between run_agent.py and
ChatCompletionsTransport.build_kwargs(). When the next PR wires profiles
to replace flags, every assertion here must still pass — any failure is
a behavioral regression.
"""

import pytest
from agent.transports.chat_completions import ChatCompletionsTransport
from providers import get_provider_profile


@pytest.fixture
def transport():
    return ChatCompletionsTransport()


def _simple_messages():
    return [{"role": "user", "content": "hello"}]


def _max_tokens_fn(n):
    return {"max_completion_tokens": n}


class TestNvidiaParity:
    """NVIDIA NIM: default max_tokens=16384."""


    def test_user_max_tokens_overrides(self, transport):
        from providers import get_provider_profile

        profile = get_provider_profile("nvidia")
        kw = transport.build_kwargs(
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            messages=_simple_messages(),
            tools=None,
            max_tokens=4096,
            max_tokens_param_fn=_max_tokens_fn,
            provider_profile=profile,
        )
        assert kw["max_completion_tokens"] == 4096  # user overrides default

    @pytest.mark.parametrize(
        ("model", "effort", "expected"),
        [
            (
                "z-ai/glm5.1",
                "ultra",
                {
                    "chat_template_kwargs": {
                        "enable_thinking": True,
                        "clear_thinking": False,
                        "reasoning_effort": "high",
                    },
                },
            ),
            (
                "deepseek-ai/deepseek-v4",
                "xhigh",
                {
                    "chat_template_kwargs": {
                        "thinking": True,
                        "reasoning_effort": "high",
                    },
                },
            ),
            (
                "moonshotai/kimi-k2.5",
                "medium",
                {
                    "chat_template_kwargs": {
                        "thinking": True,
                        "reasoning_effort": "medium",
                    },
                },
            ),
            (
                "qwen/qwen3.5-397b-a17b",
                "minimal",
                {
                    "chat_template_kwargs": {
                        "thinking": True,
                        "reasoning_effort": "low",
                    },
                },
            ),
            (
                "nvidia/nemotron-3-ultra-550b-a55b",
                "high",
                {
                    "chat_template_kwargs": {"enable_thinking": True},
                },
            ),
            (
                "nvidia/nemotron-3-ultra-550b-a55b",
                "medium",
                {
                    "chat_template_kwargs": {
                        "enable_thinking": True,
                        "medium_effort": True,
                    },
                },
            ),
            (
                "nvidia/nemotron-3-ultra-550b-a55b",
                "low",
                {
                    "chat_template_kwargs": {
                        "enable_thinking": True,
                        "medium_effort": True,
                    },
                },
            ),
        ],
    )
    def test_reasoning_uses_family_specific_nim_envelope(
        self, transport, model, effort, expected
    ):
        kw = transport.build_kwargs(
            model=model,
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("nvidia"),
            reasoning_config={"enabled": True, "effort": effort},
        )

        assert kw["extra_body"] == expected
        assert "reasoning_effort" not in kw

    def test_nemotron_3_ultra_reasoning_disabled_uses_family_specific_toggle(
        self, transport
    ):
        kw = transport.build_kwargs(
            model="nvidia/nemotron-3-ultra-550b-a55b",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("nvidia"),
            reasoning_config={"enabled": False, "effort": "none"},
        )

        assert kw["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }
        assert "reasoning_budget" not in kw["extra_body"]

    def test_nemotron_3_ultra_does_not_enable_reasoning_without_policy(
        self, transport
    ):
        kw = transport.build_kwargs(
            model="nvidia/nemotron-3-ultra-550b-a55b",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("nvidia"),
            reasoning_config=None,
        )

        assert "extra_body" not in kw

    def test_nim_chat_template_overrides_merge_without_erasing_profile_fields(
        self, transport
    ):
        kw = transport.build_kwargs(
            model="deepseek-ai/deepseek-v4",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("nvidia"),
            reasoning_config={"enabled": True, "effort": "high"},
            request_overrides={
                "extra_body": {
                    "chat_template_kwargs": {
                        "custom_flag": "keep",
                        "reasoning_effort": "low",
                    }
                }
            },
        )

        assert kw["extra_body"]["chat_template_kwargs"] == {
            "thinking": True,
            "reasoning_effort": "low",
            "custom_flag": "keep",
        }


class TestKimiParity:
    """Kimi: OMIT temperature, max_tokens=32000, thinking + reasoning_effort."""

    def test_temperature_omitted(self, transport):
        kw = transport.build_kwargs(
            model="kimi-k2",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("kimi-coding"),
            omit_temperature=True,
        )
        assert "temperature" not in kw


    def test_thinking_enabled(self, transport):
        # xor contract (fix ce4e74b3): an explicit recognized effort sends
        # reasoning_effort ONLY — never paired with extra_body.thinking.
        kw = transport.build_kwargs(
            model="kimi-k2",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("kimi-coding"),
            reasoning_config={"enabled": True, "effort": "high"},
        )
        assert kw.get("reasoning_effort") == "high"
        assert "thinking" not in kw.get("extra_body", {})



    def test_reasoning_effort_top_level(self, transport):
        """Kimi reasoning_effort is a TOP-LEVEL api_kwargs key, NOT in extra_body."""
        kw = transport.build_kwargs(
            model="kimi-k2",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("kimi-coding"),
            reasoning_config={"enabled": True, "effort": "high"},
        )
        assert kw.get("reasoning_effort") == "high"
        assert "reasoning_effort" not in kw.get("extra_body", {})



class TestOpenRouterParity:
    """OpenRouter: provider preferences, reasoning in extra_body."""

    def test_provider_preferences(self, transport):
        prefs = {"allow": ["anthropic"], "sort": "price"}
        kw = transport.build_kwargs(
            model="anthropic/claude-sonnet-4.6",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("openrouter"),
            provider_preferences=prefs,
        )
        assert kw["extra_body"]["provider"] == prefs





class TestNousParity:
    """Nous: product tags, reasoning, omit when disabled."""

    def test_tags(self, transport):
        from agent.portal_tags import nous_portal_tags
        kw = transport.build_kwargs(
            model="hermes-3-llama-3.1-405b",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("nous"),
        )
        assert kw["extra_body"]["tags"] == nous_portal_tags()





class TestQwenParity:
    """Qwen: max_tokens=65536, vl_high_resolution, metadata top-level."""


    def test_vl_high_resolution(self, transport):
        kw = transport.build_kwargs(
            model="qwen3.5-plus",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("qwen-oauth"),
        )
        assert kw["extra_body"]["vl_high_resolution_images"] is True

    def test_metadata_top_level(self, transport):
        """Qwen metadata goes to top-level api_kwargs, NOT extra_body."""
        meta = {"sessionId": "s123", "promptId": "p456"}
        kw = transport.build_kwargs(
            model="qwen3.5-plus",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("qwen-oauth"),
            qwen_session_metadata=meta,
        )
        assert kw["metadata"] == meta
        assert "metadata" not in kw.get("extra_body", {})


class TestCustomOllamaParity:
    """Custom/Ollama: num_ctx, thinking controls — now tested via profile."""

    def test_ollama_num_ctx(self, transport):
        kw = transport.build_kwargs(
            model="llama3.1",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("custom"),
            ollama_num_ctx=131072,
        )
        assert kw["extra_body"]["options"]["num_ctx"] == 131072

    def test_think_false_when_disabled(self, transport):
        kw = transport.build_kwargs(
            model="qwen3:72b",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("custom"),
            reasoning_config={"enabled": False, "effort": "none"},
            base_url="http://127.0.0.1:11434/v1",
        )
        assert kw["extra_body"]["think"] is False

    def test_think_omitted_for_mistral_custom(self, transport):
        kw = transport.build_kwargs(
            model="mistral-small-latest",
            messages=_simple_messages(),
            tools=None,
            provider_profile=get_provider_profile("custom"),
            reasoning_config={"enabled": False, "effort": "none"},
            base_url="https://api.mistral.ai/v1",
        )
        assert kw.get("extra_body", {}).get("think") is None
        assert kw.get("reasoning_effort") == "none"
