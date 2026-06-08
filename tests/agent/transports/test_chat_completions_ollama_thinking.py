"""Regression tests for disabling qwen3 "thinking" on local Ollama.

Ollama's OpenAI-compatible /v1 endpoint defaults thinking ON for qwen3-family
models; with tools present the reasoning swallows the tool call and the agent
returns empty output / no tool_calls (ollama/ollama #10976, #11381). The only
control Ollama honors on /v1 is ``reasoning_effort: "none"`` -> Think=false
(``chat_template_kwargs.enable_thinking`` is ignored, #10809).

Hermes previously emitted no reasoning field for local Ollama, so setting
``reasoning_effort: none`` was a silent no-op. These tests pin the fix for
NousResearch/hermes-agent#6152.
"""

import pytest

from agent.transports import get_transport


@pytest.fixture
def transport():
    import agent.transports.chat_completions  # noqa: F401  (self-registers)
    return get_transport("chat_completions")


def _build(transport, *, base_url, reasoning_config):
    return transport.build_kwargs(
        model="huihui_ai/Qwen3.6-abliterated:35b",
        messages=[{"role": "user", "content": "use the tool"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        base_url=base_url,
        reasoning_config=reasoning_config,
    )


class TestOllamaThinkingDisable:
    LOCAL = "http://127.0.0.1:11434/v1"
    REMOTE = "https://api.openai.com/v1"

    def test_disabled_local_emits_none(self, transport):
        # reasoning_effort:"none" -> Ollama Think=false -> tool-calling restored (#6152)
        out = _build(transport, base_url=self.LOCAL, reasoning_config={"enabled": False})
        assert out.get("reasoning_effort") == "none"

    def test_enabled_local_unchanged(self, transport):
        # When reasoning is enabled we must not alter behaviour (keep thinking).
        out = _build(
            transport,
            base_url=self.LOCAL,
            reasoning_config={"enabled": True, "effort": "medium"},
        )
        assert "reasoning_effort" not in out

    def test_disabled_remote_unchanged(self, transport):
        # Only local (Ollama) endpoints get the workaround; remote providers untouched.
        out = _build(transport, base_url=self.REMOTE, reasoning_config={"enabled": False})
        assert "reasoning_effort" not in out

    def test_no_reasoning_config_unchanged(self, transport):
        out = _build(transport, base_url=self.LOCAL, reasoning_config=None)
        assert "reasoning_effort" not in out
