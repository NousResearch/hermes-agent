"""Regression tests for disabling qwen3 "thinking" on Ollama (local and remote).

Ollama's OpenAI-compatible /v1 endpoint defaults thinking ON for qwen3-family
models; with tools present the reasoning swallows the tool call and the agent
returns empty output / no tool_calls (ollama/ollama #10976, #11381). The only
control Ollama honors on /v1 is ``reasoning_effort: "none"`` -> Think=false
(``chat_template_kwargs.enable_thinking`` is ignored, #10809).

Hermes previously emitted no reasoning field for local Ollama, so setting
``reasoning_effort: none`` was a silent no-op. These tests pin the fix for
NousResearch/hermes-agent#6152:
  * local Ollama + reasoning disabled -> reasoning_effort "none" auto-emitted
  * any host (incl. public/remote) -> model.extra_body forwards reasoning_effort
  * an explicit extra_body reasoning_effort takes precedence over the auto-path
"""

import pytest

from agent.transports import get_transport


@pytest.fixture
def transport():
    import agent.transports.chat_completions  # noqa: F401  (self-registers)
    return get_transport("chat_completions")


def _build(transport, *, base_url, reasoning_config, extra_body=None):
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
        extra_body_additions=extra_body,
    )


class TestOllamaThinkingDisable:
    LOCAL = "http://127.0.0.1:11434/v1"
    REMOTE = "https://ollama.example.com/v1"  # public domain: not is_local_endpoint

    # ── Local auto-path (existing reasoning_effort knob) ───────────────
    def test_disabled_local_emits_none(self, transport):
        out = _build(transport, base_url=self.LOCAL, reasoning_config={"enabled": False})
        assert out.get("reasoning_effort") == "none"

    def test_enabled_local_unchanged(self, transport):
        out = _build(
            transport,
            base_url=self.LOCAL,
            reasoning_config={"enabled": True, "effort": "medium"},
        )
        assert "reasoning_effort" not in out

    def test_disabled_remote_needs_optin(self, transport):
        # Auto-path is local-only; a public/remote host is not auto-fixed — it
        # requires the explicit model.extra_body opt-in (next tests).
        out = _build(transport, base_url=self.REMOTE, reasoning_config={"enabled": False})
        assert "reasoning_effort" not in out
        assert "reasoning_effort" not in out.get("extra_body", {})

    def test_no_reasoning_config_unchanged(self, transport):
        out = _build(transport, base_url=self.LOCAL, reasoning_config=None)
        assert "reasoning_effort" not in out

    # ── Universal explicit opt-in via model.extra_body ─────────────────
    def test_extra_body_disables_thinking_on_remote(self, transport):
        # Public/remote Ollama: model.extra_body forwards reasoning_effort regardless
        # of host (is_local_endpoint cannot match a public domain).
        out = _build(
            transport,
            base_url=self.REMOTE,
            reasoning_config={"enabled": False},
            extra_body={"reasoning_effort": "none"},
        )
        assert out["extra_body"]["reasoning_effort"] == "none"

    def test_extra_body_takes_precedence_over_autopath(self, transport):
        # Explicit extra_body wins over the local auto-path -> no conflicting
        # top-level reasoning_effort is emitted.
        out = _build(
            transport,
            base_url=self.LOCAL,
            reasoning_config={"enabled": False},
            extra_body={"reasoning_effort": "high"},
        )
        assert "reasoning_effort" not in out  # auto-path suppressed
        assert out["extra_body"]["reasoning_effort"] == "high"
