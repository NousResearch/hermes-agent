"""End-to-end test for the ``needs_reasoning_content`` custom-provider opt-in.

Drives the actual replay path (AIAgent._copy_reasoning_content_for_api, via
_needs_thinking_reasoning_pad) rather than just the config-parsing layer
covered in tests/hermes_cli/test_custom_provider_needs_reasoning_content.py.
"""

from __future__ import annotations

import hermes_cli.config as hermes_config
from run_agent import AIAgent


def _make_agent(base_url: str) -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.provider = "custom"
    agent.model = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF:Q4_K_XL"
    agent.base_url = base_url
    agent.verbose_logging = False
    agent.reasoning_callback = None
    agent.stream_delta_callback = None
    agent._stream_callback = None
    return agent


def _stub_custom_providers(monkeypatch, custom_providers: list[dict]) -> None:
    monkeypatch.setattr(hermes_config, "load_config_readonly", lambda: {})
    monkeypatch.setattr(
        hermes_config,
        "get_compatible_custom_providers",
        lambda _config=None: custom_providers,
    )


def _tool_call_turn() -> dict:
    return {
        "role": "assistant",
        "reasoning_content": "<think>plan the next tool call</think>",
        "tool_calls": [{"id": "c1", "function": {"name": "terminal"}}],
    }


def test_reasoning_content_preserved_when_provider_opts_in(monkeypatch) -> None:
    """needs_reasoning_content: true keeps reasoning_content on replay."""
    _stub_custom_providers(monkeypatch, [
        {
            "name": "local-llamacpp",
            "base_url": "http://localhost:8080/v1",
            "needs_reasoning_content": True,
        },
    ])
    agent = _make_agent(base_url="http://localhost:8080/v1")
    api_msg: dict = {}
    agent._copy_reasoning_content_for_api(_tool_call_turn(), api_msg)
    assert api_msg["reasoning_content"] == "<think>plan the next tool call</think>"


def test_reasoning_content_stripped_when_provider_does_not_opt_in(monkeypatch) -> None:
    """No flag set — reasoning_content is dropped, like any other custom provider."""
    _stub_custom_providers(monkeypatch, [
        {"name": "local-llamacpp", "base_url": "http://localhost:8080/v1"},
    ])
    agent = _make_agent(base_url="http://localhost:8080/v1")
    api_msg: dict = {}
    agent._copy_reasoning_content_for_api(_tool_call_turn(), api_msg)
    assert "reasoning_content" not in api_msg


def test_reasoning_content_stripped_for_unmatched_base_url(monkeypatch) -> None:
    """The flag on a different provider entry must not leak onto this one."""
    _stub_custom_providers(monkeypatch, [
        {
            "name": "other",
            "base_url": "http://other.local:8080/v1",
            "needs_reasoning_content": True,
        },
    ])
    agent = _make_agent(base_url="http://localhost:8080/v1")
    api_msg: dict = {}
    agent._copy_reasoning_content_for_api(_tool_call_turn(), api_msg)
    assert "reasoning_content" not in api_msg
