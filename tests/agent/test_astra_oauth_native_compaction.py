"""Astra native management requires both the supported model and OAuth destination."""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent.native_compaction import (
    native_compaction_context_management,
    resolve_native_compaction_capabilities,
)
from run_agent import AIAgent


@pytest.mark.parametrize("model,provider,base_url,eligible", [
    ("gpt-6-astra", "openai-codex", "https://chatgpt.com/backend-api/codex", True),
    ("GPT-6-ASTRA", "openai-codex", "https://chatgpt.com:443/backend-api/codex/", True),
    ("gpt-6-astra", "openai", "https://api.openai.com/v1", False),
    ("gpt-6-astra", "openai", "", False),
    ("gpt-6-astra", "openai", "https://chatgpt.com/backend-api/codex", False),
    ("gpt-6-astra", "openai-codex", "https://relay.example/v1", False),
    ("gpt-6-astra", "openai-codex", "http://localhost:8080/v1", False),
    ("gpt-6-astra", "openai-codex", "https://chatgpt.com.example/backend-api/codex", False),
    ("gpt-6-astra", "openai-codex", "http://chatgpt.com/backend-api/codex", False),
    ("gpt-6-astra", "openai-codex", "https://chatgpt.com:8443/backend-api/codex", False),
    ("gpt-6-astra", "openai-codex", "https://chatgpt.com/backend-api/codex-other", False),
    ("gpt-6-astra", "openai-codex", "https://chatgpt.com:invalid/backend-api/codex", False),
    ("gpt-6-astra", "openai-codex", None, False),
    ("gpt-6-astra-mini", "openai-codex", "https://chatgpt.com/backend-api/codex", False),
    ("gpt-6-other", "openai-codex", "https://chatgpt.com/backend-api/codex", False),
    ("gpt-5.6-sol", "openai", "https://api.openai.com/v1", True),
    ("gpt-5.6-sol", "openai-codex", "https://chatgpt.com/backend-api/codex", True),
])
def test_destination_capability_and_request_gate_agree(model, provider, base_url, eligible):
    is_codex = provider == "openai-codex"
    resolved = resolve_native_compaction_capabilities(
        model=model, provider=provider, base_url=base_url, is_codex_backend=is_codex,
    )
    assert resolved["native_compaction"] is eligible
    agent = SimpleNamespace(
        model=model, provider=provider, base_url=base_url,
        codex_responses_native_compaction=True, compression_enabled=True,
        capabilities={"openai_native_compaction": True},
    )
    # The request gate must also exclude Astra relays before runtime resolution,
    # even when a proxy advertises support for native compaction.
    for runtime in (None, resolved):
        agent.runtime_capabilities = runtime
        payload = native_compaction_context_management(agent, is_codex_backend=is_codex)
        assert (payload is not None) is eligible


@pytest.mark.parametrize("setting,value,enabled", [
    (None, None, True),
    ("codex_responses_native_compaction", False, False),
    ("compression_enabled", False, False),
    ("compression_checkpoint_required", True, False),
    ("runtime_capabilities", {"native_compaction": False}, False),
])
def test_configured_agent_preserves_request_safety_gates(monkeypatch, setting, value, enabled):
    Path(os.environ["HERMES_HOME"], "config.yaml").write_text(
        "compression:\n  codex_responses_native: true\n"
        "  codex_responses_compact_threshold: 999999\n  threshold_tokens: 204000\n"
        "auxiliary:\n  title_generation:\n    enabled: false\n"
    )
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kwargs: [])
    agent: Any = AIAgent(
        model="gpt-6-astra", provider="openai-codex", api_mode="codex_responses",
        base_url="https://chatgpt.com/backend-api/codex", api_key="test-key",
        quiet_mode=True, skip_context_files=True, skip_memory=True,
        enabled_toolsets=[], save_trajectories=False,
    )
    try:
        assert agent.runtime_capabilities["native_compaction"] is True
        if setting:
            setattr(agent, setting, value)
        request = agent._build_api_kwargs([{"role": "user", "content": "Continue."}])
        assert ("context_management" in request) is enabled
        if enabled:
            management = request["context_management"][0]
            assert management["type"] == "compaction"
            assert 1024 <= management["compact_threshold"] < agent.context_compressor.threshold_tokens
    finally:
        agent.close()
