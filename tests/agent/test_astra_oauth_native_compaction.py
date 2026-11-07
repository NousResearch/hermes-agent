"""Astra (including its ``-900k`` picker alias) and exact gpt-6.1-sol require official
Codex OAuth for native compaction (#103720).

The destination capability and per-request gate must agree; trusted proxy capabilities
must not make these models eligible on relays.
"""

from types import SimpleNamespace

import pytest

from agent.native_compaction import (
    native_compaction_context_management,
    resolve_native_compaction_capabilities,
)

_CODEX = "https://chatgpt.com/backend-api/codex"


@pytest.mark.parametrize("model,provider,base_url,eligible", [
    ("gpt-6-astra", "openai-codex", _CODEX, True),
    ("GPT-6-ASTRA", "openai-codex", "https://chatgpt.com:443/backend-api/codex/", True),
    ("gpt-6-astra-900k", "openai-codex", _CODEX, True),
    ("openai/gpt-6-astra-900k", "openai-codex", _CODEX, True),
    ("gpt-6.1-sol", "openai-codex", _CODEX, True),
    ("GPT-6.1-SOL", "openai-codex", "https://chatgpt.com:443/backend-api/codex/", True),
    ("gpt-6.1-sol", "openai", "https://api.openai.com/v1", False),
    ("gpt-6.1-sol", "openai", _CODEX, False),
    ("gpt-6.1-sol", "openai-codex", "https://relay.example/v1", False),
    ("gpt-6.1-sol", "openai-codex", "https://chatgpt.com.example/backend-api/codex", False),
    ("gpt-6.1-sol", "openai-codex", "http://chatgpt.com/backend-api/codex", False),
    ("gpt-6.1-sol", "openai-codex", None, False),
    ("gpt-6.1-sol-900k", "openai-codex", _CODEX, False),
    ("gpt-6.1-sol-mini", "openai-codex", _CODEX, False),
    ("openai/gpt-6.1-sol", "openai-codex", _CODEX, False),
    ("gpt-6-astra-900k", "openai-codex", "https://relay.example/v1", False),
    ("gpt-6-astra-900k", "openai", "https://api.openai.com/v1", False),
    ("gpt-6-astra", "openai", "https://api.openai.com/v1", False),
    ("gpt-6-astra", "openai", _CODEX, False),
    ("gpt-6-astra", "openai-codex", "https://relay.example/v1", False),
    ("gpt-6-astra", "openai-codex", "https://chatgpt.com.example/backend-api/codex", False),
    ("gpt-6-astra", "openai-codex", "http://chatgpt.com/backend-api/codex", False),
    ("gpt-6-astra", "openai-codex", None, False),
    ("gpt-6-astra-mini", "openai-codex", _CODEX, False),
    ("gpt-6-other", "openai-codex", _CODEX, False),
    ("gpt-5.6", "openai", "https://api.openai.com/v1", True),
    ("gpt-5.6", "openai-codex", _CODEX, True),
])
def test_capability_and_request_gate_agree(model, provider, base_url, eligible):
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
    for runtime in (None, resolved):
        agent.runtime_capabilities = runtime
        payload = native_compaction_context_management(agent, is_codex_backend=is_codex)
        assert (payload is not None) is eligible


@pytest.mark.parametrize("eligible", [True, False])
def test_sol_request_and_checkpoint_replay_follow_current_gate(eligible):
    from run_agent import AIAgent

    agent = AIAgent(
        model="gpt-6.1-sol", provider="openai-codex", base_url=_CODEX,
        api_mode="codex_responses", api_key="test-key", quiet_mode=True,
        skip_context_files=True, skip_memory=True, enabled_toolsets=[],
    )
    agent.codex_responses_native_compaction = eligible
    checkpoint = {
        "type": "compaction", "encrypted_content": "opaque-sol-checkpoint",
        "_issuer_kind": "codex_backend", "_issuer_model": "gpt-6.1-sol",
    }
    history = [
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "compact this"},
        {"role": "assistant", "content": "OK", "codex_reasoning_items": [checkpoint]},
        {"role": "user", "content": "next"},
    ]
    kwargs = agent._build_api_kwargs(history)
    assert ("context_management" in kwargs) is eligible
    replayed = [item for item in kwargs["input"] if item.get("type") == "compaction"]
    assert replayed == ([{"type": "compaction", "encrypted_content": "opaque-sol-checkpoint"}] if eligible else [])
    old_answer = {"role": "assistant", "content": [{"type": "output_text", "text": "old answer"}]}
    assert (old_answer in kwargs["input"]) is not eligible
