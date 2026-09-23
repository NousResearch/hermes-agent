"""Native Responses compaction on live-verified GPT-6 Sol/Luna Codex routes."""

from types import SimpleNamespace

import pytest

from agent.native_compaction import (
    native_compaction_context_management,
    resolve_native_compaction_capabilities,
)

_CODEX = "https://chatgpt.com/backend-api/codex"


@pytest.mark.parametrize("model,provider,base_url,eligible", [
    ("gpt-6-sol", "openai-codex", _CODEX, True),
    ("gpt-6-luna", "openai-codex", _CODEX, True),
    ("gpt-6-sol-900k", "openai-codex", _CODEX, True),
    ("gpt-6-luna-900k", "openai-codex", _CODEX, True),
    ("gpt-6-sol-pro", "openai-codex", _CODEX, False),
    ("gpt-6-terra", "openai-codex", _CODEX, False),
    ("gpt-6-sol", "openai", "https://api.openai.com/v1", False),
    ("gpt-6-luna", "openai-codex", "https://relay.example/v1", False),
    ("gpt-6-sol", "openai-codex", "https://chatgpt.com.example/backend-api/codex", False),
])
def test_codex_capability_and_request_agree(model, provider, base_url, eligible):
    resolved = resolve_native_compaction_capabilities(
        model=model, provider=provider, base_url=base_url,
        is_codex_backend=provider == "openai-codex",
    )
    assert resolved["native_compaction"] is eligible
    agent = SimpleNamespace(
        model=model, provider=provider, base_url=base_url,
        runtime_capabilities=resolved, codex_responses_native_compaction=True,
        compression_enabled=True, codex_responses_compact_threshold=700_000,
        context_compressor=SimpleNamespace(threshold_tokens=720_000),
        capabilities={},
    )
    payload = native_compaction_context_management(
        agent, is_codex_backend=provider == "openai-codex",
    )
    assert (payload is not None) is eligible
    if eligible:
        assert payload == [{"type": "compaction", "compact_threshold": 700_000}]
