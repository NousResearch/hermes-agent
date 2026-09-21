"""Exact gpt-6-astra is native-compaction eligible only on official Codex OAuth (#103720).

Both the destination capability (``resolve_native_compaction_capabilities``) and the
per-request gate (``native_compaction_context_management``) must agree, and the request
gate must exclude Astra relays even when a trusted proxy advertises native compaction.
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
def test_astra_capability_and_request_gate_agree(model, provider, base_url, eligible):
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
