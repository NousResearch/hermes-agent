from types import SimpleNamespace

from agent.native_compaction import (
    native_compaction_context_management,
    resolve_native_compaction_capabilities,
)


def _agent(capabilities):
    return SimpleNamespace(
        model="gpt-5.6",
        base_url="https://proxy.example/v1",
        codex_responses_native_compaction=True,
        compression_enabled=True,
        codex_responses_compact_threshold=200_000,
        context_compressor=None,
        capabilities=capabilities,
    )


def test_trusted_destination_capabilities_are_explicitly_enabled():
    capabilities = resolve_native_compaction_capabilities(
        model="gpt-5.6",
        base_url="https://api.openai.com/v1",
    )

    assert capabilities == {"native_compaction": True}


def test_untrusted_destination_capabilities_are_explicitly_denied():
    capabilities = resolve_native_compaction_capabilities(
        model="gpt-5.6",
        base_url="https://openrouter.ai/api/v1",
    )

    assert capabilities == {"native_compaction": False}


def test_explicit_false_capability_denies_native_payload():
    agent = _agent({"native_compaction": False})

    assert native_compaction_context_management(agent, is_codex_backend=False) is None


def test_missing_capability_keeps_default_deny():
    agent = _agent({})

    assert native_compaction_context_management(agent, is_codex_backend=False) is None
