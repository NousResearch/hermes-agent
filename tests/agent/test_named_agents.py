import pytest

from agent.archetypes import (
    ARCHETYPES_BY_NAME,
    NamedAgentContract,
    normalize_named_agent_contract,
    normalize_named_agent_registry,
    resolve_specialist_mapping,
)
from hermes_cli.config import DEFAULT_OMO_AGENTS


def test_normalize_named_agent_contract_applies_canonical_schema_defaults():
    contract = normalize_named_agent_contract(
        "oracle",
        {
            "archetype": "researcher",
            "specialist": "consultant",
            "mode": "subagent",
            "provider": "openai",
            "model": "gpt-5.4",
            "provider_options": {"reasoningEffort": "high"},
            "fallback_models": ["openai/gpt-4.1-mini"],
            "permission": {
                "edit": "deny",
                "bash": "deny",
                "webfetch": "ask",
                "doom_loop": "deny",
                "external_directory": "deny",
            },
            "description": "Evidence-only consultant.",
            "ultrawork": {"model": "openai/o3", "variant": "safe"},
        },
    )

    assert isinstance(contract, NamedAgentContract)
    assert contract.name == "oracle"
    assert contract.role == "researcher"
    assert contract.archetype == "researcher"
    assert contract.specialist == "consultant"
    assert contract.mode == "subagent-only"
    assert contract.route_category == "deep"
    assert contract.category == "deep"
    assert contract.provider == "openai"
    assert contract.model == "gpt-5.4"
    assert contract.providerOptions == {"reasoningEffort": "high"}
    assert contract.fallback_models == ({"model": "openai/gpt-4.1-mini"},)
    assert contract.ultrawork == {"model": "openai/o3", "variant": "safe"}
    assert contract.permissions == {
        "edit": "deny",
        "bash": "deny",
        "webfetch": "ask",
        "doom_loop": "deny",
        "external_directory": "deny",
    }
    assert contract.description == "Evidence-only consultant."
    assert "oracle" in contract.safe_claim_text
    assert "consultant" in contract.safe_claim_text
    assert "researcher" in contract.safe_claim_text


def test_normalize_named_agent_registry_converges_default_omo_agents():
    registry = normalize_named_agent_registry(DEFAULT_OMO_AGENTS)

    assert set(registry) == set(DEFAULT_OMO_AGENTS)

    for name, contract in registry.items():
        assert isinstance(contract, NamedAgentContract)
        assert contract.archetype in ARCHETYPES_BY_NAME
        assert contract.role == contract.archetype
        if contract.specialist is not None:
            assert resolve_specialist_mapping(contract.specialist) is not None

    assert registry["oracle"].blocked_tools
    assert registry["librarian"].blocked_tools
    assert registry["explore"].blocked_tools
    assert "terminal" in registry["explore"].blocked_tools
    assert registry["momus"].blocked_tools
    assert registry["multimodal-looker"].allowed_tools
    assert "delegate_task" in registry["atlas"].blocked_tools
    assert "task" in registry["atlas"].blocked_tools
    assert registry["multimodal-looker"].specialist == "multimodal_specialist"
    assert registry["multimodal-looker"].archetype == "generalist"
    assert registry["multimodal-looker"].route_category == "visual"
    assert registry["multimodal-looker"].specialist != registry["multimodal-looker"].archetype


@pytest.mark.parametrize(
    ("name", "entry", "message_parts"),
    [
        (
            "bad-mode",
            {"archetype": "generalist", "mode": "parent"},
            ["bad-mode", "mode", "primary | subagent-only | disabled"],
        ),
        (
            "bad-specialist",
            {"archetype": "generalist", "specialist": "unknown_specialist"},
            ["bad-specialist", "specialist", "unknown_specialist"],
        ),
        (
            "bad-permission",
            {"archetype": "generalist", "permission": {"edit": "maybe"}},
            ["bad-permission", "permission.edit", "allow", "ask", "deny"],
        ),
        (
            "bad-provider-options",
            {"archetype": "generalist", "provider_options": "fast"},
            ["bad-provider-options", "providerOptions", "mapping"],
        ),
    ],
)
def test_normalize_named_agent_contract_rejects_invalid_fields_readably(name, entry, message_parts):
    with pytest.raises(ValueError) as excinfo:
        normalize_named_agent_contract(name, entry)

    message = str(excinfo.value)
    for part in message_parts:
        assert part in message
