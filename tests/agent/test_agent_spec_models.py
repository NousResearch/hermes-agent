from agent.agent_spec_models import (
    MCP_VALIDATION_STATES,
    SANDBOX_ENFORCEMENT_STATUSES,
    SCHEMA_VERSION,
    is_valid_reasoning_effort,
    validation_status,
)


def test_exact_vocabularies_are_exported():
    assert MCP_VALIDATION_STATES == {
        "known_in_catalog_and_configured",
        "known_in_catalog_but_not_configured_optional",
        "known_in_catalog_but_required_missing",
        "unknown_server_id",
        "tool_discovery_unavailable",
        "tool_not_in_catalog_or_discovery",
    }
    assert SANDBOX_ENFORCEMENT_STATUSES == {
        "declared_only",
        "partially_enforced_by_backend",
        "enforced",
        "not_supported_on_backend",
    }
    assert SCHEMA_VERSION == "hermes.agent_spec/v1alpha1"


def test_reasoning_effort_vocabulary_includes_none():
    for effort in ["none", "minimal", "low", "medium", "high", "xhigh"]:
        assert is_valid_reasoning_effort(effort)
    assert not is_valid_reasoning_effort("galaxy-brain")


def test_validation_status_and_strict_warning_promotion():
    assert validation_status([], [], strict=False) == "pass"
    assert validation_status([], [object()], strict=False) == "warn"
    assert validation_status([], [object()], strict=True) == "fail"
    assert validation_status([object()], [], strict=False) == "fail"
