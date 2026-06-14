from hermes_os_integration.mcp_bridge import grant_tool
from hermes_os_integration.memory_boundary import classify_memory_write


def test_authoritative_memory_mutation_rejected():
    record, error = classify_memory_write("projects", "overwrite")

    assert record is None
    assert error.code == "state_conflict"


def test_runtime_cache_memory_allowed():
    record, error = classify_memory_write("projects", "cache")

    assert error is None
    assert record["authoritative"] is False


def test_mcp_permission_denied_by_default():
    grant, error = grant_tool({"allowed_tools": ["docs"]}, "filesystem")

    assert grant is None
    assert error.code == "permission_denied"


def test_mcp_permission_allowed_when_delegated():
    grant, error = grant_tool({"allowed_tools": ["filesystem"]}, "filesystem")

    assert error is None
    assert grant["permission_authority"] == "Hermes OS"
