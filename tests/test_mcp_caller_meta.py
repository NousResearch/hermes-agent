from gateway.caller_context import reset_caller, set_caller
from tools.mcp_tool import _current_caller_payload


def test_mcp_caller_payload_empty_without_gateway_caller():
    assert _current_caller_payload() is None


def test_mcp_caller_payload_includes_gateway_caller():
    token = set_caller("slack", "U07TCQBDPMJ")
    try:
        assert _current_caller_payload() == {
            "provider": "slack",
            "external_id": "U07TCQBDPMJ",
        }
    finally:
        reset_caller(token)
