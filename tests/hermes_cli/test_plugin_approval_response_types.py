"""Only an explicit boolean True from the transport authorizes a plugin gate."""

import pytest

from hermes_cli import plugins
from tools import approval


@pytest.mark.parametrize(
    "response",
    [
        None,
        [],
        "yes",
        {},
        {"approved": False},
        {"approved": 0},
        {"approved": 1},
        {"approved": 1.0},
        {"approved": "false"},
        {"approved": "true"},
        {"approved": []},
        {"approved": True},
    ],
    ids=[
        "none",
        "list",
        "string",
        "missing",
        "deny",
        "integer-zero",
        "integer-one",
        "float-one",
        "string-false",
        "string-true",
        "list-value",
        "explicit-true",
    ],
)
def test_plugin_gate_rejects_non_boolean_approval(monkeypatch, response):
    monkeypatch.setattr(approval, "request_tool_approval", lambda *args, **kwargs: response)
    details = plugins._PreToolCallDirective(action="approve", message="test gate")
    block = plugins._resolve_block_from_details(details, "terminal")
    if isinstance(response, dict) and response.get("approved") is True:
        assert block is None
    else:
        assert isinstance(block, str) and block
