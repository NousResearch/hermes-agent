"""Expanded context costs must come from the gateway owner, not the CLI stand-in."""
from unittest.mock import Mock

import pytest

from hermes_cli.cli_info_mixin import CLIInfoMixin


@pytest.mark.parametrize("arg", ["", "all", "full", "details"])
def test_gateway_context_renders_requested_details(monkeypatch, capsys, arg):
    from hermes_cli import gateway_side

    details = {"skills": [], "toolsets": [
        {"toolset": "terminal", "tool_count": 1, "schema_tokens": 123}]}
    payload = {"categories": [{"id": "system", "label": "System prompt", "tokens": 100}],
               "context_max": 1000, "estimated_total": 100, "model": "owner-model"}
    connection = Mock()
    connection.call.side_effect = lambda method, **params: {
        **payload, **({"details": details} if params.get("include_details") else {})}
    monkeypatch.setattr(gateway_side, "connection_for", lambda cli: connection)
    cli = CLIInfoMixin.__new__(CLIInfoMixin)
    cli.session_id = "owner-session"
    cli.model = "stand-in-model"
    cli._show_context_breakdown("/context" + (" " + arg if arg else ""))
    out = capsys.readouterr().out
    assert "owner-model" in out and "System prompt" in out
    assert ("terminal" in out) == bool(arg)
    assert ("Use /context all" in out) == (not arg)
    expected: dict[str, object] = {"session_id": "owner-session"}
    if arg:
        expected["include_details"] = True
    connection.call.assert_called_once_with("session.context_breakdown", **expected)
