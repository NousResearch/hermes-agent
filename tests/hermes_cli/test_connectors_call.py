"""Behaviour contracts for ``hermes connectors call``: what a cron gate script can rely on."""

from __future__ import annotations

import argparse
import json
from typing import Literal

import pytest

from hermes_cli.connectors_cmd import cmd_connectors
from hermes_cli.subcommands.connectors import build_connectors_parser
from tools.connectors.portal.wire import ConnectorTool, ConnectorToolsListing


def _parse(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    build_connectors_parser(parser.add_subparsers(dest="command"))
    return parser.parse_args(list(argv))


def _tool(slug: str, facet: Literal["read", "write"]) -> ConnectorTool:
    return ConnectorTool(slug=slug, name=slug, description="", facet=facet, hints=[], categories=[],
                         noAuth=False, deprecated=False)


@pytest.fixture
def gateway(monkeypatch):
    executed: list = []

    class FakePortal:
        def authorization_token(self):
            return None

        def origin(self):
            return "https://portal.test"

        def tools(self, slug, *, if_none_match=None):
            return ConnectorToolsListing(connector=slug, toolkitVersion="1", etag="e", tools=[
                _tool("SLACK_SEARCH_MESSAGES", "read"), _tool("SLACK_SEND_MESSAGE", "write")])

    class FakeClient:
        def schemas(self, tools):
            return {"schemas": {}}

        def execute(self, planned, **_):
            executed.extend(planned)
            return [{"data": {"messages": [{"ts": "1.0"}]}, "error": None, "account": "work"} for _ in planned]

    monkeypatch.setattr("tools.connectors.portal.client.PortalConnectorClient", FakePortal)
    monkeypatch.setattr("tools.connectors.gateway.bridge._default_client_factory", lambda call=None: FakeClient())
    monkeypatch.setattr("tools.connectors.gateway.bridge.connectors_available", lambda: True)
    monkeypatch.setattr("tools.connectors.connectors_available", lambda: True, raising=False)
    return executed


def test_read_tool_prints_one_json_document_and_passes_the_account(gateway, capsys):
    rc = cmd_connectors(_parse("connectors", "call", "slack.search_messages",
                               "--args", '{"query": "test"}', "--account", "work"))

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == {"messages": [{"ts": "1.0"}]}
    assert [(p.connector, p.tool, p.arguments) for p in gateway] == [
        ("slack", "SLACK_SEARCH_MESSAGES", {"query": "test", "connector_alias": "work"})]


def test_write_tool_without_allow_write_is_refused_before_the_gateway(gateway, capsys):
    rc = cmd_connectors(_parse("connectors", "call", "connectors__slack__send_message", "--args", '{"text": "hi"}'))

    assert rc == 5
    assert capsys.readouterr().out == ""
    assert gateway == []
