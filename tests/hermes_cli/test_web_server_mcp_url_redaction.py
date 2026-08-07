"""Regression tests for MCP URL redaction in dashboard JSON responses.

Covers #61562: saved MCP server URLs that carry secret query parameters must be
masked when serialized for the dashboard, without mutating the on-disk config.
"""

import pytest


def test_mcp_server_summary_redacts_secret_query_params():
    from hermes_cli.web_server import _mcp_server_summary

    cfg = {
        "url": (
            "http://windmill.example.com/api/mcp?token=secret&api_key=ak"
            "&key=k&access_token=at&foo=bar"
        ),
    }
    summary = _mcp_server_summary("windmill", cfg)

    expected = (
        "http://windmill.example.com/api/mcp?token=<redacted>&api_key=<redacted>"
        "&key=<redacted>&access_token=<redacted>&foo=bar"
    )
    assert summary["url"] == expected


def test_mcp_server_summary_leaves_stored_url_unchanged():
    from hermes_cli.web_server import _mcp_server_summary

    url = "http://windmill.example.com/api/mcp?token=secret"
    cfg = {"url": url}
    _mcp_server_summary("windmill", cfg)

    assert cfg["url"] == url


def test_mcp_server_summary_handles_url_without_query():
    from hermes_cli.web_server import _mcp_server_summary

    cfg = {"url": "http://windmill.example.com/api/mcp"}
    summary = _mcp_server_summary("windmill", cfg)
    assert summary["url"] == "http://windmill.example.com/api/mcp"


def test_mcp_server_summary_handles_none_url():
    from hermes_cli.web_server import _mcp_server_summary

    cfg = {"command": "npx"}
    summary = _mcp_server_summary("local", cfg)
    assert summary["url"] is None
