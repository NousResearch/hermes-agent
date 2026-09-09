"""Oneshot -t must admit portable plugin MCP servers (not only config.yaml mcp_servers)."""

from unittest.mock import patch

from hermes_cli.oneshot import (
    _configured_mcp_servers,
    _resolve_mcp_toolset_name,
    _validate_explicit_toolsets,
)


PORTABLE = "agent-plugin-rivetflo-csr-runtime-11e814b1__living-runtime"


def test_resolve_mcp_toolset_name_exact_and_suffix():
    names = {PORTABLE, "rivetflo"}
    assert _resolve_mcp_toolset_name(PORTABLE, names) == PORTABLE
    assert _resolve_mcp_toolset_name("living-runtime", names) == PORTABLE
    assert _resolve_mcp_toolset_name("rivetflo", names) == "rivetflo"
    assert _resolve_mcp_toolset_name("nope", names) is None


def test_configured_mcp_servers_unions_portable(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.oneshot._portable_mcp_server_names",
        lambda: {PORTABLE},
    )
    with patch("hermes_cli.config.read_raw_config", return_value={"mcp_servers": {"rivetflo": {}}}):
        enabled, disabled = _configured_mcp_servers()
    assert PORTABLE in enabled
    assert "rivetflo" in enabled
    assert not disabled


def test_validate_explicit_toolsets_accepts_portable_and_short_alias():
    with patch("hermes_cli.oneshot._configured_mcp_servers", return_value=({PORTABLE}, set())):
        with patch("toolsets.validate_toolset", return_value=False):
            with patch("hermes_cli.plugins.discover_plugins"):
                valid, err = _validate_explicit_toolsets(PORTABLE)
                assert err is None
                assert valid == [PORTABLE]
                valid_short, err_short = _validate_explicit_toolsets("living-runtime")
                assert err_short is None
                assert valid_short == [PORTABLE]


def test_validate_explicit_toolsets_unknown_still_errors():
    with patch("hermes_cli.oneshot._configured_mcp_servers", return_value=(set(), set())):
        with patch("toolsets.validate_toolset", return_value=False):
            with patch("hermes_cli.plugins.discover_plugins"):
                valid, err = _validate_explicit_toolsets("living-runtime")
                assert valid is None
                assert err and "did not contain any valid toolsets" in err
