"""The MCP discovery gate must see Managed Scope mcp_servers (#91073).

`_has_configured_mcp_servers()` read only the raw user config, so
`mcp_servers` published by an administrator via Managed Scope
(`/etc/hermes/config.yaml`) — present in the effective config and honored at
connect time — never started discovery: zero MCP servers on every surface,
no warning. The gate now applies `managed_scope.apply_managed_overlay()`,
the shared helper every other self-built config reader already uses.
"""

import textwrap

import pytest

from hermes_cli.mcp_startup import _has_configured_mcp_servers


@pytest.fixture
def managed(tmp_path, monkeypatch):
    md = tmp_path / "managed"
    md.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(md))
    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()
    return md


def _write_managed(md, body):
    (md / "config.yaml").write_text(textwrap.dedent(body), encoding="utf-8")
    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()


def test_managed_scope_mcp_servers_enable_discovery(managed, monkeypatch):
    """The #91073 shape: user config has no mcp_servers, the administrator
    publishes one via Managed Scope — the gate must answer True."""
    _write_managed(
        managed,
        """
        mcp_servers:
          managed-fs:
            command: fs-server
            args: ["--stdio"]
        """,
    )
    # User config stays empty of MCP servers.
    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "read_raw_config", lambda: {})
    assert _has_configured_mcp_servers() is True


def test_user_config_mcp_servers_still_enable_discovery(managed, monkeypatch):
    """Regression: the pre-existing path (raw user config carries servers)
    keeps answering True; the managed dir exists but is empty."""
    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "read_raw_config",
        lambda: {"mcp_servers": {"user-server": {"command": "x"}}},
    )
    assert _has_configured_mcp_servers() is True


def test_no_servers_anywhere_keeps_gate_closed(managed, monkeypatch):
    """Fail-closed on the decision itself: with no servers in either scope
    the gate stays False so non-MCP users still skip the MCP stack import."""
    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "read_raw_config", lambda: {"display": {}})
    assert _has_configured_mcp_servers() is False


def test_gate_and_connect_share_the_deep_merge_contract(managed, monkeypatch):
    """Pin the merge contract the gate relies on (review on #91073):
    ``apply_managed_overlay`` deep-merges ``mcp_servers`` per server name,
    so a config carrying servers in BOTH scopes resolves to the UNION —
    exactly what the connect path (which runs the same overlay before
    building its server set) sees. If the overlay ever flipped to a
    shallow replace, the managed server below would vanish and this test
    fails, turning the shared-helper assumption into an enforced
    invariant instead of a trust."""
    _write_managed(
        managed,
        """
        mcp_servers:
          managed-fs:
            command: fs-server
            args: ["--stdio"]
        """,
    )
    import hermes_cli.config as config_mod
    from hermes_cli import managed_scope

    monkeypatch.setattr(
        config_mod,
        "read_raw_config",
        lambda: {"mcp_servers": {"user-server": {"command": "x"}}},
    )
    assert _has_configured_mcp_servers() is True

    merged = managed_scope.apply_managed_overlay(
        {"mcp_servers": {"user-server": {"command": "x"}}}
    )
    assert set(merged["mcp_servers"]) == {"user-server", "managed-fs"}
