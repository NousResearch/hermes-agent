"""The MCP discovery gate must see Managed Scope mcp_servers (#91073).

`_has_configured_mcp_servers()` read only the raw user config, so
`mcp_servers` published by an administrator via Managed Scope
(`/etc/hermes/config.yaml`) — present in the effective config and honored at
connect time — never started discovery: zero MCP servers on every surface,
no warning. The gate now reads through the canonical effective loader
(``load_user_config_effective()``: user file, ``${VAR}`` expansion, managed
overlay, no defaults — hermes_cli/AGENTS.md), so the tests drive the real
loader off disk (``HERMES_HOME`` + ``HERMES_MANAGED_DIR``) instead of
monkeypatching the config readers.
"""

import textwrap

import pytest

from hermes_cli.mcp_startup import _has_configured_mcp_servers


@pytest.fixture
def homes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    md = tmp_path / "managed"
    md.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(md))
    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()
    return home, md


def _write_user(home, body):
    (home / "config.yaml").write_text(textwrap.dedent(body), encoding="utf-8")


def _write_managed(md, body):
    (md / "config.yaml").write_text(textwrap.dedent(body), encoding="utf-8")
    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()


def test_managed_scope_mcp_servers_enable_discovery(homes, monkeypatch):
    """The #91073 shape: user config has no mcp_servers, the administrator
    publishes one (env-substituted, as managed configs do) — the gate must
    answer True through the real loader pipeline."""
    monkeypatch.setenv("MCP_GATE_CMD", "fs-server")
    _write_managed(
        homes[1],
        """
        mcp_servers:
          managed-fs:
            command: ${MCP_GATE_CMD}
            args: ["--stdio"]
        """,
    )
    # No user config.yaml at all: the managed layer still applies.
    assert _has_configured_mcp_servers() is True


def test_user_config_mcp_servers_still_enable_discovery(homes):
    """Regression: the pre-existing path (raw user config carries servers)
    keeps answering True; the managed dir exists but is empty."""
    _write_user(
        homes[0],
        """
        mcp_servers:
          user-server:
            command: x
        """,
    )
    assert _has_configured_mcp_servers() is True


def test_no_servers_anywhere_keeps_gate_closed(homes):
    """Fail-closed on the decision itself: with no servers in either scope
    the gate stays False so non-MCP users still skip the MCP stack import."""
    _write_user(homes[0], "display: {}\n")
    assert _has_configured_mcp_servers() is False


def test_gate_and_connect_share_the_deep_merge_contract(homes):
    """Pin the merge contract the gate relies on (review on #91073):
    ``apply_managed_overlay`` deep-merges ``mcp_servers`` per server name,
    so a config carrying servers in BOTH scopes resolves to the UNION —
    exactly what the connect path (which runs the same overlay before
    building its server set) sees. If the overlay ever flipped to a
    shallow replace, the managed server below would vanish and this test
    fails, turning the shared-helper assumption into an enforced
    invariant instead of a trust."""
    _write_managed(
        homes[1],
        """
        mcp_servers:
          managed-fs:
            command: fs-server
            args: ["--stdio"]
        """,
    )
    _write_user(
        homes[0],
        """
        mcp_servers:
          user-server:
            command: x
        """,
    )
    assert _has_configured_mcp_servers() is True

    from hermes_cli import managed_scope

    merged = managed_scope.apply_managed_overlay(
        {"mcp_servers": {"user-server": {"command": "x"}}}
    )
    assert set(merged["mcp_servers"]) == {"user-server", "managed-fs"}


def test_torn_user_config_keeps_user_declared_servers(homes):
    """The increment the canonical loader adds over raw-read + overlay
    (review on #91076): ``read_raw_config()`` serves ``{}`` on a torn
    user config, which dropped user-declared servers from the gate;
    ``load_user_config_effective`` replays the last-known-good user layer
    instead, so the gate stays open."""
    _write_user(
        homes[0],
        """
        mcp_servers:
          user-server:
            command: x
        """,
    )
    assert _has_configured_mcp_servers() is True  # seeds the parse cache
    (homes[0] / "config.yaml").write_text("mcp_servers: [torn", encoding="utf-8")
    assert _has_configured_mcp_servers() is True
