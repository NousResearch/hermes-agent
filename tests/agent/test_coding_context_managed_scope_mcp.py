"""The session toolset allowlist must see Managed Scope mcp_servers (#91073).

`_enabled_mcp_servers()` — the second bare raw-config reader beside the
discovery gate — feeds `toolset_selection()`/`coding_selection()`, which
`cli.py` and `tui_gateway/server.py` consume as the session toolset list.
Under ``agent.coding_context: focus`` a Managed-Scope-only server was absent
from that allowlist, so it was filtered back out of the session even once
discovery (fixed separately) started it. The function now reads the same
canonical effective config as the gate (``load_user_config_effective()``),
driven off real files here so the disk pipeline is what the tests pin.
"""

import textwrap

import pytest

from agent.coding_context import _enabled_mcp_servers


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


def test_managed_only_mcp_server_stays_in_toolset_allowlist(homes, monkeypatch):
    """The sibling #91073 shape: user config has no mcp_servers, the
    administrator publishes one (env-substituted, as managed configs do) —
    its name must survive into the allowlist."""
    monkeypatch.setenv("MCP_TEST_CMD", "fs-server")
    _write_managed(
        homes[1],
        """
        mcp_servers:
          managed-fs:
            command: ${MCP_TEST_CMD}
            args: ["--stdio"]
        """,
    )
    # No user config.yaml at all: the managed layer still applies.
    assert _enabled_mcp_servers(None) == ["managed-fs"]


def test_user_config_mcp_servers_still_listed(homes):
    """Regression: the pre-existing path (raw user config carries servers)
    keeps listing them; the managed dir exists but is empty."""
    _write_user(
        homes[0],
        """
        mcp_servers:
          user-server:
            command: x
        """,
    )
    assert _enabled_mcp_servers(None) == ["user-server"]


def test_both_scopes_listed_and_disabled_managed_server_excluded(homes):
    """Deep-merge contract at the allowlist level: servers in BOTH scopes
    resolve to the union (same per-leaf merge the connect path sees), while a
    managed server explicitly disabled stays excluded."""
    _write_managed(
        homes[1],
        """
        mcp_servers:
          managed-fs:
            command: fs-server
          managed-off:
            command: other-server
            enabled: false
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
    assert _enabled_mcp_servers(None) == ["user-server", "managed-fs"]


def test_no_servers_anywhere_keeps_allowlist_empty(homes):
    """Fail-closed on the decision itself: with no servers in either scope
    the allowlist stays empty so the session shape is unchanged."""
    _write_user(homes[0], "display: {}\n")
    assert _enabled_mcp_servers(None) == []
