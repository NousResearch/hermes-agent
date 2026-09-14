"""The session toolset allowlist must see Managed Scope mcp_servers (#91073).

`_enabled_mcp_servers()` — the second bare raw-config reader beside the
discovery gate — feeds `toolset_selection()`/`coding_selection()`, which
`cli.py` and `tui_gateway/server.py` consume as the session toolset list.
Under ``agent.coding_context: focus`` a Managed-Scope-only server was absent
from that allowlist, so it was filtered back out of the session even once
discovery (fixed separately) started it. The function now applies the same
`managed_scope.apply_managed_overlay()` the discovery gate applies.
"""

import textwrap

import pytest

from agent.coding_context import _enabled_mcp_servers


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


def test_managed_only_mcp_server_stays_in_toolset_allowlist(managed, monkeypatch):
    """The sibling #91073 shape: user config has no mcp_servers, the
    administrator publishes one (env-substituted, as managed configs do) —
    its name must survive into the allowlist."""
    monkeypatch.setenv("MCP_TEST_CMD", "fs-server")
    _write_managed(
        managed,
        """
        mcp_servers:
          managed-fs:
            command: ${MCP_TEST_CMD}
            args: ["--stdio"]
        """,
    )
    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "read_raw_config", lambda: {})
    assert _enabled_mcp_servers(None) == ["managed-fs"]


def test_user_config_mcp_servers_still_listed(managed, monkeypatch):
    """Regression: the pre-existing path (raw user config carries servers)
    keeps listing them; the managed dir exists but is empty."""
    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "read_raw_config",
        lambda: {"mcp_servers": {"user-server": {"command": "x"}}},
    )
    assert _enabled_mcp_servers(None) == ["user-server"]


def test_both_scopes_listed_and_disabled_managed_server_excluded(managed, monkeypatch):
    """Deep-merge contract at the allowlist level: servers in BOTH scopes
    resolve to the union (same per-leaf merge the connect path sees), while a
    managed server explicitly disabled stays excluded."""
    _write_managed(
        managed,
        """
        mcp_servers:
          managed-fs:
            command: fs-server
          managed-off:
            command: other-server
            enabled: false
        """,
    )
    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "read_raw_config",
        lambda: {"mcp_servers": {"user-server": {"command": "x"}}},
    )
    assert _enabled_mcp_servers(None) == ["user-server", "managed-fs"]


def test_no_servers_anywhere_keeps_allowlist_empty(managed, monkeypatch):
    """Fail-closed on the decision itself: with no servers in either scope
    the allowlist stays empty so the session shape is unchanged."""
    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "read_raw_config", lambda: {"display": {}})
    assert _enabled_mcp_servers(None) == []
