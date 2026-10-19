"""Config `${env:VAR}` SecretRef parity (salvaged from PR #59516).

`${env:VAR}` already resolved in MCP server config (mcp_tool._env_ref_name);
config.yaml's expander treated it as a literal.  These tests pin the parity
plus the cache-snapshot tracking and the non-env-source warning behavior.
"""
from __future__ import annotations

import pytest

from hermes_cli.config import (
    _env_ref_snapshot,
    _env_ref_var_name,
    _expand_env_vars,
)








def test_value_containing_colon_is_not_a_source_ref(monkeypatch):
    """URL-ish or uppercase-colon refs are legacy bare names, not sources —
    only a lowercase ident prefix counts as a SecretRef source."""
    monkeypatch.delenv("MY:WEIRD", raising=False)
    # Uppercase before ':' → treated as a bare (unset) var, kept verbatim,
    # no misleading source warning.
    assert _expand_env_vars("${MY:WEIRD}") == "${MY:WEIRD}"


# ---------------------------------------------------------------------------
# _env_ref_var_name + snapshot tracking
# ---------------------------------------------------------------------------








def test_snapshot_detects_rotation_for_env_prefixed(monkeypatch):
    """The #58514 cache-invalidation contract must hold for ${env:VAR} refs:
    the snapshot records the value under the REAL var name, so a rotation
    changes the snapshot."""
    monkeypatch.setenv("PARITY_ROT", "before")
    snap1 = _env_ref_snapshot({"k": "${env:PARITY_ROT}"})
    monkeypatch.setenv("PARITY_ROT", "after")
    snap2 = _env_ref_snapshot({"k": "${env:PARITY_ROT}"})
    assert snap1 != snap2
