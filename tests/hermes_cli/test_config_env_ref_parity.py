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


def test_bare_ref_still_expands(monkeypatch):
    monkeypatch.setenv("PARITY_VAR", "val-bare")
    assert _expand_env_vars("x-${PARITY_VAR}-y") == "x-val-bare-y"


def test_env_prefixed_ref_expands(monkeypatch):
    monkeypatch.setenv("PARITY_VAR", "val-prefixed")
    assert _expand_env_vars("${env:PARITY_VAR}") == "val-prefixed"


def test_env_prefixed_ref_unset_stays_verbatim(monkeypatch):
    monkeypatch.delenv("PARITY_MISSING", raising=False)
    assert _expand_env_vars("${env:PARITY_MISSING}") == "${env:PARITY_MISSING}"


def test_empty_env_ref_stays_verbatim():
    assert _expand_env_vars("${env:}") == "${env:}"


def test_non_env_source_stays_verbatim_with_warning(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
        out = _expand_env_vars("${bitwarden:MY_KEY}")
    assert out == "${bitwarden:MY_KEY}"
    assert any("env:NAME" in r.message for r in caplog.records)


def test_nested_structures_expand(monkeypatch):
    monkeypatch.setenv("PARITY_VAR", "v")
    cfg = {"a": ["${env:PARITY_VAR}", {"b": "${PARITY_VAR}"}], "n": 3}
    out = _expand_env_vars(cfg)
    assert out == {"a": ["v", {"b": "v"}], "n": 3}


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


@pytest.mark.parametrize("ref,expected", [
    ("PLAIN_VAR", "PLAIN_VAR"),
    ("env:PLAIN_VAR", "PLAIN_VAR"),
    ("env: SPACED ", "SPACED"),
    ("env:", None),
    ("bitwarden:KEY", None),
    ("vault:path/to/key", None),
])
def test_env_ref_var_name(ref, expected):
    assert _env_ref_var_name(ref) == expected


def test_snapshot_tracks_env_prefixed_under_real_name(monkeypatch):
    monkeypatch.setenv("PARITY_SNAP", "s1")
    snap = _env_ref_snapshot({"k": "${env:PARITY_SNAP}"})
    assert snap == {"PARITY_SNAP": "s1"}


def test_snapshot_excludes_non_env_sources(monkeypatch):
    snap = _env_ref_snapshot({"k": "${bitwarden:KEY}", "j": "${PARITY_SNAP2}"})
    assert "bitwarden:KEY" not in snap
    assert "KEY" not in snap
    assert "PARITY_SNAP2" in snap


def test_snapshot_detects_rotation_for_env_prefixed(monkeypatch):
    """The #58514 cache-invalidation contract must hold for ${env:VAR} refs:
    the snapshot records the value under the REAL var name, so a rotation
    changes the snapshot."""
    monkeypatch.setenv("PARITY_ROT", "before")
    snap1 = _env_ref_snapshot({"k": "${env:PARITY_ROT}"})
    monkeypatch.setenv("PARITY_ROT", "after")
    snap2 = _env_ref_snapshot({"k": "${env:PARITY_ROT}"})
    assert snap1 != snap2
