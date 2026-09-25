"""G1 guard rail: config writes must never silently lose user-data keys.

Evidence: 2026-09-24, Nana's whole ``mcp_servers`` map was deleted by a
full-replace save (``mcp_servers`` is not in DEFAULT_CONFIG and leaf-path
preservation could not protect the node). Spec:
docs/superpowers/specs/2026-09-25-hermes-guard-rails-design.md §3.
"""
import logging

import pytest
import yaml

from hermes_cli.config import (
    DEFAULT_CONFIG,
    _explicit_config_paths,
    load_config,
    read_raw_config,
    save_config,
)


class TestExplicitConfigPathsDictNodes:
    def test_unknown_top_level_dict_records_node_and_leaves(self):
        raw = {"mcp_servers": {"github": {"command": "npx", "args": ["-y", "pkg"]}}}
        paths = _explicit_config_paths(raw)
        assert ("mcp_servers",) in paths
        assert ("mcp_servers", "github") in paths
        assert ("mcp_servers", "github", "command") in paths

    def test_known_defaulted_dict_records_leaves_only(self):
        # ``gateway`` exists in DEFAULT_CONFIG as a dict: the node must NOT be
        # preserved (that would keep caller-injected default-equal leaves too),
        # but unknown sub-dicts under it gain a node.
        assert isinstance(DEFAULT_CONFIG.get("gateway"), dict)
        raw = {"gateway": {"standalone": True, "extra": {"a": 1}}}
        paths = _explicit_config_paths(raw)
        assert ("gateway",) not in paths
        assert ("gateway", "standalone") in paths
        assert ("gateway", "extra") in paths

    def test_scalar_known_key_still_records_leaf(self):
        raw = {"proxy": {"label": "nana"}}
        paths = _explicit_config_paths(raw)
        assert ("proxy", "label") in paths
