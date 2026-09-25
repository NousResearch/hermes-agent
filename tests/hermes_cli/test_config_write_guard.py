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


SEED_CONFIG = {
    "mcp_servers": {
        "github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"], "enabled": True},
        "chrome-devtools": {"command": "npx", "args": ["-y", "chrome-devtools-mcp"], "enabled": True},
    },
    "custom_providers": {"ark": {"base_url": "https://ark.example"}},
}


@pytest.fixture
def seeded_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME whose config.yaml carries user-data keys."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(SEED_CONFIG), encoding="utf-8")
    return tmp_path


class TestSaveConfigPreservesUserData:
    def test_omitted_mcp_servers_is_represerved_with_warning(self, seeded_home, caplog):
        incoming = {"custom_providers": {"ark": {"base_url": "https://ark.example"}}}
        with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
            save_config(incoming)
        raw = read_raw_config()
        assert set(raw["mcp_servers"]) == set(SEED_CONFIG["mcp_servers"])
        assert any("mcp_servers" in record.message for record in caplog.records)

    def test_removed_keys_makes_removal_explicit_and_silent(self, seeded_home, caplog):
        incoming = {"custom_providers": {"ark": {"base_url": "https://ark.example"}}}
        with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
            save_config(incoming, removed_keys={"mcp_servers"})
        raw = read_raw_config()
        assert "mcp_servers" not in raw
        assert not [r for r in caplog.records if "mcp_servers" in r.message]

    def test_default_config_key_omission_still_strips(self, seeded_home):
        # Spec §3.2: keys IN DEFAULT_CONFIG keep today's strip-defaults behavior.
        assert "model" in DEFAULT_CONFIG
        save_config({**SEED_CONFIG, "model": {"default": "gpt-x", "provider": "openai"}})
        save_config({"mcp_servers": SEED_CONFIG["mcp_servers"],
                     "custom_providers": SEED_CONFIG["custom_providers"]})
        raw = read_raw_config()
        assert "model" not in raw

    def test_caller_dict_is_not_mutated(self, seeded_home):
        incoming = {"custom_providers": {"ark": {"base_url": "https://ark.example"}}}
        snapshot = {k: dict(v) for k, v in incoming.items()}
        save_config(incoming)
        assert incoming == snapshot
