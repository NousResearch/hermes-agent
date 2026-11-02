"""Catalog consent policy reaches the installed MCP server configuration."""

from pathlib import Path

import pytest
import hermes_yaml as yaml

from hermes_cli.mcp_catalog import CatalogError, _build_server_config, _parse_manifest


def test_clixrx_catalog_requires_each_call():
    path = Path(__file__).parents[2] / "optional-mcps/clixrx/manifest.yaml"
    config = _build_server_config(_parse_manifest(path), None)
    assert config["trust"] == "untrusted"
    assert config.get("approval_policy") == "per_call"


@pytest.mark.parametrize("value", [None, False, "", "typo", [], {}])
def test_invalid_catalog_policy_is_rejected(tmp_path, value):
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump({
        "manifest_version": 2, "name": "demo", "description": "Demo",
        "transport": {"type": "http", "url": "https://example.invalid/mcp"},
        "auth": {"type": "none"}, "approval_policy": value,
    }), encoding="utf-8")
    with pytest.raises(CatalogError, match="approval_policy"):
        _parse_manifest(path)
