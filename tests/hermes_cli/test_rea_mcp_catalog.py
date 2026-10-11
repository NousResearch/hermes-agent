"""REA's catalog candidate must use the existing MCP approval/install path."""
from pathlib import Path

from hermes_cli import mcp_catalog
from tools.connectors.mcp import validate_mcp_names

ROOT = Path(__file__).resolve().parents[2]


def test_rea_is_installable_through_existing_catalog(monkeypatch):
    monkeypatch.setenv("HERMES_OPTIONAL_MCPS", str(ROOT / "optional-mcps"))
    entry = mcp_catalog.get_entry("rea")
    assert entry is not None, "REA must be discoverable in the bundled catalog"
    assert validate_mcp_names("install", ["rea"]) is None
    assert entry.transport.version == "6.3.0"
    assert entry.transport.command == "${REA_NODE}"
    assert entry.transport.args == ["${REA_SCRIPT}", "mcp"]
    assert entry.install is None  # Never bootstrap/upstream setup or register outside Hermes.
    assert entry.auth.type == "none"
    assert all(not spec.secret for spec in entry.auth.env)
    assert {s.name for s in entry.auth.env} == {
        "REA_NODE", "REA_SCRIPT", "REA_WORKSPACE", "REA_TMPDIR",
        "GHIDRA_INSTALL_DIR", "JAVA_HOME",
    }
    assert entry.tools.default_enabled is None  # Full user-approved REA surface.
    assert entry.tools.default_excluded is None


def test_catalog_sampling_opt_out_reaches_both_install_paths(tmp_path):
    import hermes_yaml as yaml
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({
        "manifest_version": 1, "name": "sampling-demo", "description": "Sampling opt-out",
        "transport": {"type": "stdio", "command": "demo"},
        "sampling": {"enabled": False},
    }))
    entry = mcp_catalog._parse_manifest(manifest)
    assert mcp_catalog._build_server_config(entry, None)["sampling"] == {"enabled": False}
    assert mcp_catalog.card_install_config(entry)["sampling"] == {"enabled": False}


def test_rea_card_config_inlines_only_declared_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_OPTIONAL_MCPS", str(ROOT / "optional-mcps"))
    entry = mcp_catalog.get_entry("rea")
    assert entry is not None
    cfg = mcp_catalog.card_install_config(entry)
    for spec in entry.auth.env:
        cfg = mcp_catalog._inline_non_secret_value(cfg, spec.name, str(tmp_path / spec.name))
    assert cfg["command"] == str(tmp_path / "REA_NODE")
    assert cfg["args"] == [str(tmp_path / "REA_SCRIPT"), "mcp"]
    assert cfg["sampling"] == {"enabled": False}
    assert cfg["env"]["HOME"] == str(tmp_path / "REA_WORKSPACE")
    assert cfg["env"]["JAVA_HOME"] == str(tmp_path / "JAVA_HOME")
    assert cfg["enabled"] is True
    assert "tools" not in cfg


def test_sampling_manifest_rejects_non_boolean_and_unknown_fields(tmp_path):
    import pytest
    import hermes_yaml as yaml
    for sampling in (None, [], {"enabled": "false"}, {"enabled": 0}, {"model": "other"}):
        path = tmp_path / "manifest.yaml"
        path.write_text(yaml.safe_dump({
            "manifest_version": 1, "name": "demo", "description": "Demo",
            "transport": {"type": "stdio", "command": "demo"}, "sampling": sampling,
        }))
        with pytest.raises(mcp_catalog.CatalogError, match="sampling"):
            mcp_catalog._parse_manifest(path)


def test_omitted_sampling_preserves_existing_client_default(tmp_path):
    import hermes_yaml as yaml
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump({
        "manifest_version": 1, "name": "demo", "description": "Demo",
        "transport": {"type": "stdio", "command": "demo"},
    }))
    assert "sampling" not in mcp_catalog._build_server_config(mcp_catalog._parse_manifest(path), None)
