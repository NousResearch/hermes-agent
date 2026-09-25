"""G1 guard rail: the mcp.json editor's whole-map replace must state its
deletion intent explicitly (``removed_keys``), or the Task-2 guard would
re-preserve the map it is trying to clear — and a silent clear would be
exactly the 2026-09-24 loss this whole component prevents."""
import yaml

from hermes_cli.mcp_config import _replace_mcp_servers
from hermes_cli.config import read_raw_config


def _seed_home(tmp_path, monkeypatch, servers):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"mcp_servers": servers}), encoding="utf-8")


def test_empty_map_removes_mcp_servers(tmp_path, monkeypatch):
    _seed_home(tmp_path, monkeypatch, {"github": {"command": "npx", "enabled": True}})
    ok, issues = _replace_mcp_servers({})
    assert ok and issues == []
    assert "mcp_servers" not in read_raw_config()


def test_non_empty_map_replaces_wholesale(tmp_path, monkeypatch):
    _seed_home(tmp_path, monkeypatch, {
        "github": {"command": "npx", "enabled": True},
        "old": {"command": "npx", "enabled": False},
    })
    ok, issues = _replace_mcp_servers({"fresh": {"url": "https://mcp.example", "enabled": True}})
    assert ok and issues == []
    raw = read_raw_config()
    assert set(raw["mcp_servers"]) == {"fresh"}


def test_invalid_entry_rejects_whole_save(tmp_path, monkeypatch):
    _seed_home(tmp_path, monkeypatch, {"github": {"command": "npx", "enabled": True}})
    ok, issues = _replace_mcp_servers({"bad": "not-a-dict"})
    assert not ok and issues
    assert set(read_raw_config()["mcp_servers"]) == {"github"}
