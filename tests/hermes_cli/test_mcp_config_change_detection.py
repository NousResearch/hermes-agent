"""Unit tests for ``hermes_cli.mcp_startup.detect_mcp_servers_change`` — the
shared change-detection behind both the TUI and gateway MCP config watchers."""

import yaml

from hermes_cli.mcp_startup import detect_mcp_servers_change


def _write_config(path, doc):
    path.write_text(yaml.dump(doc), encoding="utf-8")
    return path.stat().st_mtime


def test_unchanged_mtime_takes_fast_path(tmp_path):
    cfg = tmp_path / "config.yaml"
    mtime = _write_config(cfg, {"mcp_servers": {"fs": {"command": "npx"}}})
    changed, new_mtime, servers = detect_mcp_servers_change(
        cfg, mtime, {"fs": {"command": "npx"}}
    )
    assert changed is False
    assert new_mtime == mtime
    assert servers == {"fs": {"command": "npx"}}


def test_added_server_triggers_change(tmp_path):
    cfg = tmp_path / "config.yaml"
    _write_config(cfg, {"mcp_servers": {"gh": {"url": "https://mcp.example.com"}}})
    changed, _mtime, servers = detect_mcp_servers_change(cfg, 0.0, {})
    assert changed is True
    assert servers == {"gh": {"url": "https://mcp.example.com"}}


def test_removed_server_triggers_change(tmp_path):
    cfg = tmp_path / "config.yaml"
    _write_config(cfg, {"mcp_servers": {}})
    changed, _mtime, servers = detect_mcp_servers_change(cfg, 0.0, {"gh": {"url": "x"}})
    assert changed is True
    assert servers == {}


def test_other_section_edit_is_not_a_change(tmp_path):
    cfg = tmp_path / "config.yaml"
    _write_config(
        cfg, {"mcp_servers": {"fs": {"command": "npx"}}, "model": {"default": "x"}}
    )
    # mtime forced stale so the parse runs, but mcp_servers is identical.
    changed, new_mtime, servers = detect_mcp_servers_change(
        cfg, 0.0, {"fs": {"command": "npx"}}
    )
    assert changed is False
    # mtime is advanced so the same file isn't re-parsed on every tick.
    assert new_mtime == cfg.stat().st_mtime
    assert servers == {"fs": {"command": "npx"}}


def test_missing_file_reports_no_change(tmp_path):
    cfg = tmp_path / "does_not_exist.yaml"
    changed, new_mtime, servers = detect_mcp_servers_change(cfg, 123.0, {"a": {}})
    assert changed is False
    assert new_mtime == 123.0
    assert servers == {"a": {}}


def test_invalid_yaml_midwrite_is_no_change_but_advances_mtime(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("mcp_servers: {oops: [unterminated\n", encoding="utf-8")  # invalid
    changed, new_mtime, servers = detect_mcp_servers_change(cfg, 0.0, {"a": {}})
    assert changed is False
    # mtime advances so a mid-write/broken file isn't re-parsed every tick; the
    # next good write bumps mtime again and is picked up.
    assert new_mtime == cfg.stat().st_mtime
    assert servers == {"a": {}}
