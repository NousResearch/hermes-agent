"""`hermes dump` must report the MCP servers that are actually configured.

The ``features:`` line read the legacy nested ``mcp.servers`` path, while MCP
server definitions live at top-level ``mcp_servers`` — so every dump printed
``mcp_servers: 0`` however many servers were configured. That line is what
``hermes debug share`` hands to support, so it read as "no MCP servers
configured" on every machine.

The count comes from ``mcp_config._get_mcp_servers``, the accessor ``hermes mcp
list`` uses, so the number cannot drift from the list again.
"""

from types import SimpleNamespace


def _mcp_count(out: str) -> str:
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("mcp_servers:"):
            return stripped.split(":", 1)[1].strip()
    raise AssertionError(f"no 'mcp_servers' line in dump output:\n{out}")


def _dump_with(monkeypatch, capsys, tmp_path, config) -> str:
    from hermes_cli import dump
    from hermes_cli.config import get_hermes_home

    monkeypatch.setattr(dump, "get_project_root", lambda: tmp_path / "noproject")
    monkeypatch.setattr(dump, "load_config", lambda: config)
    get_hermes_home().mkdir(parents=True, exist_ok=True)
    dump.run_dump(SimpleNamespace(show_keys=False))
    return capsys.readouterr().out


def test_dump_counts_configured_mcp_servers(monkeypatch, capsys, tmp_path):
    out = _dump_with(
        monkeypatch, capsys, tmp_path,
        {
            "mcp_servers": {
                "coingecko": {"url": "https://example.com/mcp"},
                "cua": {"command": "cua-driver"},
                "phone": {"command": "python", "enabled": False},
            }
        },
    )
    assert _mcp_count(out) == "3"


def test_dump_reports_zero_when_mcp_servers_absent(monkeypatch, capsys, tmp_path):
    # A bare `mcp_servers:` key in YAML arrives as None; counting it must not raise.
    out = _dump_with(monkeypatch, capsys, tmp_path, {"mcp_servers": None})
    assert _mcp_count(out) == "0"
