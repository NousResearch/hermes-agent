"""The in-tree MCP catalog is gone; its CLI and REST entry points stay one release as stubs that
point at ``hermes mcp add`` and work with the catalog modules absent."""

from __future__ import annotations

import argparse
import sys

import hermes_yaml as yaml
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def no_catalog(monkeypatch):
    """Import of either catalog module fails, as it will once the modules are deleted."""
    monkeypatch.setitem(sys.modules, "hermes_cli.mcp_catalog", None)
    monkeypatch.setitem(sys.modules, "hermes_cli.mcp_picker", None)


def _run_cli(argv: list[str]):
    from hermes_cli.mcp_config import mcp_command
    from hermes_cli.subcommands.mcp import build_mcp_parser

    parser = argparse.ArgumentParser(prog="hermes")
    build_mcp_parser(parser.add_subparsers(dest="command"), cmd_mcp=mcp_command)
    args = parser.parse_args(["mcp", *argv])
    return args.func(args)


@pytest.mark.parametrize(("argv", "rc", "hint"), [
    (["install", "deepwiki"], 1, "hermes mcp add deepwiki --url <url>"),
    (["catalog"], None, "hermes mcp add <name> --url <url>"),
    (["picker"], None, "hermes mcp add <name> --url <url>"),
])
def test_catalog_cli_commands_print_the_mcp_add_equivalent(no_catalog, capsys, argv, rc, hint):
    assert _run_cli(argv) == rc
    out = capsys.readouterr().out
    assert "MCP catalog was removed" in out
    assert hint in out


def test_bare_mcp_lists_configured_servers_and_usage(no_catalog, capsys):
    from hermes_constants import get_hermes_home

    (get_hermes_home() / "config.yaml").write_text(
        yaml.safe_dump({"mcp_servers": {"linear": {"url": "https://mcp.linear.app/mcp", "auth": "oauth"}}}),
        encoding="utf-8",
    )
    assert _run_cli([]) is None
    out = capsys.readouterr().out
    assert "linear" in out
    assert "hermes mcp add <name> --url <endpoint>" in out
    assert "catalog" not in out


def test_rest_catalog_routes_answer_old_clients(no_catalog):
    from hermes_cli.web_server import _SESSION_TOKEN, app
    from hermes_constants import get_hermes_home

    headers = {"X-Hermes-Session-Token": _SESSION_TOKEN}
    with TestClient(app) as client:
        listed = client.get("/api/mcp/catalog", headers=headers, params={"profile": "default"})
        install = client.post("/api/mcp/catalog/install", headers=headers,
                              json={"name": "demo", "env": {"DEMO_API_KEY": "must-not-land"}})

    assert listed.status_code == 200
    assert listed.json() == {"entries": [], "diagnostics": []}
    assert install.status_code == 410
    assert "hermes mcp add" in install.json()["detail"]
    env_path = get_hermes_home() / ".env"
    assert not env_path.exists() or "must-not-land" not in env_path.read_text(encoding="utf-8")
