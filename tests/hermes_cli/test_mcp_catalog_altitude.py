"""Altitude defaults stay bounded when its remote tool surface grows."""
from pathlib import Path
import sys

import pytest

from hermes_cli import mcp_catalog as mc

READ_TOOLS = [
    "whoami", "list_workspaces", "list_accounts", "search_transactions",
    "get_bill_settings", "get_receipt_settings", "list_export_options",
    "get_export_status",
]


@pytest.fixture
def altitude(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "optional-mcps/altitude/manifest.yaml"
    monkeypatch.setattr(mc, "installed_servers", dict)
    return mc._parse_manifest(path)


def test_card_default_keeps_unreviewed_tools_disabled(altitude):
    config = mc.card_install_config(altitude)
    assert config["tools"] == {"include": READ_TOOLS}
    assert config["url"] == "https://mcp.altitude.xyz/mcp"
    assert config["auth"] == "oauth"


@pytest.mark.parametrize("probe_succeeds", [False, True])
def test_cli_default_keeps_unreviewed_tools_disabled(altitude, monkeypatch, probe_succeeds):
    advertised = [(name, "") for name in READ_TOOLS + [
        "generate_exports", "download_export", "future_action",
    ]]
    monkeypatch.setattr(mc, "_probe_tools", lambda name: advertised if probe_succeeds else None)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    saved = {}
    monkeypatch.setattr(mc, "load_config", lambda: saved)
    monkeypatch.setattr(mc, "save_config", lambda config: saved.update(config))
    mc._apply_tool_selection(altitude, prior_selection=None)
    assert saved["mcp_servers"]["altitude"]["tools"] == {"include": READ_TOOLS}
