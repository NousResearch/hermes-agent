from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "mcp"
    / "screenpipe"
    / "scripts"
    / "screenpipe_helper.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("screenpipe_skill", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_install_mcp_server_writes_config(tmp_path: Path):
    mod = load_module()
    config_path = tmp_path / "config.yaml"

    result = mod.install_mcp_server(config_path=config_path)

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert result["success"] is True
    assert saved["mcp_servers"]["screenpipe"]["command"] == "npx"
    assert saved["mcp_servers"]["screenpipe"]["args"] == ["-y", "screenpipe-mcp"]


def test_install_mcp_server_rejects_existing_without_force(tmp_path: Path):
    mod = load_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"mcp_servers": {"screenpipe": {"command": "python", "args": ["other.py"]}}}),
        encoding="utf-8",
    )

    with pytest.raises(mod.ScreenpipeError):
        mod.install_mcp_server(config_path=config_path)


def test_search_uses_screenpipe_search_endpoint(monkeypatch):
    mod = load_module()
    seen = {}

    def fake_request(url, params=None):
        seen["url"] = url
        seen["params"] = params
        return {"data": [{"id": "row-1"}]}

    monkeypatch.setattr(mod, "_json_request", fake_request)

    result = mod.search("meeting notes", content_type="ocr", limit=5)

    assert result["success"] is True
    assert seen["url"].endswith("/search")
    assert seen["params"]["q"] == "meeting notes"
    assert seen["params"]["content_type"] == "ocr"
    assert seen["params"]["limit"] == 5


def test_doctor_reports_api_and_existing_mcp(monkeypatch, tmp_path: Path):
    mod = load_module()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"mcp_servers": {"screenpipe": {"command": "npx", "args": ["-y", "screenpipe-mcp"]}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "_json_request", lambda url, params=None: {"status": "ok"})

    result = mod.doctor(config_path=config_path)

    assert result["success"] is True
    assert result["api_reachable"] is True
    assert result["mcp_configured"] is True
    assert result["health"]["status"] == "ok"
