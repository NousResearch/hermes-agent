"""Tests for ACP Registry metadata shipped with Hermes."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "acp_registry" / "agent.json"
ICON = ROOT / "acp_registry" / "icon.svg"
FORBIDDEN_MANIFEST_KEYS = {"schema_version", "display_name"}
ALLOWED_DISTRIBUTIONS = {"binary", "npx", "uvx"}


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _pyproject_version() -> str:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["version"]


def test_agent_json_matches_official_registry_required_fields():
    data = _manifest()

    assert FORBIDDEN_MANIFEST_KEYS.isdisjoint(data)
    assert data["id"] == "hermes-agent"
    assert re.fullmatch(r"[a-z][a-z0-9-]*", data["id"])
    assert data["name"] == "Hermes Agent"
    assert data["description"]
    assert data["repository"] == "https://github.com/NousResearch/hermes-agent"
    assert data["website"].startswith("https://hermes-agent.nousresearch.com/")
    assert data["authors"] == ["Nous Research"]
    assert data["license"] == "MIT"
    assert set(data["distribution"]) <= ALLOWED_DISTRIBUTIONS


def test_agent_json_uses_npx_distribution_without_local_command_fields():
    data = _manifest()

    assert set(data["distribution"]) == {"npx"}
    assert set(data["distribution"]["npx"]) == {"package"}
    assert data["distribution"]["npx"]["package"] == (
        f"@nousresearch/hermes-agent-acp@{data['version']}"
    )
    assert "type" not in data["distribution"]
    assert "command" not in data["distribution"]
    assert "args" not in data["distribution"]


def test_agent_json_version_matches_pyproject():
    assert _manifest()["version"] == _pyproject_version()


def test_npm_launcher_versions_match_pyproject_and_manifest():
    version = _pyproject_version()
    package = json.loads(
        (ROOT / "packages" / "hermes-agent-acp" / "package.json").read_text(encoding="utf-8")
    )
    launcher = (ROOT / "packages" / "hermes-agent-acp" / "bin" / "hermes-agent-acp.js").read_text(
        encoding="utf-8"
    )

    assert package["version"] == version
    assert f"const HERMES_AGENT_VERSION = '{version}';" in launcher
    assert _manifest()["distribution"]["npx"]["package"] == (
        f"@nousresearch/hermes-agent-acp@{version}"
    )


def test_icon_svg_is_16x16_current_color():
    root = ET.fromstring(ICON.read_text(encoding="utf-8"))

    assert root.attrib["viewBox"] == "0 0 16 16"
    assert root.attrib["width"] == "16"
    assert root.attrib["height"] == "16"


def test_icon_svg_has_no_hardcoded_colors_or_gradients():
    text = ICON.read_text(encoding="utf-8")

    assert "linearGradient" not in text
    assert "radialGradient" not in text
    assert "url(#" not in text
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", text)

    root = ET.fromstring(text)
    for element in root.iter():
        for attr in ("fill", "stroke"):
            value = element.attrib.get(attr)
            if value is not None:
                assert value in {"currentColor", "none"}
