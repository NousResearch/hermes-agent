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


def test_agent_json_uses_uvx_distribution_without_local_command_fields():
    data = _manifest()

    assert set(data["distribution"]) == {"uvx"}
    uvx = data["distribution"]["uvx"]
    # Schema allows {package, args, env}; we use {package, args}.
    assert set(uvx) <= {"package", "args", "env"}
    assert "package" in uvx
    assert uvx["package"] == f"hermes-agent[acp]=={data['version']}"
    assert uvx["args"] == ["hermes-acp"]
    # Old command-shape fields must not leak back in.
    assert "type" not in data["distribution"]
    assert "command" not in data["distribution"]


def test_agent_json_version_matches_pyproject():
    assert _manifest()["version"] == _pyproject_version()


def test_agent_json_pins_uvx_package_to_pyproject_version():
    """The registry CI rejects ``@latest`` and floating pins; the manifest must
    always reference the exact PyPI version listed in pyproject.toml."""
    assert _manifest()["distribution"]["uvx"]["package"] == (
        f"hermes-agent[acp]=={_pyproject_version()}"
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
