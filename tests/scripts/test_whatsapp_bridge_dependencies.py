"""Contract for the WhatsApp bridge's resolved security dependency graph."""

import json
from pathlib import Path

from packaging.version import Version


REPO_ROOT = Path(__file__).parents[2]
BRIDGE = REPO_ROOT / "scripts" / "whatsapp-bridge"


VULNERABLE_BODY_PARSER = {Version("1.20.5"), Version("1.20.6")}
MINIMUM_QS = Version("6.16.0")


def test_resolved_dependencies_are_outside_advisory_ranges() -> None:
    manifest = json.loads((BRIDGE / "package.json").read_text())
    lock = json.loads((BRIDGE / "package-lock.json").read_text())
    packages = lock["packages"]

    assert "body-parser" not in manifest["overrides"]

    body_parser_versions = [
        Version(package["version"])
        for path, package in packages.items()
        if path.endswith("/body-parser") or path == "node_modules/body-parser"
    ]
    assert body_parser_versions
    assert all(version not in VULNERABLE_BODY_PARSER for version in body_parser_versions)

    qs_versions = [
        Version(package["version"])
        for path, package in packages.items()
        if path == "node_modules/qs" or path.endswith("/node_modules/qs")
    ]
    assert qs_versions
    assert all(version >= MINIMUM_QS for version in qs_versions)
