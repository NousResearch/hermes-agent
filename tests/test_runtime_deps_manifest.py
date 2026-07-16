"""Invariant tests for runtime-deps.json.

Behavior contract, NOT a change-detector. We assert the *shape* and
*relationships* of the manifest (it parses, has the right schema, the
python/node entries exist and have well-formed version strings) — never
the exact version values, which are expected to change over time.

See AGENTS.md "Don't write change-detector tests" and
docs/updater-world.md §2.6.
"""

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "runtime-deps.json"

_VERSION_RE = re.compile(r"^\d+(\.\d+)?$")


@pytest.fixture
def manifest():
    """Load and parse runtime-deps.json."""
    return json.loads(MANIFEST_PATH.read_text())


class TestManifestShape:
    def test_manifest_file_exists(self):
        assert MANIFEST_PATH.exists(), f"{MANIFEST_PATH} not found"

    def test_parses_as_json(self, manifest):
        assert isinstance(manifest, dict)

    def test_schema_is_one(self, manifest):
        assert manifest.get("schema") == 1

    def test_has_python_entry(self, manifest):
        assert "python" in manifest
        assert isinstance(manifest["python"], dict)

    def test_has_node_entry(self, manifest):
        assert "node" in manifest
        assert isinstance(manifest["node"], dict)


class TestVersionFields:
    def test_python_version_well_formed(self, manifest):
        ver = manifest["python"].get("version", "")
        assert ver, "python.version is empty"
        assert _VERSION_RE.match(ver), f"python version {ver!r} doesn't match ^\\d+(\\.\\d+)?$"

    def test_node_version_well_formed(self, manifest):
        ver = manifest["node"].get("version", "")
        assert ver, "node.version is empty"
        assert _VERSION_RE.match(ver), f"node version {ver!r} doesn't match ^\\d+(\\.\\d+)?$"

    def test_python_source_present(self, manifest):
        source = manifest["python"].get("source", "")
        assert source, "python.source is empty"

    def test_node_floor_present(self, manifest):
        floor = manifest["node"].get("floor", "")
        assert floor, "node.floor is empty"

    def test_node_floor_reason_present(self, manifest):
        reason = manifest["node"].get("floor_reason", "")
        assert reason, "node.floor_reason is empty"


class TestDepEntries:
    def test_uv_entry_present(self, manifest):
        assert "uv" in manifest
        assert manifest["uv"].get("channel"), "uv.channel is empty"

    def test_chromium_entry_present(self, manifest):
        assert "chromium" in manifest
        assert manifest["chromium"].get("on_demand") is True

    def test_ffmpeg_entry_present(self, manifest):
        assert "ffmpeg" in manifest
        assert manifest["ffmpeg"].get("on_demand") is True

    def test_ripgrep_entry_present(self, manifest):
        assert "ripgrep" in manifest
        assert manifest["ripgrep"].get("bundled") is True
