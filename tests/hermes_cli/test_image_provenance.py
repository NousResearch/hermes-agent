"""Tests for hermes_cli/image_provenance.py — dataclass, _invalid helper, and file reader."""

import json
import tempfile
from pathlib import Path


def test_dataclass_fields_and_defaults():
    from hermes_cli.image_provenance import ImageProvenance
    p = ImageProvenance(1, "docker", "compose", "img:v1", "1.0", "abc", "/tmp/x")
    assert p.valid is True
    assert p.error is None
    assert p.deployment_kind == "docker"


def test_invalid_helper():
    from hermes_cli.image_provenance import _invalid, IMAGE_PROVENANCE_PATH
    p = _invalid(IMAGE_PROVENANCE_PATH, "bad json")
    assert p.valid is False
    assert p.error == "bad json"
    assert p.marker_path == str(IMAGE_PROVENANCE_PATH)


def test_read_valid_provenance():
    from hermes_cli.image_provenance import read_image_provenance
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({
            "schema": 1,
            "deployment_kind": "docker",
            "manager": "compose",
            "image": "hermes:latest",
            "version": "1.0.0",
            "revision": "abc123"
        }, f)
        path = Path(f.name)
    try:
        result = read_image_provenance(path)
        assert result.valid is True
        assert result.deployment_kind == "docker"
        assert result.manager == "compose"
    finally:
        path.unlink(missing_ok=True)


def test_read_malformed_provenance():
    from hermes_cli.image_provenance import read_image_provenance
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("not json")
        path = Path(f.name)
    try:
        result = read_image_provenance(path)
        assert result.valid is False
        assert result.error is not None
    finally:
        path.unlink(missing_ok=True)


def test_read_missing_file():
    from hermes_cli.image_provenance import read_image_provenance
    result = read_image_provenance(Path("/tmp/nonexistent_hermes_provenance_test.json"))
    assert result is None
