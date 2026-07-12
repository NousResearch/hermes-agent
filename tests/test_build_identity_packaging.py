"""Release artifacts retain an exact Hermes source revision without Git."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
METADATA_SUFFIX = "hermes_cli/_build_metadata.json"


@pytest.mark.integration
def test_wheel_and_sdist_embed_the_build_revision(tmp_path):
    revision_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    revision = revision_result.stdout.strip()
    artifacts = tmp_path / "artifacts"
    env = os.environ.copy()
    env["HERMES_GIT_SHA"] = revision

    build = subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--sdist",
            "--out-dir",
            str(artifacts),
            ".",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert build.returncode == 0, f"uv build failed:\n{build.stderr}"

    wheels = glob.glob(str(artifacts / "*.whl"))
    tarballs = glob.glob(str(artifacts / "*.tar.gz"))
    assert len(wheels) == 1
    assert len(tarballs) == 1

    with zipfile.ZipFile(wheels[0]) as wheel:
        matches = [name for name in wheel.namelist() if name.endswith(METADATA_SUFFIX)]
        assert len(matches) == 1
        wheel_payload = json.loads(wheel.read(matches[0]))
        extract_root = tmp_path / "wheel-root"
        wheel.extractall(extract_root)

    with tarfile.open(tarballs[0]) as sdist:
        matches = [name for name in sdist.getnames() if name.endswith(METADATA_SUFFIX)]
        assert len(matches) == 1
        metadata = sdist.extractfile(matches[0])
        assert metadata is not None
        sdist_payload = json.load(metadata)
        sdist_root = tmp_path / "sdist-root"
        sdist.extractall(sdist_root, filter="data")

    assert wheel_payload == {"source_revision": revision}
    assert sdist_payload == wheel_payload

    probe_env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "HERMES_GIT_SHA"}
    }
    probe_env["PYTHONPATH"] = str(extract_root)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "from hermes_cli.build_info import get_source_revision; "
            "print(get_source_revision())",
        ],
        cwd=tmp_path,
        env=probe_env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == revision

    extracted_sources = [path for path in sdist_root.iterdir() if path.is_dir()]
    assert len(extracted_sources) == 1
    downstream_artifacts = tmp_path / "downstream-artifacts"
    downstream_env = os.environ.copy()
    downstream_env["HERMES_GIT_SHA"] = "89abcdef0123456789abcdef0123456789abcdef"
    downstream_build = subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--out-dir",
            str(downstream_artifacts),
            ".",
        ],
        cwd=extracted_sources[0],
        env=downstream_env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert downstream_build.returncode == 0, downstream_build.stderr
    downstream_wheels = glob.glob(str(downstream_artifacts / "*.whl"))
    assert len(downstream_wheels) == 1
    with zipfile.ZipFile(downstream_wheels[0]) as wheel:
        matches = [name for name in wheel.namelist() if name.endswith(METADATA_SUFFIX)]
        assert len(matches) == 1
        assert json.loads(wheel.read(matches[0])) == {"source_revision": revision}
