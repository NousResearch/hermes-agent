"""Matrix extra preserves Linux E2EE while admitting non-Linux plain clients."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

from packaging.markers import Marker, default_environment
from packaging.requirements import Requirement
import pytest


ROOT = Path(__file__).resolve().parents[1]
CRYPTO = {"python-olm", "pycryptodome", "base58", "unpaddedbase64"}
COMMON = {"mautrix", "aiosqlite", "asyncpg", "aiohttp-socks", "aiohttp"}


@pytest.mark.parametrize("system", ["linux", "darwin", "win32"])
@pytest.mark.parametrize("version", ["3.14.0", "3.15.0"])
def test_matrix_declaration_and_frozen_edges_select_the_same_dependencies(system, version):
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    environment = {**default_environment(), "sys_platform": system,
                   "python_full_version": version, "python_version": version[:4],
                   "extra": "matrix"}
    requirements = [Requirement(spec) for spec in metadata["project"]["optional-dependencies"]["matrix"]]
    selected = {req.name: req for req in requirements
                if req.marker is None or req.marker.evaluate(environment)}
    gate = metadata["tool"]["hermes"]["extras-platforms"].get("matrix")
    assert gate is None or Marker(gate).evaluate(environment)
    assert set(selected) == COMMON
    assert selected["mautrix"].extras == ({"encryption"} if system == "linux" else set())

    package = next(row for row in lock["package"] if row["name"] == "hermes-agent")
    edges = [edge for edge in package["optional-dependencies"]["matrix"]
             if "marker" not in edge or Marker(edge["marker"]).evaluate(environment)]
    assert {edge["name"] for edge in edges} == set(selected)
    assert len(edges) == len(selected)
    for edge in edges:
        assert set(edge.get("extra", [])) == selected[edge["name"]].extras
    declared = [row for row in package["metadata"]["requires-dist"]
                if row["name"] in COMMON and "extra == 'matrix'" in row.get("marker", "")
                and Marker(row["marker"]).evaluate(environment)]
    assert len(declared) == len(selected)
    for row in declared:
        requirement = selected[row["name"]]
        assert row["specifier"] == str(requirement.specifier)
        assert set(row.get("extras", [])) == requirement.extras


@pytest.mark.parametrize("system, target", [
    ("linux", "x86_64-unknown-linux-gnu"),
    ("darwin", "aarch64-apple-darwin"),
    ("win32", "x86_64-pc-windows-msvc"),
])
def test_frozen_matrix_sync_keeps_crypto_only_on_linux(system, target, tmp_path):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to exercise the frozen dependency plan")
    # Exported transitive requirements can lose their parent's platform marker.
    # Resolve the frozen graph for each target instead of reinterpreting an export.
    environment = {**os.environ,
                   "UV_PROJECT_ENVIRONMENT": str(tmp_path / "environment"),
                   "MACOSX_DEPLOYMENT_TARGET": "14.0"}
    result = subprocess.run(
        [uv, "sync", "--frozen", "--extra", "matrix", "--no-dev",
         "--no-install-project", "--dry-run", "--python", sys.executable,
         "--python-platform", target, "--output-format", "json"],
        cwd=ROOT, env=environment, check=True, capture_output=True, text=True,
        timeout=60,
    )
    plan = json.loads(result.stdout)
    assert plan["dry_run"] is True
    selected = {change["name"] for change in plan["sync"]["changes"]
                if change["action"] == "installed"}
    assert COMMON <= selected
    assert CRYPTO & selected == (CRYPTO if system == "linux" else set())
    assert not (tmp_path / "environment").exists()
