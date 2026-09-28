"""Repeated PM environment composition must not overflow cmd's PATH (#126861)."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

from pm.package import compose_env


def test_recomposition_preserves_precedence_without_growing_managed_path():
    diffs = [{"PATH": ["node-bin"]}, {"PATH": ["npm-bin"], "SETTING": "pinned"}]
    inherited = os.pathsep.join(["user-bin", "", "node-bin", "user-bin", "relative/../bin", ""])
    base = {"Path": inherited, "SETTING": "user"}
    expected = os.pathsep.join(["npm-bin", "node-bin", "user-bin", "", "user-bin", "relative/../bin", ""])
    env = compose_env(diffs, base=base)
    assert env == {"Path": expected, "SETTING": "pinned"}
    assert compose_env(diffs, base=env) == env
    assert compose_env([], base=base) == base
    assert base == {"Path": inherited, "SETTING": "user"}
    assert compose_env(diffs, base={"PATH": ""})["PATH"] == os.pathsep.join(["npm-bin", "node-bin"])


@pytest.mark.platforms("windows")
def test_composed_path_keeps_node_resolvable_by_real_cmd(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed for the native lifecycle lookup probe")
    node_dir = str(Path(node).parent)
    # The tool exists and Python can find it, but cmd drops an oversized PATH.
    inherited = os.pathsep.join([node_dir] * (8500 // (len(node_dir) + 1) + 1))
    assert len(inherited) > 8191
    assert shutil.which("node", path=inherited) is not None
    env = compose_env([{"PATH": [node_dir]}], base={**os.environ, "PATH": inherited})
    result = subprocess.run(
        [shutil.which("cmd.exe"), "/d", "/s", "/c", "node --version"],
        cwd=tmp_path, env=env, capture_output=True, text=True, errors="replace", timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().startswith("v")
