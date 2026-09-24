"""Connect-only bootstrap must not list the agent runtime stages.

The Mac welcome screen's "Connect to existing Hermes" action drives
``install.sh --manifest --desktop-only``. That print exits before any stage
runs. The local Install action keeps ``--include-desktop`` without the flag,
which still lists the agent stages.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"

AGENT_STAGES = ("venv", "python-deps", "node-deps", "path", "config", "setup", "gateway")

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def manifest_names(*args: str) -> list[str]:
    result = subprocess.run(
        ["bash", str(INSTALL_SH), "--manifest", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    return [stage["name"] for stage in payload["stages"]]


def test_desktop_only_manifest_excludes_agent_runtime():
    names = manifest_names("--include-desktop", "--desktop-only")

    assert "repository" in names
    assert "desktop" in names
    assert not (set(names) & set(AGENT_STAGES))


def test_include_desktop_manifest_keeps_agent_runtime():
    names = manifest_names("--include-desktop")

    assert "desktop" in names
    for stage in AGENT_STAGES:
        assert stage in names
