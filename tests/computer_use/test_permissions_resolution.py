"""Computer Use readiness resolves PM state even under a thin GUI PATH."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def pm_driver(tmp_path, monkeypatch):
    """Publish temp PM facts; no acquisition or native permission changes."""
    import pm
    from pm import paths
    from pm.store import tree_digest

    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    monkeypatch.setattr(pm, "ensure", MagicMock(side_effect=AssertionError("status must not install")))
    package, target = pm.get_package("cua-driver"), pm.current_target()
    lock = pm.Lockfile(paths.lockfile_path())
    version = lock.version(package.name)
    assert version is not None
    root = paths.store_root()
    entry = root / package.store_entry(version, target)
    binary = package.binary(entry, target)
    assert binary is not None
    binary.parent.mkdir(parents=True)
    binary.write_bytes(Path(sys.executable).read_bytes())
    binary.chmod(0o755)
    pm.Facts(paths.facts_path()).record(
        package.name, version, entry.name, package.env(entry, target), root,
        target=target, artifacts=[a["sha256"] for a in lock.artifacts(package.name, target)],
        digest=tree_digest(entry),
    )
    return binary


@pytest.mark.platforms("linux", "macos", "windows")
def test_status_finds_pm_driver_when_path_omits_it(pm_driver, monkeypatch):
    """Desktop status probes the same selected binary as the runtime, without installing."""
    from pm import paths
    from tools.computer_use import permissions

    monkeypatch.setenv("PATH", "")
    before = paths.facts_path().read_bytes()
    check = {"label": "Driver", "status": "pass", "message": "Driver is healthy"}
    outputs = {
        ("--version",): "cua-driver fixture",
        ("doctor", "--json"): json.dumps({"ok": True, "probes": [check]}),
        ("permissions", "status", "--json"): json.dumps({
            "accessibility": True, "screen_recording": True,
            "screen_recording_capturable": True,
        }),
    }

    def run(binary, *args, timeout):
        assert binary == str(pm_driver)
        return subprocess.CompletedProcess([binary, *args], 0, stdout=outputs[args], stderr="")

    with patch.object(permissions, "_run", side_effect=run):
        status = permissions.computer_use_status()

    assert status["installed"] is True
    assert status["version"] == outputs[("--version",)]
    assert status["checks"] == [check]
    assert status["ready"] is True
    assert status["error"] is None
    assert paths.facts_path().read_bytes() == before


def test_status_missing_pm_binary_is_unknown(pm_driver):
    from tools.computer_use import permissions

    pm_driver.unlink()
    with patch.object(permissions, "_run") as run:
        status = permissions.computer_use_status()

    run.assert_not_called()
    assert status["installed"] is False
    assert status["ready"] is None
    assert status["version"] is None
    assert status["checks"] == []
