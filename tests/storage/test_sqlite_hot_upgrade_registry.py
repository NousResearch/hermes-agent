"""Hot source updates must not split the live SQLite lock registry."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "tests" / "storage" / "sqlite_hot_upgrade_probe.py"


@pytest.mark.parametrize("mode", ["live", "consumer", "stale"])
def test_sqlite_legacy_registry_survives_new_owner_import(mode: str) -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-S" if mode != "consumer" else "-E", str(PROBE), str(ROOT), mode],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if mode == "stale":
        assert "stale-facade isolation: OK" in result.stdout
    else:
        assert "live legacy-to-canonical registry upgrade: OK" in result.stdout
