"""Windows Desktop handoff retry policy behavior."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
RETRY_POLICY = REPO_ROOT / "scripts" / "desktop-update" / "retry-policy.ps1"


@pytest.mark.windows_only
def test_retry_policy_distinguishes_self_lock_deferral(tmp_path: Path) -> None:
    install_root = tmp_path / "hermes-agent"
    install_root.mkdir()
    marker = install_root / ".update-incomplete"

    policy = str(RETRY_POLICY).replace("'", "''")
    root = str(install_root).replace("'", "''")
    command = f"""
        . '{policy}'
        $withoutMarker = @(
            (Test-HermesUpdateShouldRetry -ExitCode 0 -InstallRoot '{root}'),
            (Test-HermesUpdateShouldRetry -ExitCode 1 -InstallRoot '{root}'),
            (Test-HermesUpdateShouldRetry -ExitCode 2 -InstallRoot '{root}')
        )
        New-Item -ItemType File -Path (Join-Path '{root}' '.update-incomplete') | Out-Null
        $withMarker = Test-HermesUpdateShouldRetry -ExitCode 2 -InstallRoot '{root}'
        @{{ withoutMarker = $withoutMarker; withMarker = $withMarker }} |
            ConvertTo-Json -Compress
    """
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {
        "withoutMarker": [False, True, False],
        "withMarker": True,
    }
    assert marker.exists()
