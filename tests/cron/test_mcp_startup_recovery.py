"""Real stdio regression: recovery must precede the scheduled agent's tool snapshot.

No model, network service, credentials, or mocked MCP state. The fixture kills
only its own child, exhausts the unchanged retry ladder, then restores service.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("case", ["recover", "delayed-list"])
def test_scheduled_snapshot_dispatches_recovered_stdio_tool(tmp_path, case):
    root = Path(__file__).resolve().parents[2]
    fixture = Path(__file__).with_name("fixtures") / "mcp_recovery.py"
    home = tmp_path / "fixture-home"
    env = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT"}
    }
    env["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(
        [sys.executable, str(fixture), str(root), case, str(home)],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=210,
        check=False,
    )
    assert completed.returncode == 0, (
        completed.stdout[-5000:] + completed.stderr[-5000:]
    )
    result = json.loads((home / "result.json").read_text())
    assert result["completed"], result.get("error")
    assert result["snapshot_present"], result["snapshot_mcp_names"]
    assert result["call_ok"], result["calls"]
    assert "hello scheduled" in result["calls"]["probe"]
    if case == "delayed-list":
        assert result["session_before_tools"]
