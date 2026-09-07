"""The read-only tools summary works in pipes; the configuration UI does not."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("summary", [True, False], ids=["summary", "interactive"])
def test_tools_cli_without_terminal(tmp_path, summary):
    config = tmp_path / "config.yaml"
    original = "platform_toolsets:\n  cli: [terminal]\n"
    config.write_text(original, encoding="utf-8")
    command = [sys.executable, "-m", "hermes_cli.main", "tools"]
    if summary:
        command.append("--summary")

    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "HERMES_HOME": str(tmp_path)},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )

    if summary:
        assert result.returncode == 0, result.stderr
        assert "Tool Summary" in result.stdout
        assert "Terminal" in result.stdout
    else:
        assert result.returncode == 1
        assert "requires an interactive terminal" in result.stderr
    assert config.read_text(encoding="utf-8") == original
