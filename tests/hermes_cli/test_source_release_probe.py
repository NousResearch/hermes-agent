"""A desktop check must not reinterpret unreadable channel config as main."""
from pathlib import Path
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("content", ["update: [", "- not-a-mapping", b"\xff"])
def test_invalid_config_refuses_desktop_channel_probe(tmp_path, content):
    home = tmp_path / "home"
    home.mkdir()
    config = home / "config.yaml"
    config.write_bytes(content.encode() if isinstance(content, str) else content)
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.source_releases", "--install-root", str(root)],
        cwd=root, env={**os.environ, "HERMES_HOME": str(home), "HERMES_IGNORE_USER_CONFIG": "0"},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert '"channel": "main"' not in result.stdout
    assert "invalid" in result.stderr
    assert config.read_bytes() == (content.encode() if isinstance(content, str) else content)
