"""Regression test for chat --provider accepting named custom providers.

The CLI should not reject config-defined custom provider names at argparse
parse-time. This test intentionally exercises the real CLI entrypoint via a
subprocess because the previous failure happened before runtime provider
resolution, inside argument parsing.
"""

from __future__ import annotations

import subprocess
import sys


def test_chat_help_accepts_named_custom_provider() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "chat",
            "--provider",
            "claude-max-proxy",
            "--help",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Inference provider" in result.stdout
