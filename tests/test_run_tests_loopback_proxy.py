"""Behavioral contracts for the canonical test runner's clean environment."""

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_canonical_runner_bypasses_loopback_proxies():
    """Every child process must bypass proxies for all loopback spellings."""
    result = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / "run_tests.sh"), "--print-env"],
        check=True,
        capture_output=True,
        text=True,
    )
    emitted = dict(
        line.split("=", 1)
        for line in result.stdout.splitlines()
        if line.startswith(("NO_PROXY=", "no_proxy="))
    )

    expected = {"127.0.0.1", "localhost", "::1"}
    for name in ("NO_PROXY", "no_proxy"):
        assert expected <= {
            host.strip() for host in emitted[name].split(",") if host.strip()
        }
    assert emitted["no_proxy"] == emitted["NO_PROXY"]
