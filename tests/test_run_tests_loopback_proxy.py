"""Contracts enforced by the canonical test runner's clean environment."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_canonical_runner_bypasses_loopback_proxies():
    """Local fixture servers must never inherit an OS-level proxy route."""
    runner = (PROJECT_ROOT / "scripts" / "run_tests.sh").read_text(encoding="utf-8")
    no_proxy_line = next(
        line.strip() for line in runner.splitlines() if line.strip().startswith("NO_PROXY=")
    )
    # Drop a trailing shell line-continuation backslash (e.g. `::1 \`).
    no_proxy_value = no_proxy_line.split("=", 1)[1].rstrip(" \\")
    no_proxy = {
        host.strip() for host in no_proxy_value.split(",") if host.strip()
    }

    assert {"127.0.0.1", "localhost", "::1"} <= no_proxy
