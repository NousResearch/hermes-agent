"""Runtime smoke tests for Docker --user flag guard.

Build the real image and verify the actual runtime behavior:

  1. docker run --user <arbitrary-uid> is rejected with actionable guidance
  2. --user <hermes-uid> (10000) is allowed (supported non-root start)

Root start (the default) is covered by test_main_invocation.py.
"""
from __future__ import annotations

import subprocess


def test_arbitrary_user_uid_rejected(
    built_image: str,
) -> None:
    """docker run --user 1000 must be rejected with actionable guidance."""
    r = subprocess.run(
        ["docker", "run", "--rm", "--user", "1000:1000",
         built_image, "echo", "should_not_reach"],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode != 0, (
        f"container started with arbitrary --user UID unexpectedly: {r.stdout}"
    )
    assert "should_not_reach" not in r.stdout, (
        f"container ran despite --user rejection: {r.stdout}"
    )
    combined = r.stdout + r.stderr
    # Must mention the remediation env vars
    assert "HERMES_UID" in combined or "PUID" in combined, (
        f"rejection message missing remediation guidance: {combined[-500:]}"
    )




def test_user_pinned_to_hermes_uid_works(
    built_image: str,
) -> None:
    """docker run --user 10000:10000 (the hermes UID) must be allowed.

    This is the supported non-root start from #34648 / #34837.
    """
    r = subprocess.run(
        ["docker", "run", "--rm", "--user", "10000:10000",
         built_image, "sh", "-c", "echo OK"],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, (
        f"--user 10000:10000 (hermes UID) was rejected: {r.stderr[-500:]}"
    )
    assert "OK" in r.stdout