"""Harness: docker run <image> [cmd...] invocation patterns.

These tests MUST pass on the current tini-based image AND continue to
pass after the Phase 2 s6 migration. Any behavior drift is a regression.

The harness expects ``built_image`` and ``container_name`` fixtures from
``tests/docker/conftest.py``. When Docker isn't available every test
here is skipped at collection time.
"""
from __future__ import annotations

import subprocess




def test_chat_subcommand_passthrough(built_image: str) -> None:
    """``docker run <image> chat --help`` should exec ``hermes chat --help``.

    Uses ``--help`` so the call doesn't need an upstream model configured.
    """
    r = subprocess.run(
        ["docker", "run", "--rm", built_image, "chat", "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0
    combined = (r.stdout + r.stderr).lower()
    assert "chat" in combined or "usage" in combined




def test_bash_pattern(built_image: str) -> None:
    """``docker run <image> bash -c 'echo ok'`` should exec bash directly."""
    r = subprocess.run(
        ["docker", "run", "--rm", built_image, "bash", "-c", "echo ok"],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0
    assert "ok" in r.stdout


def test_container_exit_code_matches_inner_exit(built_image: str) -> None:
    """The container exit code must match the inner process's exit code.

    Critical for CI: ``docker run <image> hermes batch ...`` returns a
    non-zero status when batch fails. Phase 2 (s6) must preserve this.
    """
    r = subprocess.run(
        ["docker", "run", "--rm", built_image, "sh", "-c", "exit 42"],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 42
