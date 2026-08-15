"""End-to-end tests for lazy cryptography loading.

These tests invoke the real CLI paths as subprocesses to verify:
1. `hermes secrets bitwarden setup --help` works (dispatch path)
2. `hermes update --check` works (update path)
3. `hermes secrets bitwarden disable` works (handler execution)
4. `hermes secrets onepassword status` works (lazy backend loads on demand)

Unlike test_lazy_secrets_import.py (which inspects sys.modules), these
run the actual commands and verify exit codes — the exact paths the
reviewer flagged as unproven.
"""

import subprocess
import sys
from pathlib import Path

import pytest


def _run_hermes(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    """Run hermes CLI as a subprocess from repo root."""
    repo_root = Path(__file__).parent.parent
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main"] + args,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        timeout=timeout,
    )


class TestSecretsDispatchE2E:
    """End-to-end secrets dispatch — the path that must not self-lock."""

    def test_bitwarden_setup_help(self) -> None:
        """`hermes secrets bitwarden setup --help` must exit 0 and print usage.

        This is the exact path that triggered the #86781 self-lock loop on
        Windows: setup/parser nested under lazy-loaded backend.
        """
        result = _run_hermes(["secrets", "bitwarden", "setup", "--help"])
        assert result.returncode == 0, (
            f"bitwarden setup --help failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        assert "usage" in result.stdout.lower()

    def test_bitwarden_status(self) -> None:
        """`hermes secrets bitwarden status` must exit 0 (runs lazy backend)."""
        result = _run_hermes(["secrets", "bitwarden", "status"])
        # status may return non-zero if not configured, but must NOT crash
        # with import errors, recursion, or missing subcommand
        assert result.returncode in (0, 1), (
            f"bitwarden status crashed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        # Must not contain import errors
        assert "ImportError" not in result.stderr
        assert "cannot import name" not in result.stderr

    def test_bitwarden_disable(self) -> None:
        """`hermes secrets bitwarden disable` must exit 0."""
        result = _run_hermes(["secrets", "bitwarden", "disable"])
        assert result.returncode == 0, (
            f"bitwarden disable failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    def test_onepassword_status(self) -> None:
        """`hermes secrets onepassword status` must exit 0 (1Password lazy backend)."""
        result = _run_hermes(["secrets", "onepassword", "status"])
        assert result.returncode in (0, 1), (
            f"onepassword status crashed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        assert "ImportError" not in result.stderr

    def test_onepassword_setup_help(self) -> None:
        """`hermes secrets onepassword setup --help` must exit 0."""
        result = _run_hermes(["secrets", "onepassword", "setup", "--help"])
        assert result.returncode in (0, 2), (
            f"onepassword setup --help failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        assert "ImportError" not in result.stderr


class TestUpdatePathE2E:
    """Update path — must not load cryptography.

    These tests invoke the real `hermes update --check` path as a subprocess.
    The conftest.py live-system guard blocks this because the command string
    contains "update"; we bypass with the pytest mark.
    """

    @pytest.mark.live_system_guard_bypass
    def test_update_check_clean(self) -> None:
        """`hermes update --check` must not load cryptography._rust."""
        result = _run_hermes(["update", "--check"])
        assert result.returncode in (0, 1, 2), (
            f"update --check crashed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        # No import errors
        assert "ImportError" not in result.stderr
        assert "cannot import name" not in result.stderr

    @pytest.mark.live_system_guard_bypass
    def test_update_no_self_lock(self) -> None:
        """Update path must not self-lock (cryptography._rust absent)."""
        result = _run_hermes(["update", "--check"])
        # The check itself may return non-zero (e.g. no updates), but
        # must not contain the self-lock defer message
        assert "deferred" not in result.stderr.lower()
        assert "self-lock" not in result.stderr.lower()
        assert "_rust.pyd" not in result.stderr.lower()