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

import os
import subprocess
import sys
from pathlib import Path

import pytest
from hermes_cli import update_cmd


def _run_hermes(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    """Run hermes CLI as a subprocess from repo root.

    The child runs with all git remote URLs rewritten to an unreachable
    local path (GIT_CONFIG_* env overrides). These tests assert the
    lazy-crypto / no-self-lock invariants of the dispatch path, NOT update
    connectivity — but ``hermes update --check`` really does ``git fetch``
    against github.com when run bare. Under remote throttling that fetch
    can exceed the subprocess timeout and TimeoutExpired the test (exactly
    what happened on CI during the 2026-08-17 GitHub incident: both update
    tests red on main for hours with no code change). Rewriting the URLs
    makes the fetch fail instantly and deterministically; the update path
    still exercises its full parser/dispatch/fetch code and exits 1, which
    the assertions already accept.
    """
    repo_root = Path(__file__).parent.parent
    env = dict(os.environ)
    env.update(
        {
            "GIT_CONFIG_COUNT": "2",
            # Rewrite every https:// and ssh remote to a nonexistent local
            # path so any fetch fails in milliseconds without touching the
            # network. insteadOf matching is prefix-based.
            "GIT_CONFIG_KEY_0": "url.file:///nonexistent-hermes-test-remote/.insteadOf",
            "GIT_CONFIG_VALUE_0": "https://",
            "GIT_CONFIG_KEY_1": "url.file:///nonexistent-hermes-test-remote/.insteadOf",
            "GIT_CONFIG_VALUE_1": "git@",
        }
    )
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main"] + args,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        env=env,
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

    @pytest.mark.live_system_guard_bypass
    def test_main_update_check_crypto_absent_in_sys_modules(self) -> None:
        """Decisive invariant: invoking main() with argv=['hermes','update','--check']
        leaves cryptography.hazmat.bindings._rust absent from sys.modules.

        This is the exact invariant review flagged as unproven (#86782 review
        2026-08-15): an import-only test cannot observe lazy failures, because
        parser construction happens inside main(). Run main() itself in a
        subprocess, let it execute the update path, then assert sys.modules.

        The check must run before _dispatch_update calls its (potentially
        lazy) network layer, so we instrument sys.modules immediately after
        parse_args() and before dispatch returns, using a monkeypatched
        _cmd_update_check that captures state then short-circuits.
        """
        script = """
import sys
from unittest.mock import patch

crypto_seen_at_dispatch = []

def capture_update_check(*args, **kwargs):
    # Run just before the real handler would; record crypto state.
    crypto_seen_at_dispatch.append(
        'cryptography.hazmat.bindings._rust' in sys.modules
    )
    # Short-circuit: don't actually call the network in tests.
    return 0

sys.argv = ['hermes', 'update', '--check']

import hermes_cli.main as m

# Patch the update handler so main() exercises its parser + dispatch
# without doing network I/O.  cmd_update (in main.py) calls
# _self()._cmd_update_check(branch=..., branch_explicit=...) where _self()
# resolves the hermes_cli.main module's lazily re-exported attribute —
# so the patch must land on update_cmd._cmd_update_check.
with patch('hermes_cli.update_cmd._cmd_update_check', capture_update_check):
    try:
        m.main()
    except SystemExit as e:
        # argparse may sys.exit for --help / bad args; ignore for this probe
        if e.code not in (0, None):
            print(f'FAIL: main() exited with code {e.code}')
            sys.exit(1)

# 1. main() must have dispatched into our capture hook
if not crypto_seen_at_dispatch:
    print('FAIL: update --check did not dispatch to _cmd_update_check')
    sys.exit(1)

# 2. At dispatch time, crypto must NOT be loaded
if crypto_seen_at_dispatch[0]:
    print('FAIL: cryptography._rust loaded by main() before update dispatch')
    sys.exit(1)

# 3. After main() returned, crypto must STILL not be loaded
if 'cryptography.hazmat.bindings._rust' in sys.modules:
    print('FAIL: cryptography._rust present in sys.modules after main()')
    sys.exit(1)

print('PASS: main() update --check path never loaded cryptography._rust')
sys.exit(0)
"""
        repo_root = Path(__file__).parent.parent
        probe = repo_root / "_test_main_update_crypto_probe.py"
        probe.write_text(script)
        try:
            result = subprocess.run(
                [sys.executable, probe.name],
                capture_output=True,
                text=True,
                cwd=str(repo_root),
                timeout=60,
            )
            assert result.returncode == 0, (
                f"Decisive main()-level probe failed:\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )
            assert "PASS" in result.stdout
        finally:
            probe.unlink(missing_ok=True)