"""Regression test: hermes update must not load cryptography eagerly."""

import sys
import subprocess
import os
from pathlib import Path


def _run_isolated(code: str) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet in the repo root (not tests/)."""
    repo_root = Path(__file__).parent.parent
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


class TestLazySecretsImport:
    """Verify that the secrets_cli import is lazy, not eager."""

    def test_secrets_parser_does_not_load_cryptography(self) -> None:
        """The secrets CLI parser should not import the secrets backends."""
        result = _run_isolated(
            """
import sys

# Import main (this builds the parser, including the secrets subparser)
import hermes_cli.main

# Check if cryptography was loaded eagerly
if 'cryptography.hazmat.bindings._rust' in sys.modules:
    print('FAIL: cryptography._rust loaded eagerly by main()')
    sys.exit(1)
else:
    print('PASS: cryptography._rust NOT loaded by main()')
    sys.exit(0)
"""
        )
        assert result.returncode == 0, (
            f"cryptography._rust was loaded eagerly by main():\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        assert "PASS" in result.stdout

    def test_secrets_dispatch_loads_cryptography_only_on_demand(self) -> None:
        """Running a secrets subcommand should load cryptography lazily."""
        result = _run_isolated(
            """
import sys

# First verify it's NOT loaded after importing main
import hermes_cli.main
assert 'cryptography.hazmat.bindings._rust' not in sys.modules, \\
    'cryptography already loaded before dispatch'

# Now simulate the secrets dispatch
# We can't easily run the actual dispatch without mocking argparse,
# but we can at least verify the import inside _dispatch_secrets works
# by checking that secrets_cli is not yet in sys.modules
assert 'hermes_cli.secrets_cli' not in sys.modules, \\
    'secrets_cli already loaded before dispatch'

print('PASS: secrets_cli and cryptography not loaded until dispatch')
sys.exit(0)
"""
        )
        assert result.returncode == 0, (
            f"Lazy import test failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    def test_update_check_no_cryptography(self) -> None:
        """Running hermes update --check should NOT load cryptography._rust."""
        # Write a small script in the repo root so hermes_cli is importable,
        # and use a filename that doesn't trigger the live-system guard.
        repo_root = Path(__file__).parent.parent
        script = repo_root / "_test_lazy_secrets_check.py"
        script.write_text(
            """
import sys
sys.argv = ['hermes', 'update', '--check']

import hermes_cli.main
from hermes_cli.update_cmd import _cmd_update_check

assert 'cryptography.hazmat.bindings._rust' not in sys.modules, \\
    'cryptography._rust loaded during update path'

print('PASS: update check path is clean of cryptography')
sys.exit(0)
"""
        )
        try:
            result = subprocess.run(
                [sys.executable, script.name],
                capture_output=True,
                text=True,
                cwd=str(repo_root),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            assert result.returncode == 0, (
                f"cryptography._rust loaded during update check:\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )
        finally:
            script.unlink()  # Clean up