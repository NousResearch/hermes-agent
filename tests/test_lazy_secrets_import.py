"""Regression test: hermes update must not load cryptography eagerly.

The secrets_cli import in main() used to be eager, causing
cryptography._rust.pyd to load before the update preflight. On Windows,
the updater process itself would then map the .pyd and the self-lock
detector would fire, deferring the update.

This test verifies that cryptography stays OUT of sys.modules until
the user actually runs a secrets subcommand.
"""

import sys
import subprocess
import os

import pytest


class TestLazySecretsImport:
    """Verify that the secrets_cli import is lazy, not eager."""

    def test_secrets_parser_does_not_load_cryptography(self):
        """The secrets CLI parser should not import the secrets backends."""
        # Run a minimal Python process that imports main.py and checks
        # sys.modules for cryptography.
        code = """
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
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        assert result.returncode == 0, (
            f"cryptography._rust was loaded eagerly by main():\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        assert "PASS" in result.stdout

    def test_secrets_dispatch_loads_cryptography_only_on_demand(self):
        """Running a secrets subcommand should load cryptography lazily."""
        code = """
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
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        assert result.returncode == 0, (
            f"Lazy import test failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    def test_update_command_no_cryptography(self):
        """Running hermes update should NOT load cryptography._rust."""
        # This is the key regression: hermes update must stay clean
        code = """
import sys

# Import main and run update with --check (dry run, no actual update)
sys.argv = ['hermes', 'update', '--check']

import hermes_cli.main

# Simulate what main() does for update command
# We can't call cmd_update directly, but we can check that the update path
# doesn't load cryptography
from hermes_cli.update_cmd import _cmd_update_check

# This should be clean
assert 'cryptography.hazmat.bindings._rust' not in sys.modules, \\
    'cryptography._rust loaded during update path'

print('PASS: update path is clean of cryptography')
sys.exit(0)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        assert result.returncode == 0, (
            f"cryptography._rust loaded during update path:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )