"""Regression tests for `hermes chat --provider` choices.

The chat CLI provider choices must stay in sync with the canonical provider
registry. Otherwise supported providers can be blocked by argparse before the
runtime provider resolver sees them.
"""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("provider", ["opencode-go", "opencode-zen", "deepseek"])
def test_chat_provider_accepts_canonical_providers(provider):
    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "chat", "--provider", provider, "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, (
        f"provider={provider!r} should be accepted by chat parser\n"
        f"stdout: {result.stdout[:500]}\n"
        f"stderr: {result.stderr[:500]}"
    )


def test_chat_provider_help_includes_new_canonical_providers():
    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "chat", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0
    help_text = result.stdout + result.stderr
    for provider in ("opencode-go", "opencode-zen", "deepseek"):
        assert provider in help_text
