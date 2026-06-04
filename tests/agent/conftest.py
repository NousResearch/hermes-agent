"""Agent test fixtures.

Autouse fixtures that apply to every test in tests/agent/:
- no_keychain: stubs out _read_claude_code_credentials_from_keychain to return
  None so the real macOS Keychain (which holds live OAuth tokens) doesn't
  interfere with unit tests that mock Path.home or subprocess.run.
  Tests that explicitly want to exercise Keychain behavior should override
  this fixture locally with their own monkeypatch.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def no_keychain(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent macOS Keychain reads from leaking into unit tests."""
    monkeypatch.setattr(
        "agent.anthropic_adapter._read_claude_code_credentials_from_keychain",
        lambda: None,
    )
