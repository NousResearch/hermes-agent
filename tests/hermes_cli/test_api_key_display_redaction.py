"""Tests for API key display redaction in CLI configuration.

Verifies that show_config does not expose any API key fragment.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _make_minimal_hermes_cli(monkeypatch):
    """Create a minimally-initialized HermesCLI for testing.

    Avoids the full constructor complexity by directly setting the
    attributes that show_config reads.
    """
    # Prevent any real env-based config loading
    monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", "/tmp")
    monkeypatch.setenv("TERMINAL_TIMEOUT", "60")

    # Create a bare object and set just the attributes show_config reads
    import cli
    obj = cli.HermesCLI.__new__(cli.HermesCLI)
    # Minimal attributes needed by show_config
    obj.api_key = "sk-test-api-key-1234567890abcdef"
    obj.model = "test-model"
    obj.base_url = "https://test.example.com"
    obj.max_turns = 90
    obj.enabled_toolsets = ["web", "terminal"]
    obj.verbose = False
    obj.session_start = __import__("datetime").datetime.now()
    # Compression setting
    obj.compact = False
    return obj


def test_api_key_config_display_is_fully_redacted(capsys, monkeypatch):
    """show_config must display '[set]' not any key fragment."""
    import cli

    obj = _make_minimal_hermes_cli(monkeypatch)
    # show_config reads self.api_key — it should be redacted
    obj.show_config()
    captured = capsys.readouterr()

    assert "[set]" in captured.out, "API key display should show [set]"
    assert "sk-test-api-key" not in captured.out, "No part of the API key should appear in output"
    assert "...sk-test" not in captured.out, "No suffix fragment of the key should appear"


def test_api_key_config_not_set_shows_not_set(capsys, monkeypatch):
    """When api_key is empty, show_config should display 'Not set!'."""
    obj = _make_minimal_hermes_cli(monkeypatch)
    obj.api_key = ""
    obj.show_config()
    captured = capsys.readouterr()
    assert "Not set!" in captured.out


def test_api_key_config_short_key_still_redacted(capsys, monkeypatch):
    """Even a short non-empty key (<= 12 chars) must show as '[set]', never
    the value or a fragment — regression for the old len()>12 threshold
    which mislabeled short-but-real local keys as unset."""
    obj = _make_minimal_hermes_cli(monkeypatch)
    obj.api_key = "sk-short"
    obj.show_config()
    captured = capsys.readouterr()

    assert "[set]" in captured.out, "A non-empty key must display as [set]"
    assert "sk-short" not in captured.out, "No part of the API key should appear in output"
    assert "Not set!" not in captured.out, "A configured key must not display as Not set!"


def test_api_key_config_microsoft_entra_display(capsys, monkeypatch):
    """When api_key is a callable (Entra ID provider), show Microsoft Entra ID."""
    from cli import HermesCLI
    obj = _make_minimal_hermes_cli(monkeypatch)

    # Set api_key to a callable, simulating the Entra ID provider pattern
    def entra_provider():
        return "fake-token"

    # But is_token_provider detects callables... we need to mock that
    # The code checks `is_token_provider(self.api_key)` first
    with patch("agent.azure_identity_adapter.is_token_provider", return_value=True):
        obj.api_key = entra_provider
        obj.show_config()
    captured = capsys.readouterr()
    assert "Microsoft Entra ID" in captured.out
    assert "[set]" not in captured.out
